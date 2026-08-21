#!/usr/bin/env python3
"""
Pokemon Red AI V3 - Lightweight Breadcrumb RAM Manager

Design goals
------------
* One central process owns all breadcrumb route state.
* PPO workers never lock or rewrite the breadcrumb database.
* Workers send only rare "target discovered" datagrams.
* A compact read-only snapshot is published in /dev/shm (RAM-backed on Linux).
* Clients evaluate breadcrumbs only when the Game Boy map actually changes.
* Agent 1 can remain completely read-only by using writer=False.
* Disk persistence happens only after a new discovery/route update.

This module is intentionally standalone in Phase 1. It does NOT patch the PPO
environment yet. That integration is done only after this module passes its
self-test against the user's stable V3.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = 1
DEFAULT_REWARD_BUDGET = 0.25
DEFAULT_TRACE_LEN = 32
MAX_PACKET = 65535


def _atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically replace a JSON file so readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _collapse_route(route: Iterable[int], max_len: int) -> List[int]:
    """Remove consecutive duplicate maps and keep the most recent max_len maps."""
    clean: List[int] = []
    for raw in route:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if not clean or clean[-1] != value:
            clean.append(value)

    if max_len > 0 and len(clean) > max_len:
        clean = clean[-max_len:]
    return clean


def mastery_factor(successful_agents: int) -> float:
    """
    Target mastery decay used by the previous BC experiment:
      1 distinct successful agent -> 1.00
      2                            -> 0.60
      3                            -> 0.30
      4+                           -> 0.00
    """
    if successful_agents <= 1:
        return 1.0
    if successful_agents == 2:
        return 0.60
    if successful_agents == 3:
        return 0.30
    return 0.0


class BCStore:
    """All mutable breadcrumb knowledge lives only in the manager process."""

    def __init__(self, persist_path: Path, max_route_len: int = DEFAULT_TRACE_LEN):
        self.persist_path = Path(persist_path)
        self.max_route_len = int(max_route_len)
        self.version = 0
        self.targets: Dict[str, Dict[str, Any]] = {}
        self.total_discoveries = 0
        self.loaded_from_disk = False

    def load(self) -> None:
        if not self.persist_path.exists():
            return

        try:
            raw = json.loads(self.persist_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[BC] WARNING: persistence could not be read: {exc}", flush=True)
            return

        targets = raw.get("targets", {})
        if not isinstance(targets, dict):
            return

        cleaned: Dict[str, Dict[str, Any]] = {}
        for target, record in targets.items():
            if not isinstance(target, str) or not isinstance(record, dict):
                continue
            route = _collapse_route(record.get("route", []), self.max_route_len)
            agents = sorted(
                {
                    int(x)
                    for x in record.get("successful_agents", [])
                    if str(x).lstrip("-").isdigit()
                }
            )
            if len(route) < 2:
                continue
            cleaned[target] = {
                "target": target,
                "route": route,
                "successful_agents": agents,
                "discoveries": int(record.get("discoveries", len(agents))),
                "updated_at": float(record.get("updated_at", 0.0)),
            }

        self.targets = cleaned
        self.version = int(raw.get("version", 0))
        self.total_discoveries = int(raw.get("total_discoveries", 0))
        self.loaded_from_disk = True

    def _record_view(self, record: Dict[str, Any]) -> Dict[str, Any]:
        agents = sorted(set(int(x) for x in record.get("successful_agents", [])))
        return {
            "target": record["target"],
            "route": list(record["route"]),
            "successful_agents": agents,
            "successful_agent_count": len(agents),
            "mastery_factor": mastery_factor(len(agents)),
            "mastered": len(agents) >= 4,
            "discoveries": int(record.get("discoveries", 0)),
            "updated_at": float(record.get("updated_at", 0.0)),
        }

    def snapshot(self) -> Dict[str, Any]:
        views = {
            target: self._record_view(record)
            for target, record in sorted(self.targets.items())
        }
        active = sum(1 for record in views.values() if not record["mastered"])
        mastered = len(views) - active
        return {
            "schema": SCHEMA_VERSION,
            "version": self.version,
            "generated_at": time.time(),
            "total_targets": len(views),
            "active_targets": active,
            "mastered_targets": mastered,
            "total_discoveries": self.total_discoveries,
            "targets": views,
        }

    def persist(self) -> None:
        payload = self.snapshot()
        _atomic_json_write(self.persist_path, payload)

    def observe_discovery(
        self,
        *,
        agent_id: int,
        target: str,
        route: Iterable[int],
        timestamp: Optional[float] = None,
    ) -> Tuple[bool, str]:
        target = str(target).strip()
        if not target:
            return False, "empty_target"

        clean = _collapse_route(route, self.max_route_len)
        if len(clean) < 2:
            return False, "route_too_short"

        now = float(timestamp if timestamp is not None else time.time())
        agent_id = int(agent_id)

        record = self.targets.get(target)
        created = record is None
        route_replaced = False
        new_agent = False

        if record is None:
            record = {
                "target": target,
                "route": clean,
                "successful_agents": [],
                "discoveries": 0,
                "updated_at": now,
            }
            self.targets[target] = record
        else:
            current = list(record.get("route", []))
            # Shortest actually observed route wins. For equal length, keep the
            # established route so the target does not oscillate between ties.
            if len(clean) < len(current):
                record["route"] = clean
                route_replaced = True

        agents = set(int(x) for x in record.get("successful_agents", []))
        if agent_id not in agents:
            agents.add(agent_id)
            new_agent = True
        record["successful_agents"] = sorted(agents)
        record["discoveries"] = int(record.get("discoveries", 0)) + 1
        record["updated_at"] = now

        self.total_discoveries += 1
        changed = created or route_replaced or new_agent
        if changed:
            self.version += 1

        reason_bits = []
        if created:
            reason_bits.append("new_target")
        if route_replaced:
            reason_bits.append("shorter_route")
        if new_agent:
            reason_bits.append("new_agent")
        if not reason_bits:
            reason_bits.append("duplicate")

        return changed, "+".join(reason_bits)


class BCManagerServer:
    """Single-writer Unix datagram server."""

    def __init__(
        self,
        *,
        socket_path: Path,
        snapshot_path: Path,
        persist_path: Path,
        max_route_len: int = DEFAULT_TRACE_LEN,
        verbose: bool = True,
    ):
        self.socket_path = Path(socket_path)
        self.snapshot_path = Path(snapshot_path)
        self.persist_path = Path(persist_path)
        self.verbose = bool(verbose)
        self.store = BCStore(self.persist_path, max_route_len=max_route_len)
        self._stop = False
        self._sock: Optional[socket.socket] = None

    def _log(self, text: str) -> None:
        if self.verbose:
            print(f"[BC] {text}", flush=True)

    def publish(self) -> None:
        _atomic_json_write(self.snapshot_path, self.store.snapshot())

    def request_stop(self, *_: Any) -> None:
        self._stop = True

    def handle_packet(self, packet: bytes) -> None:
        try:
            event = json.loads(packet.decode("utf-8"))
        except Exception:
            return

        action = event.get("action")

        if action == "shutdown":
            self._log("shutdown requested")
            self._stop = True
            return

        if action == "publish":
            self.publish()
            return

        if action != "discover":
            return

        try:
            agent_id = int(event["agent_id"])
            target = str(event["target"])
            route = event["route"]
            timestamp = float(event.get("timestamp", time.time()))
        except Exception:
            return

        changed, reason = self.store.observe_discovery(
            agent_id=agent_id,
            target=target,
            route=route,
            timestamp=timestamp,
        )

        if changed:
            self.store.persist()
            self.publish()
            view = self.store.snapshot()["targets"][target]
            self._log(
                f"target={target!r} agent={agent_id} "
                f"route_len={len(view['route'])} "
                f"agents={view['successful_agent_count']} "
                f"mastery={view['mastery_factor']:.2f} "
                f"reason={reason}"
            )

    def serve_forever(self) -> int:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass

        self.store.load()
        self.publish()

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(str(self.socket_path))
        sock.settimeout(0.25)
        self._sock = sock

        try:
            signal.signal(signal.SIGTERM, self.request_stop)
            signal.signal(signal.SIGINT, self.request_stop)
        except ValueError:
            # Unit tests may run in a non-main thread/process.
            pass

        self._log(
            f"ready socket={self.socket_path} "
            f"snapshot={self.snapshot_path} "
            f"persist={self.persist_path} "
            f"targets={len(self.store.targets)}"
        )

        try:
            while not self._stop:
                try:
                    packet = sock.recv(MAX_PACKET)
                except socket.timeout:
                    continue
                except InterruptedError:
                    continue
                self.handle_packet(packet)
        finally:
            try:
                self.store.persist()
                self.publish()
            except Exception as exc:
                self._log(f"WARNING: final persistence failed: {exc}")
            try:
                sock.close()
            finally:
                try:
                    self.socket_path.unlink()
                except FileNotFoundError:
                    pass
            self._log("stopped")

        return 0


class BCClient:
    """
    Very small client intended to live inside a PPO environment worker.

    Important performance rule:
      reward evaluation happens ONLY when the map changes.
    """

    def __init__(
        self,
        *,
        agent_id: int,
        socket_path: Path,
        snapshot_path: Path,
        writer: bool,
        reward_enabled: bool = True,
        reward_budget: float = DEFAULT_REWARD_BUDGET,
        trace_len: int = DEFAULT_TRACE_LEN,
    ):
        self.agent_id = int(agent_id)
        self.socket_path = str(socket_path)
        self.snapshot_path = Path(snapshot_path)
        self.writer = bool(writer)
        self.reward_enabled = bool(reward_enabled)
        self.reward_budget = float(reward_budget)
        self.trace = deque(maxlen=int(trace_len))

        self._sock: Optional[socket.socket] = None
        self._last_map: Optional[int] = None
        self._snapshot_mtime_ns: Optional[int] = None
        self._snapshot_version = -1
        self._routes: Dict[str, Dict[str, Any]] = {}
        self._progress: Dict[str, int] = {}
        self._completed_routes: set[str] = set()
        self._local_seen_targets: set[str] = set()
        self.dropped_events = 0

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None

    def reset_episode(
        self,
        *,
        initial_map: Optional[int] = None,
        already_seen_targets: Optional[Iterable[str]] = None,
    ) -> None:
        self.trace.clear()
        self._progress.clear()
        self._completed_routes.clear()
        self._local_seen_targets = set(str(x) for x in (already_seen_targets or []))
        self._last_map = None

        if initial_map is not None:
            value = int(initial_map)
            self.trace.append(value)
            self._last_map = value

    def _ensure_socket(self) -> socket.socket:
        if self._sock is None:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            sock.setblocking(False)
            self._sock = sock
        return self._sock

    def _send(self, payload: Dict[str, Any]) -> bool:
        if not self.writer:
            return False

        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if len(raw) > 8192:
            self.dropped_events += 1
            return False

        try:
            self._ensure_socket().sendto(raw, self.socket_path)
            return True
        except (BlockingIOError, FileNotFoundError, ConnectionRefusedError, OSError):
            self.dropped_events += 1
            return False

    def discover(self, target: str) -> bool:
        """
        Call only when this episode has just obtained a new *real named flag*.
        The current recent map-transition trace becomes an observed route.
        """
        target = str(target).strip()
        if not target:
            return False

        self._local_seen_targets.add(target)
        route = list(self.trace)
        if len(route) < 2:
            return False

        return self._send(
            {
                "action": "discover",
                "agent_id": self.agent_id,
                "target": target,
                "route": route,
                "timestamp": time.time(),
            }
        )

    def _reload_snapshot_if_changed(self) -> None:
        try:
            stat = self.snapshot_path.stat()
        except FileNotFoundError:
            return

        if self._snapshot_mtime_ns == stat.st_mtime_ns:
            return

        try:
            data = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            return

        if int(data.get("schema", -1)) != SCHEMA_VERSION:
            return

        routes = data.get("targets", {})
        if not isinstance(routes, dict):
            return

        self._routes = routes
        self._snapshot_version = int(data.get("version", -1))
        self._snapshot_mtime_ns = stat.st_mtime_ns

        # Drop progress entries for routes that no longer exist.
        current_targets = set(routes)
        for target in list(self._progress):
            if target not in current_targets:
                self._progress.pop(target, None)

    @staticmethod
    def _advance_progress(route: List[int], progress: int, map_id: int) -> Tuple[int, bool]:
        """
        Advance one route state machine.

        Returns:
            new_progress, advanced_forward

        A mismatch restarts at route[0] when possible. This is intentionally
        simple because routes are tiny and evaluated only on map transitions.
        """
        if len(route) < 2:
            return 0, False

        progress = max(0, min(int(progress), len(route)))

        if progress == 0:
            if map_id == route[0]:
                return 1, False
            return 0, False

        if progress < len(route) and map_id == route[progress]:
            return progress + 1, True

        if map_id == route[0]:
            return 1, False

        return 0, False

    def observe_map(self, map_id: int) -> float:
        """
        Record a map transition and return at most one bounded BC reward.

        Repeated frames on the same map are O(1) and return immediately.
        The snapshot is checked only after an actual map transition.
        """
        map_id = int(map_id)

        if self._last_map == map_id:
            return 0.0

        self._last_map = map_id
        if not self.trace or self.trace[-1] != map_id:
            self.trace.append(map_id)

        self._reload_snapshot_if_changed()

        if not self.reward_enabled or not self._routes:
            return 0.0

        best_reward = 0.0

        for target, record in self._routes.items():
            if target in self._local_seen_targets:
                continue
            if target in self._completed_routes:
                continue

            try:
                factor = float(record.get("mastery_factor", 0.0))
                route = [int(x) for x in record.get("route", [])]
            except Exception:
                continue

            if factor <= 0.0 or len(route) < 2:
                continue

            old_progress = self._progress.get(target, 0)
            new_progress, advanced = self._advance_progress(route, old_progress, map_id)
            self._progress[target] = new_progress

            if not advanced:
                continue

            # The full route can award at most reward_budget before external
            # environment reward scaling. Multiple matching routes do not stack:
            # only the strongest reward for this map transition is returned.
            hop_reward = self.reward_budget * factor / max(1, len(route) - 1)
            best_reward = max(best_reward, hop_reward)

            if new_progress >= len(route):
                self._completed_routes.add(target)

        return best_reward

    @property
    def snapshot_version(self) -> int:
        return self._snapshot_version

    @property
    def known_targets(self) -> int:
        return len(self._routes)


def _send_control(socket_path: Path, action: str) -> bool:
    payload = json.dumps({"action": action}, separators=(",", ":")).encode("utf-8")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.sendto(payload, str(socket_path))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pokemon Red AI V3 breadcrumb RAM manager")
    parser.add_argument("--serve", action="store_true", help="run manager server")
    parser.add_argument("--shutdown", action="store_true", help="ask running manager to stop")
    parser.add_argument("--socket", default="runs/bc_manager.sock")
    parser.add_argument("--snapshot", default=f"/dev/shm/pokemonred_bc_{os.getuid()}.json")
    parser.add_argument("--persist", default="runs/breadcrumb_routes_v3.json")
    parser.add_argument("--max-route-len", type=int, default=DEFAULT_TRACE_LEN)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    socket_path = Path(args.socket)

    if args.shutdown:
        return 0 if _send_control(socket_path, "shutdown") else 1

    if not args.serve:
        parser.error("use --serve or --shutdown")

    server = BCManagerServer(
        socket_path=socket_path,
        snapshot_path=Path(args.snapshot),
        persist_path=Path(args.persist),
        max_route_len=args.max_route_len,
        verbose=not args.quiet,
    )
    return server.serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())

