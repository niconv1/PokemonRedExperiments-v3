#!/usr/bin/env python3
"""Standalone Phase-1 tests for the lightweight BC manager."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from bc_manager import BCClient


def wait_for(path: Path, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for {path}")


def wait_version(snapshot: Path, minimum: int, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            data = json.loads(snapshot.read_text(encoding="utf-8"))
            if int(data.get("version", -1)) >= minimum:
                return data
        except Exception:
            pass
        time.sleep(0.02)
    raise AssertionError(f"Snapshot did not reach version {minimum}")


def send_shutdown(sock_path: Path) -> None:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        s.sendto(b'{"action":"shutdown"}', str(sock_path))
    finally:
        s.close()


def run() -> None:
    here = Path(__file__).resolve().parent

    with tempfile.TemporaryDirectory(prefix="pokemonred_bc_test_") as tmp:
        tmp = Path(tmp)
        sock = tmp / "bc.sock"
        snap = tmp / "bc_snapshot.json"
        persist = tmp / "breadcrumb_routes_v3.json"

        proc = subprocess.Popen(
            [
                sys.executable,
                str(here / "bc_manager.py"),
                "--serve",
                "--socket", str(sock),
                "--snapshot", str(snap),
                "--persist", str(persist),
                "--quiet",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            wait_for(sock)
            wait_for(snap)

            # Agent 2 discovers a first route.
            a2 = BCClient(
                agent_id=2,
                socket_path=sock,
                snapshot_path=snap,
                writer=True,
            )
            a2.reset_episode()
            for m in [38, 37, 0, 39]:
                a2.observe_map(m)
            assert a2.discover("Entered Blues House")
            s1 = wait_version(snap, 1)

            r1 = s1["targets"]["Entered Blues House"]
            assert r1["route"] == [38, 37, 0, 39], r1
            assert r1["successful_agent_count"] == 1, r1
            assert abs(r1["mastery_factor"] - 1.0) < 1e-9, r1

            # Agent 3 consumes the route. Total reward across the full route
            # must be bounded by the 0.25 route budget.
            a3 = BCClient(
                agent_id=3,
                socket_path=sock,
                snapshot_path=snap,
                writer=True,
            )
            a3.reset_episode()
            reward_total = 0.0
            for m in [38, 37, 0, 39]:
                reward_total += a3.observe_map(m)

            assert 0.0 < reward_total <= 0.2500001, reward_total

            assert a3.discover("Entered Blues House")
            s2 = wait_version(snap, 2)
            r2 = s2["targets"]["Entered Blues House"]
            assert r2["successful_agent_count"] == 2, r2
            assert abs(r2["mastery_factor"] - 0.60) < 1e-9, r2

            # Agent 4 finds a genuinely shorter route. The manager must keep it.
            a4 = BCClient(
                agent_id=4,
                socket_path=sock,
                snapshot_path=snap,
                writer=True,
            )
            a4.reset_episode()
            for m in [38, 39]:
                a4.observe_map(m)
            assert a4.discover("Entered Blues House")
            s3 = wait_version(snap, 3)
            r3 = s3["targets"]["Entered Blues House"]
            assert r3["route"] == [38, 39], r3
            assert r3["successful_agent_count"] == 3, r3
            assert abs(r3["mastery_factor"] - 0.30) < 1e-9, r3

            # Fourth distinct successful agent masters the target.
            a5 = BCClient(
                agent_id=5,
                socket_path=sock,
                snapshot_path=snap,
                writer=True,
            )
            a5.reset_episode()
            for m in [38, 39]:
                a5.observe_map(m)
            assert a5.discover("Entered Blues House")
            s4 = wait_version(snap, 4)
            r4 = s4["targets"]["Entered Blues House"]
            assert r4["successful_agent_count"] == 4, r4
            assert r4["mastered"] is True, r4
            assert abs(r4["mastery_factor"] - 0.0) < 1e-9, r4

            # A mastered target no longer gives route reward.
            a6 = BCClient(
                agent_id=6,
                socket_path=sock,
                snapshot_path=snap,
                writer=True,
            )
            a6.reset_episode()
            mastered_reward = sum(a6.observe_map(m) for m in [38, 39])
            assert mastered_reward == 0.0, mastered_reward

            # Agent 1 read-only mode: discover() must never write.
            a1 = BCClient(
                agent_id=1,
                socket_path=sock,
                snapshot_path=snap,
                writer=False,
                reward_enabled=False,
            )
            a1.reset_episode()
            for m in [1, 2, 3]:
                assert a1.observe_map(m) == 0.0
            assert a1.discover("SHOULD_NOT_EXIST") is False
            time.sleep(0.1)
            final = json.loads(snap.read_text(encoding="utf-8"))
            assert "SHOULD_NOT_EXIST" not in final["targets"]

            # Persistence must exist and contain the same learned target.
            wait_for(persist)
            persisted = json.loads(persist.read_text(encoding="utf-8"))
            assert "Entered Blues House" in persisted["targets"]

            for client in (a1, a2, a3, a4, a5, a6):
                client.close()

            print("====================================================")
            print(" BC RAM MANAGER PHASE 1: ALL TESTS PASSED")
            print("====================================================")
            print("OK central single-writer manager")
            print("OK Unix datagram discovery events")
            print("OK RAM snapshot")
            print("OK shortest observed route")
            print("OK target mastery 1.00 -> 0.60 -> 0.30 -> 0.00")
            print("OK bounded total route reward <= 0.25")
            print("OK mastered target reward = 0")
            print("OK Agent 1 read-only cannot write")
            print("OK atomic persistence")

        finally:
            if proc.poll() is None:
                try:
                    send_shutdown(sock)
                except Exception:
                    pass
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            if proc.stdout is not None:
                remaining = proc.stdout.read()
                if remaining.strip():
                    print("\n[manager output]")
                    print(remaining.strip())


if __name__ == "__main__":
    run()

