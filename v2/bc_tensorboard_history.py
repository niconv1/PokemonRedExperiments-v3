#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

TAG = "trajectory/all_flags/text_summary"


def load_recorded_keys(history_path: Path):
    keys = set()
    if not history_path.exists():
        return keys
    try:
        with history_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    key = row.get("key")
                    if key:
                        keys.add(str(key))
                except Exception:
                    continue
    except Exception:
        pass
    return keys


def read_snapshot(snapshot_path: Path):
    try:
        return json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def append_jsonl(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, separators=(",", ":"), sort_keys=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


class HistoryWatcher:
    def __init__(self, logdir, snapshot_path, history_path, poll=3.0):
        self.logdir = Path(logdir)
        self.snapshot_path = Path(snapshot_path)
        self.history_path = Path(history_path)
        self.poll = float(poll)
        self.stop = False
        self.accumulators = {}
        self.recorded = load_recorded_keys(self.history_path)
        self.seen = set(self.recorded)
        self.started_at = time.time()

    def request_stop(self, *_):
        self.stop = True

    def event_files(self):
        return sorted(self.logdir.rglob("events.out.tfevents.*"))

    def _accumulator(self, path: Path):
        key = str(path.resolve())
        acc = self.accumulators.get(key)
        if acc is None:
            acc = event_accumulator.EventAccumulator(
                key,
                size_guidance={
                    event_accumulator.TENSORS: 0,
                    event_accumulator.SCALARS: 0,
                    event_accumulator.IMAGES: 0,
                    event_accumulator.HISTOGRAMS: 0,
                },
            )
            self.accumulators[key] = acc
        return acc

    def scan(self, initial=False):
        for path in self.event_files():
            try:
                acc = self._accumulator(path)
                acc.Reload()
                if TAG not in acc.Tags().get("tensors", []):
                    continue
                events = acc.Tensors(TAG)
            except Exception:
                continue

            try:
                run = str(path.parent.relative_to(self.logdir))
            except Exception:
                run = path.parent.name

            for ev in events:
                key = f"{run}:{int(ev.step)}"
                if key in self.seen:
                    continue
                self.seen.add(key)

                # Don't fake historical BC for old Text points on first launch.
                if initial and float(ev.wall_time) < self.started_at - 2.0:
                    continue

                snapshot = read_snapshot(self.snapshot_path)
                if snapshot is None:
                    self.seen.discard(key)
                    continue

                row = {
                    "key": key,
                    "run": run,
                    "step": int(ev.step),
                    "text_wall_time": float(ev.wall_time),
                    "recorded_at": time.time(),
                    "bc": snapshot,
                }
                append_jsonl(self.history_path, row)
                self.recorded.add(key)

                print(
                    f"[BC-TB] step={int(ev.step):,} run={run} "
                    f"targets={snapshot.get('total_targets', 0)} "
                    f"active={snapshot.get('active_targets', 0)} "
                    f"mastered={snapshot.get('mastered_targets', 0)}",
                    flush=True,
                )

    def run(self):
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.scan(initial=True)
        print(
            f"[BC-TB] watching {TAG} | logdir={self.logdir} | "
            f"history={self.history_path}",
            flush=True,
        )

        while not self.stop:
            time.sleep(self.poll)
            self.scan(initial=False)

        print("[BC-TB] stopped", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="runs")
    p.add_argument(
        "--snapshot",
        default=f"/dev/shm/pokemonred_bc_{os.getuid()}.json",
    )
    p.add_argument("--history", default="runs/bc_tensorboard_history.jsonl")
    p.add_argument("--poll", type=float, default=3.0)
    args = p.parse_args()

    HistoryWatcher(
        logdir=Path(args.logdir),
        snapshot_path=Path(args.snapshot),
        history_path=Path(args.history),
        poll=args.poll,
    ).run()


if __name__ == "__main__":
    main()
