# benchmarks/benchmark_scenarios/utils.py

import time
import csv
import json
from typing import List, Tuple, Optional
from PyQt5 import QtCore, QtWidgets


class FpsMeter(QtCore.QObject):
    """Approximate FPS meter using QGraphicsScene.changed signal counts."""
    def __init__(self, scene: QtWidgets.QGraphicsScene):
        super().__init__()
        self._count = 0
        self._enabled = False
        self._start = 0.0
        self._stop = 0.0
        scene.changed.connect(self._on_changed)

    def _on_changed(self, *_):
        if self._enabled:
            self._count += 1

    def start(self):
        self._count = 0
        self._start = time.perf_counter()
        self._stop = 0.0
        self._enabled = True

    def stop(self):
        self._enabled = False
        self._stop = time.perf_counter()

    def stats(self) -> Tuple[float, int, float]:
        if self._stop <= self._start:
            return (0.0, 0, 0.0)
        duration = self._stop - self._start
        fps = self._count / duration if duration > 0 else 0.0
        return (fps, self._count, duration)


def wait_until(app: QtWidgets.QApplication, cond, timeout_s: float,
               verbose: bool, tag: str) -> Optional[float]:
    """Pump Qt events until cond() is True or timeout (s)."""
    start = time.perf_counter()
    last_log = start
    while True:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        if cond():
            return time.perf_counter() - start
        now = time.perf_counter()
        if now - start >= timeout_s:
            return None
        if verbose and (now - last_log) >= 1.0:
            print(f"[{tag}] …still waiting ({now - start:.2f}s elapsed)")
            last_log = now


def write_outputs(rows: List[dict], csv_path: str, json_path: str, verbose: bool):
    if not rows:
        return

    # Collect union of all keys
    fieldnames = sorted({k for row in rows for k in row.keys()})

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            # ensure all fields exist
            complete_row = {fn: row.get(fn, None) for fn in fieldnames}
            w.writerow(complete_row)
    if verbose:
        print(f"[output] CSV written: {csv_path}")

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    if verbose:
        print(f"[output] JSON written: {json_path}")
