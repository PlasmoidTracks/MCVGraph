# benchmarks/benchmark_gui.py
#
# GUI benchmark: 3 ScatterPlots on a shared DataSource.
# Measures:
#   - Selection latency: issue select_all() on plot #1 and wait until all 3 plots
#     have fully updated highlights for that source.
#   - Clear latency: broadcast clear for that source and wait until all 3 plots
#     have cleared highlights.
#   - Approx FPS per canvas during each wait window (scene.changed counts / duration).
#
# Option A (no library edits): Polls private state (_highlight_indices) and calls
# private helpers as needed. This is acceptable for a benchmark harness.
#
# Usage (from project root):
#   python -m benchmarks.benchmark_gui --sizes 100 1000 5000 --reps 3 --timeout 10 \
#       --csv bench_results.csv --json bench_results.json --seed 0 --verbose
#
# Notes:
# - Visible windows are shown; run on a desktop environment.
# - Default sizes are small (100, 1000, 5000) per user request.
# - If a run exceeds --timeout seconds, it records DNF for that metric.

# python -m benchmark --scenario simul --sizes 100 1000 5000 --reps 3 --timeout 10 --verbose

import os
import sys
import csv
import json
import argparse
import time
from typing import List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MCVGraph.DataSource import DataSource
from MCVGraph.canvas.Canvas import Canvas
from MCVGraph.graphs.ScatterPlot import ScatterPlot
from MCVGraph.EventType import EventType


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
        """Returns (avg_fps, frames, duration_s)."""
        if self._stop <= self._start:
            return (0.0, 0, 0.0)
        duration = self._stop - self._start
        fps = self._count / duration if duration > 0 else 0.0
        return (fps, self._count, duration)


def wait_until(app: QtWidgets.QApplication, cond, timeout_s: float, verbose: bool, tag: str) -> Optional[float]:
    """Pump Qt events until cond() is True or timeout; returns elapsed seconds or None on DNF."""
    start = time.perf_counter()
    last_log = start
    while True:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        if cond():
            elapsed = time.perf_counter() - start
            return elapsed
        now = time.perf_counter()
        if now - start >= timeout_s:
            return None
        # Optional periodic heartbeat
        if verbose and (now - last_log) >= 1.0:
            print(f"[{tag}] …still waiting ({now - start:.2f}s elapsed)")
            last_log = now


def selection_done_for_source(layers: List[ScatterPlot], source_key: str, expected_count: int) -> bool:
    """All layers have an entry for source_key with exactly expected_count indices."""
    for ly in layers:
        d = getattr(ly, "_highlight_indices", {})
        if source_key not in d:
            return False
        inds, _ = d[source_key]
        try:
            n = len(inds)
        except Exception:
            return False
        if n != expected_count:
            return False
    return True


def cleared_done_for_source(layers: List[ScatterPlot], source_key: str) -> bool:
    """All layers have removed the source_key entry."""
    for ly in layers:
        d = getattr(ly, "_highlight_indices", {})
        if source_key in d:
            return False
    return True


def multi_selection_done(layers: List[ScatterPlot], source_keys: List[str], expected_count: int) -> bool:
    """All layers contain each source_key with expected_count indices."""
    for ly in layers:
        d = getattr(ly, "_highlight_indices", {})
        for sk in source_keys:
            if sk not in d:
                return False
            inds, _ = d[sk]
            try:
                n = len(inds)
            except Exception:
                return False
            if n != expected_count:
                return False
    return True


def multi_cleared_done(layers: List[ScatterPlot], source_keys: List[str]) -> bool:
    """All layers have cleared all given source_keys."""
    for ly in layers:
        d = getattr(ly, "_highlight_indices", {})
        for sk in source_keys:
            if sk in d:
                return False
    return True


def build_three_scatter_canvases(n_points: int, seed: int, verbose: bool):
    """Create 3 Canvas windows with ScatterPlot layers sharing the same DataSource."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_points, 2)).astype(np.float32)

    if verbose:
        print(f"[setup] Creating DataSource with {n_points} points")

    ds = DataSource(data)

    canvases: List[Canvas] = []
    layers: List[ScatterPlot] = []

    for i in range(3):
        c = Canvas(name=f"Canvas #{i+1}")
        # Place windows in a row (best-effort)
        c.move(80 + i * 420, 80)
        c.resize(400, 300)
        c.show()

        sp = ScatterPlot(ds, name=f"S{i+1}", focus=(i == 0))
        c.plot(sp)  # attaches to plot_item and view_box

        canvases.append(c)
        layers.append(sp)

    # give Qt a moment to lay out
    QtWidgets.QApplication.instance().processEvents()
    return canvases, layers


def run_scenario_scatter3(app: QtWidgets.QApplication,
                          sizes: List[int],
                          reps: int,
                          timeout_s: float,
                          seed: int,
                          verbose: bool):
    """Scenario: 3 scatterplots, select-all from first plot, measure propagation & clear."""
    results = []
    for n_points in sizes:
        for rep in range(1, reps + 1):
            if verbose:
                print(f"\n[scenario] 3x Scatter | points={n_points} | rep={rep}")

            canvases, layers = build_three_scatter_canvases(n_points, seed + rep - 1, verbose)
            app.processEvents()

            # FPS meters per canvas
            meters = [FpsMeter(c.plot_widget.scene()) for c in canvases]

            # Selection from first layer
            emitter = layers[0]
            source_key = emitter._source_name()  # private but stable enough for benchmark
            idx_all = np.arange(n_points, dtype=int)

            # --- selection latency ---
            for m in meters:
                m.start()
            if verbose:
                print(f"[select] Emitting select_indices({n_points}) from source='{source_key}'")
            t0 = time.perf_counter()
            emitter.select_indices(idx_all)

            sel_elapsed = wait_until(
                app,
                lambda: selection_done_for_source(layers, source_key, n_points),
                timeout_s=timeout_s,
                verbose=verbose,
                tag="select"
            )
            for m in meters:
                m.stop()

            # --- clear latency ---
            for m in meters:
                m.start()
            if verbose:
                print(f"[clear] Broadcasting clear_highlight for source='{source_key}'")
            emitter.graph.emit_broadcast(EventType.CLEAR_HIGHLIGHT, {"source": source_key})

            clr_elapsed = wait_until(
                app,
                lambda: cleared_done_for_source(layers, source_key),
                timeout_s=timeout_s,
                verbose=verbose,
                tag="clear"
            )
            for m in meters:
                m.stop()

            # collect FPS stats (average fps over the respective phases)
            # For simplicity, we report the selection-window FPS of canvas #1..#3
            fps_sel = [m.stats()[0] for m in meters]  # avg fps ≈ frames/duration
            # We could also report clear FPS; to keep it simple and compact we only
            # keep selection FPS here; extend later if desired.

            # Debug prints
            if sel_elapsed is None:
                print(f"[result] SELECT: DNF (> {timeout_s}s)")
            else:
                print(f"[result] SELECT: {sel_elapsed*1000:.2f} ms")

            if clr_elapsed is None:
                print(f"[result] CLEAR: DNF (> {timeout_s}s)")
            else:
                print(f"[result] CLEAR: {clr_elapsed*1000:.2f} ms")

            print(f"[fps] selection-window avg FPS ~ C1={fps_sel[0]:.1f}, C2={fps_sel[1]:.1f}, C3={fps_sel[2]:.1f}")

            results.append({
                "scenario": "scatter3_select_all_from_first",
                "points": n_points,
                "repetition": rep,
                "select_latency_ms": None if sel_elapsed is None else sel_elapsed * 1000.0,
                "clear_latency_ms": None if clr_elapsed is None else clr_elapsed * 1000.0,
                "fps_canvas1": fps_sel[0],
                "fps_canvas2": fps_sel[1],
                "fps_canvas3": fps_sel[2],
                "dnf": (sel_elapsed is None) or (clr_elapsed is None),
            })

            # Cleanly disconnect layers from GraphBus and scene before closing windows
            for sp in layers:
                try:
                    sp.close()  # disconnects GraphEventClient and stops further callbacks
                except Exception as e:
                    if verbose:
                        print(f"[cleanup] layer close error: {e}")

            # Close windows cleanly
            for c in canvases:
                c.close()
            app.processEvents()
            time.sleep(0.10)  # let Qt settle before next rep

    return results


def run_scenario_scatter3_simul(app: QtWidgets.QApplication,
                                sizes: List[int],
                                reps: int,
                                timeout_s: float,
                                seed: int,
                                verbose: bool):
    """Scenario: 3 scatterplots, select-all from ALL plots (three sources), measure propagation & clear."""
    results = []
    for n_points in sizes:
        for rep in range(1, reps + 1):
            if verbose:
                print(f"\n[scenario] 3x Scatter (SIMUL) | points={n_points} | rep={rep}")

            canvases, layers = build_three_scatter_canvases(n_points, seed + rep - 1, verbose)
            app.processEvents()

            meters = [FpsMeter(c.plot_widget.scene()) for c in canvases]

            # Each emitter has its own distinct source key
            sources = [ly._source_name() for ly in layers]  # private but acceptable for benchmarks
            idx_all = np.arange(n_points, dtype=int)

            # --- selection latency (simultaneous) ---
            for m in meters:
                m.start()
            if verbose:
                print(f"[select-simul] Emitting select_indices({n_points}) from sources={sources}")
            t0 = time.perf_counter()
            for ly in layers:
                ly.select_indices(idx_all)

            sel_elapsed = wait_until(
                app,
                lambda: multi_selection_done(layers, sources, n_points),
                timeout_s=timeout_s,
                verbose=verbose,
                tag="select-simul"
            )
            for m in meters:
                m.stop()

            # --- clear latency (clear all three sources) ---
            for m in meters:
                m.start()
            if verbose:
                print(f"[clear-simul] Broadcasting clear_highlight for sources={sources}")
            t1 = time.perf_counter()
            for sk in sources:
                layers[0].graph.emit_broadcast(EventType.CLEAR_HIGHLIGHT, {"source": sk})

            clr_elapsed = wait_until(
                app,
                lambda: multi_cleared_done(layers, sources),
                timeout_s=timeout_s,
                verbose=verbose,
                tag="clear-simul"
            )
            for m in meters:
                m.stop()

            fps_sel = [m.stats()[0] for m in meters]

            if sel_elapsed is None:
                print(f"[result] SELECT-SIMUL: DNF (> {timeout_s}s)")
            else:
                print(f"[result] SELECT-SIMUL: {sel_elapsed*1000:.2f} ms")

            if clr_elapsed is None:
                print(f"[result] CLEAR-SIMUL: DNF (> {timeout_s}s)")
            else:
                print(f"[result] CLEAR-SIMUL: {clr_elapsed*1000:.2f} ms")

            print(f"[fps] selection-window avg FPS (simul) ~ C1={fps_sel[0]:.1f}, C2={fps_sel[1]:.1f}, C3={fps_sel[2]:.1f}")

            results.append({
                "scenario": "scatter3_select_all_simultaneous",
                "points": n_points,
                "repetition": rep,
                "select_latency_ms": None if sel_elapsed is None else sel_elapsed * 1000.0,
                "clear_latency_ms": None if clr_elapsed is None else clr_elapsed * 1000.0,
                "fps_canvas1": fps_sel[0],
                "fps_canvas2": fps_sel[1],
                "fps_canvas3": fps_sel[2],
                "dnf": (sel_elapsed is None) or (clr_elapsed is None),
            })

            # Clean teardown
            for sp in layers:
                try:
                    sp.close()
                except Exception as e:
                    if verbose:
                        print(f"[cleanup] layer close error: {e}")
            for c in canvases:
                c.close()
            app.processEvents()
            time.sleep(0.10)

    return results


def write_outputs(rows: List[dict], csv_path: str, json_path: str, verbose: bool):
    if not rows:
        return
    # CSV
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    if verbose:
        print(f"[output] CSV written: {csv_path}")

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    if verbose:
        print(f"[output] JSON written: {json_path}")


def parse_args(argv: List[str]):
    p = argparse.ArgumentParser(description="GUI Benchmark for MCVGraph (3x ScatterPlots)")
    p.add_argument("--sizes", type=int, nargs="*", default=[1000, 5000, 15000],
                   help="Point counts for scatter dataset.")
    p.add_argument("--reps", type=int, default=3, help="Repetitions per size.")
    p.add_argument("--timeout", type=float, default=10.0,
                   help="Timeout in seconds per measurement; exceeded => DNF.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument("--csv", type=str, default="bench_results.csv", help="Output CSV path.")
    p.add_argument("--json", type=str, default="bench_results.json", help="Output JSON path.")
    p.add_argument("--verbose", action="store_true", help="Verbose debug printing.")
    p.add_argument("--scenario", type=str, choices=["single", "simul", "both"], default="both",
                   help="Which scenario to run: 'single' (select from first), 'simul' (select from all), or 'both'.")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    if args.verbose:
        print(f"[config] sizes={args.sizes}, reps={args.reps}, timeout={args.timeout}s, seed={args.seed}")
        print(f"[config] csv={args.csv}, json={args.json}")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    all_rows = []
    if args.scenario in ("single", "both"):
        all_rows += run_scenario_scatter3(
            app=app,
            sizes=args.sizes,
            reps=args.reps,
            timeout_s=args.timeout,
            seed=args.seed,
            verbose=args.verbose
        )
    if args.scenario in ("simul", "both"):
        all_rows += run_scenario_scatter3_simul(
            app=app,
            sizes=args.sizes,
            reps=args.reps,
            timeout_s=args.timeout,
            seed=args.seed,
            verbose=args.verbose
        )

    write_outputs(all_rows, args.csv, args.json, args.verbose)
    # Ensure Qt event loop processed final I/O
    app.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
