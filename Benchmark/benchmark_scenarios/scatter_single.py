# benchmarks/benchmark_scenarios/scatter_single.py
#
# Scenario: 3 ScatterPlots sharing one DataSource.
# Measure latency and FPS for select_all() from the first plot.

import numpy as np
import time
from typing import List
from PyQt5 import QtWidgets

from MCVGraph.DataSource import DataSource
from MCVGraph.canvas.Canvas import Canvas
from MCVGraph.graphs.ScatterPlot import ScatterPlot
from MCVGraph.EventType import EventType

from .utils import FpsMeter, wait_until


def build_three_scatter_canvases(n_points: int, seed: int, verbose: bool):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_points, 2)).astype(np.float32)
    ds = DataSource(data)
    canvases, layers = [], []
    for i in range(3):
        c = Canvas(name=f"Canvas #{i+1}")
        c.move(80 + i * 420, 80)
        c.resize(400, 300)
        c.show()
        sp = ScatterPlot(ds, name=f"S{i+1}", focus=(i == 0))
        c.plot(sp)
        canvases.append(c)
        layers.append(sp)
    QtWidgets.QApplication.instance().processEvents()
    return canvases, layers


def run(app: QtWidgets.QApplication,
        sizes: List[int], reps: int,
        timeout_s: float, seed: int, verbose: bool):

    results = []
    for n_points in sizes:
        for rep in range(1, reps + 1):
            if verbose:
                print(f"\n[scenario] scatter_single | points={n_points} | rep={rep}")
            canvases, layers = build_three_scatter_canvases(n_points, seed + rep - 1, verbose)
            meters = [FpsMeter(c.plot_widget.scene()) for c in canvases]

            emitter = layers[0]
            source_key = emitter._source_name()
            idx_all = np.arange(n_points, dtype=int)

            # --- selection ---
            for m in meters:
                m.start()
            emitter.select_indices(idx_all)
            sel_elapsed = wait_until(app,
                lambda: all(len(getattr(ly, "_highlight_indices", {}).get(source_key, ([],))[0]) == n_points for ly in layers),
                timeout_s, verbose, "select")
            for m in meters:
                m.stop()

            # --- clear ---
            for m in meters:
                m.start()
            emitter.graph.emit_broadcast(EventType.CLEAR_HIGHLIGHT, {"source": source_key})
            clr_elapsed = wait_until(app,
                lambda: all(source_key not in getattr(ly, "_highlight_indices", {}) for ly in layers),
                timeout_s, verbose, "clear")
            for m in meters:
                m.stop()

            fps_sel = [m.stats()[0] for m in meters]
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

            for sp in layers:
                sp.close()
            for c in canvases:
                c.close()
            app.processEvents()
            time.sleep(0.1)

    return results
