# benchmarks/benchmark_scenarios/scaling.py
#
# Scenario: Scaling with number of canvases (3, 6, 12).
# Each canvas has a ScatterPlot. Measure select/clear latency.

import numpy as np
import time
from typing import List
from PyQt5 import QtWidgets

from MCVGraph.DataSource import DataSource
from MCVGraph.canvas.Canvas import Canvas
from MCVGraph.graphs.ScatterPlot import ScatterPlot
from MCVGraph.EventType import EventType

from .utils import FpsMeter, wait_until


def build_canvases(n_points: int, n_canvases: int, seed: int):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_points, 2)).astype(np.float32)
    ds = DataSource(data)

    canvases, layers = [], []
    for i in range(n_canvases):
        c = Canvas(name=f"Canvas {i+1}")
        c.move(50 + i * 420, 80)
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
        for n_canvases in [3, 6, 12]:
            for rep in range(1, reps + 1):
                if verbose:
                    print(f"\n[scenario] scaling | points={n_points} | canvases={n_canvases} | rep={rep}")

                canvases, layers = build_canvases(n_points, n_canvases, seed + rep - 1)
                meters = [FpsMeter(c.plot_widget.scene()) for c in canvases]

                emitter = layers[0]
                source_key = emitter._source_name()
                idx_all = np.arange(n_points, dtype=int)

                # --- select ---
                for m in meters: m.start()
                emitter.select_indices(idx_all)
                sel_elapsed = wait_until(app,
                    lambda: all(len(getattr(ly, "_highlight_indices", {}).get(source_key, ([],))[0]) == n_points for ly in layers),
                    timeout_s, verbose, "select-scale")
                for m in meters: m.stop()

                # --- clear ---
                for m in meters: m.start()
                emitter.graph.emit_broadcast(EventType.CLEAR_HIGHLIGHT, {"source": source_key})
                clr_elapsed = wait_until(app,
                    lambda: all(source_key not in getattr(ly, "_highlight_indices", {}) for ly in layers),
                    timeout_s, verbose, "clear-scale")
                for m in meters: m.stop()

                fps_sel = [m.stats()[0] for m in meters]
                results.append({
                    "scenario": f"scatter_scaling_{n_canvases}_canvases",
                    "points": n_points,
                    "repetition": rep,
                    "select_latency_ms": None if sel_elapsed is None else sel_elapsed * 1000.0,
                    "clear_latency_ms": None if clr_elapsed is None else clr_elapsed * 1000.0,
                    **{f"fps_canvas{i+1}": fps_sel[i] for i in range(len(fps_sel))},
                    "dnf": (sel_elapsed is None) or (clr_elapsed is None),
                })

                for sp in layers: sp.close()
                for c in canvases: c.close()
                app.processEvents()
                time.sleep(0.1)

    return results
