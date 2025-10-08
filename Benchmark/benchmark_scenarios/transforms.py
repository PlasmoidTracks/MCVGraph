# benchmarks/benchmark_scenarios/transforms.py
#
# Scenario: Two ScatterPlots on the same DataSource,
# one raw, one with a heavy transform (complex square).
# Measure latency and FPS for selection propagation.

import numpy as np
import time
from typing import List
from PyQt5 import QtWidgets

from MCVGraph.DataSource import DataSource
from MCVGraph.canvas.Canvas import Canvas
from MCVGraph.graphs.ScatterPlot import ScatterPlot
from MCVGraph.EventType import EventType

from .utils import FpsMeter, wait_until


def complex_square(data: np.ndarray) -> np.ndarray:
    """Transform: interpret (x,y) as complex, square it, return (Re, Im)."""
    z = data[:, 0] + 1j * data[:, 1]
    z2 = z ** 2
    return np.column_stack((z2.real, z2.imag))


def run(app: QtWidgets.QApplication,
        sizes: List[int], reps: int,
        timeout_s: float, seed: int, verbose: bool):

    results = []
    for n_points in sizes:
        for rep in range(1, reps + 1):
            if verbose:
                print(f"\n[scenario] transforms | points={n_points} | rep={rep}")

            rng = np.random.default_rng(seed + rep - 1)
            data = rng.normal(size=(n_points, 2)).astype(np.float32)
            ds = DataSource(data)

            # Raw scatter
            c1 = Canvas(name="Raw")
            sp1 = ScatterPlot(ds, name="raw", focus=True)
            c1.plot(sp1)
            c1.show()

            # Transformed scatter
            c2 = Canvas(name="Squared")
            sp2 = ScatterPlot(ds, name="squared", transform=complex_square)
            c2.plot(sp2)
            c2.show()

            app.processEvents()
            meters = [FpsMeter(c1.plot_widget.scene()), FpsMeter(c2.plot_widget.scene())]

            source_key = sp1._source_name()
            idx_all = np.arange(n_points, dtype=int)

            # --- select ---
            for m in meters: m.start()
            sp1.select_indices(idx_all)
            sel_elapsed = wait_until(app,
                lambda: all(len(getattr(sp, "_highlight_indices", {}).get(source_key, ([],))[0]) == n_points for sp in [sp1, sp2]),
                timeout_s, verbose, "select-transform")
            for m in meters: m.stop()

            # --- clear ---
            for m in meters: m.start()
            sp1.graph.emit_broadcast(EventType.CLEAR_HIGHLIGHT, {"source": source_key})
            clr_elapsed = wait_until(app,
                lambda: all(source_key not in getattr(sp, "_highlight_indices", {}) for sp in [sp1, sp2]),
                timeout_s, verbose, "clear-transform")
            for m in meters: m.stop()

            fps_sel = [m.stats()[0] for m in meters]
            results.append({
                "scenario": "scatter_transform_propagation",
                "points": n_points,
                "repetition": rep,
                "select_latency_ms": None if sel_elapsed is None else sel_elapsed * 1000.0,
                "clear_latency_ms": None if clr_elapsed is None else clr_elapsed * 1000.0,
                "fps_canvas1": fps_sel[0],
                "fps_canvas2": fps_sel[1],
                "dnf": (sel_elapsed is None) or (clr_elapsed is None),
            })

            sp1.close()
            sp2.close()
            c1.close()
            c2.close()
            app.processEvents()
            time.sleep(0.1)

    return results
