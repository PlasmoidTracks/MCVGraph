# benchmarks/benchmark_scenarios/streaming.py
#
# Scenario: Streaming updates to a ScatterPlot.
# Repeatedly update DataSource for ~10s, measure average FPS.

import numpy as np
import time
from typing import List
from PyQt5 import QtWidgets, QtCore

from MCVGraph.DataSource import DataSource
from MCVGraph.canvas.Canvas import Canvas
from MCVGraph.graphs.ScatterPlot import ScatterPlot

from .utils import FpsMeter


def run(app: QtWidgets.QApplication,
        sizes: List[int], reps: int,
        timeout_s: float, seed: int, verbose: bool):

    results = []
    duration_s = 10.0  # fixed streaming window

    for n_points in sizes:
        for rep in range(1, reps + 1):
            if verbose:
                print(f"\n[scenario] streaming | points={n_points} | rep={rep}")

            rng = np.random.default_rng(seed + rep - 1)
            data = rng.normal(size=(n_points, 2)).astype(np.float32)
            ds = DataSource(data)

            c = Canvas(name="Streaming")
            sp = ScatterPlot(ds, name="stream", focus=True)
            c.plot(sp)
            c.show()
            app.processEvents()

            meter = FpsMeter(c.plot_widget.scene())
            meter.start()
            start = time.perf_counter()
            while time.perf_counter() - start < duration_s:
                new_data = rng.normal(size=(n_points, 2)).astype(np.float32)
                ds.set(new_data)
                app.processEvents(QtCore.QEventLoop.AllEvents, 50)

            meter.stop()

            fps, frames, duration = meter.stats()
            results.append({
                "scenario": "scatter_streaming_updates",
                "points": n_points,
                "repetition": rep,
                "select_latency_ms": None,
                "clear_latency_ms": None,
                "fps_canvas1": fps,
                "frames": frames,
                "duration_s": duration,
                "dnf": False,
            })

            sp.close()
            c.close()
            app.processEvents()
            time.sleep(0.1)

    return results
