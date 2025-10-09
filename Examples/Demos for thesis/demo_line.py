# demo_line.py

import os
import sys
import numpy as np
from PyQt5 import QtWidgets

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MCVGraph import DataSource, LinePlot, Canvas

def make_signal(fs=300, dur=10.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs*dur)) / fs
    y = (
        0.9*np.sin(2*np.pi*1.2*t) +
        0.5*np.sin(2*np.pi*3.7*t + 0.3) +
        0.2*np.sin(2*np.pi*12.0*t + 1.1)
    )
    y += 0.05*rng.standard_normal(len(t))
    return y.astype(np.float32), fs

def identity_transform(arr):
    return arr.reshape(-1, 1)

def moving_avg_transform(window=25):
    def _f(arr):
        y = arr.reshape(-1)
        if window <= 1:
            return y[:, None]
        k = int(window)
        kernel = np.ones(k, dtype=np.float32)/k
        z = np.convolve(y, kernel, mode="same")
        return z[:, None]
    return _f

def main():
    y, fs = make_signal()
    ds = DataSource(y)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    ca = Canvas()
    ca.set_graph_name("Line • Raw")
    lp_a = LinePlot(ds, sample_rate=fs)
    lp_a.set_transform(identity_transform)
    ca.plot(lp_a)
    ca.set_axis_label("x", "Time (s)")
    ca.set_axis_label("y", "Amplitude")
    ca.set_focus(lp_a)
    ca.resize(900, 300)
    ca.show()

    cb = Canvas()
    cb.set_graph_name("Line • Moving Average (w=25)")
    lp_b = LinePlot(ds, sample_rate=fs)
    lp_b.set_transform(moving_avg_transform(window=25))
    cb.plot(lp_b)
    cb.set_axis_label("x", "Time (s)")
    cb.set_axis_label("y", "Smoothed")
    cb.set_focus(lp_b)
    cb.resize(900, 300)
    cb.show()

    app.exec()

if __name__ == "__main__":
    main()
