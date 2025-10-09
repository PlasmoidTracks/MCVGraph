# demo_scatter.py

import os
import sys
import numpy as np
from PyQt5 import QtWidgets

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MCVGraph import DataSource, ScatterPlot, Canvas

def make_data(n=450, seed=2):
    rng = np.random.default_rng(seed)
    # Single 2D Gaussian blob centered at origin
    data = rng.normal(loc=[0.0, 0.0], scale=[1.0, 1.0], size=(n, 2))
    return data.astype(np.float32)

def cartesian(arr):
    return arr[:, :2]

def polar_xy(arr):
    xy = arr[:, :2]
    r = np.linalg.norm(xy, axis=1)
    th = np.arctan2(xy[:, 1], xy[:, 0])
    # return (theta, r) so horizontal ~ angle, vertical ~ radius
    return np.column_stack([th, r]).astype(np.float32)

def main():
    data = make_data()
    ds = DataSource(data)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    ca = Canvas()
    ca.set_graph_name("Scatter • Cartesian")
    a = ScatterPlot(ds, base_marker_size=6, color_non_selected="gray")
    a.set_transform(cartesian)
    ca.plot(a)
    ca.set_axis_label("x", "X")
    ca.set_axis_label("y", "Y")
    ca.set_focus(a)
    ca.resize(600, 600)
    ca.show()

    cb = Canvas()
    cb.set_graph_name("Scatter • Polar (θ vs r)")
    b = ScatterPlot(ds, base_marker_size=6, color_non_selected="gray")
    b.set_transform(polar_xy)
    cb.plot(b)
    cb.set_axis_label("x", "θ (rad)")
    cb.set_axis_label("y", "r")
    cb.set_focus(b)
    cb.resize(600, 600)
    cb.show()

    app.exec()

if __name__ == "__main__":
    main()
