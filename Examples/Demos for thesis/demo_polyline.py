# demo_polyline.py

import os
import sys
import numpy as np
from PyQt5 import QtWidgets

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MCVGraph import DataSource, ScatterPlot, PolylinePlot, Canvas

def grid_points(m=12, spacing=0.45):
    xs, ys = np.meshgrid(np.arange(m), np.arange(m))
    xy = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    xy = (xy - (m-1)/2.0) * spacing
    return xy

def grid_edges(m=12):
    def idx(i, j): return i*m + j
    e = []
    for i in range(m):
        for j in range(m):
            if i+1 < m: e.append([idx(i, j), idx(i+1, j)])
            if j+1 < m: e.append([idx(i, j), idx(i, j+1)])
    return np.asarray(e, dtype=int)

def swirl(xy, k=0.9):
    # angle depends on radius → swirly deformation
    x, y = xy[:, 0], xy[:, 1]
    r2 = x*x + y*y
    a = k * r2
    c, s = np.cos(a), np.sin(a)
    x2 = c*x - s*y
    y2 = s*x + c*y
    return np.column_stack([x2, y2]).astype(np.float32)

def main():
    m = 12
    verts = grid_points(m=m, spacing=0.45)
    edges = grid_edges(m=m)

    ds_vertices = DataSource(verts)
    ds_edges    = DataSource(edges)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # Canvas A: original geometry with scatter as selection driver
    ca = Canvas()
    ca.set_graph_name("Polyline • Original (select on scatter)")

    sp = ScatterPlot(ds_vertices, base_marker_size=6, color_non_selected="gray")
    sp.set_transform(lambda a: a[:, :2])

    pl_a = PolylinePlot(
        vertices=ds_vertices,
        edges=ds_edges,
        color_non_selected="gray",
        selection_color=None,
        line_width=1.6,
    )

    ca.plot(pl_a)
    ca.plot(sp)
    ca.set_axis_label("x", "X")
    ca.set_axis_label("y", "Y")
    ca.set_focus(sp)
    ca.resize(600, 600)
    ca.show()

    # Canvas B: non-linear “swirl” deformation (linked highlight)
    verts_swirl = swirl(verts, k=0.2)
    ds_vertices_swirl = DataSource(verts_swirl)

    cb = Canvas()
    cb.set_graph_name("Polyline • Swirl deformation (linked highlight)")

    pl_b = PolylinePlot(
        vertices=ds_vertices_swirl,
        edges=ds_edges,
        color_non_selected="gray",
        selection_color=None,
        line_width=1.6,
    )

    cb.plot(pl_b)
    cb.set_axis_label("x", "X′")
    cb.set_axis_label("y", "Y′")
    cb.resize(600, 600)
    cb.show()

    app.exec()

if __name__ == "__main__":
    main()
