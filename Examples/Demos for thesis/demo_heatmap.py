# demo_heatmap.py

import os
import sys
import numpy as np
from PyQt5 import QtWidgets

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MCVGraph import DataSource, ScatterPlot, HeatmapPlot, Canvas

def make_points(n=50000, seed=2):
    rng = np.random.default_rng(seed)
    # Single 2D Gaussian blob centered at origin
    data = rng.normal(loc=[0.0, 0.0], scale=[1.0, 1.0], size=(n, 2))
    return data.astype(np.float32)

def rect_binner(size, domain=((-3.0, 3.0), (-3.0, 3.0))):
    """
    Rectangular binning aligned with scatter coordinates.
    Centered bins, correct orientation, and flipped along y=x (swap axes).
    """
    (xmin, xmax), (ymin, ymax) = domain

    def _f(arr):
        xy = arr[:, :2]
        nx = np.clip((xy[:, 0] - xmin) / (xmax - xmin), 0.0, 1.0)
        ny = np.clip((xy[:, 1] - ymin) / (ymax - ymin), 0.0, 1.0)

        # Normalize first
        ixf = np.clip(nx * size, 0.0, size - 1e-6)
        iyf = np.clip(ny * size, 0.0, size - 1e-6)

        # Convert to int indices AFTER swap and centering
        iy = np.clip((ixf - 0.5).astype(int), 0, size - 1)
        ix = np.clip((iyf - 0.5).astype(int), 0, size - 1)

        mat = np.zeros((size, size), dtype=np.float32)
        np.add.at(mat, (iy, ix), 1.0)

        return mat

    # Swap scale_x and scale_y for the transposed layout
    scale_x = (ymax - ymin) / size
    scale_y = (xmax - xmin) / size
    # Shift origin to the lower-left corner of the first bin
    origin = (ymin + 0.5 * scale_x, xmin + 0.5 * scale_y)
    return _f, origin, (scale_x, scale_y)

def log_normalizer(arr):
    la = np.log1p(arr)
    vmin, vmax = la.min(), la.max()
    return (la - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(la)

def main():
    pts = make_points()
    ds = DataSource(pts)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # --- Canvas A: Scatter Plot ---
    scatter_canvas = Canvas()
    scatter_canvas.set_graph_name("Scatter Plot")
    scatter = ScatterPlot(ds, base_marker_size=1)
    scatter.set_transform(lambda a: a[:, :2])
    scatter_canvas.plot(scatter)
    scatter_canvas.set_axis_label("x", "X")
    scatter_canvas.set_axis_label("y", "Y")
    scatter_canvas.set_focus(scatter)
    scatter_canvas.resize(600, 600)
    scatter_canvas.show()

    # --- Canvas B: Heatmap Density ---
    f_rect, origin_rect, scale_rect = rect_binner(size=64, domain=((-3, 3), (-3, 3)))
    heatmap_canvas = Canvas()
    heatmap_canvas.set_graph_name("Heatmap Density")

    heatmap = HeatmapPlot(
        data_source=ds,
        transform=f_rect,
        normalizer=log_normalizer,
        scale_x=scale_rect[0],
        scale_y=scale_rect[1],
    )
    heatmap.set_translation(origin_rect[0], origin_rect[1])
    heatmap.set_opacity(1.0)

    heatmap_canvas.plot(heatmap)
    heatmap_canvas.set_axis_label("x", "X")
    heatmap_canvas.set_axis_label("y", "Y")
    heatmap_canvas.resize(600, 600)
    heatmap_canvas.show()

    app.exec()

    app.exec()

if __name__ == "__main__":
    main()
