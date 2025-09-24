# main_customplot.py

import os
import sys
import numpy as np
from PyQt5 import QtWidgets

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MCVGraph import ScatterPlot, DataSource, Canvas
from Examples.CustomPlot.CustomPlot import CustomPlot


# Create some random 2D data
rng = np.random.default_rng(123)
points = rng.uniform(-1, 1, size=(200, 2)).astype(float)
data_source = DataSource(points)

# Standard Qt application
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

# --- Window 1: CustomPlot (circle selection) ---
canvas1 = Canvas()
canvas1.set_graph_name("CustomPlot Window")
custom_plot = CustomPlot(data_source, color="green", size=10, non_selected_color="lightgray")
custom_plot._focus = True  # ensure our plot receives mouse events on this canvas
canvas1.plot(custom_plot)
canvas1.set_axis_label("x", "X")
canvas1.set_axis_label("y", "Y")
canvas1.show()

# --- Window 2: Library ScatterPlot (receives highlights) ---
canvas2 = Canvas()
canvas2.set_graph_name("ScatterPlot Window")
scatter_plot = ScatterPlot(
    data_source,
    selection_color="magenta",
    color_non_selected="gray",
    base_marker_size=7,
    focus=True,  # focused so it can also emit if you lock/unlock & select there
)
canvas2.plot(scatter_plot)
canvas2.set_axis_label("x", "X")
canvas2.set_axis_label("y", "Y")
canvas2.show()

# Run the event loop
app.exec()
