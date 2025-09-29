# main_minimal.py

import os
import sys
import numpy as np
import time
from PyQt5 import QtWidgets

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MCVGraph import DataSource, Canvas, ScatterPlot


# Create some random 2D data
rng = np.random.default_rng(42)
points = rng.uniform(-1, 1, size=(100, 2))
data_source = DataSource(points)

# Standard Qt application
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

# Create a canvas and a scatter plot
canvas = [Canvas(), Canvas(), Canvas(), Canvas(), Canvas()]
scatter = ScatterPlot(data_source)

for c in canvas:
    c.plot(scatter)
    c.show()

canvas[1]._link_to_target(canvas[0])
canvas[1].snap_right(canvas[0])
canvas[1].toggle_auto_match(True)
canvas[1]._scale_width(1)
canvas[1]._scale_height(0.5)

canvas[2]._link_to_target(canvas[1])
canvas[2].snap_below(canvas[1])
canvas[2].toggle_auto_match(True)
canvas[2]._scale_width(1)
canvas[2]._scale_height(1)

canvas[3]._link_to_target(canvas[0])
canvas[3].snap_below(canvas[0])
canvas[3].toggle_auto_match(True)
canvas[3]._scale_width(2)
canvas[3]._scale_height(0.5)

canvas[4]._link_to_target(canvas[0])
canvas[4].snap_left(canvas[0])
canvas[4].toggle_auto_match(True)
canvas[4]._scale_width(0.5)
canvas[4]._scale_height(1.5)

# Run the event loop
app.exec()

