# main_minimal.py

import os
import sys
import numpy as np
from PyQt5 import QtWidgets

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MCVGraph.DataSource import DataSource
from MCVGraph.canvas.Canvas import Canvas
from MCVGraph.graphs.ScatterPlot import ScatterPlot

# Create some random 2D data
rng = np.random.default_rng(42)
points = rng.uniform(-1, 1, size=(100, 2))
data_source = DataSource(points)

# Standard Qt application
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

# Create a canvas and a scatter plot
canvas = Canvas()
scatter = ScatterPlot(data_source)

canvas.plot(scatter)
canvas.show()

# Run the event loop
app.exec()
