import numpy as np
from MCVGraph import DataSource, ScatterPlot, Canvas
from PyQt5 import QtWidgets

# Define a transformation: interpret (x,y) as complex, square it, return new (x,y)
def complex_square(data):
    z = data[:, 0] + 1j * data[:, 1]
    z2 = z ** 2
    return np.column_stack((z2.real, z2.imag))

# Create some random 2D data
rng = np.random.default_rng(42)
points = rng.uniform(-1, 1, size=(1000, 2))
data_source = DataSource(points)

# Standard Qt application
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

# Create two scatter plots from the same data source
scatter1 = ScatterPlot(data_source)
scatter2 = ScatterPlot(data_source, transform=complex_square)

# Create canvases and show them
canvas1 = Canvas()
canvas1.plot(scatter1)
canvas1.show()

canvas2 = Canvas()
canvas2.plot(scatter2)
canvas2.show()

# Run the event loop
app.exec()
