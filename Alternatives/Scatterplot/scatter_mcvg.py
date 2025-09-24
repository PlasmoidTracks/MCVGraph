import pandas as pd
import numpy as np

np.random.seed(0)
data = pd.DataFrame({
    "x1": np.random.rand(100),
    "y1": np.random.rand(100),
    "x2": np.random.rand(100),
    "y2": np.random.rand(100)
})


from MCVGraph import DataSource, ScatterPlot, Canvas
from PyQt5 import QtWidgets

data_source1 = DataSource(data[["x1", "y1"]])
data_source2 = DataSource(data[["x2", "y2"]])

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

canvas = Canvas()
scatterplot1 = ScatterPlot(data_source1)
scatterplot2 = ScatterPlot(data_source2)
canvas.plot(scatterplot1)
canvas.plot(scatterplot2)

canvas.show()

app.exec()
