import pandas as pd

data = pd.DataFrame({
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y1": [1,4,9,16,25,36,49,64,81,100],
    "y2": [2,3,5,7,11,13,17,19,23,29]
})





from MCVGraph import DataSource, Canvas, ScatterPlot
from PyQt5 import QtWidgets

data_source1 = DataSource(data[["x", "y1"]])
data_source2 = DataSource(data[["x", "y2"]])

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

canvas = Canvas()
scatterplot1 = ScatterPlot(data_source1)
scatterplot2 = ScatterPlot(data_source2)
canvas.plot(scatterplot1)
canvas.plot(scatterplot2)

canvas.show()

app.exec()
