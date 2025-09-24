import pandas as pd

data = pd.DataFrame({
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y1": [1,4,9,16,25,36,49,64,81,100],
    "y2": [2,3,5,7,11,13,17,19,23,29]
})


from MCVGraph import DataSource, LinePlot, Canvas
from PyQt5 import QtWidgets

data_source1 = DataSource(data["y1"])
data_source2 = DataSource(data["y2"])

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

canvas = Canvas()
lineplot1 = LinePlot(data_source1, sample_rate=1)
lineplot2 = LinePlot(data_source2, sample_rate=1)
canvas.plot(lineplot1)
canvas.plot(lineplot2)

canvas.show()

app.exec()
