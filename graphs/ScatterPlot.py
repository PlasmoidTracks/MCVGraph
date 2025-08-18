# graphs/ScatterPlot.py

import numpy as np
from DataSource import DataSource

class ScatterPlot:
    def __init__(self, data_source: DataSource, name: str = None):
        self.data_source = data_source
        self.transform = lambda x: x
        self.graph_name = name or "Scatter"

        self._x_label = ""
        self._y_label = ""

    def set_transform(self, transform):
        self.transform = transform

    def set_graph_name(self, name: str):
        self.graph_name = str(name)