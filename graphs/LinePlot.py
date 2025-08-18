# graphs/LinePlot.py

import numpy as np
from DataSource import DataSource

class LinePlot:
    def __init__(self, data_source: DataSource, sample_rate: int = 44100, name: str = None):
        self.data_source = data_source
        self.sample_rate = int(sample_rate)
        self.transform = lambda x: x
        self.argsort_func = None
        self.graph_name = name or "Line"

        self._x_label = ""
        self._y_label = ""

    def set_transform(self, transform_func, argsort_func=None):
        self.transform = transform_func
        self.argsort_func = argsort_func

    def set_graph_name(self, name: str):
        self.graph_name = str(name)

    def set_axis_label(self, axis: str, text: str):
        if axis == "x":
            self._x_label = str(text)
        elif axis == "y":
            self._y_label = str(text)

    def get_axis_labels(self):
        return self._x_label, self._y_label
