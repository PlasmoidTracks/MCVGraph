# DataSource.py

from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np

class DataSource(QObject):
    data_updated = pyqtSignal()

    def __init__(self, initial_data=None):
        super().__init__()
        if initial_data is None:
            self._data = np.empty((0, 0))
        else:
            self._data = np.asarray(initial_data)

    def get(self):
        return self._data

    def set(self, new_data):
        if new_data is None:
            self._data = np.empty((0, 0))
        else:
            self._data = np.asarray(new_data)
        self.data_updated.emit()

    def apply_transform(self, func):
        return func(self._data)

    def size(self):
        return len(self._data) if self._data.ndim > 0 else 0
