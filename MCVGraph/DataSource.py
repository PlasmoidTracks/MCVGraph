# DataSource.py

from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from typing import Any, Callable, Optional

class DataSource(QObject):
    data_updated = pyqtSignal()

    def __init__(self, initial_data: Optional[Any] = None) -> None:
        super().__init__()
        self._data: np.ndarray = self._normalize(initial_data)

    def set(self, new_data: Any) -> None:
        self._data = self._normalize(new_data)
        self.data_updated.emit()

    def _normalize(self, data: Any) -> np.ndarray:
        if data is None:
            return np.empty((0, 0))
        try:
            arr = np.asarray(data).copy()
            if arr.ndim == 0:  # scalar or string
                return np.empty((0, 0))
            elif arr.ndim == 1:  # force column vector
                return arr.reshape(-1, 1)
            elif arr.ndim == 2:
                return arr
            else:
                return np.empty((0, 0))
        except Exception:
            return np.empty((0, 0))

    def get(self) -> np.ndarray:
        return self._data

    def apply_transform(self, func: Callable[[np.ndarray], Any]) -> Any:
        return func(self._data)

    def size(self) -> int:
        return len(self._data) if self._data.ndim > 0 else 0
