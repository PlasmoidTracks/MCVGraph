# widgets/ViewBox.py

import pyqtgraph as pg
from PyQt5 import QtCore

class ViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, enableMouse=True, **kwargs)

    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() & QtCore.Qt.ControlModifier:
            super().wheelEvent(ev, axis)

    def mouseDragEvent(self, ev, axis=None):
        if ev.modifiers() & QtCore.Qt.ControlModifier:
            super().mouseDragEvent(ev, axis)

