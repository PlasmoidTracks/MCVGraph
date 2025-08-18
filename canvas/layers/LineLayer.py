# canvas/layers/LineLayer.py

from PyQt5 import QtCore
import numpy as np
import pyqtgraph as pg
from GraphBus import GraphEventClient
import time
try:
    import sounddevice as sd
except:
    pass

class LineLayer:
    _DEFAULT_COLORS = ['red', 'lime', 'blue', 'magenta', 'cyan', 'orange']
    _color_counter = 0
    _COLOR_REGISTRY = {}

    def __init__(self, data_source, sample_rate=44100, transform=lambda x: x, argsort_func=None, name=None):
        self.data_source = data_source
        self.sample_rate = int(sample_rate)
        self.transform = transform
        self.argsort_func = argsort_func if argsort_func is not None else (lambda x: np.arange(len(x)))

        self.name = name

        if self.name and self.name in LineLayer._COLOR_REGISTRY:
            self._selection_color = LineLayer._COLOR_REGISTRY[self.name]
        else:
            idx = LineLayer._color_counter % len(LineLayer._DEFAULT_COLORS)
            self._selection_color = LineLayer._DEFAULT_COLORS[idx]
            LineLayer._color_counter += 1
            if self.name:
                LineLayer._COLOR_REGISTRY[self.name] = self._selection_color

        self._plot_item = None
        self._view_box = None
        self._curve = None
        self._highlight_curves = {}
        self._focus = False
        self._opacity = 1.0
        self._z = 10

        self._order_indices = None
        self._duration = 0.0
        self._time_axis = None

        self._selection = None
        self._selection_origin = None

        self._highlight_indices = {}

        self.graph = GraphEventClient(self, "LinePlot")
        self.data_source.data_updated.connect(self._update_plot)

        self._cursor = None
        self._play_timer = QtCore.QTimer()
        self._play_timer.timeout.connect(self._update_cursor)
        self._playback_start_time = None
        self._waveform = None

    def handle_event(self, event_type, payload):
        if event_type == "subset_indices":
            indices = np.array(payload["indices"], dtype=int)
            source = payload["source"]
            color = payload.get("color", "r")
            self._highlight_indices[source] = (indices, color)
            self._update_plot()

        elif event_type == "clear_highlight":
            source = payload["source"]
            if source in self._highlight_indices:
                del self._highlight_indices[source]
                self._update_plot()

        elif event_type == "data_update":
            self._update_plot()

    def add_to(self, plot_item, view_box):
        self._plot_item = plot_item
        self._view_box = view_box

        self._curve = pg.PlotCurveItem(pen='gray')
        self._curve.setOpacity(self._opacity)
        self._curve.setZValue(self._z)
        self._plot_item.addItem(self._curve)

        self._zero = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(150, 150, 150, style=pg.QtCore.Qt.DashLine))
        self._zero.setOpacity(self._opacity)
        self._zero.setZValue(self._z)
        self._plot_item.addItem(self._zero)

        if self._cursor is None:
            self._cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=1.5))
            self._cursor.setVisible(False)
            self._plot_item.addItem(self._cursor)

        self._update_plot()

    def remove_from(self, plot_item):
        if self._curve is not None and self._curve.scene() is not None:
            self._plot_item.removeItem(self._curve)
        if hasattr(self, "_zero") and self._zero.scene() is not None:
            self._plot_item.removeItem(self._zero)
        for it in list(self._highlight_curves.values()):
            if it.scene() is not None:
                self._plot_item.removeItem(it)
        self._highlight_curves.clear()
        if self._selection is not None and self._selection.scene() is not None:
            self._plot_item.removeItem(self._selection)
        if self._cursor is not None and self._cursor.scene() is not None:
            self._plot_item.removeItem(self._cursor)
        self._cursor = None
        self._play_timer.stop()
        self._plot_item = None
        self._view_box = None

    def set_opacity(self, alpha: float):
        self._opacity = float(alpha)
        if self._curve is not None:
            self._curve.setOpacity(self._opacity)
        if hasattr(self, "_zero"):
            self._zero.setOpacity(self._opacity)
        for it in self._highlight_curves.values():
            it.setOpacity(self._opacity)

    def set_z(self, z: int):
        self._z = int(z)
        if self._curve is not None:
            self._curve.setZValue(self._z)
        if hasattr(self, "_zero"):
            self._zero.setZValue(self._z)
        for it in self._highlight_curves.values():
            it.setZValue(self._z + 1)
        if self._selection is not None:
            self._selection.setZValue(self._z + 2)

    def set_focus(self, focus: bool):
        self._focus = bool(focus)
        if not self._focus and self._selection is not None:
            pass

    def set_transform(self, transform, argsort=lambda x: np.arange(len(x))):
        self.transform = transform
        self.argsort_func = argsort
        self._update_plot()

    def select_range(self, t_min, t_max, source=None):
        if self._order_indices is None or self._time_axis is None:
            return
        mask = (self._time_axis >= t_min) & (self._time_axis <= t_max)
        indices = self._order_indices[mask]
        payload = {"indices": indices.copy(), "source": self._source_name()}
        self.graph.emit("ScatterPlot", "subset_indices", payload)
        self.graph.emit("LinePlot", "subset_indices", payload)

    def get_selection_indices(self):
        src = self._source_name()
        return self._highlight_indices.get(src, (np.array([], dtype=int), None))[0]

    def bounds(self):
        raw = self.data_source.get()
        if raw.shape[0] == 0 or raw.shape[1] < 1:
            return None
        y = self.transform(raw).reshape(-1)
        if y.size == 0:
            return None
        x0, x1 = 0.0, max(1e-9, len(y) / float(self.sample_rate))
        y0, y1 = float(np.min(y)), float(np.max(y))
        return (x0, x1, y0, y1)

    def _update_plot(self):
        if self._plot_item is None:
            return
        data = self.data_source.get()
        if data.shape[0] == 0 or data.shape[1] < 1:
            self._curve.setData([], [])
            return

        order = self.argsort_func(data)
        reverse = np.empty_like(order)
        reverse[order] = np.arange(len(order))

        y_full = self.transform(data).reshape(-1).astype(np.float32)
        n = len(y_full)
        if n == 0:
            self._curve.setData([], [])
            self._waveform = None
            return

        self._duration = n / float(self.sample_rate)
        self._time_axis = np.linspace(0.0, self._duration, n, endpoint=False)
        self._order_indices = order

        self._curve.setData(self._time_axis, y_full)
        self._waveform = y_full

        for it in list(self._highlight_curves.values()):
            if it.scene() is not None:
                self._plot_item.removeItem(it)
        self._highlight_curves.clear()

        for src, (global_idx, color) in self._highlight_indices.items():
            gi = np.asarray(global_idx, dtype=int)
            gi = gi[(gi >= 0) & (gi < n)]
            if gi.size == 0:
                continue
            pos = np.sort(reverse[gi])
            y_overlay = np.full(n, np.nan, dtype=np.float32)
            y_overlay[pos] = y_full[pos]
            curve = pg.PlotCurveItem(self._time_axis, y_overlay, pen=pg.mkPen(color=color, width=1.5))
            curve.setOpacity(self._opacity)
            curve.setZValue(self._z + 1)
            self._plot_item.addItem(curve)
            self._highlight_curves[src] = curve

    def handle_scene_event(self, event, view_box):
        if not self._focus:
            return False

        et = event.type()

        if et == QtCore.QEvent.GraphicsSceneMousePress:
            if event.button() != QtCore.Qt.LeftButton:
                return False

            click_x = view_box.mapSceneToView(event.scenePos()).x()

            if self._selection is not None:
                sel_min, sel_max = self._selection.getRegion()
                eps = max(1e-9, (sel_max - sel_min) * 0.01)
                if (sel_min - eps) <= click_x <= (sel_max + eps):
                    return False
                else:
                    if self._selection.scene() is not None:
                        self._plot_item.removeItem(self._selection)
                    self._selection = None
                    self._selection_origin = event.scenePos()

                    payload = {"source": self._source_name()}
                    self.graph.emit("ScatterPlot", "clear_highlight", payload)
                    self.graph.emit("LinePlot", "clear_highlight", payload)
                    return True

            self._selection_origin = event.scenePos()
            return True

        elif et == QtCore.QEvent.GraphicsSceneMouseMove:
            if self._selection_origin is None:
                return False

            start_x = view_box.mapSceneToView(self._selection_origin).x()
            current_x = view_box.mapSceneToView(event.scenePos()).x()
            lo, hi = (min(start_x, current_x), max(start_x, current_x))

            if self._selection is None:
                self._selection = pg.LinearRegionItem(values=[lo, hi], orientation=pg.LinearRegionItem.Vertical)
                self._selection.setZValue(self._z + 2)
                self._plot_item.addItem(self._selection)

                self._selection.sigRegionChanged.connect(self._on_selection_region_changed)
            else:
                self._selection.setRegion([lo, hi])

            self.select_range(lo, hi)
            return True

        elif et == QtCore.QEvent.GraphicsSceneMouseRelease:
            if self._selection_origin is None:
                return False
            self._selection_origin = None
            return True

        return False

    def _on_selection_region_changed(self):
        if self._selection is None:
            return
        lo, hi = self._selection.getRegion()
        self.select_range(lo, hi)

    def _source_name(self):
        desc = getattr(self, "_descriptor", None)
        uid = getattr(self, "_canvas_uid", None)
        if desc is not None and hasattr(desc, "graph_name") and desc.graph_name:
            return f"{desc.graph_name}@{uid}" if uid is not None else desc.graph_name
        name = getattr(getattr(self, "graph", None), "owner", None)
        if name and hasattr(name, "graph_name"):
            return name.graph_name
        return f"Layer@{hex(id(self))}"

    def close(self):
        if hasattr(self, "graph") and hasattr(self.graph, "disconnect"):
            self.graph.disconnect()

    def get_color(self):
        return self._selection_color

    def set_selection_color(self, color: str):
        self._selection_color = str(color)

    def set_color(self, color: str):
        self.set_selection_color(color)

    def expose_actions(self, parent=None):
        actions = []

        play_action = pg.QtWidgets.QAction("Play waveform", parent)
        play_action.triggered.connect(self._play_audio)
        actions.append(play_action)

        return actions

    def clone(self):
        dup = LineLayer(self.data_source,
                        sample_rate=self.sample_rate,
                        transform=self.transform,
                        argsort_func=self.argsort_func)
        dup.set_selection_color(self._selection_color)
        return dup

    def _play_audio(self):
        if self._waveform is None or self._time_axis is None or self._duration <= 0:
            return

        wf = self._waveform
        peak = float(np.max(np.abs(wf))) if wf.size else 1.0
        wf = (wf / peak) if peak > 0 else wf

        if self._cursor is not None:
            self._cursor.setVisible(True)
            self._cursor.setPos(0.0)

        self._playback_start_time = time.perf_counter()
        sd.play(wf, self.sample_rate, blocking=False)
        self._play_timer.start(16)

    def _update_cursor(self):
        if self._playback_start_time is None or self._cursor is None:
            self._play_timer.stop()
            return
        elapsed = time.perf_counter() - self._playback_start_time
        if elapsed >= self._duration:
            self._cursor.setVisible(False)
            self._play_timer.stop()
            self._playback_start_time = None
            return
        self._cursor.setPos(elapsed)
