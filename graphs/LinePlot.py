# graphs/LinePlot.py

from PyQt5 import QtCore
import numpy as np
import pyqtgraph as pg
from GraphBus import GraphEventClient
import time
try:
    import sounddevice as sd
except:
    pass

class LinePlot:
    _DEFAULT_COLORS = ['red', 'lime', 'blue', 'magenta', 'cyan', 'orange']
    _color_counter = 0
    _COLOR_REGISTRY = {}

    def __init__(
        self,
        data_source,
        *,
        sample_rate=44100,
        transform=None,
        argsort_func=None,
        name=None,
        selection_color=None,
        focus=False,
        opacity=1.0,
        z=10,
        base_pen='gray',
        highlight_pen_width=1.5,
        zero_line_visible=True,
        zero_line_pen=None,
        cursor_pen='r',
        playback_timer_ms=16,
    ):
        self.data_source = data_source
        self.sample_rate = sample_rate
        self.transform = (transform if transform is not None else (lambda x: x))
        self.argsort_func = (argsort_func if argsort_func is not None else (lambda x: np.arange(len(x))))
        self.name = name

        if self.name and self.name in LinePlot._COLOR_REGISTRY and selection_color is None:
            self._selection_color = LinePlot._COLOR_REGISTRY[self.name]
        else:
            if selection_color is None:
                idx = LinePlot._color_counter % len(LinePlot._DEFAULT_COLORS)
                self._selection_color = LinePlot._DEFAULT_COLORS[idx]
                LinePlot._color_counter += 1
            else:
                self._selection_color = str(selection_color)
            if self.name:
                LinePlot._COLOR_REGISTRY[self.name] = self._selection_color

        self._base_pen = base_pen
        self._highlight_pen_width = highlight_pen_width
        self._zero_line_visible = zero_line_visible
        self._zero_line_pen = zero_line_pen
        self._cursor_pen = cursor_pen
        self._playback_timer_ms = playback_timer_ms

        self._plot_item = None
        self._view_box = None
        self._curve = None
        self._highlight_curves = {}
        self._focus = focus
        self._opacity = opacity
        self._z = z

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

        self._curve = pg.PlotCurveItem(pen=self._base_pen)
        self._curve.setOpacity(self._opacity)
        self._curve.setZValue(self._z)
        self._plot_item.addItem(self._curve)

        if self._zero_line_visible:
            pen = self._zero_line_pen
            if pen is None:
                pen = pg.mkPen(150, 150, 150, style=pg.QtCore.Qt.DashLine)
            self._zero = pg.InfiniteLine(angle=0, movable=False, pen=pen)
            self._zero.setOpacity(self._opacity)
            self._zero.setZValue(self._z)
            self._plot_item.addItem(self._zero)
        else:
            self._zero = None

        if self._cursor is None:
            self._cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(self._cursor_pen, width=1.5))
            self._cursor.setVisible(False)
            self._plot_item.addItem(self._cursor)

        self._update_plot()

    def remove_from(self, plot_item):
        if self._curve is not None and self._curve.scene() is not None:
            self._plot_item.removeItem(self._curve)
        if getattr(self, "_zero", None) is not None and self._zero.scene() is not None:
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

    def set_opacity(self, alpha):
        self._opacity = alpha
        if self._curve is not None:
            self._curve.setOpacity(self._opacity)
        if hasattr(self, "_zero"):
            self._zero.setOpacity(self._opacity)
        for it in self._highlight_curves.values():
            it.setOpacity(self._opacity)

    def set_z(self, z):
        self._z = z
        if self._curve is not None:
            self._curve.setZValue(self._z)
        if hasattr(self, "_zero"):
            self._zero.setZValue(self._z)
        for it in self._highlight_curves.values():
            it.setZValue(self._z + 1)
        if self._selection is not None:
            self._selection.setZValue(self._z + 2)

    def set_focus(self, focus):
        self._focus = focus

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
        y = self.transform(raw).reshape(-1)
        x0, x1 = 0.0, max(1e-9, len(y) / self.sample_rate)
        y0, y1 = np.min(y), np.max(y)
        return (x0, x1, y0, y1)

    def _update_plot(self):
        data = self.data_source.get()

        order = self.argsort_func(data)
        reverse = np.empty_like(order)
        reverse[order] = np.arange(len(order))

        y_full = self.transform(data).reshape(-1).astype(np.float32)
        n = len(y_full)

        self._duration = n / self.sample_rate
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
            curve = pg.PlotCurveItem(self._time_axis, y_overlay, pen=pg.mkPen(color=color, width=self._highlight_pen_width))
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
        uid = getattr(self, "_canvas_uid", None)
        if getattr(self, "name", None):
            return f"{self.name}@{uid}" if uid is not None else self.name

        desc = getattr(self, "_descriptor", None)
        if desc is not None and hasattr(desc, "graph_name") and desc.graph_name:
            return f"{desc.graph_name}@{uid}" if uid is not None else desc.graph_name

        owner = getattr(getattr(self, "graph", None), "owner", None)
        if owner and hasattr(owner, "graph_name"):
            return owner.graph_name
        return f"Layer@{hex(id(self))}"

    def close(self):
        if hasattr(self, "graph") and hasattr(self.graph, "disconnect"):
            self.graph.disconnect()

    def get_color(self):
        return self._selection_color

    def set_selection_color(self, color):
        self._selection_color = str(color)

    def set_color(self, color):
        self.set_selection_color(color)

    def expose_actions(self, parent=None):
        actions = []

        play_action = pg.QtWidgets.QAction("Play waveform", parent)
        play_action.triggered.connect(self._play_audio)
        actions.append(play_action)

        return actions

    def clone(self):
        dup = LinePlot(
            self.data_source,
            sample_rate=self.sample_rate,
            transform=self.transform,
            argsort_func=self.argsort_func,
            name=self.name,
            selection_color=self._selection_color,
            focus=self._focus,
            opacity=self._opacity,
            z=self._z,
            base_pen=self._base_pen,
            highlight_pen_width=self._highlight_pen_width,
            zero_line_visible=self._zero_line_visible,
            zero_line_pen=self._zero_line_pen,
            cursor_pen=self._cursor_pen,
            playback_timer_ms=self._playback_timer_ms,
        )
        return dup

    def _play_audio(self):
        if self._waveform is None or self._time_axis is None or self._duration <= 0:
            return

        wf = self._waveform
        peak = np.max(np.abs(wf)) if wf.size else 1.0
        wf = (wf / peak) if peak > 0 else wf

        if self._cursor is not None:
            self._cursor.setVisible(True)
            self._cursor.setPos(0.0)

        self._playback_start_time = time.perf_counter()
        sd.play(wf, self.sample_rate, blocking=False)
        self._play_timer.start(self._playback_timer_ms)

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
