# graphs/LinePlot.py

from PyQt5 import QtCore, QtGui
from typing import Any, Optional
import numpy as np
import pyqtgraph as pg
from MCVGraph.GraphBus import GraphEventClient
from MCVGraph.EventType import EventType
from MCVGraph.BasePlot import GraphBase
import time
import warnings


class LinePlot(GraphBase):
    _DEFAULT_COLORS = ['red', 'lime', 'blue', 'magenta', 'cyan', 'orange']
    _color_counter = 0
    _COLOR_REGISTRY = {}

    def __init__(
        self,
        data_source: Any,
        *,
        sample_rate: int = 44100,
        transform: Optional[Any] = None,
        argsort_func: Optional[Any] = None,
        name: Optional[str] = None,
        selection_color: Optional[str] = None,
        focus: bool = False,
        opacity: float = 1.0,
        z: int = 10,
        base_pen: str = "gray",
        highlight_pen_width: float = 1.5,
        zero_line_visible: bool = True,
        zero_line_pen: Optional[Any] = None,
        cursor_pen: str = "r",
        playback_timer_ms: int = 16,
    ) -> None:
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

        self._subset_only_mode = False

        self._cursor = None
        self._play_timer = QtCore.QTimer()
        self._play_timer.timeout.connect(self._update_cursor)
        self._playback_start_time = None
        self._waveform = None

    def handle_event(self, event_type: str, payload: dict[str, Any]) -> None:
        super().handle_event(event_type, payload)

    def add_to(self, plot_item: Any, view_box: Any) -> None:
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

    def remove_from(self, plot_item: Any) -> None:
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

    def set_opacity(self, alpha: float) -> None:
        self._opacity = alpha
        if self._curve is not None:
            self._curve.setOpacity(self._opacity)
        if hasattr(self, "_zero"):
            self._zero.setOpacity(self._opacity)
        for it in self._highlight_curves.values():
            it.setOpacity(self._opacity)

    def set_z(self, z: int) -> None:
        self._z = z
        if self._curve is not None:
            self._curve.setZValue(self._z)
        if hasattr(self, "_zero"):
            self._zero.setZValue(self._z)
        for it in self._highlight_curves.values():
            it.setZValue(self._z + 1)
        if self._selection is not None:
            self._selection.setZValue(self._z + 2)

    def set_focus(self, focus: bool) -> None:
        self._focus = focus

    def set_transform(self, transform: Any, argsort: Any = lambda x: np.arange(len(x))) -> None:
        self.transform = transform
        self.argsort_func = argsort
        self._update_plot()

    def select_range(self, t_min: float, t_max: float, source: Optional[str] = None) -> None:
        """
        Select a contiguous time range on the waveform.
        - Uses the computed time axis to map selection [t_min, t_max]
          back to raw data indices.
        - Broadcasts SUBSET_INDICES so linked plots highlight the same region.
        """
        if self._order_indices is None or self._time_axis is None:
            return

        mask = (self._time_axis >= t_min) & (self._time_axis <= t_max)
        raw_indices = self._order_indices[mask]
        payload = {"indices": raw_indices.astype(int).copy(), "source": self._source_name()}
        self.graph.emit_broadcast(EventType.SUBSET_INDICES, payload)

    def get_selection_indices(self) -> np.ndarray:
        src = self._source_name()
        return self._highlight_indices.get(src, (np.array([], dtype=int), None))[0]

    def bounds(self) -> tuple[float, float, float, float]:
        raw = self.data_source.get()
        y = self.transform(raw).reshape(-1)
        x0, x1 = 0.0, max(1e-9, len(y) / self.sample_rate)
        y0, y1 = np.min(y), np.max(y)
        return (x0, x1, y0, y1)

    def _update_plot(self) -> None:
        """
        Redraw the line plot with current data and highlights.
        Steps:
        1. Fetch raw waveform samples and apply optional subset filtering.
        2. Apply argsort_func to determine plotting order (e.g., sorting).
        3. Transform waveform (e.g., scaling/normalization).
        4. Build time axis (x = time, y = waveform amplitude).
        5. Draw the base curve.
        6. Overlay highlight curves (each in its own color).
        """
        if self._plot_item is None:
            return

        raw = self.data_source.get()
        n_raw = self.data_source.size()
        if n_raw <= 0:
            return

        subset_only = getattr(self, "_subset_only_mode", False)

        # Restrict allowed indices if subset-only mode is active
        allowed_mask = np.ones(n_raw, dtype=bool)
        if subset_only:
            combined = np.zeros(n_raw, dtype=bool)
            for src, (inds, _) in self._highlight_indices.items():
                if src != self._source_name():
                    ii = np.asarray(inds, dtype=int)
                    ii = ii[(ii >= 0) & (ii < n_raw)]
                    combined[ii] = True
            if np.any(combined):
                allowed_mask = combined

        # Select raw samples to display
        if subset_only and np.any(allowed_mask):
            view_raw_idx = np.where(allowed_mask)[0]
            data_view = raw[allowed_mask]
        else:
            view_raw_idx = np.arange(n_raw)
            data_view = raw

        # Compute plotting order (argsort) and ensure validity
        order_view = self.argsort_func(data_view)
        order_view = np.asarray(order_view, dtype=int)
        order_view = order_view[(order_view >= 0) & (order_view < len(data_view))]
        if order_view.size != len(data_view):
            order_view = np.arange(len(data_view), dtype=int)

        # Transform and prepare data
        y_view = self.transform(data_view).reshape(-1).astype(np.float32)
        n_view = len(y_view)
        if n_view == 0:
            return
        y_plot = y_view[order_view]

        # Build mappings between raw and view indices
        sub_index_of_raw = np.full(n_raw, -1, dtype=int)
        sub_index_of_raw[view_raw_idx] = np.arange(n_view)
        reverse_view = np.empty_like(order_view)
        reverse_view[order_view] = np.arange(n_view)

        # Build time axis (x in seconds)
        self._duration = n_view / self.sample_rate
        self._time_axis = np.linspace(0.0, self._duration, n_view, endpoint=False)
        self._order_indices = view_raw_idx[order_view]

        # Update base curve
        if self._curve is not None:
            self._curve.setData(self._time_axis, y_plot)
        self._waveform = y_plot

        # Remove old highlight overlays
        for it in list(self._highlight_curves.values()):
            if it.scene() is not None:
                self._plot_item.removeItem(it)
        self._highlight_curves.clear()

        # Draw highlights as separate curves
        for src, (global_idx_raw, color) in self._highlight_indices.items():
            gi = np.asarray(global_idx_raw, dtype=int)
            gi = gi[(gi >= 0) & (gi < n_raw)]
            if gi.size == 0:
                continue

            sub_idx = sub_index_of_raw[gi]
            sub_idx = sub_idx[sub_idx >= 0]
            if sub_idx.size == 0:
                continue

            pos = reverse_view[sub_idx]
            pos = np.sort(pos)

            overlay = np.full(n_view, np.nan, dtype=np.float32)
            overlay[pos] = y_plot[pos]

            curve = pg.PlotCurveItem(self._time_axis, overlay,
                                     pen=pg.mkPen(color=color, width=self._highlight_pen_width))
            curve.setOpacity(self._opacity)
            curve.setZValue(self._z + 1)
            self._plot_item.addItem(curve)
            self._highlight_curves[src] = curve

    def handle_scene_event(self, event: QtCore.QEvent, view_box: Any) -> bool:
        """
        Handle mouse events for interactive selection on the waveform.
        - Press: start or clear a LinearRegionItem.
        - Move: drag out a new selection range and emit highlight event.
        - Release: finalize the region.
        Returns True if event was consumed.
        """
        # Ignore if Ctrl is pressed (reserved for multi-select in ScatterPlot)
        if QtGui.QGuiApplication.keyboardModifiers() & QtCore.Qt.ControlModifier:
            return False
        if not self._focus:
            return False

        et = event.type()

        if et == QtCore.QEvent.GraphicsSceneMousePress:
            if event.button() != QtCore.Qt.LeftButton:
                return False

            click_x = view_box.mapSceneToView(event.scenePos()).x()

            # If selection exists, check if click is inside it
            if self._selection is not None:
                sel_min, sel_max = self._selection.getRegion()
                eps = max(1e-9, (sel_max - sel_min) * 0.01)
                if (sel_min - eps) <= click_x <= (sel_max + eps):
                    return False
                else:
                    # Clear old selection and broadcast CLEAR_HIGHLIGHT
                    if self._selection.scene() is not None:
                        self._plot_item.removeItem(self._selection)
                    self._selection = None
                    self._selection_origin = event.scenePos()
                    payload = {"source": self._source_name()}
                    self.graph.emit_broadcast(EventType.CLEAR_HIGHLIGHT, payload)
                    return True

            # Otherwise start new selection
            self._selection_origin = event.scenePos()
            return True

        elif et == QtCore.QEvent.GraphicsSceneMouseMove:
            if self._selection_origin is None:
                return False

            # Convert scene coords to time values
            start_x = view_box.mapSceneToView(self._selection_origin).x()
            current_x = view_box.mapSceneToView(event.scenePos()).x()
            lo, hi = (min(start_x, current_x), max(start_x, current_x))

            # Create new selection region if not exists
            if self._selection is None:
                self._selection = pg.LinearRegionItem(values=[lo, hi], orientation=pg.LinearRegionItem.Vertical)
                self._selection.setZValue(self._z + 2)
                self._plot_item.addItem(self._selection)
                self._selection.sigRegionChanged.connect(self._on_selection_region_changed)
            else:
                self._selection.setRegion([lo, hi])

            # Emit highlight for the current drag
            self.select_range(lo, hi)
            return True

        elif et == QtCore.QEvent.GraphicsSceneMouseRelease:
            if self._selection_origin is None:
                return False
            self._selection_origin = None
            return True

        return False

    def _on_selection_region_changed(self) -> None:
        if self._selection is None:
            return
        lo, hi = self._selection.getRegion()
        self.select_range(lo, hi)

    def _source_name(self) -> str:
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

    def close(self) -> None:
        if hasattr(self, "graph") and hasattr(self.graph, "disconnect"):
            self.graph.disconnect()

    def get_color(self) -> str:
        return self._selection_color

    def set_selection_color(self, color: str) -> None:
        self._selection_color = str(color)

    def set_color(self, color: str) -> None:
        self.set_selection_color(color)

    def expose_actions(self, parent: Optional[Any] = None) -> list[Any]:
        actions = []

        subset_action = pg.QtWidgets.QAction("Filter by Selection (subset-only mode)", parent, checkable=True)
        subset_action.setChecked(getattr(self, "_subset_only_mode", False))
        def _toggle_subset():
            self._subset_only_mode = subset_action.isChecked()
            self._update_plot()
        subset_action.toggled.connect(_toggle_subset)
        actions.append(subset_action)

        play_action = pg.QtWidgets.QAction("Play waveform", parent)
        play_action.triggered.connect(self._play_audio)
        actions.append(play_action)

        base_color_action = pg.QtWidgets.QAction("Set Non-selected Color…", parent)
        def _set_base_color():
            col = pg.QtWidgets.QColorDialog.getColor(QtGui.QColor(self._base_pen), None, "Pick base color")
            if col.isValid():
                self._base_pen = col.name()
                if self._curve is not None:
                    self._curve.setPen(pg.mkPen(self._base_pen))
                self._update_plot()
        base_color_action.triggered.connect(_set_base_color)
        actions.append(base_color_action)

        sel_color_action = pg.QtWidgets.QAction("Set Selection Color…", parent)
        def _set_sel_color():
            col = pg.QtWidgets.QColorDialog.getColor(QtGui.QColor(self._selection_color), None, "Pick selection color")
            if col.isValid():
                self.set_selection_color(col.name())
                self._update_plot()
        sel_color_action.triggered.connect(_set_sel_color)
        actions.append(sel_color_action)

        return actions

    def clone(self) -> "LinePlot":
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

    def _play_audio(self) -> None:
        """
        Playback the waveform using sounddevice.
        - Normalizes amplitude to [-1, 1].
        - Makes cursor visible and starts it at t=0.
        - Starts a timer to update cursor position in sync with playback.
        """
        if self._waveform is None or self._time_axis is None or self._duration <= 0:
            return

        wf = self._waveform
        peak = np.max(np.abs(wf)) if wf.size else 1.0
        wf = (wf / (peak * 2.0)) if peak > 0 else wf

        if self._cursor is not None:
            self._cursor.setVisible(True)
            self._cursor.setPos(0.0)

        self._playback_start_time = time.perf_counter()
        try:
            import sounddevice as sd
            sd.play(wf, self.sample_rate, blocking=False)
            self._play_timer.start(self._playback_timer_ms)
        except Exception as e:
            warnings.warn(f"Waveform playback unavailable: {e}")
            self._cursor.setVisible(False)
            self._playback_start_time = None


    def _update_cursor(self) -> None:
        """
        Timer callback to animate the playback cursor.
        - Computes elapsed playback time.
        - Hides and stops timer when end is reached.
        - Otherwise moves cursor to current time.
        """
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
