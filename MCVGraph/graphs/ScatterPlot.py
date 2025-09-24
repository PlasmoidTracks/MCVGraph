# graphs/ScatterPlot.py

from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Any, Optional
from PyQt5.QtWidgets import QGraphicsRectItem
import numpy as np
import pyqtgraph as pg
from MCVGraph.GraphBus import GraphEventClient
from MCVGraph.EventType import EventType
from MCVGraph.BasePlot import GraphBase

class ScatterPlot(GraphBase):
    _DEFAULT_COLORS = ['red', 'lime', 'blue', 'magenta', 'cyan', 'orange']
    _color_counter = 0

    def __init__(
        self,
        data_source: Any,
        *,
        transform: Optional[Any] = None,
        base_marker_size: int = 6,
        color_non_selected: str = "gray",
        selection_color: Optional[str] = None,
        name: Optional[str] = None,
        focus: bool = False,
        opacity: float = 1.0,
        z: int = 10,
        selection_locked: bool = False,
        x_label: str = "",
        y_label: str = "",
        axis_labels_visible: bool = True,
    ) -> None:
        self.data_source = data_source
        self.transform = (transform if transform is not None else (lambda x: x))
        self._base_marker_size = base_marker_size if base_marker_size is not None else 6
        self.color_non_selected = color_non_selected

        self._last_ds_version = -1
        self._cached_xy = None

        self._update_pending = False

        self._last_emitted_indices = None

        if selection_color is None:
            idx = ScatterPlot._color_counter % len(ScatterPlot._DEFAULT_COLORS)
            self._selection_color = ScatterPlot._DEFAULT_COLORS[idx]
            ScatterPlot._color_counter += 1
        else:
            self._selection_color = str(selection_color)

        self.name = name

        self._plot_item = None
        self._view_box = None
        self._scatter_base = None
        self._highlight_items = {}
        self._focus = focus
        self._opacity = opacity
        self._z = z

        self.selection_locked = selection_locked
        self._selection_rect = None
        self._selection_origin = None
        self._dragging = False
        self._drag_offset = None
        self._click_clear_pending = False
        self._click_press_pos = None
        self.selection_rect_view_coords = None

        self._x_label = str(x_label)
        self._y_label = str(y_label)
        self._axis_labels_visible = axis_labels_visible

        self._highlight_indices = {}

        self.graph = GraphEventClient(self, "ScatterPlot")

        self.data_source.data_updated.connect(self._update_plot)

        self._subset_only_mode = False

    def handle_event(self, event_type: str, payload: dict[str, Any]) -> None:
        super().handle_event(event_type, payload)

    def add_to(self, plot_item: Any, view_box: Any) -> None:
        self._plot_item = plot_item
        self._view_box = view_box
        self._scatter_base = pg.ScatterPlotItem(pen=None, size=self._base_marker_size)
        self._plot_item.addItem(self._scatter_base)
        self._scatter_base.setOpacity(self._opacity)
        self._scatter_base.setZValue(self._z)

        if hasattr(self._view_box, "sigRangeChanged"):
            self._view_box.sigRangeChanged.connect(self._on_view_changed)

        self._update_plot()

    def remove_from(self, plot_item: Any) -> None:
        if self._scatter_base is not None and self._scatter_base.scene() is not None:
            self._plot_item.removeItem(self._scatter_base)
        for it in list(self._highlight_items.values()):
            if it.scene() is not None:
                self._plot_item.removeItem(it)
        self._highlight_items.clear()
        self._plot_item = None
        self._view_box = None

    def set_opacity(self, alpha: float) -> None:
        self._opacity = alpha
        if self._scatter_base is not None:
            self._scatter_base.setOpacity(self._opacity)
        for it in self._highlight_items.values():
            it.setOpacity(self._opacity)

    def set_z(self, z: int) -> None:
        self._z = z
        if self._scatter_base is not None:
            self._scatter_base.setZValue(self._z)
        for it in self._highlight_items.values():
            it.setZValue(self._z + 1)

    def set_focus(self, focus: bool) -> None:
        self._focus = focus

    def select_rectangle(self, x0: float, y0: float, x1: float, y1: float, source: Optional[str] = None) -> None:
        """
        Select points within a rectangular region in data space.
        - Supports "subset-only mode": restricts selection to points
          already highlighted by other plots.
        - Otherwise applies selection over the full dataset.
        - Emits a SUBSET_INDICES event with the raw indices of selected points.
        """
        # Normalize rectangle so x0<x1 and y0<y1
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])

        raw = self.data_source.get()
        n_raw = len(raw)
        subset_only = getattr(self, "_subset_only_mode", False)

        # Default: all points allowed
        allowed_mask = np.ones(n_raw, dtype=bool)

        # Restrict allowed set if subset-only mode is active
        if subset_only:
            combined = np.zeros(n_raw, dtype=bool)
            for src, (inds, _) in self._highlight_indices.items():
                if src != self._source_name():
                    ii = np.asarray(inds, dtype=int)
                    ii = ii[(ii >= 0) & (ii < n_raw)]
                    combined[ii] = True
            if np.any(combined):
                allowed_mask = combined

        # Apply selection to either subset view or full data
        if subset_only and np.any(allowed_mask):
            view_raw_idx = np.where(allowed_mask)[0]
            data_view = self.transform(raw[allowed_mask])
            rect_mask = (
                (data_view[:, 0] >= x0) & (data_view[:, 0] <= x1) &
                (data_view[:, 1] >= y0) & (data_view[:, 1] <= y1)
            )
            indices_raw = view_raw_idx[np.where(rect_mask)[0]]
        else:
            data_full = self.transform(raw)
            rect_mask = (
                (data_full[:, 0] >= x0) & (data_full[:, 0] <= x1) &
                (data_full[:, 1] >= y0) & (data_full[:, 1] <= y1)
            )
            indices_raw = np.where(rect_mask)[0]

        # Broadcast the selection
        payload = {"indices": indices_raw.astype(int).copy(), "source": self._source_name()}
        self.graph.emit_broadcast(EventType.SUBSET_INDICES, payload)

    def select_indices(self, indices: Any, source: Optional[str] = None) -> None:
        indices = np.asarray(indices, dtype=int)
        payload = {"indices": indices.copy(), "source": self._source_name()}
        self.graph.emit_broadcast(EventType.SUBSET_INDICES, payload)

    def get_selection_indices(self) -> np.ndarray:
        src = self._source_name()
        if src in self._highlight_indices:
            return self._highlight_indices[src][0].copy()
        return np.array([], dtype=int)

    def bounds(self) -> tuple[float, float, float, float]:
        raw = self.data_source.get()
        data = self.transform(raw)
        x0, x1 = np.min(data[:, 0]), np.max(data[:, 0])
        y0, y1 = np.min(data[:, 1]), np.max(data[:, 1])
        return (x0, x1, y0, y1)

    def _update_plot(self) -> None:
        """
        Redraw the scatter plot based on the current dataset and highlight state.
        Steps:
        1. Fetch raw data from the source and transform it.
        2. Apply subset-only filtering if enabled.
        3. Build inverse index mapping from raw indices to view indices.
        4. Draw non-highlighted points in base color.
        5. For each highlight set:
           - Compute overlapping indices.
           - Create ScatterPlotItem layers with unique colors/sizes.
        6. Ensure highlight layers are stacked above base points.
        """
        if self._plot_item is None:
            return

        raw = self.data_source.get()
        n_raw = self.data_source.size()
        if n_raw == 0:
            return

        subset_only = getattr(self, "_subset_only_mode", False)

        # Determine which raw points are allowed
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

        # Map raw indices to transformed view indices
        if subset_only and np.any(allowed_mask):
            view_raw_idx = np.where(allowed_mask)[0]
            data = self.transform(raw[allowed_mask])
            n_view = len(data)
            inv_map = np.full(n_raw, -1, dtype=int)
            inv_map[view_raw_idx] = np.arange(n_view)
        else:
            data = self.transform(raw)
            n_view = len(data)
            inv_map = np.arange(n_raw, dtype=int)
            if n_view != n_raw:
                inv_map = np.clip(inv_map, 0, max(0, n_view - 1))

        if n_view == 0:
            return

        # Draw base layer (non-highlighted points)
        combined_highlight_mask = np.zeros(n_view, dtype=bool)
        for inds, _ in self._highlight_indices.values():
            ii_raw = np.asarray(inds, dtype=int)
            ii_raw = ii_raw[(ii_raw >= 0) & (ii_raw < n_raw)]
            pos = inv_map[ii_raw]
            pos = pos[pos >= 0]
            combined_highlight_mask[pos] = True

        base_mask = ~combined_highlight_mask
        self._scatter_base.setData(
            x=data[base_mask, 0],
            y=data[base_mask, 1],
            brush=self.color_non_selected,
            size=self._base_marker_size
        )

        # Clear previous highlight layers
        for it in list(self._highlight_items.values()):
            if it.scene() is not None:
                self._plot_item.removeItem(it)
        self._highlight_items.clear()

        # Build new highlight layers
        max_draw_counts = np.zeros(n_view, dtype=int)
        for inds, _ in self._highlight_indices.values():
            ii_raw = np.asarray(inds, dtype=int)
            ii_raw = ii_raw[(ii_raw >= 0) & (ii_raw < n_raw)]
            pos = inv_map[ii_raw]
            pos = pos[pos >= 0]
            np.add.at(max_draw_counts, pos, 1)

        draw_counts = np.zeros(n_view, dtype=int)
        highlight_base_size = self._base_marker_size + 2
        size_decrement = 3

        for src, (inds, color) in self._highlight_indices.items():
            ii_raw = np.asarray(inds, dtype=int)
            ii_raw = ii_raw[(ii_raw >= 0) & (ii_raw < n_raw)]
            pos = inv_map[ii_raw]
            pos = pos[pos >= 0]
            if pos.size == 0:
                continue

            spots = []
            for p in pos:
                total = max_draw_counts[p]
                size = max(2, highlight_base_size + ((total - 1) * 2) - draw_counts[p] * size_decrement)
                spots.append({'pos': data[p], 'brush': color, 'size': size})
                draw_counts[p] += 1

            scat = pg.ScatterPlotItem(pen=None)
            scat.setData(spots)
            scat.setOpacity(self._opacity)
            scat.setZValue(self._z + 1)
            self._plot_item.addItem(scat)
            self._highlight_items[src] = scat

    def handle_scene_event(self, event: QtCore.QEvent, view_box: Any) -> bool:
        """
        Handle mouse interactions for selection.
        - MousePress: start new selection rectangle or prepare drag of existing one.
        - MouseMove: update rectangle geometry, apply selection, support dragging.
        - MouseRelease: finalize or clear selection.
        Returns True if the event was consumed.
        """
        if not self._focus:
            return False

        et = event.type()
        if self.selection_locked:
            return False

        if et == QtCore.QEvent.GraphicsSceneMousePress:
            # Left-click to start new rectangle, unless Ctrl is held
            if event.button() != QtCore.Qt.LeftButton:
                return False
            if QtGui.QGuiApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
                return False
            pos = event.scenePos()

            # If clicking inside an existing rect → start dragging it
            if self._selection_rect is not None:
                rect = self._selection_rect.rect()
                if rect.contains(pos):
                    self._dragging = True
                    self._drag_offset = pos - rect.topLeft()
                    self._click_clear_pending = False
                    self._click_press_pos = None
                    return True

            # Otherwise prepare to start a new selection
            self._click_clear_pending = True
            self._click_press_pos = pos
            return True

        elif et == QtCore.QEvent.GraphicsSceneMouseMove:
            # If dragging an existing rectangle
            if self._dragging and self._selection_rect is not None and self._drag_offset is not None:
                rect = self._selection_rect.rect()
                size = rect.size()
                new_top_left = event.scenePos() - self._drag_offset
                new_rect = QtCore.QRectF(new_top_left, size)
                self._selection_rect.setRect(new_rect)

                tl = view_box.mapSceneToView(new_rect.topLeft())
                br = view_box.mapSceneToView(new_rect.bottomRight())
                self.selection_rect_view_coords = (tl.x(), tl.y(), br.x(), br.y())
                self.select_rectangle(tl.x(), tl.y(), br.x(), br.y())
                return True

            # If starting a new selection rectangle
            if self._click_clear_pending and self._click_press_pos is not None:
                if (event.scenePos() - self._click_press_pos).manhattanLength() >= 4:
                    self._selection_origin = self._click_press_pos
                    self._click_clear_pending = False

                    # Remove previous rect if still present
                    if self._selection_rect is not None and self._selection_rect.scene() is not None:
                        self._plot_item.scene().removeItem(self._selection_rect)

                    # Create a new rectangle item
                    self._selection_rect = QGraphicsRectItem()
                    self._selection_rect.setPen(QtGui.QPen(QtGui.QColor("gold"), 2))
                    self._selection_rect.setBrush(QtGui.QBrush(QtGui.QColor(255, 215, 0, 50)))
                    self._plot_item.scene().addItem(self._selection_rect)

            # Update geometry while dragging out the rectangle
            if self._selection_origin is not None and self._selection_rect is not None:
                rect = QtCore.QRectF(self._selection_origin, event.scenePos()).normalized()
                self._selection_rect.setRect(rect)
                tl = view_box.mapSceneToView(rect.topLeft())
                br = view_box.mapSceneToView(rect.bottomRight())
                self.selection_rect_view_coords = (tl.x(), tl.y(), br.x(), br.y())
                self.select_rectangle(tl.x(), tl.y(), br.x(), br.y())
                return True

            return False

        elif et == QtCore.QEvent.GraphicsSceneMouseRelease:
            if self._click_clear_pending:
                # Click release without drag → clear selection
                self._emit_clear_highlight()
                self._remove_selection_rect()
                self._click_clear_pending = False
                self._click_press_pos = None
                return True

            if self._dragging or self._selection_origin is not None:
                # Finish rectangle drag
                self._dragging = False
                self._drag_offset = None
                self._selection_origin = None
                return True

            return False

        return False

    def _emit_clear_highlight(self) -> None:
        payload = {"source": self._source_name()}
        self.graph.emit_broadcast(EventType.CLEAR_HIGHLIGHT, payload)

    def _remove_selection_rect(self) -> None:
        if self._selection_rect is not None and self._selection_rect.scene() is not None:
            self._plot_item.scene().removeItem(self._selection_rect)
        self._selection_rect = None

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

    def set_transform(self, transform: Any) -> None:
        self.transform = transform
        self._update_plot()

    def set_base_marker_size(self, size: int) -> None:
        self._base_marker_size = max(1, size)
        if self._scatter_base is not None:
            self._scatter_base.setSize(self._base_marker_size)
        self._update_plot()

    def set_color_non_selected(self, color: str) -> None:
        self.color_non_selected = color
        self._update_plot()

    def expose_actions(self, parent: Optional[Any] = None) -> list[Any]:
        parent = parent or None

        subset_action = QtWidgets.QAction("Filter by Selection (subset-only mode)", parent, checkable=True)
        subset_action.setChecked(getattr(self, "_subset_only_mode", False))
        def _toggle_subset():
            self._subset_only_mode = subset_action.isChecked()
            self._update_plot()
        subset_action.toggled.connect(_toggle_subset)

        lock_action = QtWidgets.QAction("Lock Selection", parent, checkable=True)
        lock_action.setChecked(self.selection_locked)
        def _toggle_lock():
            self.set_selection_locked(lock_action.isChecked())
        lock_action.toggled.connect(_toggle_lock)

        marker_action = QtWidgets.QAction(f"Set Base Marker Size (current: {self._base_marker_size})", parent)
        def _set_marker():
            val, ok = QtWidgets.QInputDialog.getInt(
                None, "Set Base Marker Size", "Base marker size (1 - 16):",
                self._base_marker_size, 1, 16, 1
            )
            if ok:
                self.set_base_marker_size(val)
        marker_action.triggered.connect(_set_marker)

        color_action = QtWidgets.QAction("Set Non-selected Color…", parent)
        def _set_color():
            col = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.color_non_selected), None, "Pick color")
            if col.isValid():
                self.set_color_non_selected(col.name())
        color_action.triggered.connect(_set_color)

        sel_color_action = QtWidgets.QAction("Set Selection Color…", parent)
        def _set_sel_color():
            col = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._selection_color), None, "Pick selection color")
            if col.isValid():
                self.set_selection_color(col.name())
                self._update_plot()
        sel_color_action.triggered.connect(_set_sel_color)

        return [subset_action, lock_action, marker_action, color_action, sel_color_action]

    def set_selection_locked(self, locked: bool) -> None:
        self.selection_locked = bool(locked)
        if self._selection_rect is not None:
            if self.selection_locked:
                self._selection_rect.setPen(QtGui.QPen(QtGui.QColor("darkred"), 2))
                self._selection_rect.setBrush(QtGui.QBrush(QtGui.QColor(139, 0, 0, 60)))
            else:
                self._selection_rect.setPen(QtGui.QPen(QtGui.QColor("gold"), 2))
                self._selection_rect.setBrush(QtGui.QBrush(QtGui.QColor(255, 215, 0, 50)))

    def _on_view_changed(self) -> None:
        if self._selection_rect is None or self.selection_rect_view_coords is None:
            return
        x0, y0, x1, y1 = self.selection_rect_view_coords
        tl = self._view_box.mapViewToScene(QtCore.QPointF(x0, y0))
        br = self._view_box.mapViewToScene(QtCore.QPointF(x1, y1))
        rect = QtCore.QRectF(tl, br).normalized()
        self._selection_rect.setRect(rect)

    def get_selection_data(self) -> np.ndarray:
        idx = self.get_selection_indices()
        data = self.data_source.get()
        return data[idx]

    def get_color(self) -> str:
        return self._selection_color

    def set_selection_color(self, color: str) -> None:
        self._selection_color = str(color)

    def set_color(self, color: str) -> None:
        self.set_selection_color(color)

    def clone(self) -> "ScatterPlot":
        dup = ScatterPlot(
            self.data_source,
            transform=self.transform,
            base_marker_size=self._base_marker_size,
            color_non_selected=self.color_non_selected,
            selection_color=self._selection_color,
            name=getattr(self, "name", None),
            focus=self._focus,
            opacity=self._opacity,
            z=self._z,
            selection_locked=self.selection_locked,
            x_label=self._x_label,
            y_label=self._y_label,
            axis_labels_visible=self._axis_labels_visible,
        )
        return dup
