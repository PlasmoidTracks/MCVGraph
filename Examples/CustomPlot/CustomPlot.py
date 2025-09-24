import numpy as np
import pyqtgraph as pg
from typing import Any, Optional
from PyQt5 import QtCore, QtGui, QtWidgets
from MCVGraph.BasePlot import GraphBase
from MCVGraph.GraphBus import GraphEventClient
from MCVGraph.EventType import EventType

class CustomPlot(GraphBase):
    """
    Minimal custom plot with:
      - Base scatter rendering
      - Circle selection (click-drag from first click as center)
      - Broadcasting of selected indices via GraphBus
      - Receiving highlights from other plots and overlaying them
    """

    def __init__(self, data_source: Any, color: str = "blue", size: int = 8, non_selected_color: str = "gray") -> None:
        self.data_source = data_source
        self.color = color
        self.size = size
        self.non_selected_color = non_selected_color

        self._plot_item = None
        self._view_box = None
        self._scatter = None

        # overlays keyed by source name -> ScatterPlotItem
        self._highlight_items: dict[str, pg.ScatterPlotItem] = {}
        self._highlight_indices: dict[str, tuple[np.ndarray, str]] = {}

        # selection state
        self._circle_center_scene: Optional[QtCore.QPointF] = None
        self._circle_center_view: Optional[QtCore.QPointF] = None
        self._circle_radius_view: Optional[float] = None
        self._circle_item: Optional[QtWidgets.QGraphicsEllipseItem] = None

        # Connect to the GraphBus
        self.graph = GraphEventClient(self, "CustomPlot")

        # Re-draw when data changes
        self.data_source.data_updated.connect(self._update_plot)

    # -------------------- GraphBase required hooks --------------------

    def add_to(self, plot_item: Any, view_box: Any) -> None:
        self._plot_item = plot_item
        self._view_box = view_box

        self._scatter = pg.ScatterPlotItem(pen=None, size=self.size)
        self._plot_item.addItem(self._scatter)

        if hasattr(self._view_box, "sigRangeChanged"):
            self._view_box.sigRangeChanged.connect(self._on_view_changed)

        self._update_plot()

    def remove_from(self, plot_item: Any) -> None:
        if self._scatter is not None and self._scatter.scene() is not None:
            plot_item.removeItem(self._scatter)
        for it in list(self._highlight_items.values()):
            if it.scene() is not None:
                plot_item.removeItem(it)
        self._highlight_items.clear()

        if self._circle_item is not None and self._circle_item.scene() is not None:
            plot_item.scene().removeItem(self._circle_item)

        self._scatter = None
        self._plot_item = None
        self._view_box = None
        self._circle_item = None
        self._circle_center_scene = None
        self._circle_center_view = None
        self._circle_radius_view = None

    def _update_plot(self) -> None:
        if self._scatter is None:
            return
        pts = self.data_source.get()
        if pts is None or len(pts) == 0:
            return

        # Base layer: draw all as non-selected
        self._scatter.setData(
            x=pts[:, 0],
            y=pts[:, 1],
            brush=self.non_selected_color,
            size=self.size
        )

        # Clear existing highlight overlays
        for it in list(self._highlight_items.values()):
            if it.scene() is not None and self._plot_item is not None:
                self._plot_item.removeItem(it)
        self._highlight_items.clear()

        # Draw per-source highlight overlays
        n = len(pts)
        for src, (indices, color) in self._highlight_indices.items():
            ii = np.asarray(indices, dtype=int)
            ii = ii[(ii >= 0) & (ii < n)]
            if ii.size == 0:
                continue
            scat = pg.ScatterPlotItem(pen=None)
            scat.setData(
                x=pts[ii, 0],
                y=pts[ii, 1],
                brush=color,
                size=max(1, self.size + 2)
            )
            self._plot_item.addItem(scat)
            self._highlight_items[src] = scat

        # Keep selection circle in sync with view transforms
        self._on_view_changed()

    def _source_name(self) -> str:
        uid = getattr(self, "_canvas_uid", None)
        return f"CustomPlot@{uid}" if uid is not None else "CustomPlot"

    def clone(self) -> "CustomPlot":
        return CustomPlot(self.data_source, color=self.color, size=self.size, non_selected_color=self.non_selected_color)

    # -------------------- Interaction & Bus integration --------------------

    def get_color(self) -> str:
        """Used by GraphEventClient to auto-fill highlight color if not provided."""
        return self.color

    def handle_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """
        Delegate to GraphBase for common behavior (maintains _highlight_indices).
        Then refresh our plot to reflect changes.
        """
        super().handle_event(event_type, payload)
        # GraphBase updates self._highlight_indices and calls _update_plot(),
        # but we ensure refresh in case of future extensions.
        # (No-op here since _update_plot already called.)

    # Circle selection API ----------------------------------------------------

    def _emit_selection(self, center_x: float, center_y: float, radius: float) -> None:
        """Compute indices in circle and broadcast via GraphBus."""
        pts = self.data_source.get()
        if pts is None or len(pts) == 0:
            return

        dx = pts[:, 0] - center_x
        dy = pts[:, 1] - center_y
        mask = (dx * dx + dy * dy) <= (radius * radius)
        indices = np.where(mask)[0].astype(int)

        payload = {"indices": indices.copy(), "source": self._source_name()}
        # color is auto-inserted by GraphEventClient if get_color() exists
        self.graph.emit_broadcast(EventType.SUBSET_INDICES, payload)

    def _clear_selection(self) -> None:
        payload = {"source": self._source_name()}
        self.graph.emit_broadcast(EventType.CLEAR_HIGHLIGHT, payload)

    # Scene event handling (Canvas will forward to us when focused) ----------

    def handle_scene_event(self, event: QtCore.QEvent, view_box: Any) -> bool:
        et = event.type()

        # CTRL modifiers are reserved for pan/zoom in your ViewBox
        if QtGui.QGuiApplication.keyboardModifiers() & QtCore.Qt.ControlModifier:
            return False

        if et == QtCore.QEvent.GraphicsSceneMousePress:
            if event.button() != QtCore.Qt.LeftButton:
                return False

            pos_scene = event.scenePos()
            self._circle_center_scene = pos_scene
            self._circle_center_view = view_box.mapSceneToView(pos_scene)
            self._circle_radius_view = 0.0

            # Create the visual circle item
            if self._circle_item is not None and self._circle_item.scene() is not None:
                view_box.scene().removeItem(self._circle_item)
            self._circle_item = QtWidgets.QGraphicsEllipseItem()
            self._circle_item.setPen(QtGui.QPen(QtGui.QColor("gold"), 2))
            self._circle_item.setBrush(QtGui.QBrush(QtGui.QColor(255, 215, 0, 50)))
            view_box.scene().addItem(self._circle_item)

            # Single click without drag will clear selection on release
            return True

        elif et == QtCore.QEvent.GraphicsSceneMouseMove:
            if self._circle_center_view is None or self._circle_item is None:
                return False

            cur_view = view_box.mapSceneToView(event.scenePos())
            cx, cy = self._circle_center_view.x(), self._circle_center_view.y()
            r = np.hypot(cur_view.x() - cx, cur_view.y() - cy)
            self._circle_radius_view = float(r)

            # Update circle geometry (convert view to scene for drawing)
            self._update_circle_item_from_view(view_box)

            # Emit selection continuously while dragging
            self._emit_selection(cx, cy, r)
            return True

        elif et == QtCore.QEvent.GraphicsSceneMouseRelease:
            # If there was no drag (very small radius), interpret as "clear"
            if self._circle_center_view is not None and (self._circle_radius_view is None or self._circle_radius_view < 1e-6):
                self._clear_selection()

            self._circle_center_scene = None
            self._circle_center_view = None
            self._circle_radius_view = None
            return True

        return False

    def _update_circle_item_from_view(self, view_box: Any) -> None:
        if self._circle_item is None or self._circle_center_view is None or self._circle_radius_view is None:
            return
        cx, cy = self._circle_center_view.x(), self._circle_center_view.y()
        r = self._circle_radius_view

        tl_view = QtCore.QPointF(cx - r, cy - r)
        br_view = QtCore.QPointF(cx + r, cy + r)
        tl_scene = view_box.mapViewToScene(tl_view)
        br_scene = view_box.mapViewToScene(br_view)
        rect = QtCore.QRectF(tl_scene, br_scene).normalized()
        self._circle_item.setRect(rect)

    def _on_view_changed(self) -> None:
        # Keep the circle overlay aligned to the data-space circle when zooming/panning
        if self._circle_item is None:
            return
        if self._view_box is None:
            return
        if self._circle_center_view is None or self._circle_radius_view is None:
            return
        self._update_circle_item_from_view(self._view_box)
