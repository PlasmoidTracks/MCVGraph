# graphs/HeatmapPlot.py

from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Any, Optional
import numpy as np
import pyqtgraph as pg
from MCVGraph.GraphBus import GraphEventClient
from MCVGraph.EventType import EventType
from MCVGraph.BasePlot import GraphBase

class HeatmapPlot(GraphBase):
    _DEFAULT_COLORS = ['red', 'lime', 'blue', 'magenta', 'cyan', 'orange']
    _color_counter = 0

    def __init__(
        self,
        data_source: Any,
        *,
        transform: Any = lambda x: x,
        normalizer: Optional[Any] = None,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        graph_name: str = "Heatmap",
    ) -> None:

        self.data_source = data_source
        self.transform = transform
        self.normalizer = normalizer or self._linear_normalizer
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.graph_name = graph_name

        idx = HeatmapPlot._color_counter % len(HeatmapPlot._DEFAULT_COLORS)
        self._selection_color = HeatmapPlot._DEFAULT_COLORS[idx]
        HeatmapPlot._color_counter += 1

        self._plot_item = None
        self._view_box = None
        self._img = None
        self._focus = False
        self._opacity = 1.0
        self._z = 10

        self._translation_x = 0.0
        self._translation_y = 0.0
        self._H = 0
        self._W = 0

        self._selection_rect = None
        self._selection_origin = None
        self.selection_locked = False
        self.selection_rect_view_coords = None

        self._highlight_overlay = None
        self.graph = GraphEventClient(self, "HeatmapPlot")
        self._highlight_indices = {}

        self._canvas_uid = getattr(self, "_canvas_uid", None)

        self.data_source.data_updated.connect(self._update_plot)

        self._subset_only_mode = False

    def set_transform(self, transform: Any) -> None:
        self.transform = transform
        self._update_plot()

    def set_normalizer(self, normalizer: Any) -> None:
        self.normalizer = normalizer
        self._update_plot()

    def _linear_normalizer(self, arr: np.ndarray) -> np.ndarray:
        """
        Normalize matrix values to [0,1] linearly.
        - Computes min/max, ignoring NaN.
        - Handles edge cases where range is invalid (returns zeros).
        - Result feeds into colormap function (_viridis_lut).
        """
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return np.zeros_like(arr)
        return (arr - vmin) / (vmax - vmin)

    def _viridis_lut(self, normalized: np.ndarray) -> np.ndarray:
        """
        Simple viridis-like colormap mapping normalized values [0,1] → RGBA.
        - R channel increases linearly with t.
        - G channel peaks around t≈0.6 with quadratic falloff.
        - B channel decreases with t.
        - Alpha is always 1.0 (opaque).
        Returns uint8 RGBA array for use in ImageItem.
        """
        x = np.clip((normalized * 255), 0, 255)
        t = x / 255.0
        r = np.clip(1.5 * t - 0.5, 0.0, 1.0)
        g = np.clip(-4.0 * (t - 0.6) ** 2 + 1.0, 0.0, 1.0)
        b = np.clip(1.0 - t * 0.7, 0.0, 1.0)
        rgba = np.stack([r, g, b, np.ones_like(r)], axis=-1)
        return (rgba * 255)

    def set_opacity(self, a: float) -> None:
        self._opacity = a
        if self._img is not None:
            self._img.setOpacity(self._opacity)

    def set_z(self, z: int) -> None:
        self._z = z
        if self._img is not None:
            self._img.setZValue(self._z)

    def set_focus(self, focus: bool) -> None:
        self._focus = focus

    def get_color(self) -> str:
        return self._selection_color

    def set_selection_color(self, color: str) -> None:
        self._selection_color = str(color)

    def set_color(self, color: str) -> None:
        self.set_selection_color(color)

    def set_translation(self, tx: float, ty: float) -> None:
        self._translation_x = tx
        self._translation_y = ty
        self._update_plot()

    def set_scale(self, tx: float, ty: float) -> None:
        self.scale_x = tx
        self.scale_y = ty
        self._update_plot()

    def add_to(self, plot_item: Any, view_box: Any) -> None:
        self._plot_item = plot_item
        self._view_box = view_box
        self._img = pg.ImageItem()
        self._img.setOpacity(self._opacity)
        self._img.setZValue(self._z)
        self._plot_item.addItem(self._img)
        self._update_plot()

    def remove_from(self, plot_item: Any) -> None:
        if self._img is not None and self._img.scene() is not None:
            self._plot_item.removeItem(self._img)
        self._img = None
        self._plot_item = None
        self._view_box = None

    def bounds(self) -> tuple[float, float, float, float]:
        arr = self._get_norm_image()
        H, W = arr.shape[:2]
        return (0.0, W * self.scale_x, 0.0, H * self.scale_y)

    def handle_event(self, event_type: str, payload: dict[str, Any]) -> None:
        super().handle_event(event_type, payload)

    def expose_actions(self, parent=None):
        actions = []

        subset_action = QtWidgets.QAction("Filter by Selection (subset-only mode)", parent, checkable=True)
        subset_action.setChecked(getattr(self, "_subset_only_mode", False))
        def _toggle_subset():
            self._subset_only_mode = subset_action.isChecked()
            self._update_plot()
        subset_action.toggled.connect(_toggle_subset)
        actions.append(subset_action)

        act_scale = QtWidgets.QAction("Set cell scale…", parent)
        def _set_scale():
            dlg = QtWidgets.QInputDialog(None)
            dlg.setInputMode(QtWidgets.QInputDialog.TextInput)
            dlg.setLabelText("scale_x:")
            dlg.setTextValue(str(self.scale_x))
            if not dlg.exec_():
                return
            try:
                sx = float(dlg.textValue())
            except ValueError:
                return

            dlg = QtWidgets.QInputDialog(None)
            dlg.setInputMode(QtWidgets.QInputDialog.TextInput)
            dlg.setLabelText("scale_y:")
            dlg.setTextValue(str(self.scale_y))
            if not dlg.exec_():
                return
            try:
                sy = float(dlg.textValue())
            except ValueError:
                return

            self.scale_x, self.scale_y = sx, sy
            self._update_plot()
        act_scale.triggered.connect(_set_scale)
        actions.append(act_scale)

        return actions

    def _update_plot(self) -> None:
        """
        Redraw the heatmap image.
        Steps:
        1. Fetch raw matrix from data source.
        2. If subset-only mode is active, filter down to highlighted rows.
        3. Apply transform (e.g., preprocessing).
        4. Normalize matrix values (default linear normalization).
        5. Map normalized matrix → RGBA via viridis colormap.
        6. Update ImageItem with pixel data and apply translation/scale transform.
        """
        if self._img is None:
            return

        raw = self.data_source.get()
        n_raw = self.data_source.size()
        if n_raw <= 0:
            return

        subset_only = getattr(self, "_subset_only_mode", False)

        # Restrict rows if subset-only mode
        if subset_only and n_raw > 0:
            allowed_mask = np.ones(n_raw, dtype=bool)
            combined = np.zeros(n_raw, dtype=bool)
            for src, (inds, _) in self._highlight_indices.items():
                if src != self._source_name():
                    ii = np.asarray(inds, dtype=int)
                    ii = ii[(ii >= 0) & (ii < n_raw)]
                    combined[ii] = True
            if np.any(combined):
                allowed_mask = combined
                raw = raw[allowed_mask]

        # Transform, normalize, colormap
        mat = self.transform(raw)
        norm = self.normalizer(mat)
        rgba = self._viridis_lut(norm)

        # Update image dimensions and data
        self._H, self._W = norm.shape
        self._img.setImage(rgba, levels=(0, 255), autoLevels=False)

        # Apply position/scale transformation for display
        tr = QtGui.QTransform()
        tr.translate(self._translation_x, self._translation_y)
        tr.scale(self.scale_x, self.scale_y)
        self._img.setTransform(tr)

    def _get_norm_image(self) -> np.ndarray:
        raw = self.data_source.get()
        mat = self.transform(raw)
        return self.normalizer(mat)

    def _source_name(self) -> str:
        uid = getattr(self, "_canvas_uid", None)
        return f"{self.graph_name}@{uid}" if uid is not None else self.graph_name

    def handle_scene_event(self, event: QtCore.QEvent, view_box: Any) -> bool:
        """
        Handle mouse interactions for drawing a selection rectangle on the heatmap.
        - MousePress: start a new rectangle, removing any existing one.
        - MouseMove: update rectangle geometry, track its coordinates in view space.
        - MouseRelease: finalize selection (does not emit directly here).
        Returns True if the event was consumed.
        """
        # Ctrl+drag reserved for scatter/line plots → ignore here
        if QtGui.QGuiApplication.keyboardModifiers() & QtCore.Qt.ControlModifier:
            return False

        if not self._focus or self.selection_locked:
            return False

        et = event.type()
        if et == QtCore.QEvent.GraphicsSceneMousePress:
            if event.button() != QtCore.Qt.LeftButton:
                return False
            self._selection_origin = event.scenePos()
            # Remove old rectangle if present
            if self._selection_rect is not None and self._selection_rect.scene() is not None:
                view_box.scene().removeItem(self._selection_rect)
            # Create new rectangle overlay
            self._selection_rect = QtWidgets.QGraphicsRectItem()
            self._selection_rect.setPen(QtGui.QPen(QtGui.QColor("gold"), 2))
            self._selection_rect.setBrush(QtGui.QBrush(QtGui.QColor(255, 215, 0, 50)))
            view_box.scene().addItem(self._selection_rect)
            return True

        elif et == QtCore.QEvent.GraphicsSceneMouseMove:
            if self._selection_origin is None or self._selection_rect is None:
                return False
            rect = QtCore.QRectF(self._selection_origin, event.scenePos()).normalized()
            self._selection_rect.setRect(rect)
            # Convert scene rect into data/view coordinates
            tl = view_box.mapSceneToView(rect.topLeft())
            br = view_box.mapSceneToView(rect.bottomRight())
            self.selection_rect_view_coords = (tl.x(), tl.y(), br.x(), br.y())
            return True

        elif et == QtCore.QEvent.GraphicsSceneMouseRelease:
            # End of drag
            self._selection_origin = None
            return True

        return False

    def close(self) -> None:
        if hasattr(self, "graph") and hasattr(self.graph, "disconnect"):
            self.graph.disconnect()

    def clone(self) -> "HeatmapPlot":
        dup = HeatmapPlot(
            data_source=self.data_source,
            transform=self.transform,
            normalizer=self.normalizer,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            graph_name=self.graph_name,
        )
        dup._canvas_uid = getattr(self, "_canvas_uid", None)
        dup.set_selection_color(self._selection_color)
        return dup
