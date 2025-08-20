# graphs/HeatmapPlot.py

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg
from GraphBus import GraphEventClient

class HeatmapPlot:
    _DEFAULT_COLORS = ['red', 'lime', 'blue', 'magenta', 'cyan', 'orange']
    _color_counter = 0

    def __init__(
        self, 
        data_source, 
        *, 
        transform=lambda x: x,
        normalizer=None, 
        scale_x = 1.0,
        scale_y = 1.0, 
        graph_name = "Heatmap"):

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

    def set_transform(self, transform):
        self.transform = transform
        self._update_plot()
    
    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
        self._update_plot()
    
    def _linear_normalizer(self, arr):
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return np.zeros_like(arr)
        return (arr - vmin) / (vmax - vmin)
        
    def _viridis_lut(self, normalized):
        x = np.clip((normalized * 255), 0, 255)
        t = x / 255.0
        r = np.clip(1.5 * t - 0.5, 0.0, 1.0)
        g = np.clip(-4.0 * (t - 0.6) ** 2 + 1.0, 0.0, 1.0)
        b = np.clip(1.0 - t * 0.7, 0.0, 1.0)
        rgba = np.stack([r, g, b, np.ones_like(r)], axis=-1)
        return (rgba * 255)

    def set_opacity(self, a):
        self._opacity = a
        if self._img is not None:
            self._img.setOpacity(self._opacity)

    def set_z(self, z):
        self._z = z
        if self._img is not None:
            self._img.setZValue(self._z)

    def set_focus(self, focus):
        self._focus = focus

    def get_color(self): 
        return self._selection_color

    def set_selection_color(self, color): 
        self._selection_color = str(color)

    def set_color(self, color): 
        self.set_selection_color(color)

    def set_translation(self, tx, ty):
        self._translation_x = tx
        self._translation_y = ty
        self._update_plot()

    def set_scale(self, tx, ty):
        self.scale_x = tx
        self.scale_y = ty
        self._update_plot()


    def add_to(self, plot_item, view_box):
        self._plot_item = plot_item
        self._view_box = view_box
        self._img = pg.ImageItem()
        self._img.setOpacity(self._opacity)
        self._img.setZValue(self._z)
        self._plot_item.addItem(self._img)
        self._update_plot()

    def remove_from(self, plot_item):
        if self._img is not None and self._img.scene() is not None:
            self._plot_item.removeItem(self._img)
        self._img = None
        self._plot_item = None
        self._view_box = None

    def bounds(self):
        arr = self._get_norm_image()
        H, W = arr.shape[:2]
        return (0.0, W * self.scale_x, 0.0, H * self.scale_y)

    def handle_event(self, event_type, payload):
        return

    def expose_actions(self, parent=None):
        act_scale = QtWidgets.QAction("Set cell scale…", parent)
        def _set_scale():
            sx, okx = QtWidgets.QInputDialog.getDouble(None, "Cell width", "scale_x:", self.scale_x, -1000, 1000, 3)
            if not okx: return
            sy, oky = QtWidgets.QInputDialog.getDouble(None, "Cell height", "scale_y:", self.scale_y, -1000, 1000, 3)
            if not oky: return
            self.scale_x, self.scale_y = sx, sy
            self._update_plot()
        act_scale.triggered.connect(_set_scale)

        return [act_scale]

    def _update_plot(self):
        if self._img is None:
            return
        raw = self.data_source.get()
        mat = self.transform(raw)
        norm = self.normalizer(mat)
        rgba = self._viridis_lut(norm)
        self._H, self._W = norm.shape
        self._img.setImage(rgba, levels=(0, 255), autoLevels=False)
        tr = QtGui.QTransform()
        tr.translate(self._translation_x, self._translation_y)
        tr.scale(self.scale_x, self.scale_y)
        self._img.setTransform(tr)

    def _get_norm_image(self):
        raw = self.data_source.get()
        mat = self.transform(raw)
        return self.normalizer(mat)

    def _source_name(self):
        uid = getattr(self, "_canvas_uid", None)
        return f"{self.graph_name}@{uid}" if uid is not None else self.graph_name

    def handle_scene_event(self, event, view_box):
        if QtGui.QGuiApplication.keyboardModifiers() & QtCore.Qt.ControlModifier:
            return False

        if not self._focus or self.selection_locked:
            return False

        et = event.type()
        if et == QtCore.QEvent.GraphicsSceneMousePress:
            if event.button() != QtCore.Qt.LeftButton:
                return False
            self._selection_origin = event.scenePos()
            if self._selection_rect is not None and self._selection_rect.scene() is not None:
                view_box.scene().removeItem(self._selection_rect)
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
            tl = view_box.mapSceneToView(rect.topLeft())
            br = view_box.mapSceneToView(rect.bottomRight())
            self.selection_rect_view_coords = (tl.x(), tl.y(), br.x(), br.y())
            return True

        elif et == QtCore.QEvent.GraphicsSceneMouseRelease:
            self._selection_origin = None
            return True

        return False

    def close(self):
        if hasattr(self, "graph") and hasattr(self.graph, "disconnect"):
            self.graph.disconnect()
    
    def clone(self):
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
