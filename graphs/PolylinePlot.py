# graphs/PolylinePlot.py

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg
from GraphBus import GraphEventClient

class PolylinePlot:
    _DEFAULT_COLORS = ['red', 'lime', 'blue', 'magenta', 'cyan', 'orange']
    _color_counter = 0

    def __init__(
        self,
        vertices,
        edges,
        *,
        color="blue",
        line_width=1.0,
        name=None,
        focus=False,
        opacity=1.0,
        z=10,
    ):
        self.vertices = vertices
        self.edges = edges
        self.color = color
        self.line_width = line_width
        self.name = name

        self._plot_item = None
        self._view_box = None
        self._curves = []
        self._focus = focus
        self._opacity = opacity
        self._z = z

        self._highlight_indices = {}

        self.graph = GraphEventClient(self, "PolylinePlot")

        self.vertices.data_updated.connect(self._update_plot)
        self.edges.data_updated.connect(self._update_plot)

        self._subset_only_mode = False

    def add_to(self, plot_item, view_box):
        self._plot_item = plot_item
        self._view_box = view_box
        self._update_plot()

    def remove_from(self, plot_item):
        for c in self._curves:
            if c.scene() is not None:
                plot_item.removeItem(c)
        self._curves.clear()
        self._plot_item = None
        self._view_box = None

    def set_opacity(self, alpha):
        self._opacity = alpha
        for c in self._curves:
            c.setOpacity(self._opacity)

    def set_z(self, z):
        self._z = z
        for c in self._curves:
            c.setZValue(self._z)

    def set_focus(self, focus):
        self._focus = focus

    def bounds(self):
        verts = self.vertices.get()
        if verts.size == 0:
            return None
        x0, x1 = np.min(verts[:, 0]), np.max(verts[:, 0])
        y0, y1 = np.min(verts[:, 1]), np.max(verts[:, 1])
        return (x0, x1, y0, y1)

    def handle_event(self, event_type, payload):
        if event_type == "subset_indices":
            indices = np.array(payload["indices"], dtype=int)
            source = payload["source"]
            color = payload.get("color", self.color)
            self._highlight_indices[source] = (indices, color)
            self._update_plot()

        elif event_type == "clear_highlight":
            source = payload["source"]
            if source in self._highlight_indices:
                del self._highlight_indices[source]
                self._update_plot()

    def _update_plot(self):
        if self._plot_item is None:
            return

        for c in self._curves:
            if c.scene() is not None:
                self._plot_item.removeItem(c)
        self._curves.clear()

        verts = self.vertices.get()
        edges = self.edges.get()

        if verts.size == 0 or edges.size == 0:
            return

        n_raw = len(verts)
        subset_only = getattr(self, "_subset_only_mode", False)

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

        view_raw_idx = np.where(allowed_mask)[0] if subset_only and np.any(allowed_mask) else np.arange(n_raw)
        inv_map = np.full(n_raw, -1, dtype=int)
        inv_map[view_raw_idx] = np.arange(len(view_raw_idx))

        edges = np.asarray(edges, dtype=int)
        if edges.ndim != 2 or edges.shape[1] != 2:
            return

        keep = []
        for i, j in edges:
            if 0 <= i < n_raw and 0 <= j < n_raw and inv_map[i] >= 0 and inv_map[j] >= 0:
                keep.append((inv_map[i], inv_map[j]))
        if not keep:
            return

        verts_view = verts[view_raw_idx]
        pts = []
        for ii, jj in keep:
            pts.append([verts_view[ii, 0], verts_view[ii, 1]])
            pts.append([verts_view[jj, 0], verts_view[jj, 1]])
            pts.append([np.nan, np.nan])

        pts = np.array(pts, dtype=float)
        if pts.size == 0:
            return

        curve = pg.PlotCurveItem(pts[:, 0], pts[:, 1], pen=pg.mkPen(self.color, width=self.line_width))
        curve.setOpacity(self._opacity)
        curve.setZValue(self._z)
        self._plot_item.addItem(curve)
        self._curves.append(curve)

    def expose_actions(self, parent=None):
        actions = []

        subset_action = QtWidgets.QAction("Filter by Selection (subset-only mode)", parent, checkable=True)
        subset_action.setChecked(getattr(self, "_subset_only_mode", False))
        def _toggle_subset():
            self._subset_only_mode = subset_action.isChecked()
            self._update_plot()
        subset_action.toggled.connect(_toggle_subset)
        actions.append(subset_action)

        return actions

    def close(self):
        if hasattr(self, "graph") and hasattr(self.graph, "disconnect"):
            self.graph.disconnect()

    def get_color(self):
        return self.color

    def set_color(self, color):
        self.color = str(color)
        self._update_plot()

    def clone(self):
        dup = PolylinePlot(
            self.vertices,
            self.edges,
            color=self.color,
            line_width=self.line_width,
            name=self.name,
            focus=self._focus,
            opacity=self._opacity,
            z=self._z,
        )
        return dup

    def _source_name(self):
        uid = getattr(self, "_canvas_uid", None)
        return f"{self.name}@{uid}" if uid is not None else (self.name or f"Polyline@{hex(id(self))}")

