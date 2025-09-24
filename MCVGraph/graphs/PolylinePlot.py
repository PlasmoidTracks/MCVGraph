# graphs/PolylinePlot.py

from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Any, Optional
import numpy as np
import pyqtgraph as pg
from MCVGraph.GraphBus import GraphEventClient
from MCVGraph.EventType import EventType
from MCVGraph.BasePlot import GraphBase

class PolylinePlot(GraphBase):
    _DEFAULT_COLORS = ['red', 'lime', 'blue', 'magenta', 'cyan', 'orange']
    _color_counter = 0

    def __init__(
        self,
        vertices: Any,
        edges: Any,
        *,
        color: str = "blue",
        line_width: float = 1.0,
        name: Optional[str] = None,
        focus: bool = False,
        opacity: float = 1.0,
        z: int = 10,
    ) -> None:
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

    def add_to(self, plot_item: Any, view_box: Any) -> None:
        self._plot_item = plot_item
        self._view_box = view_box
        self._update_plot()

    def remove_from(self, plot_item: Any) -> None:
        for c in self._curves:
            if c.scene() is not None:
                plot_item.removeItem(c)
        self._curves.clear()
        self._plot_item = None
        self._view_box = None

    def set_opacity(self, alpha: float) -> None:
        self._opacity = alpha
        for c in self._curves:
            c.setOpacity(self._opacity)

    def set_z(self, z: int) -> None:
        self._z = z
        for c in self._curves:
            c.setZValue(self._z)

    def set_focus(self, focus: bool) -> None:
        self._focus = focus

    def bounds(self) -> Optional[tuple[float, float, float, float]]:
        verts = self.vertices.get()
        if verts.size == 0:
            return None
        x0, x1 = np.min(verts[:, 0]), np.max(verts[:, 0])
        y0, y1 = np.min(verts[:, 1]), np.max(verts[:, 1])
        return (x0, x1, y0, y1)

    def handle_event(self, event_type: str, payload: dict[str, Any]) -> None:
        super().handle_event(event_type, payload)

    def _update_plot(self) -> None:
        """
        Redraw polyline network.
        Steps:
        1. Clear old curve items from the scene.
        2. Fetch vertices (Nx2 coordinates) and edges (Mx2 index pairs).
        3. If subset-only mode is active, restrict allowed vertices
           to those highlighted by other plots.
        4. Build an inverse mapping raw_idx → view_idx.
        5. Filter edges: keep only those whose endpoints are still valid.
        6. Build a polyline path (sequence of line segments, NaN separators).
        7. Add as a single PlotCurveItem with the configured color/width.
        """
        if self._plot_item is None:
            return

        # Remove old curve items
        for c in self._curves:
            if c.scene() is not None:
                self._plot_item.removeItem(c)
        self._curves.clear()

        verts = self.vertices.get()
        edges = self.edges.get()
        if verts.size == 0 or edges.size == 0:
            return

        n_raw = self.vertices.size()
        subset_only = getattr(self, "_subset_only_mode", False)

        # Allowed vertices mask (all by default)
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

        # Indices of allowed vertices and inverse mapping
        view_raw_idx = np.where(allowed_mask)[0] if subset_only and np.any(allowed_mask) else np.arange(n_raw)
        inv_map = np.full(n_raw, -1, dtype=int)
        inv_map[view_raw_idx] = np.arange(len(view_raw_idx))

        # Ensure edges are valid pairs
        edges = np.asarray(edges, dtype=int)
        if edges.ndim != 2 or edges.shape[1] != 2:
            return

        # Keep only valid edges where both endpoints are in allowed set
        keep = []
        for i, j in edges:
            if 0 <= i < n_raw and 0 <= j < n_raw and inv_map[i] >= 0 and inv_map[j] >= 0:
                keep.append((inv_map[i], inv_map[j]))
        if not keep:
            return

        # Build vertex view coordinates
        verts_view = verts[view_raw_idx]
        pts = []
        for ii, jj in keep:
            pts.append([verts_view[ii, 0], verts_view[ii, 1]])
            pts.append([verts_view[jj, 0], verts_view[jj, 1]])
            pts.append([np.nan, np.nan])  # separator for discontinuous segments

        pts = np.array(pts, dtype=float)
        if pts.size == 0:
            return

        # Create and add polyline curve
        curve = pg.PlotCurveItem(pts[:, 0], pts[:, 1],
                                 pen=pg.mkPen(self.color, width=self.line_width))
        curve.setOpacity(self._opacity)
        curve.setZValue(self._z)
        self._plot_item.addItem(curve)
        self._curves.append(curve)

    def expose_actions(self, parent: Optional[Any] = None) -> list[Any]:
        actions = []

        subset_action = QtWidgets.QAction("Filter by Selection (subset-only mode)", parent, checkable=True)
        subset_action.setChecked(getattr(self, "_subset_only_mode", False))
        def _toggle_subset():
            self._subset_only_mode = subset_action.isChecked()
            self._update_plot()
        subset_action.toggled.connect(_toggle_subset)
        actions.append(subset_action)

        return actions

    def close(self) -> None:
        if hasattr(self, "graph") and hasattr(self.graph, "disconnect"):
            self.graph.disconnect()

    def get_color(self) -> str:
        return self.color

    def set_color(self, color: str) -> None:
        self.color = str(color)
        self._update_plot()

    def clone(self) -> "PolylinePlot":
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

    def _source_name(self) -> str:
        uid = getattr(self, "_canvas_uid", None)
        return f"{self.name}@{uid}" if uid is not None else (self.name or f"Polyline@{hex(id(self))}")

