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
        color: str = "gray",  # kept for backward-compat; treated as non-selected color
        color_non_selected: Optional[str] = None,
        selection_color: Optional[str] = None,
        line_width: float = 1.0,
        name: Optional[str] = None,
        focus: bool = False,
        opacity: float = 1.0,
        z: int = 10,
    ) -> None:
        self.vertices = vertices
        self.edges = edges

        # Non-selected/base color (match other plots' default "gray")
        self.color_non_selected = str(color_non_selected if color_non_selected is not None else color)

        # Selection/highlight color (match ScatterPlot/LinePlot rotating palette)
        if selection_color is None:
            idx = PolylinePlot._color_counter % len(PolylinePlot._DEFAULT_COLORS)
            self._selection_color = PolylinePlot._DEFAULT_COLORS[idx]
            PolylinePlot._color_counter += 1
        else:
            self._selection_color = str(selection_color)

        # Maintain a generic "color" attribute for GraphBase defaults if ever used
        self.color = self._selection_color

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
        Redraw polyline network with highlight overlays.
        Base layer uses non-selected color.
        For each highlight source, draw an overlay of edges incident to
        any selected vertex in that source color.
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
        valid = (
            (edges[:, 0] >= 0) & (edges[:, 0] < n_raw) &
            (edges[:, 1] >= 0) & (edges[:, 1] < n_raw) &
            (inv_map[edges[:, 0]] >= 0) &
            (inv_map[edges[:, 1]] >= 0)
        )
        if not np.any(valid):
            return

        edges_raw_valid = edges[valid]
        # Map endpoints to view indices
        edges_view = np.stack(
            [inv_map[edges_raw_valid[:, 0]], inv_map[edges_raw_valid[:, 1]]],
            axis=1
        )

        verts_view = verts[view_raw_idx]

        # Helper to build a NaN-separated polyline path from a list/array of edges (view indices)
        def build_pts(ev: np.ndarray) -> np.ndarray:
            if ev.size == 0:
                return np.empty((0, 2), dtype=float)
            n = len(ev)
            pts = np.empty((n * 3, 2), dtype=float)
            pts[0::3] = verts_view[ev[:, 0], :2]
            pts[1::3] = verts_view[ev[:, 1], :2]
            pts[2::3, 0] = np.nan
            pts[2::3, 1] = np.nan
            return pts

        # Determine highlight membership per edge per source
        highlighted_any = np.zeros(len(edges_raw_valid), dtype=bool)

        # Draw overlays per highlight source
        for src, (inds_raw, color) in self._highlight_indices.items():
            ii = np.asarray(inds_raw, dtype=int)
            ii = ii[(ii >= 0) & (ii < n_raw)]
            if ii.size == 0:
                continue

            # Edge is highlighted if any endpoint is selected
            sel_mask = np.isin(edges_raw_valid[:, 0], ii) | np.isin(edges_raw_valid[:, 1], ii)
            if not np.any(sel_mask):
                continue

            highlighted_any |= sel_mask
            ev = edges_view[sel_mask]
            pts = build_pts(ev)
            if pts.size == 0:
                continue

            curve_h = pg.PlotCurveItem(pts[:, 0], pts[:, 1],
                                       pen=pg.mkPen(color, width=self.line_width))
            curve_h.setOpacity(self._opacity)
            curve_h.setZValue(self._z + 1)
            self._plot_item.addItem(curve_h)
            self._curves.append(curve_h)

        # Base layer: edges not highlighted by any source
        base_mask = ~highlighted_any
        if np.any(base_mask):
            ev_base = edges_view[base_mask]
            pts_base = build_pts(ev_base)
            if pts_base.size > 0:
                curve_base = pg.PlotCurveItem(pts_base[:, 0], pts_base[:, 1],
                                              pen=pg.mkPen(self.color_non_selected, width=self.line_width))
                curve_base.setOpacity(self._opacity)
                curve_base.setZValue(self._z)
                self._plot_item.addItem(curve_base)
                self._curves.append(curve_base)

    def expose_actions(self, parent: Optional[Any] = None) -> list[Any]:
        actions = []

        subset_action = QtWidgets.QAction("Filter by Selection (subset-only mode)", parent, checkable=True)
        subset_action.setChecked(getattr(self, "_subset_only_mode", False))
        def _toggle_subset():
            self._subset_only_mode = subset_action.isChecked()
            self._update_plot()
        subset_action.toggled.connect(_toggle_subset)
        actions.append(subset_action)

        base_color_action = QtWidgets.QAction("Set Non-selected Color…", parent)
        def _set_base_color():
            col = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.color_non_selected), None, "Pick color")
            if col.isValid():
                self.set_color_non_selected(col.name())
        base_color_action.triggered.connect(_set_base_color)
        actions.append(base_color_action)

        sel_color_action = QtWidgets.QAction("Set Selection Color…", parent)
        def _set_sel_color():
            col = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._selection_color), None, "Pick selection color")
            if col.isValid():
                self.set_selection_color(col.name())
                # No immediate visual change unless this layer emits selections,
                # but keep consistent with other plots.
                self._update_plot()
        sel_color_action.triggered.connect(_set_sel_color)
        actions.append(sel_color_action)

        return actions

    def close(self) -> None:
        if hasattr(self, "graph") and hasattr(self.graph, "disconnect"):
            self.graph.disconnect()

    def get_color(self) -> str:
        # Color used when *this* plot emits selection events
        return self._selection_color

    def set_color_non_selected(self, color: str) -> None:
        self.color_non_selected = str(color)
        self._update_plot()

    def set_selection_color(self, color: str) -> None:
        self._selection_color = str(color)
        # Keep generic attribute in sync for any default usage
        self.color = self._selection_color

    # Backward-compat: treat set_color as setting the selection color
    def set_color(self, color: str) -> None:
        self.set_selection_color(color)
        self._update_plot()

    def clone(self) -> "PolylinePlot":
        dup = PolylinePlot(
            self.vertices,
            self.edges,
            color=self.color_non_selected,
            color_non_selected=self.color_non_selected,
            selection_color=self._selection_color,
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

