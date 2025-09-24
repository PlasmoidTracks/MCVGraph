# canvas/Canvas.py

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut, QAction, QActionGroup, QToolButton, QMenu
import pyqtgraph as pg
from typing import Any, Dict, List, Optional

from MCVGraph.widgets.GraphWidget import GraphWidget
from MCVGraph.widgets.ViewBox import ViewBox
from MCVGraph.graphs.ScatterPlot import ScatterPlot
from MCVGraph.graphs.LinePlot import LinePlot

class Canvas(GraphWidget):
    _uid_counter: int = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._canvas_uid: int = Canvas._uid_counter
        Canvas._uid_counter += 1

        self._nonfocus_alpha: float = 0.2
        self._layers: List[Any] = []
        self._focused: Optional[Any] = None

        self._view_menu: Optional[QMenu] = getattr(self, "_view_menu", None)

        self.plot_widget: pg.PlotWidget = pg.PlotWidget(viewBox=ViewBox())
        self.plot_item: Any = self.plot_widget.getPlotItem()
        self.view_box: Any = self.plot_widget.getViewBox()

        self.content_layout.addWidget(self.plot_widget)

        self.plot_widget.scene().installEventFilter(self)

        self._x_label: str = ""
        self._y_label: str = ""
        self._axis_labels_visible: bool = True

        self._install_focus_shortcuts()

        self.resize(900, 600)

        self._rebuild_focus_menu()

    def extend_toolbar(self, toolbar: Any) -> None:
        super().extend_toolbar(toolbar)

        self._view_menu = None
        for action in toolbar.actions():
            widget = toolbar.widgetForAction(action)
            if isinstance(widget, QToolButton) and widget.text() == "View":
                self._view_menu = widget.menu()
                break

        if self._view_menu is not None:
            self._focus_submenu_shell()
            self._layer_actions_submenu_shell()

    def set_view_port(self, x1: float, y1: float, x2: float, y2: float) -> None:
        try:
            self.plot_widget.setXRange(x1, x2, padding=0)
            self.plot_widget.setYRange(y1, y2, padding=0)
            self.view_box.disableAutoRange()
        except Exception as e:
            print(f"[Canvas.set_view_port] Failed to set view: {e}")

    def _focus_submenu_shell(self) -> None:
        if self._view_menu is None:
            return
        for act in list(self._view_menu.actions()):
            sub = act.menu()
            if sub is not None and sub.title() == "Focus":
                self._view_menu.removeAction(act)
        self._focus_menu: QMenu = QtWidgets.QMenu("Focus", self)
        self._view_menu.addMenu(self._focus_menu)

    def _layer_actions_submenu_shell(self) -> None:
        if self._view_menu is None:
            return
        for act in list(self._view_menu.actions()):
            sub = act.menu()
            if sub is not None and sub.title() == "Layer Actions":
                self._view_menu.removeAction(act)
        self._layer_actions_menu: QMenu = QtWidgets.QMenu("Layer Actions", self)
        self._view_menu.addMenu(self._layer_actions_menu)

    def _rebuild_layer_actions_menu(self) -> None:
        if self._view_menu is None or not hasattr(self, "_focused"):
            return
        self._layer_actions_submenu_shell()
        self._layer_actions_menu.clear()

        layer = self._focused
        if layer is None:
            act = QAction("(no focused layer)", self, enabled=False)
            self._layer_actions_menu.addAction(act)
            return

        exposer = getattr(layer, "expose_actions", None)
        if not callable(exposer):
            act = QAction("(no actions)", self, enabled=False)
            self._layer_actions_menu.addAction(act)
            return

        items = exposer(parent=self)
        if not items:
            act = QAction("(no actions)", self, enabled=False)
            self._layer_actions_menu.addAction(act)
            return

        for it in items:
            if isinstance(it, QtWidgets.QMenu):
                self._layer_actions_menu.addMenu(it)
            else:
                self._layer_actions_menu.addAction(it)

    def _rebuild_focus_menu(self) -> None:
        if self._view_menu is None:
            return
        self._focus_submenu_shell()
        self._focus_menu.clear()

        group = QActionGroup(self)
        group.setExclusive(True)

        for idx, layer in enumerate(self._layers):
            name = self._layer_display_name(layer, default=f"Layer {idx+1}")
            act = QAction(name, self, checkable=True)
            act.setChecked(layer is self._focused)
            act.triggered.connect(lambda _=False, i=idx: self._focus_by_index(i))
            group.addAction(act)
            self._focus_menu.addAction(act)

        if not self._layers:
            act = QAction("(no plots)", self, enabled=False)
            self._focus_menu.addAction(act)

        self._rebuild_layer_actions_menu()

    def plot(self, layer_or_plot: Any) -> Any:
        """
        Add a new graph layer to the Canvas.
        - Converts the input into a proper layer (via `_coerce_to_layer`).
        - If already plotted, just refresh focus state and menus.
        - Otherwise: attach the layer to the PlotItem + ViewBox,
          track it in `_layers`, focus it if none exists,
          update opacity/focus visuals, fit view to data, and
          rebuild menus.
        Returns the layer instance.
        """
        layer = self._coerce_to_layer(layer_or_plot)

        # Avoid duplicates: if already present, just refresh UI state
        if layer in self._layers:
            self._apply_focus_and_opacity()
            self._rebuild_focus_menu()
            return layer

        # Attach new layer to underlying pyqtgraph structures
        layer.add_to(self.plot_item, self.view_box)
        self._layers.append(layer)

        # First plotted layer becomes default focus
        if self._focused is None:
            self._focused = layer

        # Refresh visual state, zoom, and menus
        self._apply_focus_and_opacity()
        self._fit_bounds_on_layer(layer)
        self._rebuild_focus_menu()
        return layer

    def unplot(self, layer_or_plot: Any) -> None:
        """
        Remove a plotted layer from the Canvas.
        - Looks up the layer object (supports passing either the object or alias).
        - Safely calls `remove_from` on the PlotItem to detach its graphics.
        - Removes it from the internal `_layers` list.
        - If it was focused, shift focus to the last remaining layer (or None).
        - Updates opacity/focus visuals and rebuilds the focus menu.
        """
        layer = self.get_graph(layer_or_plot)
        if layer is None:
            return

        try:
            layer.remove_from(self.plot_item)
        finally:
            if layer in self._layers:
                self._layers.remove(layer)

        # Reassign focus if the removed layer was active
        if self._focused is layer:
            self._focused = self._layers[-1] if self._layers else None

        # Refresh visuals and menus
        self._apply_focus_and_opacity()
        self._rebuild_focus_menu()

    def set_focus(self, layer_or_plot: Any) -> None:
        layer = self.get_graph(layer_or_plot)
        if layer is None:
            return
        self._focused = layer
        self._apply_focus_and_opacity()
        self._rebuild_focus_menu()
        self._rebuild_layer_actions_menu()

    def get_graph(self, layer_or_plot: Any) -> Optional[Any]:
        for layer in self._layers:
            if layer is layer_or_plot:
                return layer
        return None

    def set_nonfocus_alpha(self, alpha: float) -> None:
        self._nonfocus_alpha = max(0.0, min(1.0, alpha))
        self._apply_focus_and_opacity()

    def set_axis_label(self, axis: str, text: str) -> None:
        if axis == "x":
            self._x_label = text
            self.plot_widget.setLabel("bottom", text if self._axis_labels_visible else "")
        elif axis == "y":
            self._y_label = text
            self.plot_widget.setLabel("left", text if self._axis_labels_visible else "")

    def update_axis_labels(self) -> None:
        if self._axis_labels_visible:
            self.plot_widget.setLabel("bottom", self._x_label)
            self.plot_widget.setLabel("left", self._y_label)
        else:
            self.plot_widget.setLabel("bottom", "")
            self.plot_widget.setLabel("left", "")

    def eventFilter(self, source: Any, event: Any) -> bool:
        if source is self.plot_widget.scene() and self._focused is not None:
            if hasattr(self._focused, "handle_scene_event"):
                handled = self._focused.handle_scene_event(event, self.view_box)
                if handled:
                    return True
        return super().eventFilter(source, event)

    def _fit_bounds_on_layer(self, layer: Any) -> None:
        """
        When the first layer is plotted, auto-fit the Canvas view to its bounds.
        - Queries the layer for its bounding box (x0, x1, y0, y1).
        - Expands the PlotWidget ranges slightly with padding.
        - Disables auto-range afterwards to prevent future overrides.
        Only applies on the *first* layer, so user retains manual control later.
        """
        if len(self._layers) != 1:
            return

        if hasattr(layer, "bounds"):
            b = layer.bounds()
            if b is not None:
                x0, x1, y0, y1 = b

                # Adjust x-axis view if bounds are defined
                if x0 is not None and x1 is not None:
                    self.plot_widget.setXRange(x0, x1, padding=0.05)

                # Adjust y-axis view if bounds are defined
                if y0 is not None and y1 is not None:
                    self.plot_widget.setYRange(y0, y1, padding=0.05)

                # Prevent automatic range resets after manual fit
                self.view_box.disableAutoRange()

    def _apply_focus_and_opacity(self) -> None:
        """
        Update all layers so that only the currently focused one is highlighted.
        - Calls `set_focus` on the active layer so it can adapt its appearance.
        - Applies opacity: fully visible if focused, faded otherwise.
        - Adjusts Z-order: focused layer is drawn above all others.
        This ensures the Canvas always emphasizes the active layer visually.
        """
        for i, layer in enumerate(self._layers):
            is_focus = (layer is self._focused)
            if hasattr(layer, "set_focus"):
                layer.set_focus(is_focus)
            if hasattr(layer, "set_opacity"):
                layer.set_opacity(1.0 if is_focus else self._nonfocus_alpha)
            if hasattr(layer, "set_z"):
                z = 10 + i if not is_focus else 1000
                layer.set_z(z)

    def _coerce_to_layer(self, obj: Any) -> Any:
        """
        Ensure the given object is a valid graph layer that can be added to the Canvas.
        Requirements for a layer:
          - Must implement `add_to(plot_item, view_box)` and `remove_from(plot_item)`.
          - If already attached to a *different* PlotItem, then clone it (if possible)
            to avoid sharing one instance across multiple canvases.
        Returns a layer instance ready to be plotted.
        Raises:
          TypeError if the object does not meet the expected graph interface.
        """
        if hasattr(obj, "add_to") and hasattr(obj, "remove_from"):

            # If object is already bound to another PlotItem, try to clone
            if getattr(obj, "_plot_item", None) is not None and obj._plot_item is not self.plot_item:
                if hasattr(obj, "clone"):
                    return obj.clone()

            # Otherwise, safe to reuse the same object
            return obj

        # Not a valid graph type
        raise TypeError("Canvas._coerce_to_layer() expects a graph that satisfies qualifications for graphs.")

    def _layer_display_name(self, layer: Any, default: str = "Layer") -> str:
        try:
            sp = getattr(layer, "_source_plot", None)
            if sp is not None and hasattr(sp, "graph_name") and sp.graph_name:
                return sp.graph_name
        except Exception:
            pass

        if hasattr(layer, "name") and layer.name:
            return str(layer.name)
        return default

    def _focus_by_index(self, idx: int) -> None:
        if not self._layers:
            return
        idx = max(0, min(idx, len(self._layers) - 1))
        self._focused = self._layers[idx]
        self._apply_focus_and_opacity()
        self._rebuild_focus_menu()
        self._rebuild_layer_actions_menu()

    def _cycle_focus(self, step: int) -> None:
        if not self._layers:
            return
        try:
            cur = self._layers.index(self._focused) if self._focused in self._layers else 0
        except ValueError:
            cur = 0
        new_idx = (cur + step) % len(self._layers)
        self._focus_by_index(new_idx)

    def _install_focus_shortcuts(self) -> None:
        sc_next = QShortcut(QKeySequence("Ctrl+F"), self)
        sc_next.activated.connect(lambda: self._cycle_focus(+1))

        sc_prev = QShortcut(QKeySequence("Ctrl+Shift+F"), self)
        sc_prev.activated.connect(lambda: self._cycle_focus(-1))

        self._focus_digit_shortcuts: List[QShortcut] = []
        for i in range(1, 10):
            seq = QKeySequence(f"Ctrl+F, {i}")
            sc = QShortcut(seq, self)
            sc.activated.connect(lambda i=i: self._focus_by_index(i - 1))
            self._focus_digit_shortcuts.append(sc)

    def to_layout_dict(self) -> Dict[str, Any]:
        base = super().to_layout_dict()
        try:
            (x0, x1), (y0, y1) = self.view_box.viewRange()
            base["view"] = {
                "x": [x0, x1],
                "y": [y0, y1],
            }
        except Exception:
            base["view"] = None
        return base
