# canvas/Canvas.py

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut, QAction, QActionGroup, QToolButton
import pyqtgraph as pg
from widgets.GraphWidget import GraphWidget
from widgets.ViewBox import ViewBox
from graphs.ScatterPlot import ScatterPlot
from graphs.LinePlot import LinePlot
from canvas.layers.ScatterLayer import ScatterLayer
from canvas.layers.LineLayer import LineLayer

class Canvas(GraphWidget):
    _uid_counter = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._canvas_uid = Canvas._uid_counter
        Canvas._uid_counter += 1

        self._nonfocus_alpha = 0.2
        self._layers = []
        self._focused = None

        self._view_menu = getattr(self, "_view_menu", None)

        self.plot_widget = pg.PlotWidget(viewBox=ViewBox())
        self.plot_item = self.plot_widget.getPlotItem()
        self.view_box = self.plot_widget.getViewBox()

        self.content_layout.addWidget(self.plot_widget)

        self.plot_widget.scene().installEventFilter(self)

        self._x_label = ""
        self._y_label = ""
        self._axis_labels_visible = True

        self._install_focus_shortcuts()

        self.resize(900, 600)

        self._rebuild_focus_menu()

    def extend_toolbar(self, toolbar):
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

    def _focus_submenu_shell(self):
        if self._view_menu is None:
            return
        for act in list(self._view_menu.actions()):
            sub = act.menu()
            if sub is not None and sub.title() == "Focus":
                self._view_menu.removeAction(act)
        self._focus_menu = QtWidgets.QMenu("Focus", self)
        self._view_menu.addMenu(self._focus_menu)

    def _layer_actions_submenu_shell(self):
        if self._view_menu is None:
            return
        for act in list(self._view_menu.actions()):
            sub = act.menu()
            if sub is not None and sub.title() == "Layer Actions":
                self._view_menu.removeAction(act)
        self._layer_actions_menu = QtWidgets.QMenu("Layer Actions", self)
        self._view_menu.addMenu(self._layer_actions_menu)

    def _rebuild_layer_actions_menu(self):
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

    def _rebuild_focus_menu(self):
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

    def plot(self, layer_or_plot):
        layer = self._coerce_to_layer(layer_or_plot)
        if layer in self._layers:
            self._apply_focus_and_opacity()
            self._rebuild_focus_menu()
            return layer

        layer.add_to(self.plot_item, self.view_box)
        self._layers.append(layer)
        if self._focused is None:
            self._focused = layer
        self._apply_focus_and_opacity()
        self._fit_bounds_on_layer(layer)
        self._rebuild_focus_menu()
        return layer

    def unplot(self, layer_or_plot):
        layer = self.get_graph(layer_or_plot)
        if layer is None:
            return
        try:
            layer.remove_from(self.plot_item)
        finally:
            if layer in self._layers:
                self._layers.remove(layer)
        if self._focused is layer:
            self._focused = self._layers[-1] if self._layers else None
        self._apply_focus_and_opacity()
        self._rebuild_focus_menu()

    def set_focus(self, layer_or_plot):
        layer = self.get_graph(layer_or_plot)
        if layer is None:
            return
        self._focused = layer
        self._apply_focus_and_opacity()
        self._rebuild_focus_menu()
        self._rebuild_layer_actions_menu()

    def get_graph(self, layer_or_plot):
        if obj in self._layers:
            return obj

        for layer in self._layers:
            if getattr(layer, "_source_plot", None) is obj:
                return layer
        return None

    def set_nonfocus_alpha(self, alpha: float):
        self._nonfocus_alpha = max(0.0, min(1.0, float(alpha)))
        self._apply_focus_and_opacity()

    def set_axis_label(self, axis, text):
        if axis == "x":
            self._x_label = text
            self.plot_widget.setLabel("bottom", text if self._axis_labels_visible else "")
        elif axis == "y":
            self._y_label = text
            self.plot_widget.setLabel("left", text if self._axis_labels_visible else "")

    def update_axis_labels(self):
        if self._axis_labels_visible:
            self.plot_widget.setLabel("bottom", self._x_label)
            self.plot_widget.setLabel("left", self._y_label)
        else:
            self.plot_widget.setLabel("bottom", "")
            self.plot_widget.setLabel("left", "")

    def eventFilter(self, source, event):
        if source is self.plot_widget.scene() and self._focused is not None:
            if hasattr(self._focused, "handle_scene_event"):
                handled = self._focused.handle_scene_event(event, self.view_box)
                if handled:
                    return True
        return super().eventFilter(source, event)

    def _fit_bounds_on_layer(self, layer):
        if len(self._layers) != 1:
            return
        if hasattr(layer, "bounds"):
            b = layer.bounds()
            if b is not None:
                x0, x1, y0, y1 = b
                if x0 is not None and x1 is not None:
                    self.plot_widget.setXRange(x0, x1, padding=0.05)
                if y0 is not None and y1 is not None:
                    self.plot_widget.setYRange(y0, y1, padding=0.05)
                self.view_box.disableAutoRange()

    def _apply_focus_and_opacity(self):
        for i, layer in enumerate(self._layers):
            is_focus = (layer is self._focused)
            if hasattr(layer, "set_focus"):
                layer.set_focus(is_focus)
            if hasattr(layer, "set_opacity"):
                layer.set_opacity(1.0 if is_focus else self._nonfocus_alpha)
            if hasattr(layer, "set_z"):
                z = 10 + i if not is_focus else 1000
                layer.set_z(z)

    def _coerce_to_layer(self, obj):
        if hasattr(obj, "add_to") and hasattr(obj, "remove_from"):
            if getattr(obj, "_plot_item", None) is not None and obj._plot_item is not self.plot_item:
                if hasattr(obj, "clone"):
                    return obj.clone()
            return obj

        if isinstance(obj, ScatterPlot):
            layer = ScatterLayer(obj.data_source)
            layer.set_transform(getattr(obj, "transform", lambda x: x))

            layer._descriptor = obj
            layer._canvas_uid = getattr(self, "_canvas_uid", None)
            layer._source_plot = obj

            return layer

        if isinstance(obj, LinePlot):
            layer = LineLayer(obj.data_source, sample_rate=getattr(obj, "sample_rate", 44100),
                              transform=getattr(obj, "transform", lambda x: x),
                              argsort_func=getattr(obj, "argsort_func", None),
                              name=getattr(obj, "graph_name", None))
            try:
                layer.set_selection_color(getattr(layer, "_selection_color", "red"))
            except Exception:
                pass
            layer._descriptor = obj
            layer._canvas_uid = getattr(self, "_canvas_uid", None)
            layer._source_plot = obj
            try:
                xlbl, ylbl = obj.get_axis_labels()
                self._x_label = xlbl or self._x_label
                self._y_label = ylbl or self._y_label
                self.update_axis_labels()
            except Exception:
                pass
            return layer

        raise TypeError("Unsupported object passed to Canvas.plot()")

    def _layer_display_name(self, layer, default="Layer"):
        try:
            sp = getattr(layer, "_source_plot", None)
            if sp is not None and hasattr(sp, "graph_name") and sp.graph_name:
                return sp.graph_name
        except Exception:
            pass

        if hasattr(layer, "name") and layer.name:
            return str(layer.name)
        return default

    def _focus_by_index(self, idx: int):
        if not self._layers:
            return
        idx = max(0, min(idx, len(self._layers) - 1))
        self._focused = self._layers[idx]
        self._apply_focus_and_opacity()
        self._rebuild_focus_menu()
        self._rebuild_layer_actions_menu()

    def _cycle_focus(self, step: int):
        if not self._layers:
            return
        try:
            cur = self._layers.index(self._focused) if self._focused in self._layers else 0
        except ValueError:
            cur = 0
        new_idx = (cur + step) % len(self._layers)
        self._focus_by_index(new_idx)

    def _install_focus_shortcuts(self):
        sc_next = QShortcut(QKeySequence("Ctrl+F"), self)
        sc_next.activated.connect(lambda: self._cycle_focus(+1))

        sc_prev = QShortcut(QKeySequence("Ctrl+Shift+F"), self)
        sc_prev.activated.connect(lambda: self._cycle_focus(-1))

        self._focus_digit_shortcuts = []
        for i in range(1, 10):
            seq = QKeySequence(f"Ctrl+F, {i}")
            sc = QShortcut(seq, self)
            sc.activated.connect(lambda i=i: self._focus_by_index(i - 1))
            self._focus_digit_shortcuts.append(sc)
