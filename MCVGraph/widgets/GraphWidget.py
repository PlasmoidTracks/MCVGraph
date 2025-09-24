# widgets/GraphWidget.py

from PyQt5 import QtWidgets, QtCore, QtGui
from typing import Any, Optional
import weakref
from PyQt5.QtWidgets import QToolBar, QAction, QMenu, QToolButton, QShortcut
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QKeySequence
import numpy as np
import json
from datetime import datetime

SNAP_LEFT = 0
SNAP_RIGHT = 1
SNAP_ABOVE = 2
SNAP_BELOW = 3
SNAP_AUTO = 4

class GraphWidget(QtWidgets.QWidget):
    _instances = weakref.WeakSet()
    _name_counter = 1
    _suspend_link_propagation = False

    def __init__(self, name: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        GraphWidget._instances.add(self)

        default_colors = ['red', 'lime', 'blue', 'magenta', 'cyan', 'orange']
        self._color_index = (GraphWidget._name_counter - 1) % len(default_colors)
        self._selection_color = default_colors[self._color_index]

        if name is None:
            name = f"Plot#{GraphWidget._name_counter}"
            GraphWidget._name_counter += 1
        self.graph_name = name
        self.setWindowTitle(self.graph_name)

        self._scale_width_factor = 1.0
        self._scale_height_factor = 1.0

        self.link_target = None
        self.snap_on_release = False

        self.toolbar = QToolBar("Graph Toolbar", self)
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setFixedHeight(24)

        self._create_actions()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)

        self.content_widget = QtWidgets.QWidget(self)
        self.content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_widget.setLayout(self.content_layout)

        content_wrapper = QtWidgets.QWidget(self)
        hbox = QtWidgets.QHBoxLayout(content_wrapper)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)
        hbox.addWidget(self.content_widget)

        layout.addWidget(content_wrapper)
        self.setLayout(layout)

        self._create_shortcuts()

        self.extend_toolbar(self.toolbar)
        self.OVERLAP_WEIGHT = 0

        self._is_snapping = False
        self._position_constraint = "auto"
        self._anchor_offset = QtCore.QPoint(0, 0)
        self._awaiting_link_digit = False
        self._auto_match_width = False
        self._auto_match_height = False

    def _create_actions(self) -> None:
        window_menu = QMenu("Window", self)
        window_menu.addAction("Maximize", self.showMaximized)
        window_menu.addAction("Minimize", self.showMinimized)
        window_menu.addAction("Restore", self.showNormal)
        window_menu.addAction("Screenshot", self.take_screenshot)

        stay_on_top_action = QAction("Stay on top", self, checkable=True)
        def toggle_stay(checked):
            self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, checked)
            self.show()
        stay_on_top_action.toggled.connect(toggle_stay)
        window_menu.addAction(stay_on_top_action)

        export_action = QAction("Export Layout…", self)
        export_action.triggered.connect(self._export_layout_dialog)
        window_menu.addAction(export_action)

        import_action = QAction("Import Layout…", self)
        import_action.triggered.connect(self._import_layout_dialog)
        window_menu.addAction(import_action)

        window_menu.addAction("Close", self.close)

        window_button = QToolButton(self)
        window_button.setText("Window")
        window_button.setMenu(window_menu)
        window_button.setPopupMode(QToolButton.InstantPopup)
        self.toolbar.addWidget(window_button)

        self.control_button = QToolButton(self)
        self.control_button.setText("Control")
        self.control_button.setPopupMode(QToolButton.InstantPopup)
        self.control_button.clicked.connect(self._show_control_menu)
        self.toolbar.addWidget(self.control_button)

        selection_menu = QMenu("Selection", self)
        selection_menu.addAction("Mode", lambda: None)

        def clear_selection():
            if hasattr(self, "graph"):
                self.graph.emit_broadcast(CLEAR_HIGHLIGHT, {"source": self.graph.owner.graph_name})

        selection_menu.addAction("Deselect all", clear_selection)

        selection_button = QToolButton(self)
        selection_button.setText("Selection")
        selection_button.setMenu(selection_menu)
        selection_button.setPopupMode(QToolButton.InstantPopup)
        self.toolbar.addWidget(selection_button)

    def _show_control_menu(self) -> None:
        control_menu = QMenu(self)

        link_menu = QMenu("Link Position", self)

        sorted_all = sorted(GraphWidget._instances, key=lambda x: x.graph_name)

        for inst in sorted_all:
            if inst is self:
                continue
            allowed = (
                not self.is_recursively_linking_to(inst) and
                not inst.is_recursively_linking_to(self)
            )

            idx_1based = sorted_all.index(inst) + 1
            digit_hint = f" (ctrl+L+{idx_1based})" if 1 <= idx_1based <= 9 else ""

            label = f"{inst.graph_name}{digit_hint}"
            action = QAction(label, self)
            action.setEnabled(allowed)

            def make_link(target):
                return lambda: self._link_to_target(target)
            action.triggered.connect(make_link(inst))
            link_menu.addAction(action)

        unlink_label = "Unlink (ctrl+U)"
        unlink_action = QAction(unlink_label, self)
        unlink_action.setEnabled(self.link_target is not None)
        unlink_action.triggered.connect(self._unlink)
        link_menu.addSeparator()
        link_menu.addAction(unlink_action)

        anchor_menu = QMenu("Anchor Mode", self)
        anchor_group = QtWidgets.QActionGroup(self)
        anchor_group.setExclusive(True)

        def make_anchor_action(name, label):
            act = QAction(label, self, checkable=True)
            act.setChecked(self._position_constraint == name)
            act.triggered.connect(lambda: self._set_anchor_mode(name))
            anchor_group.addAction(act)
            return act

        anchor_menu.addAction(make_anchor_action("auto", "Auto"))
        anchor_menu.addAction(make_anchor_action("left", "Left"))
        anchor_menu.addAction(make_anchor_action("right", "Right"))
        anchor_menu.addAction(make_anchor_action("above", "Above"))
        anchor_menu.addAction(make_anchor_action("below", "Below"))

        lock_edge_action = QAction("Lock Current Edge", self)
        lock_edge_action.setEnabled(self.link_target is not None)
        def lock_current_edge():
            if self.link_target:
                self._position_constraint = self._infer_relative_edge(self.link_target)
        lock_edge_action.triggered.connect(lock_current_edge)
        anchor_menu.addAction(lock_edge_action)

        link_menu.addMenu(anchor_menu)
        control_menu.addMenu(link_menu)

        offset_menu = QMenu("Anchor Offset", self)

        set_offset_action = QAction("Set Offset Manually... (ctrl+shift+O)", self)
        def set_offset():
            dx, ok1 = QtWidgets.QInputDialog.getInt(self, "Anchor Offset X", "X Offset (px):", self._anchor_offset.x(), -2000, 2000)
            if not ok1: return
            dy, ok2 = QtWidgets.QInputDialog.getInt(self, "Anchor Offset Y", "Y Offset (px):", self._anchor_offset.y(), -2000, 2000)
            if not ok2: return
            self._set_anchor_offset(dx, dy)
        set_offset_action.triggered.connect(set_offset)
        offset_menu.addAction(set_offset_action)

        use_current_action = QAction("Use Current Offset (ctrl+alt+O)", self)
        use_current_action.setEnabled(self.link_target is not None)
        use_current_action.triggered.connect(self._use_current_offset)
        offset_menu.addAction(use_current_action)

        reset_offset_action = QAction("Reset Offset (ctrl+shift+R)", self)
        reset_offset_action.triggered.connect(self._reset_anchor_offset)
        offset_menu.addAction(reset_offset_action)

        link_menu.addMenu(offset_menu)

        link_size_menu = QMenu("Link Window Size", self)

        width_scale_label = f"Set Width Scale (current: {self._scale_width_factor:.2f}) (ctrl+alt+W)"
        width_scale_action = QAction(width_scale_label, self)
        def set_width_scale():
            val, ok = QtWidgets.QInputDialog.getDouble(
                self, "Set Width Scale", "Width scale factor (0.1 - 10.0):",
                self._scale_width_factor, 0.1, 10.0, 2
            )
            if ok:
                self._scale_width_factor = val
                self._scale_width(val)
        width_scale_action.triggered.connect(set_width_scale)
        link_size_menu.addAction(width_scale_action)

        height_scale_label = f"Set Height Scale (current: {self._scale_height_factor:.2f}) (ctrl+alt+H)"
        height_scale_action = QAction(height_scale_label, self)
        def set_height_scale():
            val, ok = QtWidgets.QInputDialog.getDouble(
                self, "Set Height Scale", "Height scale factor (0.1 - 10.0):",
                self._scale_height_factor, 0.1, 10.0, 2
            )
            if ok:
                self._scale_height_factor = val
                self._scale_height(val)
        height_scale_action.triggered.connect(set_height_scale)
        link_size_menu.addAction(height_scale_action)

        if not hasattr(self, '_auto_match_width'):
            self._auto_match_width = False
        if not hasattr(self, '_auto_match_height'):
            self._auto_match_height = False
        if not hasattr(self, '_scale_width_factor'):
            self._scale_width_factor = 1.0
        if not hasattr(self, '_scale_height_factor'):
            self._scale_height_factor = 1.0

        auto_match_width_action = QAction("Auto Match Width (ctrl+shift+W)", self, checkable=True)
        auto_match_width_action.setChecked(self._auto_match_width)
        def toggle_auto_match_width():
            self._auto_match_width = not self._auto_match_width
            if self._auto_match_width and self.link_target:
                self._scale_width(self._scale_width_factor)
            QtCore.QTimer.singleShot(0, lambda: self._show_control_menu())

        auto_match_width_action.triggered.connect(toggle_auto_match_width)
        link_size_menu.addAction(auto_match_width_action)

        auto_match_height_action = QAction("Auto Match Height (ctrl+shift+H)", self, checkable=True)
        auto_match_height_action.setChecked(self._auto_match_height)
        def toggle_auto_match_height():
            self._auto_match_height = not self._auto_match_height
            if self._auto_match_height and self.link_target:
                self._scale_height(self._scale_height_factor)
            QtCore.QTimer.singleShot(0, lambda: self._show_control_menu())

        auto_match_height_action.triggered.connect(toggle_auto_match_height)
        link_size_menu.addAction(auto_match_height_action)

        link_size_menu.setEnabled(self.link_target is not None)
        control_menu.addMenu(link_size_menu)

        control_menu.exec_(self.control_button.mapToGlobal(QtCore.QPoint(0, self.control_button.height())))

    def _match_size(self) -> None:
        if not self.link_target:
            return
        self.resize(self.link_target.size())

    def _match_width(self) -> None:
        if not self.link_target:
            return
        self.resize(self.link_target.width(), self.height())

    def _match_height(self) -> None:
        if not self.link_target:
            return
        self.resize(self.width(), self.link_target.height())

    def _scale_size(self, factor: float) -> None:
        if not self.link_target:
            return
        self._scale_width_factor = max(0.1, min(10.0, factor))
        self._scale_height_factor = max(0.1, min(10.0, factor))
        new_size = self.link_target.size() * factor
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        new_width = min(new_size.width(), screen.width())
        new_height = min(new_size.height(), screen.height())
        self.resize(new_width, new_height)

    def _scale_width(self, factor: float) -> None:
        if not self.link_target:
            return
        self._scale_width_factor = max(0.1, min(10.0, factor))
        new_width = self.link_target.width() * self._scale_width_factor
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        new_width = min(new_width, screen.width())
        self.resize(int(new_width), self.height())

    def _scale_height(self, factor: float) -> None:
        if not self.link_target:
            return
        self._scale_height_factor = max(0.1, min(10.0, factor))
        new_height = self.link_target.height() * self._scale_height_factor
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        new_height = min(new_height, screen.height())
        self.resize(self.width(), int(new_height))

    def _custom_scale_dialog(self) -> None:
        if not self.link_target:
            return
        factor_x, ok_x = QtWidgets.QInputDialog.getDouble(
            self, "Custom Width Scale", "Width scale factor (0.1 - 10.0):", self._scale_width_factor, 0.1, 10.0, 2)
        if not ok_x:
            return
        factor_y, ok_y = QtWidgets.QInputDialog.getDouble(
            self, "Custom Height Scale", "Height scale factor (0.1 - 10.0):", self._scale_height_factor, 0.1, 10.0, 2)
        if not ok_y:
            return

        self._scale_width_factor = factor_x
        self._scale_height_factor = factor_y

        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        new_width = min(self.link_target.width() * factor_x, screen.width())
        new_height = min(self.link_target.height() * factor_y, screen.height())
        self.resize(new_width, new_height)

    def take_screenshot(self) -> None:
        pixmap = self.grab(QRect(0, 0, self.width(), self.height()))
        filename = f"screenshot_{id(self)}.png"
        pixmap.save(filename)

    def extend_toolbar(self, toolbar: QtWidgets.QToolBar) -> None:
        view_menu = QtWidgets.QMenu("View", self)

        axis_menu = QtWidgets.QMenu("Axis", self)

        self._axis_labels_visible = True
        toggle_axis_labels_action = QtWidgets.QAction("Show Axis Labels", self, checkable=True)
        toggle_axis_labels_action.setChecked(self._axis_labels_visible)

        def on_toggle_axis_labels():
            self._axis_labels_visible = toggle_axis_labels_action.isChecked()
            if hasattr(self, "update_axis_labels"):
                self.update_axis_labels()

        toggle_axis_labels_action.triggered.connect(on_toggle_axis_labels)
        axis_menu.addAction(toggle_axis_labels_action)

        set_x_label_action = QtWidgets.QAction("Set X Label...", self)
        def set_x_label():
            text, ok = QtWidgets.QInputDialog.getText(self, "Set X Axis Label", "X Label:")
            if ok and hasattr(self, "set_axis_label"):
                self.set_axis_label("x", text)
        set_x_label_action.triggered.connect(set_x_label)
        axis_menu.addAction(set_x_label_action)

        set_y_label_action = QtWidgets.QAction("Set Y Label...", self)
        def set_y_label():
            text, ok = QtWidgets.QInputDialog.getText(self, "Set Y Axis Label", "Y Label:")
            if ok and hasattr(self, "set_axis_label"):
                self.set_axis_label("y", text)
        set_y_label_action.triggered.connect(set_y_label)
        axis_menu.addAction(set_y_label_action)

        view_menu.addMenu(axis_menu)

        view_button = QtWidgets.QToolButton(self)
        view_button.setText("View")
        view_button.setMenu(view_menu)
        view_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        toolbar.addWidget(view_button)

    def snap(self, target: Optional["GraphWidget"] = None, direction: int = SNAP_AUTO) -> None:
        """
        Align (snap) this window relative to another GraphWidget.
        - If no target is given, auto-selects the *nearest* window by center distance.
        - Supports fixed directions (LEFT/RIGHT/ABOVE/BELOW).
        - SNAP_AUTO tries all directions and chooses the best alignment
          (prefers non-overlapping, then minimizes distance).
        - Applies anchor offset to allow fine-tuned positioning.
        """
        if target is None:
            # Find nearest other window
            nearest = None
            min_distance = float('inf')
            my_geom = self.geometry()

            for inst in GraphWidget._instances:
                if inst is self:
                    continue
                dist = (my_geom.center() - inst.geometry().center()).manhattanLength()
                if dist < min_distance:
                    nearest = inst
                    min_distance = dist

            if nearest is None:
                return
            target = nearest

        target_geom = target.geometry()
        my_geom = self.geometry()
        spacing = 0

        # Explicit snap direction (skip auto-search)
        if direction != SNAP_AUTO:
            if direction == SNAP_LEFT:
                new_x = target_geom.x() - my_geom.width() - spacing
                new_y = target_geom.y() - 32
            elif direction == SNAP_RIGHT:
                new_x = target_geom.x() + target_geom.width() + spacing
                new_y = target_geom.y() - 32
            elif direction == SNAP_ABOVE:
                new_x = target_geom.x()
                new_y = target_geom.y() - my_geom.height() - spacing - 32
            elif direction == SNAP_BELOW:
                new_x = target_geom.x()
                new_y = target_geom.y() + target_geom.height() + spacing - 32
            else:
                return
            self.move(new_x + self._anchor_offset.x(), new_y + self._anchor_offset.y())
            return

        my_geom = self.geometry()
        original_pos = my_geom.topLeft()

        # Auto mode: try all directions and pick the best candidate
        candidates = []
        for dir_candidate in [SNAP_LEFT, SNAP_RIGHT, SNAP_ABOVE, SNAP_BELOW]:
            if dir_candidate == SNAP_LEFT:
                new_x = target_geom.x() - my_geom.width() - spacing
                new_y = target_geom.y()
            elif dir_candidate == SNAP_RIGHT:
                new_x = target_geom.x() + target_geom.width() + spacing
                new_y = target_geom.y()
            elif dir_candidate == SNAP_ABOVE:
                new_x = target_geom.x()
                new_y = target_geom.y() - my_geom.height() - spacing
            elif dir_candidate == SNAP_BELOW:
                new_x = target_geom.x()
                new_y = target_geom.y() + target_geom.height() + spacing
            else:
                continue

            trial_rect = QtCore.QRect(new_x, new_y, my_geom.width(), my_geom.height())
            intersection = trial_rect.intersected(target_geom)
            overlap_area = 0 if intersection.isEmpty() else intersection.width() * intersection.height()
            new_center = QtCore.QRect(new_x, new_y, my_geom.width(), my_geom.height()).center()
            dist = (my_geom.center() - new_center).manhattanLength()

            candidates.append({
                "pos": (new_x, new_y - 32),
                "overlap": overlap_area,
                "distance": dist
            })

        # Prefer candidates with no overlap, otherwise pick closest
        non_overlap = [c for c in candidates if c["overlap"] == 0]
        if non_overlap:
            best = min(non_overlap, key=lambda c: c["distance"])
        else:
            best = min(candidates, key=lambda c: c["distance"])

        pos = QtCore.QPoint(*best["pos"]) + self._anchor_offset
        self.move(pos)

    def snap_left(self, target: "GraphWidget") -> None:
        self.snap(target, SNAP_LEFT)

    def snap_right(self, target: "GraphWidget") -> None:
        self.snap(target, SNAP_RIGHT)

    def snap_above(self, target: "GraphWidget") -> None:
        self.snap(target, SNAP_ABOVE)

    def snap_below(self, target: "GraphWidget") -> None:
        self.snap(target, SNAP_BELOW)

    def moveEvent(self, event: QtGui.QMoveEvent) -> None:
        """
        When this window moves, propagate the move to any windows
        that are *linked* to follow it.
        - Skips if link propagation is suspended (e.g., during layout import).
        - Each dependent window calls `snap` to maintain its relative position.
        """
        super().moveEvent(event)
        if getattr(GraphWidget, "_suspend_link_propagation", False):
            return
        for instance in GraphWidget._instances:
            if instance is self:
                continue
            if instance.link_target is self:
                instance._is_snapping = True
                instance.snap(self, instance._resolve_snap_direction())
                instance._is_snapping = False

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """
        When this window is resized, update any windows that are
        linked to it.
        - If a linked window has auto-match width/height enabled,
          recompute its size relative to this window (respecting scale factors).
        - Re-snaps the linked window afterward to maintain alignment.
        """
        super().resizeEvent(event)
        if getattr(GraphWidget, "_suspend_link_propagation", False):
            return
        for instance in GraphWidget._instances:
            if instance is self:
                continue
            if instance.link_target is self:
                screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
                width = instance.width()
                height = instance.height()
                if getattr(instance, '_auto_match_width', False):
                    width = max(0.1, min(screen.width(), self.width() * getattr(instance, '_scale_width_factor', 1.0)))
                if getattr(instance, '_auto_match_height', False):
                    height = max(0.1, min(screen.height(), self.height() * getattr(instance, '_scale_height_factor', 1.0)))
                instance.resize(int(width), int(height))
                instance._is_snapping = True
                instance.snap(self, instance._resolve_snap_direction())
                instance._is_snapping = False

    def _link_to_target(self, target: "GraphWidget") -> None:
        self.link_target = target
        self.snap(target, self._resolve_snap_direction())

    def _unlink(self) -> None:
        if self.link_target:
            self.link_target = None
            self._scale_width_factor = 1.0
            self._scale_height_factor = 1.0

    def link(self, target: "GraphWidget") -> None:
        if target is None:
            raise ValueError("target must not be None")
        if target is self:
            raise ValueError("cannot link a widget to itself")
        if self.is_recursively_linking_to(target) or target.is_recursively_linking_to(self):
            raise ValueError("link would create a cycle")
        self._link_to_target(target)

    def unlink(self, target: Optional["GraphWidget"] = None) -> None:
        if target is None or target is self.link_target:
            self._unlink()

    def set_graph_name(self, name: str) -> None:
        self.graph_name = name
        self.setWindowTitle(self.graph_name)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        for instance in list(GraphWidget._instances):
            if instance.link_target is self:
                if self.link_target is not None and self.link_target is not instance:
                    instance.link_target = self.link_target
                else:
                    instance.link_target = None

        if hasattr(self, "graph"):
            self.graph.emit("ScatterPlot", CLEAR_HIGHLIGHT, {
                "source": self.graph.owner.graph_name
            })
            self.graph.disconnect()

        GraphWidget._instances.discard(self)
        super().closeEvent(event)

    def is_linking(self) -> bool:
        return self.link_target is not None

    def is_linking_to(self, other: "GraphWidget") -> bool:
        return self.link_target is other

    def is_linked_to(self) -> bool:
        return any(g.link_target is self for g in GraphWidget._instances if g is not self)

    def is_recursively_linking_to(self, other: "GraphWidget") -> bool:
        current = self.link_target
        while current is not None:
            if current is other:
                return True
            current = current.link_target
        return False

    def set_color(self, color: str) -> None:
        self._selection_color = color

        if hasattr(self, "graph"):
            indices = self.get_selection_indices()
            if indices is not None and len(indices) > 0:
                payload = {
                    "indices": indices.copy(),
                    "source": self.graph.owner.graph_name,
                    "color": self.get_color()
                }
                self.graph.emit("ScatterPlot", EventType.SUBSET_INDICES, payload)
                self.graph.emit("LinePlot", EventType.SUBSET_INDICES, payload)

    def get_color(self) -> str:
        return self._selection_color

    def _set_anchor_mode(self, mode: str) -> None:
        self._position_constraint = mode
        if self.link_target:
            self.snap(self.link_target, self._resolve_snap_direction())

    def _infer_relative_edge(self, target: "GraphWidget") -> str:
        my_geom = self.geometry()
        target_geom = target.geometry()

        dx = my_geom.center().x() - target_geom.center().x()
        dy = my_geom.center().y() - target_geom.center().y()

        if abs(dx) > abs(dy):
            return "left" if dx < 0 else "right"
        else:
            return "above" if dy < 0 else "below"

    def _resolve_snap_direction(self) -> int:
        mapping = {
            "left": SNAP_LEFT,
            "right": SNAP_RIGHT,
            "above": SNAP_ABOVE,
            "below": SNAP_BELOW
        }
        return mapping.get(self._position_constraint, SNAP_AUTO)

    def _set_anchor_offset(self, dx: int, dy: int) -> None:
        self._anchor_offset = QtCore.QPoint(dx, dy)
        if self.link_target:
            self.snap(self.link_target, self._resolve_snap_direction())

    def _use_current_offset(self) -> None:
        if not self.link_target:
            return
        snap_dir = self._resolve_snap_direction()
        anchor_point = self._get_anchor_position(self.link_target, snap_dir)
        current_top_left = self.geometry().topLeft()
        self._anchor_offset = current_top_left - anchor_point
        self.snap(self.link_target, snap_dir)

    def _reset_anchor_offset(self) -> None:
        self._anchor_offset = QtCore.QPoint(0, 0)
        if self.link_target:
            self.snap(self.link_target, self._resolve_snap_direction())

    def _get_anchor_position(self, target: "GraphWidget", direction: int) -> QtCore.QPoint:
        tg = target.geometry()
        mg = self.geometry()

        if direction == SNAP_LEFT:
            return QtCore.QPoint(tg.x() - mg.width(), tg.y() - 32)
        elif direction == SNAP_RIGHT:
            return QtCore.QPoint(tg.x() + tg.width(), tg.y() - 32)
        elif direction == SNAP_ABOVE:
            return QtCore.QPoint(tg.x(), tg.y() - mg.height() - 32)
        elif direction == SNAP_BELOW:
            return QtCore.QPoint(tg.x(), tg.y() + tg.height() - 32)
        else:
            return QtCore.QPoint(tg.x() + mg.width(), tg.y())

    def _create_shortcuts(self) -> None:
        def bind(seq, func):
            QShortcut(QKeySequence(seq), self).activated.connect(func)

        bind("Ctrl+Shift+M", self.showMaximized)
        bind("Ctrl+Shift+N", self.showMinimized)
        bind("Ctrl+Shift+R", self.showNormal)
        bind("Ctrl+Shift+S", self.take_screenshot)
        bind("Ctrl+T", lambda: self._toggle_stay_on_top())
        bind("Ctrl+Q", self.close)

        bind("Ctrl+Alt+S", lambda: self.snap())
        bind("Ctrl+Alt+Left", lambda: self.snap_left(self.link_target) if self.link_target else None)
        bind("Ctrl+Alt+Right", lambda: self.snap_right(self.link_target) if self.link_target else None)
        bind("Ctrl+Alt+Up", lambda: self.snap_above(self.link_target) if self.link_target else None)
        bind("Ctrl+Alt+Down", lambda: self.snap_below(self.link_target) if self.link_target else None)

        bind("Ctrl+U", self._unlink)
        bind("Ctrl+L", self._start_link_mode)
        bind("Ctrl+Shift+O", self._prompt_set_anchor_offset)
        bind("Ctrl+Alt+O", self._use_current_offset)
        bind("Ctrl+Shift+R", self._reset_anchor_offset)

        bind("Ctrl+Alt+W", self._prompt_width_scale)
        bind("Ctrl+Alt+H", self._prompt_height_scale)
        bind("Ctrl+Shift+W", self.toggle_auto_match_width)
        bind("Ctrl+Shift+H", self.toggle_auto_match_height)

    def _toggle_stay_on_top(self) -> None:
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, not self.windowFlags() & QtCore.Qt.WindowStaysOnTopHint)
        self.show()

    def _lock_current_edge(self) -> None:
        if self.link_target:
            self._position_constraint = self._infer_relative_edge(self.link_target)

    def _prompt_set_anchor_offset(self) -> None:
        dx, ok1 = QtWidgets.QInputDialog.getInt(self, "Anchor Offset X", "X Offset (px):", self._anchor_offset.x(), -2000, 2000)
        if not ok1:
            return
        dy, ok2 = QtWidgets.QInputDialog.getInt(self, "Anchor Offset Y", "Y Offset (px):", self._anchor_offset.y(), -2000, 2000)
        if not ok2:
            return
        self._set_anchor_offset(dx, dy)

    def _prompt_width_scale(self) -> None:
        val, ok = QtWidgets.QInputDialog.getDouble(
            self, "Set Width Scale", "Width scale factor (0.1 - 10.0):",
            self._scale_width_factor, 0.1, 10.0, 2
        )
        if ok:
            self._scale_width_factor = val
            self._scale_width(val)

    def _prompt_height_scale(self) -> None:
        val, ok = QtWidgets.QInputDialog.getDouble(
            self, "Set Height Scale", "Height scale factor (0.1 - 10.0):",
            self._scale_height_factor, 0.1, 10.0, 2
        )
        if ok:
            self._scale_height_factor = val
            self._scale_height(val)

    def toggle_auto_match_width(self) -> None:
        self._auto_match_width = not self._auto_match_width
        if self._auto_match_width and self.link_target:
            self._scale_width(self._scale_width_factor)

    def toggle_auto_match_height(self) -> None:
        self._auto_match_height = not self._auto_match_height
        if self._auto_match_height and self.link_target:
            self._scale_height(self._scale_height_factor)
    
    def toggle_auto_match(self) -> None:
        if not self._auto_match_width or not self._auto_match_height:
            self._auto_match_width = True
            self._auto_match_height = True
        if self._auto_match_width and self._auto_match_height:
            self._auto_match_width = False
            self._auto_match_height = False
        if self.link_target:
            if self._auto_match_width:
                self._scale_width(self._scale_width_factor)
            if self._auto_match_height:
                self._scale_height(self._scale_height_factor)

    def _start_link_mode(self) -> None:
        self._awaiting_link_digit = True

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self._awaiting_link_digit:
            key = event.key()
            if QtCore.Qt.Key_1 <= key <= QtCore.Qt.Key_9:
                index = key - QtCore.Qt.Key_1
                sorted_instances = sorted(GraphWidget._instances, key=lambda x: x.graph_name)
                if 0 <= index < len(sorted_instances):
                    self._unlink()
                    target = sorted_instances[index]
                    if target is not self and not self.is_recursively_linking_to(target) and not target.is_recursively_linking_to(self):
                        self._link_to_target(target)
                self._awaiting_link_digit = False
                return
            else:
                self._awaiting_link_digit = False
        super().keyPressEvent(event)

    def get_selection_indices(self) -> Optional[Any]:
        return None

    def get_selection_data(self) -> Optional[Any]:
        return None



    def _window_is_stay_on_top(self) -> bool:
        return self.windowFlags() & QtCore.Qt.WindowStaysOnTopHint

    def _stay_on_top(self, value: bool) -> None:
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, value)
        self.show()

    def to_layout_dict(self) -> dict[str, Any]:
        g = self.geometry()
        link_name = self.link_target.graph_name if self.link_target is not None else None
        return {
            "name": self.graph_name,
            "geometry": {"x": g.x(), "y": g.y(), "width": g.width(), "height": g.height()},
            "stay_on_top": self._window_is_stay_on_top(),
            "link": {"target_name": link_name} if link_name else None,
            "anchor_mode": self._position_constraint,
            "anchor_offset": {"dx": self._anchor_offset.x(), "dy": self._anchor_offset.y()},
            "auto_match_width": self._auto_match_width,
            "auto_match_height": self._auto_match_height,
            "scale_width_factor": self._scale_width_factor,
            "scale_height_factor": self._scale_height_factor,
        }

    def export_all_layout(self) -> dict[str, Any]:
        windows = [inst.to_layout_dict() for inst in sorted(GraphWidget._instances, key=lambda x: x.graph_name)]
        return {"windows": windows}

    def _export_layout_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Layout", "layout.json", "JSON Files (*.json)")
        if not path:
            return
        data = self.export_all_layout()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

    def import_layout_from_dict(self, data: dict[str, Any]) -> None:
        """
        Restore window arrangement from a serialized layout dictionary.
        - Maps saved names to existing GraphWidget instances.
        - Suspends link propagation while restoring to avoid cascades.
        - First pass: reset all links, then apply geometry, flags, anchors, scale.
        - Second pass: re-establish link relationships by name.
        - Finally: snap all linked windows into place.
        """
        name_to_inst = {inst.graph_name: inst for inst in GraphWidget._instances}

        try:
            GraphWidget._suspend_link_propagation = True

            # Reset all links
            for inst in list(GraphWidget._instances):
                inst._unlink()

            # Restore geometry and properties for each saved window
            for w in data["windows"]:
                name = w.get("name")
                inst = name_to_inst.get(name)
                if inst is None:
                    continue

                geom = w.get("geometry", {})
                x = geom.get("x"); y = geom.get("y"); width = geom.get("width"); height = geom.get("height")
                if None not in (x, y, width, height):
                    inst.setGeometry(x, y, width, height)

                inst._stay_on_top(w.get("stay_on_top", False))
                inst._position_constraint = w.get("anchor_mode", "auto")

                off = w.get("anchor_offset", {})
                inst._anchor_offset = QtCore.QPoint(off.get("dx", 0), off.get("dy", 0))
                inst._auto_match_width = bool(w.get("auto_match_width", False))
                inst._auto_match_height = bool(w.get("auto_match_height", False))
                inst._scale_width_factor = w.get("scale_width_factor", 1.0)
                inst._scale_height_factor = w.get("scale_height_factor", 1.0)

                # Restore viewport ranges if present
                view = w.get("view", None)
                xr = view.get("x")
                yr = view.get("y")
                x0, x1 = xr[0], xr[1]
                y0, y1 = yr[0], yr[1]
                inst.plot_widget.setXRange(x0, x1, padding=0)
                inst.plot_widget.setYRange(y0, y1, padding=0)
                inst.view_box.disableAutoRange()

            # Second pass: re-establish links between windows
            for w in data["windows"]:
                name = w.get("name")
                inst = name_to_inst.get(name)
                if inst is None:
                    continue
                link = w.get("link")
                if not link:
                    continue
                target_name = link.get("target_name") or link.get("target_id") or link.get("target")
                target = name_to_inst.get(target_name)
                if target is None or target is inst:
                    continue
                if inst.is_recursively_linking_to(target) or target.is_recursively_linking_to(inst):
                    continue
                inst.link_target = target

        finally:
            GraphWidget._suspend_link_propagation = False

        # Snap all linked windows into place
        for inst in list(GraphWidget._instances):
            if inst.link_target is not None:
                inst.snap(inst.link_target, inst._resolve_snap_direction())

    def _import_layout_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import Layout", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.import_layout_from_dict(data)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import failed", str(e))

    def export_all_layout_to_file(self, path: str) -> None:
        data = self.export_all_layout()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def import_layout_from_file(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.import_layout_from_dict(data)

