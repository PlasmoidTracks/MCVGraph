# main_demo.py

import os
import sys
import time
import threading
import numpy as np
from PyQt5 import QtWidgets, QtCore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import all required components from the MCVGraph library
import MCVGraph
from MCVGraph.graphs.ScatterPlot import ScatterPlot
from MCVGraph.graphs.LinePlot import LinePlot
from MCVGraph.graphs.HeatmapPlot import HeatmapPlot
from MCVGraph.graphs.PolylinePlot import PolylinePlot
from MCVGraph.DataSource import DataSource
from MCVGraph.GraphBus import GraphBus
from MCVGraph.canvas.Canvas import Canvas


# =============================================================================
# 1. DATA GENERATION: High-dimensional (6D) periodic signal
# =============================================================================

def generate_6d_data(n=200, t=None):
    """
    Generates 6D state data: [x, y, vx, vy, ax, ay]
    - Position (x, y) oscillates in a figure-8 pattern.
    - Velocity and acceleration follow phase-locked dynamics.
    """
    if t is None:
        t = np.linspace(0, 4 * np.pi, n)

    # Figure-8 trajectory
    x = np.sin(t) * (1 + 0.5 * np.cos(2 * t))
    y = np.sin(2 * t) * 0.5

    # Derivatives (velocity and acceleration)
    dx = np.cos(t) * (1 + 0.5 * np.cos(2 * t)) - np.sin(t) * np.sin(2 * t)
    dy = 2 * np.cos(2 * t) * 0.5

    ax = -np.sin(t) * (1 + 0.5 * np.cos(2 * t)) - 2 * np.sin(2 * t) * 0.5
    ay = -4 * np.sin(2 * t) * 0.5

    return np.column_stack((x, y, dx, dy, ax, ay))


# =============================================================================
# 2. TRANSFORM FUNCTIONS: Map 6D data to 2D views
# =============================================================================

def pos_transform(data):
    """Map 6D data → 2D position (x, y)"""
    return data[:, 0:2]

def vel_transform(data):
    """Map 6D data → 2D velocity (vx, vy)"""
    return data[:, 2:4]

def acc_transform(data):
    """Map 6D data → 2D acceleration (ax, ay)"""
    return data[:, 4:6]

def density_transform(data, size=64):
    """Convert 2D position to density heatmap (64x64 grid)"""
    xy = data[:, 0:2]
    nx = (xy[:, 0] + 2.0) / 4.0  # Map [-2, 2] → [0, 1]
    ny = (xy[:, 1] + 1.0) / 2.0
    ix = np.clip((nx * size).astype(int), 0, size - 1)
    iy = np.clip((ny * size).astype(int), 0, size - 1)
    mat = np.zeros((size, size), dtype=np.float32)
    np.add.at(mat, (ix, iy), 1.0)
    return mat

def log_normalizer(arr):
    """Log-normalize array for heatmap intensity"""
    log_arr = np.log1p(arr)
    vmin = np.min(log_arr)
    vmax = np.max(log_arr)
    return (log_arr - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(arr)


# =============================================================================
# 3. ANIMATION LOOP: Update data in a background thread
# =============================================================================

def run_animation_loop(data_source, edges_ds, *, dt=0.016, stop_event=None):
    """
    Updates the 6D state in a loop using a smooth periodic trajectory.
    - Simulates dynamic system (e.g., a particle in motion).
    - Updates `data_source` and `edges_ds` for PolylinePlot.
    """
    t = 0.0
    while stop_event is None or not stop_event.is_set():
        # Generate new state
        # Generate a time vector around t
        time_vector = np.linspace(t, t + 4 * np.pi, 200)
        state = generate_6d_data(n=200, t=time_vector)

        data_source.set(state)

        # Update edges: connect each point to its 3 nearest neighbors (excluding self)
        n_points = len(state)
        pos = state[:, 0:2]
        diff = pos[None, :, :] - pos[:, None, :]
        dist2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist2, np.inf)
        dist = np.sqrt(dist2)
        nearest = np.argsort(dist, axis=1)[:, 1:4]  # Top 3
        edges = []
        for i in range(n_points):
            for j in nearest[i]:
                edges.append([i, j])
        edges_ds.set(np.array(edges, dtype=int))

        t += dt * 0.1
        time.sleep(dt)  # Simulate real-time update


# =============================================================================
# 4. MAIN: Create and link all canvases with full feature set
# =============================================================================

if __name__ == "__main__":
    # Initialize data source
    initial_state = generate_6d_data(n=200)
    data_source = DataSource(initial_state)
    edges_ds = DataSource(np.zeros((0, 2), dtype=int))  # For polyline edges

    # Initialize Qt app
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # =============================================================================
    # 5. Create canvases for each view
    # =============================================================================

    # Canvas 1: Position & Velocity (scatter plots + heatmap)
    canvas_position = Canvas()
    canvas_position.setWindowTitle("Position & Velocity Views")

    # Canvas 2: Acceleration (scatter plot)
    canvas_acceleration = Canvas()
    canvas_acceleration.setWindowTitle("Acceleration View")

    # Canvas 3: Sonification (line plot)
    canvas_sonify = Canvas()
    canvas_sonify.setWindowTitle("Sonification: Acceleration (Y)")

    # Canvas 4: Density (heatmap)
    canvas_density = Canvas()
    canvas_density.setWindowTitle("Density (Heatmap)")

    # Canvas 5: Network (polyline plot)
    canvas_polyline = Canvas()
    canvas_polyline.setWindowTitle("Network (3-Nearest Neighbor)")

    # =============================================================================
    # 6. Create graphs and assign transforms
    # =============================================================================

    # Scatter plots
    scatter_pos = ScatterPlot(data_source, name="Position")
    scatter_pos.set_transform(pos_transform)

    scatter_vel = ScatterPlot(data_source, name="Velocity")
    scatter_vel.set_transform(vel_transform)

    scatter_acc = ScatterPlot(data_source, name="Acceleration")
    scatter_acc.set_transform(acc_transform)

    # Line plot (sonification)
    line_sonify = LinePlot(
        data_source=data_source,
        sample_rate=1000,
        transform=lambda d: d[:, 5],  # Use y-acceleration
        name="Acceleration (Y)"
    )

    # Heatmap plot (density)
    heatmap_plot = HeatmapPlot(
        data_source=data_source,
        transform=lambda d: density_transform(d, size=64),
        normalizer=log_normalizer,
        scale_x=4.0 / 64.0,
        scale_y=2.0 / 64.0,
        graph_name="Density"
    )
    heatmap_plot.set_translation(-2.0, -1.0)  # Align with data range

    # Polyline plot (network)
    polyline_plot = PolylinePlot(
        vertices=data_source,
        edges=edges_ds,
        color="orange",
        line_width=1.5,
        name="3-Nearest-Neighbor Graph"
    )

    # =============================================================================
    # 7. Plot graphs on canvases
    # =============================================================================

    # Position & Velocity canvas
    canvas_position.plot(scatter_pos)
    canvas_position.plot(scatter_vel)
    canvas_position.plot(heatmap_plot)
    canvas_position.plot(polyline_plot)
    canvas_position.show()

    # Acceleration canvas
    canvas_acceleration.plot(scatter_acc)
    canvas_acceleration.show()

    # Sonification canvas
    canvas_sonify.plot(line_sonify)
    canvas_sonify.show()

    # Density canvas
    canvas_density.plot(heatmap_plot)
    canvas_density.show()

    # Polyline canvas
    canvas_polyline.plot(polyline_plot)
    canvas_polyline.show()

    # =============================================================================
    # 8. LINK CANVASES (optional): Use GraphWidget linking features
    # =============================================================================

    # Link position to all others for alignment
    # (For demo: just show how it's done)
    from MCVGraph.widgets.GraphWidget import SNAP_LEFT, SNAP_RIGHT, SNAP_ABOVE, SNAP_BELOW, SNAP_AUTO
    canvas_position.snap(canvas_acceleration, SNAP_BELOW)
    canvas_position.link_target = canvas_acceleration
    canvas_position.toggle_auto_match()

    # =============================================================================
    # 9. Start animation thread
    # =============================================================================

    stop_event = threading.Event()
    anim_thread = threading.Thread(
        target=run_animation_loop,
        kwargs=dict(
            data_source=data_source,
            edges_ds=edges_ds,
            dt=0.016,
            stop_event=stop_event,
        ),
        daemon=True,
    )
    anim_thread.start()

    # =============================================================================
    # 10. Run the event loop
    # =============================================================================

    app.exec()

    # Clean shutdown
    stop_event.set()
    anim_thread.join(timeout=2.0)
