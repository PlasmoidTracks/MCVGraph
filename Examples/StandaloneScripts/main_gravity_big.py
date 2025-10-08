# main_gravity.py

import os
import sys
import time
import threading
import numpy as np
from PyQt5 import QtWidgets, QtCore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MCVGraph import ScatterPlot, LinePlot, HeatmapPlot, PolylinePlot, DataSource, GraphBus, Canvas


def position_transform(data):
    return data[:, 0:2]

def velocity_transform(data):
    return data[:, 2:4]

def acceleration_transform(data):
    return data[:, 4:6]

def acceleration_sonify_transform(data):
    return data[:, 5]

def density_transform(data, size):
    xy = data[:, 0:2]

    nx = (xy[:, 0] + 5.0) / 10.0
    ny = (xy[:, 1] + 5.0) / 10.0

    ix = np.clip((nx * size).astype(int), 0, size-1)
    iy = np.clip((ny * size).astype(int), 0, size-1)

    mat = np.zeros((size, size), dtype=np.float32)
    np.add.at(mat, (ix, iy), 1.0)
    return mat

def log_normalizer(arr):
    log_arr = np.log1p(arr)
    vmin = np.min(log_arr)
    vmax = np.max(log_arr)
    return (log_arr - vmin) / (vmax - vmin)


def init_state(n = 500, pos_scale = 1.0, vel_scale = 0.1, seed = None):
    rng = np.random.default_rng(seed)

    x = rng.uniform(-pos_scale, pos_scale, size=n)
    y = rng.uniform(-pos_scale, pos_scale, size=n)

    vx = rng.normal(0.0, vel_scale, size=n)
    vy = rng.normal(0.0, vel_scale, size=n)

    ax = np.zeros(n)
    ay = np.zeros(n)

    return np.column_stack((x, y, vx, vy, ax, ay))

def run_simulation_loop(data_source, edges_ds, *, dt_seconds=1/120.0, target_hz=60.0,
                        G=0.0005, r_min=0.1, k=3, stop_event=None):
    while stop_event is None or not stop_event.is_set():
        state = data_source.get()
        pos = state[:, 0:2]
        vel = state[:, 2:4]

        diff = pos[None, :, :] - pos[:, None, :]
        dist2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist2, np.inf)

        dist_geom = np.sqrt(dist2)

        dist_phys = np.maximum(dist_geom, r_min)

        inv_r3 = 1.0 / (dist_phys * dist_phys * dist_phys)
        acc = G * np.sum(diff * inv_r3[:, :, None], axis=1)

        vel = vel + acc * dt_seconds
        pos = pos + vel * dt_seconds

        out_x = (pos[:, 0] < -5) | (pos[:, 0] > 5)
        out_y = (pos[:, 1] < -5) | (pos[:, 1] > 5)

        pos[:, 0] = np.clip(pos[:, 0], -5, 5)
        pos[:, 1] = np.clip(pos[:, 1], -5, 5)

        vel[out_x, 0] *= -0.5
        vel[out_y, 1] *= -0.5
        new_state = np.column_stack((pos, vel, acc))
        data_source.set(new_state)

        edges = []
        n_points = len(pos)

        for i in range(n_points):
            nearest = np.argsort(dist_geom[i])[:3]
            for j in nearest[1:3]:
                edges.append([i, j])

        edges_ds.set(np.array(edges, dtype=int))

        time.sleep(0.016)

N = 100
state0 = init_state(n=N, pos_scale=1.0, vel_scale=0.06, seed=None)
data_source = DataSource(state0)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

canvas_position = Canvas()
canvas_velocity = Canvas()
canvas_acceleration = Canvas()
canvas_sonify = Canvas()
canvas_density = Canvas()
canvas_polyline = Canvas()

scatter_pos = ScatterPlot(data_source)
scatter_pos.set_transform(position_transform)

scatter_vel = ScatterPlot(data_source)
scatter_vel.set_transform(velocity_transform)

scatter_acc = ScatterPlot(data_source)
scatter_acc.set_transform(acceleration_transform)

line_sonify = LinePlot(
    data_source=data_source,
    sample_rate=1000,
    transform=acceleration_sonify_transform,
)

heatmap_plot = HeatmapPlot(
    data_source=data_source,
    transform=lambda data: density_transform(data, size=64),
    normalizer=log_normalizer,
    scale_x=10.0 / 64.0,
    scale_y=10.0 / 64.0,
)
heatmap_plot.set_translation(-5.0, -5.0)

vertices_ds = data_source  
edges_ds = DataSource(np.zeros((0, 2), dtype=int))

polyline_plot = PolylinePlot(
    vertices=vertices_ds,
    edges=edges_ds,
    line_width=1.5,
)

canvas_velocity.plot(scatter_vel)
canvas_velocity.show()

canvas_position.plot(scatter_pos)
canvas_position.plot(scatter_vel)
canvas_position.plot(scatter_acc)
canvas_position.plot(heatmap_plot)
canvas_position.plot(polyline_plot)
canvas_position.show()

canvas_acceleration.plot(scatter_acc)
canvas_acceleration.show()

canvas_sonify.plot(line_sonify)
canvas_sonify.show()

canvas_density.plot(heatmap_plot)
canvas_density.show()

canvas_polyline.plot(polyline_plot)
canvas_polyline.show()

stop_event = threading.Event()
sim_thread = threading.Thread(
    target=run_simulation_loop,
    kwargs=dict(
        data_source=data_source,
        edges_ds=edges_ds,
        dt_seconds=1/120.0,
        target_hz=60.0,
        G=0.005,
        r_min=0.1,
        k=1,
        stop_event=stop_event,
    ),
    daemon=True,
)
sim_thread.start()

app.exec()

stop_event.set()
sim_thread.join(timeout=2.0)

