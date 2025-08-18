import time
import threading
import numpy as np
from PyQt5 import QtWidgets, QtCore
from canvas.layers.ScatterLayer import ScatterLayer
from canvas.layers.LineLayer import LineLayer
from DataSource import DataSource
from canvas.Canvas import Canvas

def position_transform(data: np.ndarray) -> np.ndarray:
    if data.ndim != 2 or data.shape[1] < 2:
        return np.empty((0, 2))
    return data[:, 0:2]

def velocity_transform(data: np.ndarray) -> np.ndarray:
    if data.ndim != 2 or data.shape[1] < 4:
        return np.empty((0, 2))
    return data[:, 2:4]

def acceleration_transform(data: np.ndarray) -> np.ndarray:
    if data.ndim != 2 or data.shape[1] < 6:
        return np.empty((0, 2))
    return data[:, 4:6]

def acceleration_sonify_transform(data: np.ndarray) -> np.ndarray:
    if data.ndim != 2 or data.shape[1] < 6:
        return np.empty((0, 2))
    return data[:, 5]

def init_state(n: int = 500, pos_scale: float = 1.0, vel_scale: float = 0.1, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)

    x = rng.uniform(-pos_scale, pos_scale, size=n)
    y = rng.uniform(-pos_scale, pos_scale, size=n)

    vx = rng.normal(0.0, vel_scale, size=n)
    vy = rng.normal(0.0, vel_scale, size=n)

    ax = np.zeros(n, dtype=float)
    ay = np.zeros(n, dtype=float)

    return np.column_stack((x, y, vx, vy, ax, ay))

def run_simulation_loop(data_source, *, dt_seconds=1/120.0, target_hz=60.0, G=0.0005, r_min=0.1, stop_event=None):
    while stop_event is None or not stop_event.is_set():
        state = data_source.get()
        pos = state[:, 0:2]
        vel = state[:, 2:4]

        diff = pos[None, :, :] - pos[:, None, :]
        dist2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist2, np.inf)
        dist = np.sqrt(dist2)
        dist = np.maximum(dist, r_min)

        inv_r3 = 1.0 / (dist * dist * dist)

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
        time.sleep(0.016)

N = 500
state0 = init_state(n=N, pos_scale=1.0, vel_scale=0.06, seed=None)
data_source = DataSource(state0)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

canvas_position = Canvas()
canvas_velocity = Canvas()
canvas_acceleration = Canvas()
canvas_sonify = Canvas()

scatter_pos = ScatterLayer(data_source)
scatter_pos.set_transform(position_transform)

scatter_vel = ScatterLayer(data_source)
scatter_vel.set_transform(velocity_transform)

scatter_acc = ScatterLayer(data_source)
scatter_acc.set_transform(acceleration_transform)

line_sonify = LineLayer(data_source, 1000)
line_sonify.set_transform(acceleration_sonify_transform)

canvas_position.plot(scatter_pos)
canvas_position.plot(scatter_vel)
canvas_position.plot(scatter_acc)
canvas_position.show()

canvas_velocity.plot(scatter_vel)
canvas_velocity.show()

canvas_acceleration.plot(scatter_acc)
canvas_acceleration.show()

canvas_sonify.plot(line_sonify)
canvas_sonify.show()

stop_event = threading.Event()
sim_thread = threading.Thread(
    target=run_simulation_loop,
    kwargs=dict(
        data_source=data_source,
        dt_seconds=1/120.0,
        target_hz=60.0,
        G=0.0005,
        r_min=0.1,
        stop_event=stop_event,
    ),
    daemon=True,
)
sim_thread.start()

app.exec()

stop_event.set()
sim_thread.join(timeout=2.0)

