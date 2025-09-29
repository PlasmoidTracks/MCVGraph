import time
import threading
import numpy as np
from PyQt5 import QtWidgets

from MCVGraph import Canvas, ScatterPlot, HeatmapPlot, DataSource

# =========================
# Config
# =========================
GRID_N = 100            # resolution for decision surface heatmap
H1 = 8                 # hidden layer 1 width
H2 = 8                 # hidden layer 2 width
LR = 0.1                # learning rate

# =========================
# Data / Task (ring classification)
# =========================
def make_ring_dataset(n_per_axis=20, r_thresh=0.5):
    xs = np.linspace(-1, 1, n_per_axis)
    ys = np.linspace(-1, 1, n_per_axis)
    xv, yv = np.meshgrid(xs, ys)
    X = np.stack([xv.flatten(), yv.flatten()], axis=1)  # (N,2)
    r = np.sqrt(np.sum(X**2, axis=1))
    y = (r <= r_thresh).astype(np.float32)[:, None]    # (N,1) inside=1, outside=0
    return X, y

def make_grid(n=GRID_N):
    g = np.linspace(-1, 1, n)
    gx, gy = np.meshgrid(g, g)
    G = np.stack([gx, gy], axis=2)
    G_flat = G.reshape(-1, 2)
    return g, gx, gy, G_flat

# =========================
# NN
# =========================
def nn_init(input_dim=2, hidden1=H1, hidden2=H2, output_dim=1, seed=0):
    rng = np.random.RandomState(seed)
    def z(shape):
        return rng.randn(*shape)
    return {
        "W1": z((hidden1, input_dim)),
        "b1": z((hidden1, 1)),
        "W2": z((hidden2, hidden1)),
        "b2": z((hidden2, 1)),
        "W3": z((output_dim, hidden2)),
        "b3": z((output_dim, 1)),
    }

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def nn_forward(params, x):
    z1 = params["W1"] @ x.T + params["b1"]
    h1 = np.tanh(z1)
    z2 = params["W2"] @ h1 + params["b2"]
    h2 = np.tanh(z2)
    z3 = params["W3"] @ h2 + params["b3"]
    y = sigmoid(z3).T
    cache = (x, z1, h1, z2, h2, z3)
    return y, cache

def nn_backward(params, cache, dy, lr=LR):
    x, z1, h1, z2, h2, z3 = cache
    m = x.shape[0]

    dz3 = dy.T
    dW3 = (dz3 @ h2.T) / m
    db3 = np.mean(dz3, axis=1, keepdims=True)

    dh2 = (params["W3"].T @ dz3) * (1 - np.tanh(z2) ** 2)
    dW2 = (dh2 @ h1.T) / m
    db2 = np.mean(dh2, axis=1, keepdims=True)

    dh1 = (params["W2"].T @ dh2) * (1 - np.tanh(z1) ** 2)
    dW1 = (dh1 @ x) / m
    db1 = np.mean(dh1, axis=1, keepdims=True)

    params["W1"] -= lr * dW1
    params["b1"] -= lr * db1
    params["W2"] -= lr * dW2
    params["b2"] -= lr * db2
    params["W3"] -= lr * dW3
    params["b3"] -= lr * db3

# =========================
# Transforms
# =========================
def transform_inputs(_):
    return X

def transform_outputs_prob_x(_):
    # map predicted probability onto x-axis in range [-1, 1]
    # map distance from origin onto y-axis
    if Y_pred is None or Y_pred.shape[0] == 0:
        return np.empty((0, 2))
    probs = Y_pred.flatten()
    mapped_x = 2 * probs - 1
    radii = np.linalg.norm(X, axis=1)
    out_points = np.column_stack([mapped_x, radii])
    return out_points

def transform_w1(_):
    return params["W1"]

def transform_w2(_):
    return params["W2"]

def transform_w3(_):
    return params["W3"]

def transform_pred_heatmap(_):
    return pred_grid.reshape(GRID_N, GRID_N)

# =========================
# Training loop
# =========================
def training_loop():
    global params, Y_pred, pred_grid
    while True:
        Y_pred, cache = nn_forward(params, X)
        dy = (Y_pred - Y_true)
        nn_backward(params, cache, dy, lr=LR)

        Yg, _ = nn_forward(params, G_flat)
        pred_grid = Yg.squeeze(axis=1)

        ds_inputs.set(X)
        ds_w1.set(params["W1"])
        ds_w2.set(params["W2"])
        ds_w3.set(params["W3"])
        ds_pred_heat.set(pred_grid)

        time.sleep(0.05)

# =========================
# Main
# =========================
if __name__ == "__main__":
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    X, Y_true = make_ring_dataset(n_per_axis=20, r_thresh=0.8)
    g, gx, gy, G_flat = make_grid(GRID_N)

    params = nn_init()
    Y_pred = np.zeros_like(Y_true)
    pred_grid = np.zeros((GRID_N * GRID_N,), dtype=np.float32)

    ds_inputs    = DataSource(X)
    ds_w1        = DataSource(params["W1"])
    ds_w2        = DataSource(params["W2"])
    ds_w3        = DataSource(params["W3"])
    ds_pred_heat = DataSource(pred_grid)

    # Inputs canvas
    canvas_in = Canvas(name="Inputs")
    scatter_in = ScatterPlot(data_source=ds_inputs)
    scatter_in.set_transform(lambda data: data)

    heat_decision = HeatmapPlot(
        data_source=ds_pred_heat,
        transform=transform_pred_heatmap,
        graph_name="P(class=1)",
        scale_x=2.0 / GRID_N,
        scale_y=2.0 / GRID_N,
    )
    heat_decision.set_translation(-1.0, -1.0)
    canvas_in.plot(scatter_in)
    canvas_in.plot(heat_decision)
    canvas_in.show()

    # Outputs canvas
    canvas_out = Canvas(name="Outputs")

    # Probability as scatter (prob on x-axis)
    scatter_out_probx = ScatterPlot(data_source=ds_inputs)
    scatter_out_probx.set_transform(transform_outputs_prob_x)
    canvas_out.plot(scatter_out_probx)

    canvas_out.set_view_port(-1.2, -0.2, 1.2, 1.2)
    canvas_out.show()


    # Weight canvases
    canvas_w2 = Canvas(name="W2")
    heat_w2 = HeatmapPlot(
        data_source=ds_w2,
        transform=transform_w2,
        graph_name="W2 (H2×H1)",
        scale_x=1.0, scale_y=1.0
    )
    canvas_w2.plot(heat_w2)
    canvas_w2.show()

    t = threading.Thread(target=training_loop, daemon=True)
    t.start()

    app.exec_()
