import time
import threading
import numpy as np

import matplotlib.pyplot as plt


# Update thread
def run_update_loop(scatter, stop_event):
    rng = np.random.default_rng(0)
    while not stop_event.is_set():
        points = rng.uniform(0, 1, size=(100, 2))
        scatter.set_offsets(points)
        plt.gcf().canvas.draw_idle()
        time.sleep(0.016)

if __name__ == "__main__":
    # Initial data
    rng = np.random.default_rng(42)
    points = rng.uniform(0, 1, size=(100, 2))

    # Setup interactive mode
    plt.ion()
    fig, ax = plt.subplots()
    scatter = ax.scatter(points[:, 0], points[:, 1], c="blue")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Start background thread
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_update_loop,
        args=(scatter, stop_event),
        daemon=True,
    )
    thread.start()

    try:
        # Keep window open
        plt.show(block=True)
    finally:
        # Clean shutdown
        stop_event.set()
        thread.join(timeout=1.0)
