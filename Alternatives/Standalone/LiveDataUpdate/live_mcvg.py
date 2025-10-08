import time
import threading
import numpy as np

from PyQt5 import QtWidgets
from MCVGraph import DataSource, Canvas, ScatterPlot

# Update thread
def run_update_loop(data_source, stop_event):
    rng = np.random.default_rng(0)
    while not stop_event.is_set():
        points = rng.uniform(0, 1, size=(100, 2))
        data_source.set(points)
        time.sleep(0.016)


if __name__ == "__main__":
    # Initial random data
    rng = np.random.default_rng(42)
    points = rng.uniform(0, 1, size=(100, 2))
    data_source = DataSource(points)

    # Standard Qt application
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # Create canvas and scatter plot
    canvas = Canvas()
    scatter = ScatterPlot(data_source)

    canvas.plot(scatter)
    canvas.show()

    # Start background thread
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_update_loop,
        args=(data_source, stop_event),
        daemon=True,
    )
    thread.start()

    # Run Qt event loop
    app.exec()

    # Clean shutdown
    stop_event.set()
    thread.join(timeout=1.0)
