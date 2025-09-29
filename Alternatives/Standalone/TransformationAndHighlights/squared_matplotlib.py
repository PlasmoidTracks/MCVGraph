import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

# Generate random points
rng = np.random.default_rng(42)
points = rng.uniform(-1, 1, size=(100, 2))

# Complex square transformation
z = points[:,0] + 1j*points[:,1]
z2 = z**2
points_sq = np.column_stack((z2.real, z2.imag))

# Shared selection state
selected = np.zeros(len(points), dtype=bool)

# Create two figures
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

# Original scatter
sc1 = ax1.scatter(points[:,0], points[:,1], c="gray")
highlight1 = ax1.scatter([], [], c="red")

# Transformed scatter
sc2 = ax2.scatter(points_sq[:,0], points_sq[:,1], c="gray")
highlight2 = ax2.scatter([], [], c="red")

def update_highlights():
    highlight1.set_offsets(points[selected])
    highlight2.set_offsets(points_sq[selected])
    fig1.canvas.draw_idle()
    fig2.canvas.draw_idle()

def onselect1(verts):
    global selected
    path = Path(verts)
    selected = path.contains_points(points)
    update_highlights()

def onselect2(verts):
    global selected
    path = Path(verts)
    selected = path.contains_points(points_sq)
    update_highlights()

# Add lasso selectors
lasso1 = LassoSelector(ax1, onselect1)
lasso2 = LassoSelector(ax2, onselect2)

plt.show()
