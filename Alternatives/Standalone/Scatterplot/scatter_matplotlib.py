import pandas as pd
import numpy as np

np.random.seed(0)
data = pd.DataFrame({
    "x1": np.random.rand(100),
    "y1": np.random.rand(100),
    "x2": np.random.rand(100),
    "y2": np.random.rand(100)
})


import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.scatter(data["x1"], data["y1"], alpha=0.7)

ax2.scatter(data["x2"], data["y2"], alpha=0.7)

plt.show()
