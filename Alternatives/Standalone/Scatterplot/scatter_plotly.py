import pandas as pd
import numpy as np

np.random.seed(0)
data = pd.DataFrame({
    "x1": np.random.rand(100),
    "y1": np.random.rand(100),
    "x2": np.random.rand(100),
    "y2": np.random.rand(100)
})


import plotly.express as px

fig1 = px.scatter(data, x="x1", y="y1")
fig2 = px.scatter(data, x="x2", y="y2")

fig1.show()
fig2.show()
