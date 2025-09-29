import pandas as pd
import numpy as np

np.random.seed(0)
data = pd.DataFrame({
    "x1": np.random.rand(100),
    "y1": np.random.rand(100),
    "x2": np.random.rand(100),
    "y2": np.random.rand(100)
})


import altair as alt

brush = alt.selection_interval()

scatter1 = alt.Chart(data).mark_circle(size=80).encode(
    x="x1",
    y="y1",
    color=alt.condition(brush, alt.value("steelblue"), alt.value("lightgray"))
).add_params(brush)

scatter2 = alt.Chart(data).mark_circle(size=80).encode(
    x="x2",
    y="y2",
    color=alt.condition(brush, alt.value("steelblue"), alt.value("lightgray"))
).add_params(brush)

chart = scatter1 | scatter2
chart.save("altair_linked.html")
