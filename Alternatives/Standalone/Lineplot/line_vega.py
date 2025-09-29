import pandas as pd

data = pd.DataFrame({
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y1": [1,4,9,16,25,36,49,64,81,100],
    "y2": [2,3,5,7,11,13,17,19,23,29]
})





import altair as alt

brush = alt.selection_interval()

points1 = alt.Chart(data).mark_circle(size=80).encode(
    x="x",
    y="y1",
    color=alt.condition(brush, alt.value("steelblue"), alt.value("lightgray"))
).add_params(brush)

points2 = alt.Chart(data).mark_circle(size=80).encode(
    x="x",
    y="y2",
    color=alt.condition(brush, alt.value("steelblue"), alt.value("lightgray"))
).add_params(brush)

chart = points1 | points2
chart.save("altair_linked.html")
