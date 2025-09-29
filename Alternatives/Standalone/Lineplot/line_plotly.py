import pandas as pd

data = pd.DataFrame({
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y1": [1,4,9,16,25,36,49,64,81,100],
    "y2": [2,3,5,7,11,13,17,19,23,29]
})





import plotly.express as px

fig1 = px.scatter(data, x="x", y="y1")
fig2 = px.scatter(data, x="x", y="y2")

fig1.show()
fig2.show()
