import pandas as pd

data = pd.DataFrame({
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y1": [1,4,9,16,25,36,49,64,81,100],
    "y2": [2,3,5,7,11,13,17,19,23,29]
})


from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.layouts import row

output_file("bokeh_linked.html")

source = ColumnDataSource(data)

p1 = figure(title="y1 vs x", width=400, height=400, tools="box_select,lasso_select")
p1.circle("x", "y1", size=8, source=source)

p2 = figure(title="y2 vs x", width=400, height=400, tools="box_select,lasso_select")
p2.circle("x", "y2", size=8, source=source)

show(row(p1, p2))
