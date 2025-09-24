import pandas as pd
import numpy as np

np.random.seed(0)
data = pd.DataFrame({
    "x1": np.random.rand(100),
    "y1": np.random.rand(100),
    "x2": np.random.rand(100),
    "y2": np.random.rand(100)
})


from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.layouts import row

output_file("bokeh_scatter_linked.html")

source = ColumnDataSource(data)

p1 = figure(title="Scatterplot 1", width=400, height=400, tools="box_select,lasso_select")
p1.circle("x1", "y1", source=source, size=6, alpha=0.6)

p2 = figure(title="Scatterplot 2", width=400, height=400, tools="box_select,lasso_select")
p2.circle("x2", "y2", source=source, size=6, alpha=0.6)

show(row(p1, p2))
