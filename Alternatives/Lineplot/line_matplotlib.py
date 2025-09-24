import pandas as pd

data = pd.DataFrame({
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y1": [1,4,9,16,25,36,49,64,81,100],
    "y2": [2,3,5,7,11,13,17,19,23,29]
})


import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
ax1.scatter(data["x"], data["y1"])
ax1.set_title("y1 vs x")

ax2.scatter(data["x"], data["y2"])
ax2.set_title("y2 vs x")

plt.show()
