# Repo Overview

## High-Level Structure

The repository implements a modular visualization framework built on **PyQt5** and **pyqtgraph**. It enables users to visualize, interact with, and synchronize multiple views of the same underlying dataset across different graph types, facilitating deep data exploration and cross-referential analysis. The framework provides:  

- **DataSource** -- encapsulates a dataset and emits signals when the data changes.  
- **Graphs** -- visualization objects (e.g., `ScatterPlot`) that connect to a `DataSource` and render the data.  
- **Canvas** -- container for graphs, responsible for layering and displaying them in a Qt widget.  
- **GraphBus** -- singleton communication bus for coordinating events and selections across graphs.  
- **Events & Selections** -- mechanism for propagating user interaction (like selecting points) through connected graphs.  
- **GraphWidget** -- wrapper around graph classes which provides macro control of windows. 

This design allows multiple visualizations to remain synchronized when exploring or transforming datasets.

---

## Internal Mechanisms

### Data Flow and Communication
1. **DataSource**  
    - Wraps a NumPy array, or any structure that can be cast into a numpy.array. 
    - Emits **change signals** whenever the dataset is modified.  
    - Graphs connected to this `DataSource` automatically update when the data changes.  

2. **GraphBus (Singleton)**  
    - Central event dispatcher, ensuring there is **only one instance** across the application.  
    - Graphs and canvases register themselves with the `GraphBus`.  
    - Events (e.g., selection updates, redraw triggers) are broadcast to all registered listeners.  

3. **Graphs**  
    - Each graph (scatter plot, line plot, etc.) is a **layer** on a `Canvas`.  
    - They subscribe to data updates from their `DataSource`.  
    - They also subscribe to global events from the `GraphBus`.  
    - Graphs can emit their own events (like a user selection) which propagate through the `GraphBus` to other components.  

4. **Canvas and Layering**  
    - A `Canvas` is a PyQt widget embedding one or more graphs.  
    - Graphs are **coerced into layers**, allowing multiple visualizations to be drawn on the same coordinate system.  
        - This is done by creating a clone of the underlying graph
    - The `Canvas` registers itself with the `GraphBus` to receive and forward events.  

---

# Setup Instructions 

This library depends on PyQt5, which must come from the system packages on Linux to avoid ABI issues. Other dependencies are installed via pip in a virtual environment. On Windows, PyQt5 is handled via pip inside the venv. 

--- 

## System requirements

This library is compatible with python versions from 3.8 up to 3.14, which is the latest version as of the time of writing. 

---

## Setup using source code

### Linux

#### 1. Install system dependencies 
```bash 
sudo apt update 
sudo apt install -y python3-pyqt5 python3-venv 
``` 

#### 2. Create a virtual environment with system site packages 
```bash 
python3 -m venv --system-site-packages venv 
source venv/bin/activate 
``` 

#### 3. Install Python dependencies (excluding PyQt5) 
Your requirements file (`requirements_linux.txt`) should **not** include PyQt5. Example contents: 
``` 
pyqtgraph==0.13.7 
numpy==1.26.4 
sounddevice==0.5.2 
``` 

Then install: 
```bash 
pip install --upgrade pip 
pip install -r requirements_linux.txt 
``` 

#### 4. Run the script 
```bash 
python "main.py" 
``` 

--- 

### Windows

#### 1. Create a virtual environment
```powershell 
python.exe -m venv venv 
``` 

#### 2. Activate the venv In PowerShell: 
```powershell 
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass 
.\venv\Scripts\Activate.ps1 
``` 

#### 3. Install Python dependencies 
Your requirements file (`requirements_windows.txt`) should include all pip dependencies (PyQt5 included). Example contents: 
``` 
PyQt5==5.15.10 
PyQt5-sip==12.13.0 
pyqtgraph==0.13.7 
numpy==1.26.4 
sounddevice==0.5.2 
``` 

Then install: 
```powershell 
python.exe -m pip install --upgrade pip
pip install -r requirements_windows.txt 
``` 

#### 4. Run the script 
```powershell 
python "main.py" 
```

--- 

## Setup using wheel package

### Linux

#### 1. Install system dependencies 
```bash 
sudo apt update 
sudo apt install -y python3-pyqt5 python3-venv
``` 

#### 2. Create a virtual environment with system site packages 
```bash 
python3 -m venv --system-site-packages venv 
source venv/bin/activate 
``` 

#### 3. Install Python package
```bash 
pip install --upgrade pip 
pip install [PATH TO PACKAGE\]/mcvgraph-0.1.0-py3-none-any.whl
``` 

#### 4. Run the script
```bash 
python "main.py" 
``` 

--- 

### Windows

#### 1. Create a virtual environment with system site packages 
```powershell 
python.exe -m venv venv 
``` 

#### 2. Activate the venv In PowerShell: 
```powershell 
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass 
.\venv\Scripts\Activate.ps1 
``` 

#### 3. Install Python package
```
python.exe -m pip install --upgrade pip
pip install "[PATH TO PACKAGE\]/mcvgraph-0.1.0-py3-none-any.whl[windows]"
``` 

#### 4. Run the script
```powershell 
python "main.py" 
```

---

## Building the package (.whl and .tar.gz)

In order to generate the package as a pip-installable file, the following script can be executed
```
pip install build
python -m build
```

---

# Getting started

### To get started with the package you can execute the following minimal setup

```python
import numpy as np
from PyQt5 import QtWidgets

from MCVGraph.DataSource import DataSource
from MCVGraph.canvas.Canvas import Canvas
from MCVGraph.graphs.ScatterPlot import ScatterPlot

# Create some random 2D data
rng = np.random.default_rng(42)
points = rng.uniform(-1, 1, size=(100, 2))
data_source = DataSource(points)

# Standard Qt application
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

# Create a canvas and a scatter plot
canvas = Canvas()
scatter = ScatterPlot(data_source)

canvas.plot(scatter)
canvas.show()

# Run the event loop
app.exec()
```

---

# Troubleshooting

## Linux

### `ModuleNotFoundError: No module named 'PyQt5.sip'`

Should something like this show up: 

```python
(venv) usr@usr:~/MCVGraph$ python main.py
Traceback (most recent call last): 
  File "/home/usr/MCVGraph/main.py", line 4, in <module>
    from PyQt5 import QtWidgets, QtCore
ModuleNotFoundError: No module named 'PyQt5.sip'
```

then this could mean that the python instance was trying to use the system-wide, or otherwise installed `PyQt5-sip` python package, instead of the venv one. 
To fix this issue, remove or rename the `PyQt5-sip` package from the system-wide python environment and install it manually in the venv. This is **NOT** recommended. 
The name is typically something like `PyQt5_sip-x.y.z.egg-info`

```bash
sudo mv /usr/lib/python3/dist-packages/PyQt5_sip-12.13.0.egg-info /usr/lib/python3/dist-packages/PyQt5_sip-12.13.0.egg-info_DISABLED
# while in venv
pip install pyqt5-sip
```

### `Failed to build PyQt5-sip`

`PyQt5-sip` does not come prebuild with all python versions. As of the time of creating, python 3.13 is the last python version that does not need to build `PyQt5-sip` itself. Should an error occur that looks something like this

```sh
Building wheels for collected packages: PyQt5-sip
  Building wheel for PyQt5-sip (pyproject.toml) ... error
  error: subprocess-exited-with-error
  
  × Building wheel for PyQt5-sip (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [11 lines of output]
      running bdist_wheel
      running build
      running build_ext
      building 'PyQt5.sip' extension
      creating build/temp.linux-x86_64-cpython-3XY
      x86_64-linux-gnu-gcc -fno-strict-overflow -Wsign-compare -DNDEBUG -g -O2 -Wall -g -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -fstack-protector-strong -fstack-clash-protection -Wformat -Werror=format-security -fcf-protection -g -fwrapv -O2 -fPIC -I/home/lukas/Schreibtisch/UNI/Bachelor/Project_cutdown/venv/include -I/usr/include/python3.XY -c apiversions.c -o build/temp.linux-x86_64-cpython-3XY/apiversions.o
      apiversions.c:12:10: fatal error: Python.h: Datei oder Verzeichnis nicht gefunden
         12 | #include <Python.h>
            |          ^~~~~~~~~~
      compilation terminated.
      error: command '/usr/bin/x86_64-linux-gnu-gcc' failed with exit code 1
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for PyQt5-sip
Failed to build PyQt5-sip
error: failed-wheel-build-for-install

× Failed to build installable wheels for some pyproject.toml based projects
╰─> PyQt5-sip
```

then the solution is to install the respective `Header files and static libraries` and try again

```sh
sudo apt install python3.XY-dev
pip install "[PATH TO PACKAGE\]/mcvgraph-0.1.0-py3-none-any.whl[windows]"
```

---

# Tips and general usage

In general, each plot can be interacted with by the mouse in two ways. In `selection mode`, which is the default mode, and `panning mode`. In `selection mode`, dragging the mouse over the canvas while holding the `left mouse button` usually results in a selection being started, though this is not defined for all graph types. `panning mode` can be entered by holding Control `[CTRL]` while dragging the mouse and holding either the left mouse button `[LMB]` or the right mouse button `[RMB]` will pan or scale the view of the canvas. In `panning mode`, using the `mouse wheel` also scales the view. 



# Future work

- Logical selection 
    - Union (an accumulative selection that compounds over multiple selections)
    - Difference (subtracting the current selection from the previously selected points) 
    - Xor (inverting selection, where previously selected points are unselected and previously unselected point become selected)
- Continuous selection
    - Currently, selections only update the selected set of point while the selection bound is updated
    - A toggle might make that update continuous, updating the selection list even after finalizing the selection boundary
- Performance
    - Sparse updates may help distribute computation over a period of time, while keeping the window fluid and responsive
        - An example would be to update 10% of the selection highlights per frame, thus distributing the workload over 10 frames. 
        - At high refresh rates, this should not be noticable
        - This would also counteract fast changing selection where the full selection set needs to be computed before the next selection box can be calculated
    - Throttling
        - Graphs that are not displayed could be throttled
- 3D visualization
    - Adding interactive projections, orthographic or perspective, would allow for greater data analysis
    - Simple controls for pitch, yaw, roll and positional translation could open up to emersive exploration of data
- Automated view
    - Adding predetermined or dynamic viewport paths, such as following a spline trajectory as the viewport center, automated zooming over time, or dynamically tracking a point or set of points, could enable programmatic control of camera motion. This would allow for reproducible visualizations, scripted exploration of datasets, and integration into automated analysis pipelines
    - A subset of this functionality is already build-in through the use of pyqtgraph, through the auto-scaling option of the axis
- Quality of Life
    - To add an undo- and redo-operation, a history stack of operations could be added 
    - Custom keybinds that trigger either custom functions, or execute a sequence of operations
- Datasource flexibility
    - Adding more convenience in how DataSource can be instantiated and used
    - Allowing higher dimensional data as input and outputting multiple datasources at once
    - Using one Datasource to feed multiple different graphs using different data sources
- Timbral flexibility
    - Add different sonification modes
    - Let the user set a convolution function, which is applied to the sonification


# Known issues

- when the DataSource is being updated too fast, the update queue of the `PyQt graph` may overflow and the graphs become unresponsive. This can be circumvented by keeping updates reasonably spaced. 60hz is a very safe option. 

- Selections can be slow, especially with many updating graphs. 

- Currently, Canvases generate clones of graphs when plotting them. This results in background graphs that still recieve events und updates, causing additional computation and lag. 
