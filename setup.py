from setuptools import setup, find_packages

setup(
    name="MCVGraph",
    version="0.1.0",
    author="Lukas Penner",
    description="Multiple Coordinated View Graph plotting utilities",
    packages=find_packages(),
    install_requires=[
        # Pin 1.24.4 for Python < 3.12
        "numpy==1.24.4; python_version<'3.12'",
        # Allow newer NumPy for Python >= 3.12
        "numpy>=1.26.0,<2.0; python_version>='3.12'",
        "pyqtgraph>=0.13.3",
        "sounddevice>=0.5.2",
        "PyQt5-sip>=12.15.0",
    ],
    extras_require={
        "windows": [
            "PyQt5>=5.15.10",
        ],
    },
    python_requires=">=3.8",
)
