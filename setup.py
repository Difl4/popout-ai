"""Editable-install configuration for the PopOut AI package.

The set of `install_requires` below mirrors the runtime dependencies declared
in `environment.yml`, so `pip install -e .` produces a working environment
for both the GUI/CLI and the optimised Numba kernels.
"""

from setuptools import setup, find_packages

setup(
    name="popout_ai",
    version="0.1.0",
    description="PopOut AI — MCTS + ID3 agents for the PopOut Connect-4 variant.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "notebook",
        "ipykernel",
        "pygame",
        "numba",
        "scikit-learn",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov"],
    },
)
