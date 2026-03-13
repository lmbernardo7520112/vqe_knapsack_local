"""vqe_knapsack — VQE Knapsack Optimizer, standalone local project."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("vqe-knapsack")
except PackageNotFoundError:
    __version__ = "dev"

__all__ = ["__version__"]
