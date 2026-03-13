"""
OptimizerFactory — creates qiskit_algorithms optimizer instances from OptimizerConfig.

Supported optimizers:
    COBYLA — gradient-free, good for small n (≤10 items, ≤30 params)
    SPSA   — stochastic gradient, scales O(p), recommended for n≥10 items
"""
from __future__ import annotations

from qiskit_algorithms.optimizers import COBYLA, SPSA, Optimizer

from vqe_knapsack.config import OptimizerConfig


class OptimizerFactory:
    """Factory to create Qiskit optimizer instances from OptimizerConfig."""

    _SUPPORTED: set[str] = {"COBYLA", "SPSA"}

    @staticmethod
    def create(config: OptimizerConfig) -> Optimizer:
        """
        Instantiate a Qiskit optimizer.

        Args:
            config: OptimizerConfig with name and maxiter

        Returns:
            Qiskit Optimizer instance

        Raises:
            ValueError: if optimizer name is not supported
        """
        if config.name == "COBYLA":
            return COBYLA(maxiter=config.maxiter)
        if config.name == "SPSA":
            return SPSA(maxiter=config.maxiter)
        raise ValueError(
            f"Unsupported optimizer '{config.name}'. "
            f"Supported: {OptimizerFactory._SUPPORTED}"
        )
