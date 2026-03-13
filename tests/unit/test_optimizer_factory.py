"""TDD unit tests for OptimizerFactory — COBYLA + SPSA."""
from __future__ import annotations

import pytest

from vqe_knapsack.config import OptimizerConfig
from vqe_knapsack.optimizer_factory import OptimizerFactory


class TestOptimizerFactory:
    def test_creates_cobyla(self):
        opt = OptimizerFactory.create(OptimizerConfig("COBYLA", maxiter=100))
        from qiskit_algorithms.optimizers import COBYLA
        assert isinstance(opt, COBYLA)

    def test_creates_spsa(self):
        opt = OptimizerFactory.create(OptimizerConfig("SPSA", maxiter=500))
        from qiskit_algorithms.optimizers import SPSA
        assert isinstance(opt, SPSA)

    def test_raises_on_unsupported_optimizer(self):
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            OptimizerFactory.create(OptimizerConfig("ADAM", maxiter=100))

    def test_cobyla_maxiter_passed_correctly(self):
        opt = OptimizerFactory.create(OptimizerConfig("COBYLA", maxiter=42))
        assert opt.settings["maxiter"] == 42

    def test_spsa_maxiter_passed_correctly(self):
        opt = OptimizerFactory.create(OptimizerConfig("SPSA", maxiter=777))
        # SPSA stores maxiter in settings
        assert opt.settings.get("maxiter") == 777
