"""
ALL tests in this file import qiskit_aer which causes a 10-30 min cold-start
freeze in new Python processes on this machine (Qiskit Aer has heavy Rust extensions).

This entire file is marked @pytest.mark.aer and excluded from `make test` (fast mode).

Run with:
    make test-aer
    # or manually:
    pytest -m aer tests/unit/test_aer_estimator_factory.py
"""
from __future__ import annotations

import pytest

from vqe_knapsack.config import BackendConfig, CPU_BACKEND, GPU_PARALLEL_BACKEND

# All tests in this file are slow (qiskit_aer cold import)
pytestmark = pytest.mark.aer


class TestAerEstimatorFactoryCPU:
    """Verifies AerEstimator creation with CPU backend (no GPU needed)."""

    def test_creates_cpu_estimator(self):
        from vqe_knapsack.aer_estimator_factory import AerEstimatorFactory
        estimator = AerEstimatorFactory.create(CPU_BACKEND)
        assert estimator is not None

    def test_detect_device_returns_cpu_or_gpu(self):
        from vqe_knapsack.aer_estimator_factory import AerEstimatorFactory
        result = AerEstimatorFactory.detect_device()
        assert result in ("CPU", "GPU")


class TestAerEstimatorFactoryGPU:
    """GPU-specific tests — run on machines with RTX 5050 / CUDA."""

    def test_gpu_fallback_to_cpu_logs_warning(self):
        from unittest.mock import patch
        import warnings
        from vqe_knapsack.aer_estimator_factory import AerEstimatorFactory
        with patch("vqe_knapsack.aer_estimator_factory._gpu_available", return_value=False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                estimator = AerEstimatorFactory.create(
                    BackendConfig(device="GPU", strict_gpu=False)
                )
                assert estimator is not None
                assert any("falling back to CPU" in str(x.message).lower() for x in w)

    def test_gpu_strict_raises_when_unavailable(self):
        from unittest.mock import patch
        from vqe_knapsack.aer_estimator_factory import AerEstimatorFactory
        with patch("vqe_knapsack.aer_estimator_factory._gpu_available", return_value=False):
            with pytest.raises(RuntimeError, match="GPU not available"):
                AerEstimatorFactory.create(BackendConfig(device="GPU", strict_gpu=True))
