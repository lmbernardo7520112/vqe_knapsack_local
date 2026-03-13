"""
AerEstimatorFactory — creates Qiskit Aer EstimatorV2 with GPU or CPU backend.

CORRECT API (verified from qiskit_aer/primitives/estimator_v2.py source):
    EstimatorV2(options={"backend_options": {"device": "GPU", "method": "statevector"}})

The `backend_options` dict is passed directly to AerSimulator(**backend_options).
Setting `estimator.options.device` does NOT work — that field doesn't exist.
"""
from __future__ import annotations

import logging
import warnings

from vqe_knapsack.config import BackendConfig

logger = logging.getLogger(__name__)


def _gpu_available() -> bool:
    """Check if Qiskit Aer CUDA GPU is available at runtime."""
    try:
        from qiskit_aer import AerSimulator
        devices = AerSimulator().available_devices()
        return "GPU" in devices
    except Exception:
        return False


class AerEstimatorFactory:
    """
    Factory for creating Qiskit Aer EstimatorV2 with GPU support.

    Uses the backend_options API:
        EstimatorV2(options={"backend_options": {"device": "GPU", "method": "statevector"}})
    """

    @staticmethod
    def create(config: BackendConfig):
        """
        Create a V2-compatible Estimator for the given BackendConfig.

        Returns:
            EstimatorV2 instance with GPU or CPU backend.

        Raises:
            RuntimeError: if device="GPU", strict_gpu=True, and GPU unavailable.
        """
        requested_device = config.device
        actual_device = requested_device

        if requested_device == "GPU":
            if not _gpu_available():
                if config.strict_gpu:
                    raise RuntimeError(
                        "GPU requested (strict_gpu=True) but CUDA not available. "
                        "Check NVIDIA driver and qiskit-aer-gpu installation."
                    )
                warnings.warn(
                    "GPU not available — falling back to CPU statevector.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                actual_device = "CPU"

        try:
            from qiskit_aer.primitives import EstimatorV2

            # backend_options is passed directly to AerSimulator(**backend_options)
            backend_opts = {
                "device": actual_device,
                "method": "statevector",
            }
            estimator = EstimatorV2(options={"backend_options": backend_opts})
            logger.info("AerEstimatorV2 created (device=%s, method=statevector)", actual_device)
            return estimator

        except (ImportError, AttributeError) as exc:
            warnings.warn(
                f"qiskit_aer.primitives.EstimatorV2 not available ({exc}) — "
                "falling back to StatevectorEstimator (CPU exact).",
                RuntimeWarning,
                stacklevel=2,
            )
            from qiskit.primitives import StatevectorEstimator
            logger.info("StatevectorEstimator fallback (device=CPU)")
            return StatevectorEstimator()

    @staticmethod
    def detect_device() -> str:
        """Return 'GPU' if CUDA available, else 'CPU'."""
        return "GPU" if _gpu_available() else "CPU"
