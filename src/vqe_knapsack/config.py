"""KnapsackConfig — immutable dataclass for all experiment parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class AnsatzConfig:
    """Configuration for a single ansatz variant."""

    reps: int
    rotation_blocks: tuple[str, ...]
    entanglement: str
    entanglement_blocks: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "reps": self.reps,
            "rotation_blocks": list(self.rotation_blocks),
            "entanglement": self.entanglement,
            "entanglement_blocks": list(self.entanglement_blocks),
        }


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for a single optimizer variant."""

    name: str
    maxiter: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "options": {"maxiter": self.maxiter},
        }


@dataclass(frozen=True)
class BackendConfig:
    """
    Backend configuration for quantum simulation.

    Controls device (CPU/GPU), Ray parallelism, and shot mode.
    GPU mode uses Qiskit Aer with CUDA (RTX 5050 8GB on this machine).

    Invariants:
    - n_workers >= 1
    - shots must be None (exact statevector) or a positive integer
    """

    device: Literal["CPU", "GPU"] = "CPU"
    n_workers: int = 1  # 1 = sequential; up to 16 on this machine
    shots: int | None = None  # None = exact statevector mode
    strict_gpu: bool = False  # if True, raise if GPU unavailable; else fallback CPU

    def __post_init__(self) -> None:
        if self.n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {self.n_workers}")
        if self.shots is not None and self.shots < 1:
            raise ValueError(f"shots must be None or >= 1, got {self.shots}")
        if self.device not in ("CPU", "GPU"):
            raise ValueError(f"device must be 'CPU' or 'GPU', got {self.device!r}")

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "n_workers": self.n_workers,
            "shots": self.shots,
            "strict_gpu": self.strict_gpu,
        }


# Convenience presets
CPU_BACKEND = BackendConfig(device="CPU", n_workers=1)
GPU_PARALLEL_BACKEND = BackendConfig(device="GPU", n_workers=16)


@dataclass(frozen=True)
class KnapsackConfig:
    """
    Immutable specification of the knapsack problem and VQE experiment parameters.

    Invariants (enforced in __post_init__):
    - len(values) == len(weights)
    - All values and weights are positive integers
    - capacity > 0
    - a_schedule is non-empty and all entries are positive
    """

    values: tuple[int, ...]
    weights: tuple[int, ...]
    capacity: int
    a_schedule: tuple[int, ...]
    ansatz_configs: tuple[AnsatzConfig, ...]
    optimizer_configs: tuple[OptimizerConfig, ...]
    output_dir: str = "outputs"

    def __post_init__(self) -> None:
        if len(self.values) != len(self.weights):
            raise ValueError(
                f"len(values)={len(self.values)} != len(weights)={len(self.weights)}"
            )
        if any(v <= 0 for v in self.values):
            raise ValueError("All values must be positive integers.")
        if any(w <= 0 for w in self.weights):
            raise ValueError("All weights must be positive integers.")
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}.")
        if not self.a_schedule:
            raise ValueError("a_schedule must not be empty.")
        if any(a <= 0 for a in self.a_schedule):
            raise ValueError("All A values must be positive.")

    @property
    def n_items(self) -> int:
        """Number of items in the knapsack problem."""
        return len(self.values)

    def to_dict(self) -> dict:
        return {
            "values": list(self.values),
            "weights": list(self.weights),
            "capacity": self.capacity,
            "a_schedule": list(self.a_schedule),
            "ansatz_configs": [a.to_dict() for a in self.ansatz_configs],
            "optimizer_configs": [o.to_dict() for o in self.optimizer_configs],
            "output_dir": self.output_dir,
        }


# ── Default configurations matching the original notebook exactly ─────────────

DEFAULT_ANSATZ_CONFIGS: tuple[AnsatzConfig, ...] = (
    AnsatzConfig(reps=2, rotation_blocks=("ry",), entanglement="linear", entanglement_blocks=("cx",)),
    AnsatzConfig(reps=3, rotation_blocks=("ry", "rz"), entanglement="full", entanglement_blocks=("cx",)),
    AnsatzConfig(reps=4, rotation_blocks=("ry",), entanglement="full", entanglement_blocks=("cx",)),
    AnsatzConfig(reps=5, rotation_blocks=("ry",), entanglement="full", entanglement_blocks=("cx",)),
    AnsatzConfig(reps=6, rotation_blocks=("ry",), entanglement="full", entanglement_blocks=("cx",)),
)

DEFAULT_OPTIMIZER_CONFIGS: tuple[OptimizerConfig, ...] = (
    OptimizerConfig(name="COBYLA", maxiter=100),
    OptimizerConfig(name="COBYLA", maxiter=400),
    OptimizerConfig(name="COBYLA", maxiter=800),
    OptimizerConfig(name="COBYLA", maxiter=2000),
)

DEFAULT_A_SCHEDULE: tuple[int, ...] = (1, 3, 5, 10, 20, 50, 100, 200, 500, 1000)

# Exact 10-item instance from the notebook
NOTEBOOK_CONFIG = KnapsackConfig(
    values=(12, 1, 4, 1, 2, 7, 8, 3, 11, 30),
    weights=(4, 2, 10, 1, 2, 8, 5, 1, 3, 4),
    capacity=18,
    a_schedule=DEFAULT_A_SCHEDULE,
    ansatz_configs=DEFAULT_ANSATZ_CONFIGS,
    optimizer_configs=DEFAULT_OPTIMIZER_CONFIGS,
)

# Fast instance for CI/testing (n=3, trivially small)
FAST_TEST_CONFIG = KnapsackConfig(
    values=(3, 4, 5),
    weights=(2, 3, 4),
    capacity=5,
    a_schedule=(1, 5, 10),
    ansatz_configs=(
        AnsatzConfig(reps=2, rotation_blocks=("ry",), entanglement="linear", entanglement_blocks=("cx",)),
    ),
    optimizer_configs=(
        OptimizerConfig(name="COBYLA", maxiter=200),
    ),
)

# ── n=20 instance: use SPSA (scales O(p) vs COBYLA O(p²)) ────────────────────
# Shallow ansätze (reps=1-3) to mitigate barren plateaus at larger n
N20_ANSATZ_CONFIGS: tuple[AnsatzConfig, ...] = (
    AnsatzConfig(reps=1, rotation_blocks=("ry",), entanglement="linear", entanglement_blocks=("cx",)),
    AnsatzConfig(reps=2, rotation_blocks=("ry",), entanglement="linear", entanglement_blocks=("cx",)),
    AnsatzConfig(reps=2, rotation_blocks=("ry",), entanglement="full",   entanglement_blocks=("cx",)),
    AnsatzConfig(reps=3, rotation_blocks=("ry", "rz"), entanglement="linear", entanglement_blocks=("cx",)),
    AnsatzConfig(reps=3, rotation_blocks=("ry",), entanglement="full",   entanglement_blocks=("cx",)),
)

N20_OPTIMIZER_CONFIGS: tuple[OptimizerConfig, ...] = (
    OptimizerConfig(name="SPSA", maxiter=200),
    OptimizerConfig(name="SPSA", maxiter=500),
    OptimizerConfig(name="SPSA", maxiter=1000),
    OptimizerConfig(name="SPSA", maxiter=2000),
)

N20_CONFIG = KnapsackConfig(
    values=(12, 1, 4, 1, 2, 7, 8, 3, 11, 30, 5, 9, 6, 14, 2, 8, 11, 4, 7, 20),
    weights=( 4, 2,10, 1, 2, 8, 5, 1,  3,  4, 3, 6, 4,  7, 2, 5,  8, 2, 4,  6),
    capacity=30,
    a_schedule=DEFAULT_A_SCHEDULE,
    ansatz_configs=N20_ANSATZ_CONFIGS,
    optimizer_configs=N20_OPTIMIZER_CONFIGS,
    output_dir="outputs",
)
