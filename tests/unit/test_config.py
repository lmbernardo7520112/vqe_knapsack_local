"""Unit tests for KnapsackConfig — TDD Red→Green→Refactor."""
from __future__ import annotations

import pytest

from vqe_knapsack.config import (
    AnsatzConfig,
    KnapsackConfig,
    OptimizerConfig,
    NOTEBOOK_CONFIG,
    FAST_TEST_CONFIG,
)


@pytest.fixture
def minimal_config() -> KnapsackConfig:
    return KnapsackConfig(
        values=(3, 4, 5),
        weights=(2, 3, 4),
        capacity=5,
        a_schedule=(1, 10),
        ansatz_configs=(
            AnsatzConfig(reps=2, rotation_blocks=("ry",), entanglement="linear", entanglement_blocks=("cx",)),
        ),
        optimizer_configs=(
            OptimizerConfig(name="COBYLA", maxiter=100),
        ),
    )


class TestKnapsackConfigValidation:
    """Config must enforce invariants at construction time."""

    def test_valid_config_creates_successfully(self, minimal_config):
        assert minimal_config.n_items == 3

    def test_raises_when_values_weights_length_mismatch(self):
        with pytest.raises(ValueError, match="len\\(values\\)"):
            KnapsackConfig(
                values=(1, 2, 3),
                weights=(1, 2),  # mismatch!
                capacity=5,
                a_schedule=(1,),
                ansatz_configs=(AnsatzConfig(reps=2, rotation_blocks=("ry",), entanglement="linear", entanglement_blocks=("cx",)),),
                optimizer_configs=(OptimizerConfig(name="COBYLA", maxiter=100),),
            )

    def test_raises_when_capacity_zero(self):
        with pytest.raises(ValueError, match="capacity must be positive"):
            KnapsackConfig(
                values=(1,), weights=(1,), capacity=0,
                a_schedule=(1,),
                ansatz_configs=(AnsatzConfig(reps=2, rotation_blocks=("ry",), entanglement="linear", entanglement_blocks=("cx",)),),
                optimizer_configs=(OptimizerConfig(name="COBYLA", maxiter=100),),
            )

    def test_raises_when_a_schedule_empty(self):
        with pytest.raises(ValueError, match="a_schedule must not be empty"):
            KnapsackConfig(
                values=(1,), weights=(1,), capacity=5,
                a_schedule=(),
                ansatz_configs=(AnsatzConfig(reps=2, rotation_blocks=("ry",), entanglement="linear", entanglement_blocks=("cx",)),),
                optimizer_configs=(OptimizerConfig(name="COBYLA", maxiter=100),),
            )

    def test_config_is_frozen(self, minimal_config):
        with pytest.raises((AttributeError, TypeError)):
            minimal_config.capacity = 99  # type: ignore


class TestKnapsackConfigProperties:
    def test_n_items_returns_length_of_values(self, minimal_config):
        assert minimal_config.n_items == 3

    def test_to_dict_round_trip_preserves_values(self, minimal_config):
        d = minimal_config.to_dict()
        assert d["values"] == [3, 4, 5]
        assert d["capacity"] == 5
        assert d["a_schedule"] == [1, 10]


class TestPrebuiltConfigs:
    def test_notebook_config_has_10_items(self):
        assert NOTEBOOK_CONFIG.n_items == 10
        assert NOTEBOOK_CONFIG.capacity == 18
        assert len(NOTEBOOK_CONFIG.ansatz_configs) == 5
        assert len(NOTEBOOK_CONFIG.optimizer_configs) == 4
        assert len(NOTEBOOK_CONFIG.a_schedule) == 10

    def test_notebook_config_total_experiments(self):
        total = (
            len(NOTEBOOK_CONFIG.ansatz_configs)
            * len(NOTEBOOK_CONFIG.optimizer_configs)
            * len(NOTEBOOK_CONFIG.a_schedule)
        )
        assert total == 200

    def test_fast_test_config_has_3_items(self):
        assert FAST_TEST_CONFIG.n_items == 3


class TestBackendConfig:
    """Fast tests for BackendConfig — no qiskit_aer import."""

    def test_default_is_cpu_sequential(self):
        from vqe_knapsack.config import BackendConfig
        cfg = BackendConfig()
        assert cfg.device == "CPU"
        assert cfg.n_workers == 1
        assert cfg.shots is None

    def test_gpu_config_valid(self):
        from vqe_knapsack.config import BackendConfig
        cfg = BackendConfig(device="GPU", n_workers=8)
        assert cfg.device == "GPU"

    def test_raises_when_n_workers_zero(self):
        from vqe_knapsack.config import BackendConfig
        with pytest.raises(ValueError, match="n_workers must be >= 1"):
            BackendConfig(n_workers=0)

    def test_raises_when_shots_zero(self):
        from vqe_knapsack.config import BackendConfig
        with pytest.raises(ValueError, match="shots must be None or >= 1"):
            BackendConfig(shots=0)

    def test_is_frozen(self):
        from vqe_knapsack.config import BackendConfig
        cfg = BackendConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.n_workers = 99  # type: ignore

    def test_cpu_backend_preset(self):
        from vqe_knapsack.config import CPU_BACKEND
        assert CPU_BACKEND.device == "CPU"
        assert CPU_BACKEND.n_workers == 1

    def test_gpu_parallel_preset(self):
        from vqe_knapsack.config import GPU_PARALLEL_BACKEND
        assert GPU_PARALLEL_BACKEND.device == "GPU"
        assert GPU_PARALLEL_BACKEND.n_workers == 16


class TestN20Config:
    """Tests for the 20-item knapsack configuration."""

    def test_n20_has_20_items(self):
        from vqe_knapsack.config import N20_CONFIG
        assert N20_CONFIG.n_items == 20
        assert N20_CONFIG.capacity == 30

    def test_n20_uses_spsa_optimizers(self):
        from vqe_knapsack.config import N20_CONFIG
        for opt in N20_CONFIG.optimizer_configs:
            assert opt.name == "SPSA", f"Expected SPSA, got {opt.name}"

    def test_n20_ansatz_are_shallow(self):
        from vqe_knapsack.config import N20_CONFIG
        for ansatz in N20_CONFIG.ansatz_configs:
            assert ansatz.reps <= 3, f"Expected reps<=3 for n=20, got {ansatz.reps}"

    def test_n20_total_experiments_is_200(self):
        from vqe_knapsack.config import N20_CONFIG
        total = (
            len(N20_CONFIG.ansatz_configs)
            * len(N20_CONFIG.optimizer_configs)
            * len(N20_CONFIG.a_schedule)
        )
        assert total == 200
