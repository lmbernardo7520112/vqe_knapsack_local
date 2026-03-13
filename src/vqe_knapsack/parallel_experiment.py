"""
ParallelExperimentRunner — runs VQE experiments distributed across Ray workers.

Architecture:
- Decomposes the 200-experiment grid into 20 independent (ansatz × optimizer) combos
- Each combo runs sequentially (warm-start within), but combos run in parallel via Ray
- Falls back to sequential execution when n_workers=1 (no Ray required)

Clean Code:
- Open/Closed: extends ExperimentRunner without modifying it
- Single Responsibility: only handles worker dispatch, not VQE logic
- No global state: each Ray worker creates its own solver + estimator
"""
from __future__ import annotations

import logging
import os
from typing import Callable

import numpy as np

from vqe_knapsack.aer_estimator_factory import AerEstimatorFactory
from vqe_knapsack.ansatz_factory import AnsatzFactory
from vqe_knapsack.classical import ClassicalSolution, brute_force_solver
from vqe_knapsack.config import BackendConfig, KnapsackConfig, AnsatzConfig, OptimizerConfig
from vqe_knapsack.experiment import ExperimentResult, ExperimentRunner
from vqe_knapsack.hamiltonian import build_knapsack_hamiltonian
from vqe_knapsack.optimizer_factory import OptimizerFactory
from vqe_knapsack.solver import SolverResult, VQESolver

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


def _run_single_combo(
    ansatz_cfg: AnsatzConfig,
    optimizer_cfg: OptimizerConfig,
    knapsack_config_dict: dict,
    backend_config_dict: dict,
) -> list[dict]:
    """
    Worker function: runs all A values for one (ansatz, optimizer) combination.
    Designed to run inside a Ray remote task — accepts only serializable args (dicts).

    IMPORTANT: Ray overrides CUDA_VISIBLE_DEVICES="" in workers when num_gpus=0.
    We restore it here so each worker can access the GPU.

    Returns list of experiment result dicts.
    """
    import os
    # Restore GPU visibility — Ray hides it by default in CPU-only workers
    if backend_config_dict.get("device", "CPU").upper() == "GPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from vqe_knapsack.config import (
        AnsatzConfig, OptimizerConfig, BackendConfig,
    )
    from vqe_knapsack.aer_estimator_factory import AerEstimatorFactory
    from vqe_knapsack.ansatz_factory import AnsatzFactory
    from vqe_knapsack.hamiltonian import build_knapsack_hamiltonian
    from vqe_knapsack.optimizer_factory import OptimizerFactory
    from vqe_knapsack.solver import VQESolver

    cfg = knapsack_config_dict
    values = tuple(cfg["values"])
    weights = tuple(cfg["weights"])
    capacity = cfg["capacity"]

    backend = BackendConfig(
        device=backend_config_dict["device"],
        n_workers=backend_config_dict["n_workers"],
        shots=backend_config_dict.get("shots"),
        strict_gpu=backend_config_dict.get("strict_gpu", False),
    )

    # EstimatorV2 from AerEstimatorFactory (GPU if available, CPU fallback)
    estimator = AerEstimatorFactory.create(backend)
    solver = VQESolver(values=values, weights=weights, capacity=capacity)

    results = []
    current_initial_point = None

    for penalty in cfg["a_schedule"]:
        hamiltonian = build_knapsack_hamiltonian(values, weights, capacity, penalty)
        ansatz = AnsatzFactory.create(ansatz_cfg, len(values))
        optimizer = OptimizerFactory.create(optimizer_cfg)

        solver_result = solver.solve(
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            optimizer=optimizer,
            penalty=float(penalty),
            initial_point=current_initial_point,
            estimator=estimator,
        )
        current_initial_point = solver_result.optimal_point

        results.append({
            "ansatz_config": ansatz_cfg,
            "optimizer_config": optimizer_cfg,
            "penalty": float(penalty),
            "solver_result": solver_result,
        })

    return results


class ParallelExperimentRunner:
    """
    Runs VQE experiments across Ray workers for GPU + parallelism.

    Strategy:
    - n_workers=1  → sequential (no Ray), exact same as ExperimentRunner
    - n_workers>1  → dispatches each (ansatz × optimizer) combo to a Ray worker
                     Each worker gets its own AerEstimator (GPU or CPU)

    Args:
        config:          KnapsackConfig (problem + hyperparameter grids)
        backend_config:  BackendConfig (device, n_workers, shots)
        progress_callback: optional (current, total, msg) callback
    """

    def __init__(
        self,
        config: KnapsackConfig,
        backend_config: BackendConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._config = config
        self._backend = backend_config
        self._progress_callback = progress_callback

    def run_all_experiments(self) -> list[ExperimentResult]:
        """Run all experiments sequentially or in parallel based on n_workers."""
        if self._backend.n_workers == 1:
            return self._run_sequential()
        return self._run_parallel_ray()

    def _run_sequential(self) -> list[ExperimentResult]:
        """Sequential execution — uses AerEstimatorV2 (GPU if available, else CPU)."""
        estimator = AerEstimatorFactory.create(self._backend)
        results: list[ExperimentResult] = []
        total = (
            len(self._config.ansatz_configs)
            * len(self._config.optimizer_configs)
            * len(self._config.a_schedule)
        )
        current = 0

        for ansatz_cfg in self._config.ansatz_configs:
            for optimizer_cfg in self._config.optimizer_configs:
                current_initial_point: np.ndarray | None = None

                for penalty in self._config.a_schedule:
                    current += 1
                    msg = (
                        f"reps={ansatz_cfg.reps} | {optimizer_cfg.name}(maxiter={optimizer_cfg.maxiter})"
                        f" | A={penalty}"
                    )
                    if self._progress_callback:
                        self._progress_callback(current, total, msg)

                    from vqe_knapsack.hamiltonian import build_knapsack_hamiltonian
                    hamiltonian = build_knapsack_hamiltonian(
                        self._config.values, self._config.weights,
                        self._config.capacity, penalty,
                    )
                    ansatz = AnsatzFactory.create(ansatz_cfg, self._config.n_items)
                    optimizer = OptimizerFactory.create(optimizer_cfg)
                    solver = VQESolver(
                        values=self._config.values,
                        weights=self._config.weights,
                        capacity=self._config.capacity,
                    )
                    solver_result = solver.solve(
                        hamiltonian=hamiltonian,
                        ansatz=ansatz,
                        optimizer=optimizer,
                        penalty=float(penalty),
                        initial_point=current_initial_point,
                        estimator=estimator,
                    )
                    current_initial_point = solver_result.optimal_point
                    results.append(ExperimentResult(
                        ansatz_config=ansatz_cfg,
                        optimizer_config=optimizer_cfg,
                        penalty=float(penalty),
                        solver_result=solver_result,
                    ))

        return results

    def _run_parallel_ray(self) -> list[ExperimentResult]:
        """Parallel execution via Ray — one worker per (ansatz × optimizer) combo.

        Uses ray.wait() for incremental progress updates instead of
        ray.get(all) which would block until everything completes.
        """
        try:
            import ray
        except ImportError:
            logger.warning("Ray not installed — falling back to sequential.")
            return self._run_sequential()

        if not ray.is_initialized():
            # Prevent Ray from hiding GPU in workers
            os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
            ray.init(num_cpus=self._backend.n_workers, ignore_reinit_error=True)
            logger.info("Ray initialized with %d CPUs", self._backend.n_workers)

        run_remote = ray.remote(_run_single_combo)

        cfg_dict = self._config.to_dict()
        backend_dict = self._backend.to_dict()

        # Build (future, metadata) pairs for progress tracking
        combos = [
            (ansatz_cfg, optimizer_cfg)
            for ansatz_cfg in self._config.ansatz_configs
            for optimizer_cfg in self._config.optimizer_configs
        ]
        futures = [
            run_remote.remote(ansatz_cfg, optimizer_cfg, cfg_dict, backend_dict)
            for ansatz_cfg, optimizer_cfg in combos
        ]
        # Map future → combo index for progress messages
        future_to_idx = {f: i for i, f in enumerate(futures)}

        total_combos = len(futures)
        n_a_values = len(self._config.a_schedule)
        total_experiments = total_combos * n_a_values

        logger.info(
            "Dispatched %d Ray tasks (%d experiments) across %d workers",
            total_combos, total_experiments, self._backend.n_workers,
        )

        # Poll for completed futures incrementally
        results: list[ExperimentResult] = []
        remaining = list(futures)
        completed_combos = 0

        while remaining:
            done, remaining = ray.wait(remaining, num_returns=1, timeout=None)
            for future in done:
                combo_results = ray.get(future)
                idx = future_to_idx[future]
                ansatz_cfg, optimizer_cfg = combos[idx]
                completed_combos += 1

                for r in combo_results:
                    results.append(ExperimentResult(
                        ansatz_config=r["ansatz_config"],
                        optimizer_config=r["optimizer_config"],
                        penalty=r["penalty"],
                        solver_result=r["solver_result"],
                    ))

                # Incremental progress update
                if self._progress_callback:
                    msg = (
                        f"[{completed_combos}/{total_combos}] "
                        f"reps={ansatz_cfg.reps} | "
                        f"{optimizer_cfg.name}(maxiter={optimizer_cfg.maxiter})"
                    )
                    
                    # Convert raw dicts to ExperimentResult objects so they can be saved
                    new_results = []
                    for r in combo_results:
                        new_results.append(ExperimentResult(
                            ansatz_config=r["ansatz_config"],
                            optimizer_config=r["optimizer_config"],
                            penalty=r["penalty"],
                            solver_result=r["solver_result"],
                        ))
                    
                    self._progress_callback(
                        completed_combos * n_a_values,
                        total_experiments,
                        msg,
                        new_results
                    )

        logger.info("All Ray workers completed. Total results: %d", len(results))
        return results

    def run_classical_reference(self) -> ClassicalSolution:
        """Brute-force classical reference solver."""
        return brute_force_solver(
            self._config.values, self._config.weights, self._config.capacity,
        )

    @staticmethod
    def best_valid_result(results: list[ExperimentResult]) -> ExperimentResult | None:
        """Return the best valid VQE result (highest value, then highest probability)."""
        valid = [r for r in results if r.is_valid]
        if not valid:
            return None
        return max(valid, key=lambda r: (r.value, r.solver_result.probability))
