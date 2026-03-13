"""
ExperimentRunner — orchestrates all 200 VQE experiments with warm-start.

Design:
- Reads KnapsackConfig (immutable) → no global state
- Emits progress via callback → decoupled from UI/logging
- Returns list[ExperimentResult] — fully serializable
- Faithful reproduction of the original notebook's triple loop:
    for ansatz_cfg in ansatz_configs:
        for optimizer_cfg in optimizer_configs:
            current_initial_point = None   # reset per ansatz×optimizer combo
            for A in a_schedule:           # warm-start within this combo
                ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from vqe_knapsack.ansatz_factory import AnsatzFactory
from vqe_knapsack.classical import ClassicalSolution, brute_force_solver
from vqe_knapsack.config import AnsatzConfig, KnapsackConfig, OptimizerConfig
from vqe_knapsack.hamiltonian import build_knapsack_hamiltonian
from vqe_knapsack.optimizer_factory import OptimizerFactory
from vqe_knapsack.solver import SolverResult, VQESolver

logger = logging.getLogger(__name__)
console = Console()

ProgressCallback = Callable[[int, int, str], None]  # (current, total, message)


@dataclass
class ExperimentResult:
    """Full result record for a single experiment in the 200-experiment grid."""

    ansatz_config: AnsatzConfig
    optimizer_config: OptimizerConfig
    penalty: float
    solver_result: SolverResult

    def to_dict(self) -> dict:
        return {
            **self.ansatz_config.to_dict(),
            **self.optimizer_config.to_dict(),
            "A": self.penalty,
            **self.solver_result.to_dict(),
        }

    @property
    def is_valid(self) -> bool:
        return self.solver_result.valid

    @property
    def value(self) -> int:
        return self.solver_result.value


class ExperimentRunner:
    """
    Runs the full grid of VQE experiments for Knapsack with penalty annealing.

    Args:
        config: KnapsackConfig defining the problem and all hyperparameter grids
        progress_callback: Optional callback(current, total, message) for progress reporting
    """

    def __init__(
        self,
        config: KnapsackConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._config = config
        self._progress_callback = progress_callback
        self._solver = VQESolver(
            values=config.values,
            weights=config.weights,
            capacity=config.capacity,
        )

    def run_all_experiments(self) -> list[ExperimentResult]:
        """
        Execute all experiments and return results.

        Mirrors the notebook's triple loop exactly:
            ansatz_cfg → optimizer_cfg → A_schedule (with warm-start)

        Returns:
            list of ExperimentResult, one per (ansatz, optimizer, A) combination
        """
        results: list[ExperimentResult] = []
        total = (
            len(self._config.ansatz_configs)
            * len(self._config.optimizer_configs)
            * len(self._config.a_schedule)
        )
        current = 0

        for ansatz_cfg in self._config.ansatz_configs:
            for optimizer_cfg in self._config.optimizer_configs:
                current_initial_point: np.ndarray | None = None  # reset for each combo

                for penalty in self._config.a_schedule:
                    current += 1
                    msg = (
                        f"reps={ansatz_cfg.reps} | {optimizer_cfg.name}(maxiter={optimizer_cfg.maxiter})"
                        f" | A={penalty}"
                    )
                    logger.info("Running experiment %d/%d: %s", current, total, msg)
                    if self._progress_callback:
                        self._progress_callback(current, total, msg)

                    # Build Hamiltonian for this A
                    hamiltonian = build_knapsack_hamiltonian(
                        self._config.values,
                        self._config.weights,
                        self._config.capacity,
                        penalty,
                    )

                    # Create fresh ansatz and optimizer for this experiment
                    ansatz = AnsatzFactory.create(ansatz_cfg, self._config.n_items)
                    optimizer = OptimizerFactory.create(optimizer_cfg)

                    # Solve with warm-start from previous A iteration
                    solver_result = self._solver.solve(
                        hamiltonian=hamiltonian,
                        ansatz=ansatz,
                        optimizer=optimizer,
                        penalty=float(penalty),
                        initial_point=current_initial_point,
                    )

                    # Update warm-start for next A
                    current_initial_point = solver_result.optimal_point

                    results.append(
                        ExperimentResult(
                            ansatz_config=ansatz_cfg,
                            optimizer_config=optimizer_cfg,
                            penalty=float(penalty),
                            solver_result=solver_result,
                        )
                    )

        logger.info("Finished all experiments. Total: %d", len(results))
        return results

    @staticmethod
    def best_valid_result(results: list[ExperimentResult]) -> ExperimentResult | None:
        """Return the best valid VQE result (highest value, then highest probability)."""
        valid = [r for r in results if r.is_valid]
        if not valid:
            return None
        return max(valid, key=lambda r: (r.value, r.solver_result.probability))

    def run_classical_reference(self) -> ClassicalSolution:
        """Run the brute-force classical solver for comparison."""
        return brute_force_solver(
            self._config.values,
            self._config.weights,
            self._config.capacity,
        )
