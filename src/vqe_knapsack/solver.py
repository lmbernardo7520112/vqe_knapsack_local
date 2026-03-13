"""
VQESolver — encapsulates VQE execution with warm-start and result extraction.

Design:
- All dependencies (estimator, VQE factory) injected via constructor → testable
- Returns typed SolverResult dataclass — no raw dict returned
- Handles bitstring extraction via Statevector after optimal parameter assignment
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import Optimizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolverResult:
    """Result of a single VQE solve call."""

    bitstring: str
    value: int
    weight: int
    probability: float
    valid: bool
    penalty: float
    optimal_point: np.ndarray | None

    def to_dict(self) -> dict:
        return {
            "bitstring": self.bitstring,
            "value": self.value,
            "weight": self.weight,
            "probability": round(float(self.probability), 4),
            "valid": self.valid,
            "penalty": self.penalty,
            "optimal_point": (
                self.optimal_point.tolist() if self.optimal_point is not None else None
            ),
        }


class VQESolver:
    """
    Encapsulates a single VQE solve for the Knapsack Hamiltonian.

    Args:
        values:   item values (for decoding bitstring → objective value)
        weights:  item weights
        capacity: knapsack capacity W
    """

    def __init__(
        self,
        values: tuple[int, ...] | list[int],
        weights: tuple[int, ...] | list[int],
        capacity: int,
    ) -> None:
        self._values = tuple(values)
        self._weights = tuple(weights)
        self._capacity = capacity
        self._n = len(values)

    def solve(
        self,
        hamiltonian: SparsePauliOp,
        ansatz: QuantumCircuit,
        optimizer: Optimizer,
        penalty: float,
        initial_point: np.ndarray | None = None,
        estimator=None,  # type: ignore[assignment]
    ) -> SolverResult:
        """
        Run VQE and decode the result.

        Args:
            hamiltonian:   The Ising Hamiltonian for the current penalty A
            ansatz:        Parametrised ansatz circuit
            optimizer:     Qiskit optimizer instance
            penalty:       Current penalty factor A (for result metadata)
            initial_point: Optional warm-start parameter vector
            estimator:     Optional estimator (StatevectorEstimator or AerEstimator).
                           If None, uses StatevectorEstimator (CPU, exact).
                           Pass AerEstimator for GPU acceleration.

        Returns:
            SolverResult with full metadata including new optimal_point for warm-start
        """
        if estimator is None:
            estimator = StatevectorEstimator()

        vqe = VQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
        )

        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        optimal_point: np.ndarray = result.optimal_point

        # Decode: assign optimal parameters → Statevector → most probable bitstring
        optimal_circuit = ansatz.assign_parameters(optimal_point)
        state = Statevector(optimal_circuit)
        bitstring, prob = self._most_probable_bitstring(state)

        # Evaluate the decoded solution
        bits = tuple(int(b) for b in bitstring)
        weight = sum(self._weights[i] * bits[i] for i in range(self._n))
        value = sum(self._values[i] * bits[i] for i in range(self._n))
        valid = weight <= self._capacity

        logger.debug(
            "VQE result",
            bitstring=bitstring,
            value=value,
            weight=weight,
            valid=valid,
            prob=round(float(prob), 4),
            penalty=penalty,
        )

        return SolverResult(
            bitstring=bitstring,
            value=value,
            weight=weight,
            probability=float(prob),
            valid=valid,
            penalty=penalty,
            optimal_point=optimal_point,
        )

    def _most_probable_bitstring(self, statevector: Statevector) -> tuple[str, float]:
        """Return the bitstring with the highest probability amplitude."""
        probs = np.abs(np.array(statevector)) ** 2
        idx = int(np.argmax(probs))
        bitstring = format(idx, f"0{self._n}b")
        return bitstring, float(probs[idx])
