"""Unit tests for VQESolver — using mocks to avoid real quantum execution."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vqe_knapsack.solver import SolverResult, VQESolver


@pytest.fixture
def solver_n5() -> VQESolver:
    return VQESolver(
        values=[12, 1, 4, 1, 2],
        weights=[4, 2, 10, 1, 2],
        capacity=15,
    )


class TestSolverResultDataclass:
    def test_to_dict_contains_all_keys(self):
        result = SolverResult(
            bitstring="10110",
            value=17,
            weight=15,
            probability=0.9998,
            valid=True,
            penalty=10.0,
            optimal_point=np.array([0.1, 0.2]),
        )
        d = result.to_dict()
        assert "bitstring" in d
        assert "value" in d
        assert "weight" in d
        assert "probability" in d
        assert "valid" in d
        assert "optimal_point" in d

    def test_optimal_point_serialized_as_list(self):
        result = SolverResult(
            bitstring="10", value=3, weight=2, probability=0.9,
            valid=True, penalty=5.0, optimal_point=np.array([1.0, 2.0]),
        )
        d = result.to_dict()
        assert isinstance(d["optimal_point"], list)

    def test_none_optimal_point_serialized_as_none(self):
        result = SolverResult(
            bitstring="10", value=3, weight=2, probability=0.9,
            valid=True, penalty=5.0, optimal_point=None,
        )
        assert result.to_dict()["optimal_point"] is None


class TestVQESolverMostProbableBitstring:
    """Test the private bitstring extraction without running real VQE."""

    def test_most_probable_bitstring_all_zeros(self, solver_n5):
        # State |00000⟩ has amplitude 1.0 at index 0 → bitstring "00000"
        from qiskit.quantum_info import Statevector
        state = Statevector.from_label("0" * 5)
        bitstring, prob = solver_n5._most_probable_bitstring(state)
        assert prob == pytest.approx(1.0)
        assert len(bitstring) == 5

    def test_most_probable_bitstring_single_computational_state(self, solver_n5):
        """State |10110⟩ should decode to bitstring '10110'."""
        from qiskit.quantum_info import Statevector
        state = Statevector.from_label("10110")
        bitstring, prob = solver_n5._most_probable_bitstring(state)
        assert prob == pytest.approx(1.0)
        assert bitstring == "10110"


class TestVQESolverSolveWithMock:
    """Test solve() method using mocked VQE to avoid real quantum simulation."""

    def test_solve_returns_solver_result_type(self, solver_n5):
        with patch("vqe_knapsack.solver.VQE") as MockVQE, \
             patch("vqe_knapsack.solver.StatevectorEstimator"):
            # Mock VQE.compute_minimum_eigenvalue
            mock_instance = MagicMock()
            mock_instance.compute_minimum_eigenvalue.return_value = MagicMock(
                optimal_point=np.zeros(15),  # 15 params for reps=2, n=5, ry, linear
            )
            MockVQE.return_value = mock_instance

            from qiskit.circuit.library.n_local import n_local
            from vqe_knapsack.hamiltonian import build_knapsack_hamiltonian
            from vqe_knapsack.optimizer_factory import OptimizerFactory
            from vqe_knapsack.config import OptimizerConfig

            hamiltonian = build_knapsack_hamiltonian([12, 1, 4, 1, 2], [4, 2, 10, 1, 2], 15, 10.0)
            ansatz = n_local(5, reps=2, rotation_blocks=["ry"], entanglement="linear", entanglement_blocks=["cx"])
            optimizer = OptimizerFactory.create(OptimizerConfig("COBYLA", maxiter=100))

            result = solver_n5.solve(hamiltonian, ansatz, optimizer, penalty=10.0)
            assert isinstance(result, SolverResult)
            assert result.penalty == pytest.approx(10.0)

    def test_warm_start_initial_point_passed_to_vqe(self, solver_n5):
        """Verifies that initial_point is forwarded to VQE constructor."""
        with patch("vqe_knapsack.solver.VQE") as MockVQE, \
             patch("vqe_knapsack.solver.StatevectorEstimator"):
            mock_instance = MagicMock()
            mock_instance.compute_minimum_eigenvalue.return_value = MagicMock(
                optimal_point=np.ones(15),
            )
            MockVQE.return_value = mock_instance

            from qiskit.circuit.library.n_local import n_local
            from vqe_knapsack.hamiltonian import build_knapsack_hamiltonian
            from vqe_knapsack.optimizer_factory import OptimizerFactory
            from vqe_knapsack.config import OptimizerConfig

            hamiltonian = build_knapsack_hamiltonian([12, 1, 4, 1, 2], [4, 2, 10, 1, 2], 15, 10.0)
            ansatz = n_local(5, reps=2, rotation_blocks=["ry"], entanglement="linear", entanglement_blocks=["cx"])
            optimizer = OptimizerFactory.create(OptimizerConfig("COBYLA", maxiter=100))

            warm_point = np.full(15, 0.7)
            solver_n5.solve(hamiltonian, ansatz, optimizer, penalty=10.0, initial_point=warm_point)

            # Verify VQE was called with the initial_point
            _, call_kwargs = MockVQE.call_args
            assert call_kwargs.get("initial_point") is warm_point
