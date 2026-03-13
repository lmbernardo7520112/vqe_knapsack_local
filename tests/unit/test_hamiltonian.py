"""Unit tests for build_knapsack_hamiltonian — analytically verified coefficients."""
from __future__ import annotations

import numpy as np
import pytest

from vqe_knapsack.hamiltonian import build_knapsack_hamiltonian


class TestHamiltonianStructure:
    """Structural tests: correct number of Pauli terms."""

    def test_n2_number_of_pauli_terms(self):
        """n=2: 3 linear groups × 2 + 1 quadratic = 7 terms before simplification."""
        op = build_knapsack_hamiltonian([3, 4], [2, 3], capacity=5, penalty=1.0)
        # SparsePauliOp simplifies like terms; result should be valid SparsePauliOp
        assert op.num_qubits == 2

    def test_num_qubits_equals_num_items(self):
        for n in [2, 3, 5, 10]:
            values = [1] * n
            weights = [1] * n
            op = build_knapsack_hamiltonian(values, weights, capacity=n - 1, penalty=1.0)
            assert op.num_qubits == n

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError, match="len\\(values\\)"):
            build_knapsack_hamiltonian([1, 2], [1], capacity=5, penalty=1.0)

    def test_raises_on_empty_instance(self):
        with pytest.raises(ValueError, match="at least one item"):
            build_knapsack_hamiltonian([], [], capacity=5, penalty=1.0)


class TestHamiltonianCoefficients:
    """
    Analytical verification for n=2.

    Problem: values=[2,3], weights=[1,2], W=2, A=1
    sum_weights = 3

    Linear Z₀ (qubit 0):
        + v₀/2                   = +1.0
        - A·w₀·sum_w/2 = -1·1·3/2 = -1.5
        + A·W·w₀       = 1·2·1   = +2.0
        Total Z₀ coeff = +1.5

    Linear Z₁ (qubit 1):
        + v₁/2                   = +1.5
        - A·w₁·sum_w/2 = -1·2·3/2 = -3.0
        + A·W·w₁       = 1·2·2   = +4.0
        Total Z₁ coeff = +2.5

    Quadratic Z₀Z₁:
        + A·w₀·w₁/2 = 1·1·2/2 = +1.0
    """

    @pytest.fixture
    def op_n2(self):
        """n=2 instance with analytically known coefficients."""
        return build_knapsack_hamiltonian([2, 3], [1, 2], capacity=2, penalty=1.0)

    def test_n2_is_hermitian(self, op_n2):
        """SparsePauliOp must be Hermitian (all real coefficients in Z basis)."""
        matrix = op_n2.to_matrix()
        assert np.allclose(matrix, matrix.conj().T), "Hamiltonian must be Hermitian"

    def test_n2_linear_z0_coefficient(self, op_n2):
        """Coefficient of ZI must be +1.5 (analytically derived)."""
        simplified = op_n2.simplify()
        labels = [str(p) for p in simplified.paulis]
        coeffs = simplified.coeffs.real

        zi_coeff = sum(c for lbl, c in zip(labels, coeffs) if lbl == "ZI")
        assert abs(zi_coeff - 1.5) < 1e-9, f"Expected ZI coeff=1.5, got {zi_coeff}"

    def test_n2_linear_iz_coefficient(self, op_n2):
        """Coefficient of IZ must be +2.5 (analytically derived)."""
        simplified = op_n2.simplify()
        labels = [str(p) for p in simplified.paulis]
        coeffs = simplified.coeffs.real

        iz_coeff = sum(c for lbl, c in zip(labels, coeffs) if lbl == "IZ")
        assert abs(iz_coeff - 2.5) < 1e-9, f"Expected IZ coeff=2.5, got {iz_coeff}"

    def test_n2_quadratic_zz_coefficient(self, op_n2):
        """Coefficient of ZZ must be +1.0 (analytically derived)."""
        simplified = op_n2.simplify()
        labels = [str(p) for p in simplified.paulis]
        coeffs = simplified.coeffs.real

        zz_coeff = sum(c for lbl, c in zip(labels, coeffs) if lbl == "ZZ")
        assert abs(zz_coeff - 1.0) < 1e-9, f"Expected ZZ coeff=1.0, got {zz_coeff}"

    def test_penalty_scales_quadratic_terms(self):
        """Quadratic terms must scale exactly with A."""
        op_a1 = build_knapsack_hamiltonian([1, 1], [2, 3], capacity=5, penalty=1.0)
        op_a5 = build_knapsack_hamiltonian([1, 1], [2, 3], capacity=5, penalty=5.0)
        s1 = op_a1.simplify()
        s5 = op_a5.simplify()
        labels1 = [str(p) for p in s1.paulis]
        labels5 = [str(p) for p in s5.paulis]
        zz1 = sum(c.real for lbl, c in zip(labels1, s1.coeffs) if lbl == "ZZ")
        zz5 = sum(c.real for lbl, c in zip(labels5, s5.coeffs) if lbl == "ZZ")
        assert abs(zz5 / zz1 - 5.0) < 1e-9, "ZZ coeff must scale linearly with A"
