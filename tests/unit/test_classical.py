"""Unit tests for brute_force_solver — deterministic verification."""
from __future__ import annotations

import pytest

from vqe_knapsack.classical import ClassicalSolution, brute_force_solver


class TestBruteForceSolverKnownInstances:
    """All expected solutions are analytically verified by hand."""

    def test_notebook_5_item_instance(self):
        """Original 5-item notebook instance — known optimal is bits=(1,0,1,1,0), value=17."""
        result = brute_force_solver(
            values=[12, 1, 4, 1, 2],
            weights=[4, 2, 10, 1, 2],
            capacity=15,
        )
        assert result.value == 17
        assert result.is_valid
        assert result.weight <= 15

    def test_notebook_10_item_instance(self):
        """Full 10-item notebook instance — known optimal is value=65."""
        result = brute_force_solver(
            values=[12, 1, 4, 1, 2, 7, 8, 3, 11, 30],
            weights=[4, 2, 10, 1, 2, 8, 5, 1, 3, 4],
            capacity=18,
        )
        assert result.value == 65
        assert result.is_valid
        assert result.weight <= 18

    def test_single_item_fits(self):
        result = brute_force_solver([5], [3], capacity=4)
        assert result.value == 5
        assert result.bitstring == (1,)

    def test_single_item_does_not_fit(self):
        result = brute_force_solver([5], [10], capacity=4)
        assert result.value == 0
        assert result.bitstring == (0,)

    def test_all_items_too_heavy(self):
        result = brute_force_solver([10, 10], [100, 100], capacity=5)
        assert result.value == 0
        assert result.weight == 0

    def test_simple_2_item_greedy_invalid(self):
        """Higher value item doesn't fit; must pick lower value one."""
        result = brute_force_solver([10, 3], [6, 2], capacity=5)
        assert result.value == 3
        assert result.bitstring == (0, 1)


class TestBruteForceSolverValidation:
    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError, match="len\\(values\\)"):
            brute_force_solver([1, 2], [1], capacity=5)

    def test_raises_on_empty_instance(self):
        with pytest.raises(ValueError, match="at least one item"):
            brute_force_solver([], [], capacity=5)


class TestClassicalSolutionDataclass:
    def test_is_valid_when_weight_equal_capacity(self):
        sol = ClassicalSolution(bitstring=(1, 1), value=5, weight=5, capacity=5)
        assert sol.is_valid

    def test_is_invalid_when_weight_exceeds_capacity(self):
        sol = ClassicalSolution(bitstring=(1, 1), value=5, weight=6, capacity=5)
        assert not sol.is_valid

    def test_str_representation_includes_bits_and_value(self):
        sol = ClassicalSolution(bitstring=(1, 0), value=3, weight=2, capacity=5)
        s = str(sol)
        assert "10" in s
        assert "3" in s
