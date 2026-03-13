"""
brute_force_solver — O(2^n) exhaustive search for the 0/1 Knapsack problem.

This is the exact classical method used in the original notebook via itertools.product.
Used as the reference oracle for correctness validation of VQE results.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassicalSolution:
    """Result of the classical brute-force knapsack solver."""

    bitstring: tuple[int, ...]
    value: int
    weight: int
    capacity: int

    @property
    def is_valid(self) -> bool:
        """Returns True if solution respects the weight capacity."""
        return self.weight <= self.capacity

    def __str__(self) -> str:
        bits = "".join(str(b) for b in self.bitstring)
        return f"ClassicalSolution(bits={bits}, value={self.value}, weight={self.weight}, valid={self.is_valid})"


def brute_force_solver(
    values: tuple[int, ...] | list[int],
    weights: tuple[int, ...] | list[int],
    capacity: int,
) -> ClassicalSolution:
    """
    Solve the 0/1 Knapsack problem by exhaustive search over all 2^n combinations.

    This is a O(2^n) algorithm — tractable for n ≤ 30 approx.
    Faithfully reproduces the itertools.product loop from the original notebook.

    Args:
        values:   item values (length n, all positive)
        weights:  item weights (length n, all positive)
        capacity: knapsack weight limit W

    Returns:
        ClassicalSolution with the optimal valid bitstring and its value/weight.

    Raises:
        ValueError: if inputs are inconsistent.
    """
    n = len(values)
    if n != len(weights):
        raise ValueError(f"len(values)={n} != len(weights)={len(weights)}")
    if n == 0:
        raise ValueError("Problem must have at least one item.")

    best_value = -1
    best_bits: tuple[int, ...] = tuple(0 for _ in range(n))

    for bits in itertools.product([0, 1], repeat=n):
        weight = sum(weights[i] * bits[i] for i in range(n))
        value = sum(values[i] * bits[i] for i in range(n))
        if weight <= capacity and value > best_value:
            best_value = value
            best_bits = bits

    best_weight = sum(weights[i] * best_bits[i] for i in range(n))
    return ClassicalSolution(
        bitstring=best_bits,
        value=max(best_value, 0),
        weight=best_weight,
        capacity=capacity,
    )
