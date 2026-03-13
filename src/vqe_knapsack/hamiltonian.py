"""
build_knapsack_hamiltonian — pure function mapping 0/1 Knapsack to Ising Hamiltonian.

QUBO → Ising mapping:
    xᵢ = (I - Zᵢ) / 2

Cost function to MINIMISE (VQE finds ground state):
    H = -Σᵢ vᵢ xᵢ  +  A·(Σᵢ wᵢ xᵢ - W)²

After substitution xᵢ = (I - Zᵢ)/2:
    Linear terms (Zᵢ):
        +vᵢ/2                       (value contribution)
        -A·wᵢ·(Σⱼ wⱼ)/2            (quadratic penalty, linear part)
        +A·W·wᵢ                     (cross term with capacity W)
    Quadratic terms (ZᵢZⱼ):
        +A·wᵢ·wⱼ/2
    Constant: ignored (does not affect argmin)
"""
from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp


def build_knapsack_hamiltonian(
    values: tuple[int, ...] | list[int],
    weights: tuple[int, ...] | list[int],
    capacity: int,
    penalty: float,
) -> SparsePauliOp:
    """
    Build the Ising Hamiltonian for the 0/1 Knapsack problem.

    Args:
        values:   item values  (length n)
        weights:  item weights (length n)
        capacity: knapsack weight limit W
        penalty:  penalty coefficient A (annealing parameter)

    Returns:
        SparsePauliOp representing H in the Pauli Z basis.

    Raises:
        ValueError: if len(values) != len(weights) or inputs are invalid.
    """
    n = len(values)
    if n != len(weights):
        raise ValueError(f"len(values)={n} != len(weights)={len(weights)}")
    if n == 0:
        raise ValueError("Problem must have at least one item.")

    sum_weights = sum(weights)
    paulis: list[str] = []
    coeffs: list[float] = []

    # ── Linear terms for value: +vᵢ/2 · Zᵢ ─────────────────────────────────
    for i in range(n):
        z = ["I"] * n
        z[i] = "Z"
        paulis.append("".join(z))
        coeffs.append(values[i] / 2.0)

    # ── Penalty linear term 1: -A·wᵢ·sum_w/2 · Zᵢ ──────────────────────────
    for i in range(n):
        z = ["I"] * n
        z[i] = "Z"
        paulis.append("".join(z))
        coeffs.append(-penalty * weights[i] * sum_weights / 2.0)

    # ── Penalty linear term 2: +A·W·wᵢ · Zᵢ ────────────────────────────────
    for i in range(n):
        z = ["I"] * n
        z[i] = "Z"
        paulis.append("".join(z))
        coeffs.append(penalty * capacity * weights[i])

    # ── Quadratic terms: +A·wᵢ·wⱼ/2 · ZᵢZⱼ ────────────────────────────────
    for i in range(n):
        for j in range(i + 1, n):
            z = ["I"] * n
            z[i] = "Z"
            z[j] = "Z"
            paulis.append("".join(z))
            coeffs.append(penalty * weights[i] * weights[j] / 2.0)

    return SparsePauliOp(paulis, coeffs)
