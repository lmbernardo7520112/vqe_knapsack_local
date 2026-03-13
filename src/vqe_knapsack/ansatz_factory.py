"""
AnsatzFactory — creates n_local ansatz circuits from AnsatzConfig specs.

Wraps Qiskit's n_local function to decouple experiment configuration
from circuit creation, enabling clean dependency injection.
"""
from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.n_local import n_local

from vqe_knapsack.config import AnsatzConfig


class AnsatzFactory:
    """Factory to create n_local ansatz circuits from AnsatzConfig."""

    @staticmethod
    def create(config: AnsatzConfig, n_qubits: int) -> QuantumCircuit:
        """
        Build a parametrised n_local ansatz circuit.

        Args:
            config:    AnsatzConfig specifying reps, rotations, entanglement
            n_qubits:  number of qubits (= number of items)

        Returns:
            QuantumCircuit — the parametrised ansatz (unbound parameters)
        """
        return n_local(
            n_qubits,
            reps=config.reps,
            rotation_blocks=list(config.rotation_blocks),
            entanglement=config.entanglement,
            entanglement_blocks=list(config.entanglement_blocks),
        )
