"""Integration test — runs real VQE on the fast n=3 instance."""
from __future__ import annotations

import pytest

from vqe_knapsack.config import FAST_TEST_CONFIG
from vqe_knapsack.experiment import ExperimentRunner


@pytest.mark.integration
class TestEndToEndFastInstance:
    """
    Runs the full VQE pipeline on the n=3 fast instance.
    Expected classical optimal: values=(3,4,5), weights=(2,3,4), W=5
        → best feasible: bits=(1,1,0) value=7, weight=5
    """

    @pytest.fixture(scope="class")
    def results(self):
        runner = ExperimentRunner(FAST_TEST_CONFIG)
        return runner.run_all_experiments()

    def test_experiments_ran_correct_count(self, results):
        expected = (
            len(FAST_TEST_CONFIG.ansatz_configs)
            * len(FAST_TEST_CONFIG.optimizer_configs)
            * len(FAST_TEST_CONFIG.a_schedule)
        )
        assert len(results) == expected

    def test_at_least_one_valid_solution_found(self, results):
        valid = [r for r in results if r.is_valid]
        assert len(valid) > 0, "VQE must find at least one valid solution"

    def test_best_valid_result_is_non_none(self, results):
        best = ExperimentRunner.best_valid_result(results)
        assert best is not None

    def test_classical_reference_matches_known_optimal(self):
        runner = ExperimentRunner(FAST_TEST_CONFIG)
        classical = runner.run_classical_reference()
        assert classical.value == 7   # 3+4=7 with weights 2+3=5 ≤ W=5
        assert classical.is_valid

    def test_best_vqe_value_positive(self, results):
        best = ExperimentRunner.best_valid_result(results)
        assert best is not None
        assert best.value > 0
