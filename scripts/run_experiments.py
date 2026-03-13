#!/usr/bin/env python3
"""
run_experiments.py — CLI entry point for VQE Knapsack experiments.

Supports GPU + Ray parallel execution via BackendConfig in the JSON config file.

Usage:
    python scripts/run_experiments.py --config configs/n20_gpu.json   # GPU + Ray + SPSA
    python scripts/run_experiments.py --config configs/fast.json       # CPU, n=3, quick test
    python scripts/run_experiments.py                                  # default 10-item notebook config
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vqe_knapsack.config import (
    NOTEBOOK_CONFIG,
    BackendConfig,
    CPU_BACKEND,
    AnsatzConfig,
    KnapsackConfig,
    OptimizerConfig,
)
from vqe_knapsack.experiment import ExperimentResult, ExperimentRunner
from vqe_knapsack.parallel_experiment import ParallelExperimentRunner

console = Console()
logging.basicConfig(level=logging.WARNING)


def load_config_from_json(path: Path) -> tuple[KnapsackConfig, BackendConfig]:
    """Load KnapsackConfig + BackendConfig from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    ansatz_configs = tuple(
        AnsatzConfig(
            reps=a["reps"],
            rotation_blocks=tuple(a["rotation_blocks"]),
            entanglement=a["entanglement"],
            entanglement_blocks=tuple(a["entanglement_blocks"]),
        )
        for a in data["ansatz_configs"]
    )
    optimizer_configs = tuple(
        OptimizerConfig(name=o["name"], maxiter=o["options"]["maxiter"])
        for o in data["optimizer_configs"]
    )
    knapsack_cfg = KnapsackConfig(
        values=tuple(data["values"]),
        weights=tuple(data["weights"]),
        capacity=data["capacity"],
        a_schedule=tuple(data["a_schedule"]),
        ansatz_configs=ansatz_configs,
        optimizer_configs=optimizer_configs,
        output_dir=data.get("output_dir", "outputs"),
    )

    # Read optional backend section (GPU, n_workers, shots)
    backend_data = data.get("backend", {})
    backend_cfg = BackendConfig(
        device=backend_data.get("device", "CPU"),
        n_workers=backend_data.get("n_workers", 1),
        shots=backend_data.get("shots"),
        strict_gpu=backend_data.get("strict_gpu", False),
    )

    return knapsack_cfg, backend_cfg


def save_results_to_csv(results: list[ExperimentResult], output_path: Path) -> None:
    """Save experiment results to a CSV file."""
    _append_to_csv(results, output_path, append=False)
    console.print(f"[green]✅ Results saved to:[/green] {output_path}")

def append_results_to_csv(results: list[ExperimentResult], output_path: Path) -> None:
    """Appends new results to the CSV file, creating it with a header if it doesn't exist."""
    _append_to_csv(results, output_path, append=True)

def _append_to_csv(results: list[ExperimentResult], output_path: Path, append: bool) -> None:
    flat_rows = []
    for r in results:
        d = r.to_dict()
        row = {
            "reps": d.get("reps"),
            "rotation_blocks": str(d.get("rotation_blocks", [])),
            "entanglement": d.get("entanglement"),
            "optimizer": d.get("name"),
            "maxiter": d.get("options", {}).get("maxiter"),
            "A": d.get("A"),
            "bitstring": d.get("bitstring"),
            "value": d.get("value"),
            "weight": d.get("weight"),
            "probability": d.get("probability"),
            "valid": d.get("valid"),
        }
        flat_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    mode = "a" if append else "w"
    
    with open(output_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        if not append or not file_exists or output_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(flat_rows)


def print_summary(results: list[ExperimentResult], classical_value: int) -> None:
    """Print a rich summary table."""
    valid_results = [r for r in results if r.is_valid]
    best = ParallelExperimentRunner.best_valid_result(results)

    table = Table(title="VQE Experiment Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim", width=35)
    table.add_column("Value", justify="right")

    table.add_row("Total experiments", str(len(results)))
    table.add_row("Valid solutions found", str(len(valid_results)))
    table.add_row("Classical optimal (brute force)", str(classical_value))
    if best:
        table.add_row("Best VQE valid value", str(best.value))
        table.add_row("Best VQE bitstring", best.solver_result.bitstring)
        table.add_row("Best VQE probability", f"{best.solver_result.probability:.4f}")
        match = best.value == classical_value
        table.add_row(
            "Matched classical?",
            "✅ YES" if match else f"❌ NO ({best.value} vs {classical_value})",
        )
        table.add_row(
            "Best config",
            f"reps={best.ansatz_config.reps}, "
            f"{best.optimizer_config.name}(maxiter={best.optimizer_config.maxiter}), "
            f"A={best.penalty}",
        )
    else:
        table.add_row("Best VQE valid value", "None found ❌")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VQE Knapsack experiments (GPU + Ray supported).")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to JSON config. Uses 10-item notebook default if not specified.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    # Load config
    if args.config and args.config.exists():
        console.print(f"[cyan]Loading config from:[/cyan] {args.config}")
        config, backend = load_config_from_json(args.config)
    else:
        console.print("[cyan]Using default NOTEBOOK_CONFIG (10 items, 200 experiments, CPU)[/cyan]")
        config = NOTEBOOK_CONFIG
        backend = CPU_BACKEND

    total = (
        len(config.ansatz_configs)
        * len(config.optimizer_configs)
        * len(config.a_schedule)
    )

    console.print(f"[bold]Problem:[/bold] n={config.n_items} items, W={config.capacity}")
    console.print(f"[bold]Total experiments:[/bold] {total}")
    console.print(
        f"[bold]Backend:[/bold] device=[green]{backend.device}[/green], "
        f"workers=[green]{backend.n_workers}[/green], "
        f"shots={'exact' if backend.shots is None else backend.shots}"
    )

    # Classical reference
    runner = ParallelExperimentRunner(config, backend)
    classical = runner.run_classical_reference()
    console.print(
        f"[bold green]Classical optimal:[/bold green] {classical.bitstring} → value={classical.value}"
    )

    # Prepare CSV output path early for appending
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"results_{timestamp}.csv"
    console.print(f"[bold]Results will be incrementally saved to:[/bold] {output_path}")

    # Run VQE experiments with rich progress bar
    results: list[ExperimentResult] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running VQE experiments...", total=total)

        def callback(current: int, total_: int, msg: str, new_results: list[ExperimentResult] | None = None) -> None:
            progress.update(task, advance=1, description=msg[:65])
            if new_results:
                append_results_to_csv(new_results, output_path)

        runner2 = ParallelExperimentRunner(config, backend, progress_callback=callback)
        results = runner2.run_all_experiments()

    # Final Summary Display
    print_summary(results, classical.value)


if __name__ == "__main__":
    main()
