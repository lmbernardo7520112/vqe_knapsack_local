"""
MCP Server for VQE Knapsack results.

Exposes VQE experiment results as Model Context Protocol resources,
allowing AI assistants to query them directly.

Usage:
    make mcp                         # runs the server
    mcp dev mcp_server/server.py     # development mode with MCP inspector
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ── Server Definition ─────────────────────────────────────────────────────────
mcp = FastMCP("vqe-knapsack")

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))


def _load_latest_results() -> list[dict]:
    """Load the most recent CSV results file."""
    import csv
    csvs = sorted(OUTPUT_DIR.glob("results_*.csv"), reverse=True)
    if not csvs:
        return []
    with open(csvs[0]) as f:
        return list(csv.DictReader(f))


# ── MCP Resources ─────────────────────────────────────────────────────────────

@mcp.resource("config://notebook")
def get_notebook_config() -> str:
    """Returns the KnapsackConfig used in the original notebook (JSON)."""
    from vqe_knapsack.config import NOTEBOOK_CONFIG
    return json.dumps(NOTEBOOK_CONFIG.to_dict(), indent=2)


@mcp.resource("results://latest")
def get_latest_results() -> str:
    """Returns all experiment results from the latest run (JSON array)."""
    rows = _load_latest_results()
    if not rows:
        return json.dumps({"error": "No results found. Run 'make run' first."})
    return json.dumps(rows, indent=2)


@mcp.resource("results://best")
def get_best_result() -> str:
    """Returns the best valid VQE result from the latest run (JSON)."""
    rows = _load_latest_results()
    valid = [r for r in rows if r.get("valid", "").lower() == "true"]
    if not valid:
        return json.dumps({"error": "No valid results found."})
    best = max(valid, key=lambda r: (int(r.get("value", 0)), float(r.get("probability", 0))))
    return json.dumps(best, indent=2)


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
def list_result_files() -> str:
    """List all available result CSV files in the output directory."""
    csvs = sorted(OUTPUT_DIR.glob("results_*.csv"), reverse=True)
    files = [{"filename": f.name, "size_kb": round(f.stat().st_size / 1024, 1)} for f in csvs]
    return json.dumps(files, indent=2)


@mcp.tool()
def get_classical_optimal() -> str:
    """Run the brute-force classical solver and return the optimal solution."""
    from vqe_knapsack.classical import brute_force_solver
    from vqe_knapsack.config import NOTEBOOK_CONFIG
    result = brute_force_solver(
        NOTEBOOK_CONFIG.values,
        NOTEBOOK_CONFIG.weights,
        NOTEBOOK_CONFIG.capacity,
    )
    return json.dumps({
        "bitstring": list(result.bitstring),
        "value": result.value,
        "weight": result.weight,
        "capacity": result.capacity,
        "is_valid": result.is_valid,
    }, indent=2)


@mcp.tool()
def summarize_results_by_ansatz() -> str:
    """Summarize best valid result per ansatz (reps) from the latest run."""
    rows = _load_latest_results()
    valid = [r for r in rows if r.get("valid", "").lower() == "true"]
    if not valid:
        return json.dumps({"error": "No valid results found."})

    by_reps: dict[str, dict] = {}
    for r in valid:
        reps = str(r.get("reps", "?"))
        value = int(r.get("value", 0))
        prob = float(r.get("probability", 0))
        if reps not in by_reps or (value, prob) > (by_reps[reps]["value"], by_reps[reps]["probability"]):
            by_reps[reps] = {
                "reps": int(reps),
                "best_value": value,
                "probability": prob,
                "bitstring": r.get("bitstring"),
                "A": r.get("A"),
                "maxiter": r.get("maxiter"),
            }

    return json.dumps(list(by_reps.values()), indent=2)


if __name__ == "__main__":
    mcp.run()
