# 🎒 VQE_Knapsack — Hybrid Quantum-Classical Framework for Combinatorial Optimization

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Qiskit](https://img.shields.io/badge/Qiskit-2.3.0-purple?style=for-the-badge&logo=qiskit)
![Ray](https://img.shields.io/badge/Ray-Parallel-red?style=for-the-badge&logo=ray)
![Scipy](https://img.shields.io/badge/SciPy-Optimization-orange?style=for-the-badge&logo=scipy)
![FastMCP](https://img.shields.io/badge/FastMCP-Server-green?style=for-the-badge&logo=fastapi)
![NumPy](https://img.shields.io/badge/NumPy-Array-lightblue?style=for-the-badge&logo=numpy)

> [!WARNING]
> **Work In Progress (WIP)**: This project is under active architectural and scientific development. Quantum modules, penalty annealing schedules, and distributed execution mechanisms via Ray are continuously being refined. Experimental results should be interpreted within the context of controlled combinatorial validation.

---

## 📘 General Description

**VQE Knapsack** is a rigorously structured hybrid quantum-classical platform designed for finding heuristic solutions to the $NP$-Hard 0/1 Knapsack problem. 

Replacing typical Jupyter Notebook fragility with robust **Specification Driven Development (SDD)**, this project integrates:

* Classical combinatorial baselines (O(2^n) Brute Force)
* Variational Quantum Eigensolver (VQE) architecture
* Penalty-based Ising Hamiltonian modeling
* Distributed execution under Model Context Protocol (MCP) and Ray
* Formal governance through Clean Code and Test-Driven Development (TDD)

The core objective is to investigate the feasibility, robustness, and ground-state convergence of quantum optimization methods within constrained decision problems, specifically tracking the effects of dynamic penalty annealing.

This is not a mere script repository — it is a **scientifically governed experimental framework**.

---

## 🧩 Hybrid Architecture

```text
┌────────────────────────────────┐
│     Classical Preprocessing     │
│  (Config parsing + Brute Force) │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│    Quantum Module Layer         │
│(Hamiltonian + Ansatz + Solver)  │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│   Statistical Evaluation Layer  │
│ (Penalty Annealing + CSV Dump)  │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│      MCP Server (FastMCP)       │
│ Persistent Registry & Audit Log │
└────────────────────────────────┘
```

---

## 🧠 Quantum Modules

### 1️⃣ Ising Hamiltonian Formulation

Implements the transformation from the QUBO formulation of the Knapsack problem to a Pauli Z-basis observable.

Used for:
* Encoding the objective function: $H = -\sum v_i x_i + A \cdot (\sum w_i x_i - W)^2$
* Guaranteeing invalid solutions are driven out of the ground state via dynamic penalty factor $A$.

---

### 2️⃣ Ansatz Factory (Hardware Efficient)

Parameterised quantum circuits trained via classical optimization.

* Ansatz configurations: Dynamic depth (`reps`: 1 to 6)
* Rotation blocks: `RY`, `RZ`
* Entanglement: Linear or Full connectivity (`CX` blocks)

Paradigm: Exploring the Hilbert space to locate the eigenstate that minimizes the problem Hamiltonian.

---

### 3️⃣ VQE Solver & Optimizer Pipeline

Implements the Variational Quantum Eigensolver loop, abstracting the Qiskit Primitives.

* Estimators: `StatevectorEstimator` (exact) or `AerEstimator` (GPU accelerated - under evaluation)
* Optimizers: COBYLA, SPSA (max iterations from 100 to 2000)
* Execution: Utilizes parameter warm-starting to carry over optimized parameters across increasing penalty steps ($A$-schedule).

Classification rule:
$$ \text{Solution}(\vec{x}) = \arg\min_{\vec{\theta}} \langle \psi(\vec{\theta}) | H(A) | \psi(\vec{\theta}) \rangle $$

This modular separation preserves full architectural stability, allowing seamless swapping of optimization backends (CPU vs. GPU via Ray).

---

# 🔬 Experimental Results — Phase II (n=20 Scaling Update)

Recent experiments expanded the problem dimensionality from $n=10$ to $n=20$. Due to the exponential scaling of the Hilbert space ($2^{20} = 1,048,576$ states), the workload had to be split into manageable blocks (e.g., `configs/n20_gpu_block1.json`).

While the initial intent was to leverage GPU acceleration via `AerEstimator`, **multiple technical issues and crashes occurred during the GPU initialization attempts**. Consequently, the execution strategy was pivoted backward to guarantee stability.

The final, successful execution for Block 1 was run purely on **CPU using Ray with 16 parallel workers**, bypassing the unstable GPU state.

Execution Command:
```bash
python scripts/run_experiments.py --config configs/n20_gpu_block1.json
```
*(Note: Despite the `gpu` nomenclature in the filename, the runtime environment enforced CPU execution to bypass CUDA/Aer allocation errors).*

### 🏷 Status
`EXECUTED_WITH_HARDWARE_LIMITATIONS`

### 📊 Performance Metrics (Block 1)

| Metric | Details | Verdict |
|---|---|---|
| Parallel Backend | **CPU-only via 16 Ray Workers** (GPU failed) | 🟡 DEGRADED |
| Ansatz Depth | `reps=1` (Linear entanglement) | — |
| Optimizer | `SPSA` ($N_{iter} \in \{200, 500, 1000, 2000\}$) | — |
| Hilbert Space | $2^{20} = 1,048,576$ states | 🔵 BASELINE |

---

## 📈 Scientific Interpretation (Phase II Initial Assessment)

The transition to $n=20$ revealed significant hardware and algorithmic bottlenecks:

* **GPU Allocation Failures**: The attempt to simulate 20 qubits densely on GPU memory failed, demonstrating that off-the-shelf `AerEstimator` requires careful precision tuning (e.g., single-precision or chunking) to prevent out-of-memory or driver crashes at this scale. 
* **CPU Scalability**: Falling back to 16 CPU workers allowed the experiment to complete, but the wall-clock time required for $2^{20}$ statevector manipulations per optimizer step is unsustainable for deeper circuits.
* **Expressivity Constraint**: Block 1 restricted the ansatz to `reps=1`. SPSA optimizer struggled to find the absolute global optimum within this highly restricted parameter space, as a single layer of linear entanglement lacks the capacity to express the complex correlation between 20 items.

### Conclusion
The architecture robustly handles distributed scaling (Ray framework), but simulating density at $n=20$ strictly on CPU is computationally expensive. The immediate technical priority must be stabilizing the GPU execution pathway before progressing to deeper ansatz blocks (`reps` 2–5).

---

## 🧪 Scientific Implications

This scaled result reflects the harsh reality of quantum simulation:
* Naive scaling of statevector dimension ($2^n$) quickly exhausts both GPU VRAM and CPU time.
* The SDD protocol prevented catastrophic failure by allowing a graceful fallback to a known-valid CPU paradigm, ensuring that at least Block 1 generated data.
* Variational algorithms at this scale require either specialized tensor-network simulators, aggressive noise mitigation on real hardware, or problem-specific ansatzes like QAOA.

VQE Knapsack functions precisely as intended: an observable, scientifically honest testbed that highlights hardware limits over theoretical hype.

---

## 🏗 MCP — Model Context Protocol

The system operates under distributed interaction capabilities:
* **FastMCP Integration**: Exposes tools for LLMs to query experiment configurations and read top outputs.
* **Deterministic Tracking**: CSV incremental persistence (`results_[timestamp].csv`).
* **Environment Configuration**: Robust state management via `.env` and `configs/`.

Server execution:
```bash
make mcp
```

Guarantees:
| Property | Guarantee |
|---|---|
| Persistence | Atomic CSV appends |
| Determinism | Typed dataclass enforcement |
| Isolation | Pure Hamiltonian functions |
| Reproducibility| Factory-driven seeding |

---

## 🧪 Validation Philosophy

VQE Knapsack does **not** assume quantum supremacy for $NP$-Hard combinatorial problems at small scale. 
Instead, it enforces:
* Exact classical benchmarking (Brute-force baseline).
* Test-Driven tracking of ground state extraction (`test_e2e.py`).
* Incremental logging of valid/invalid physical states.
* Registry-backed results suitable for future statistical tests.

---

## 🧩 Folder Structure

```text
vqe_knapsack_local/
├── src/vqe_knapsack/
│   ├── config.py
│   ├── hamiltonian.py
│   ├── classical.py
│   ├── ansatz_factory.py
│   ├── solver.py
│   ├── experiment.py
│   └── parallel_experiment.py
├── configs/
│   ├── default.json (n=10)
│   ├── fast.json (n=3)
│   └── n20_gpu_block*.json (n=20 splits)
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
│   └── run_experiments.py
├── mcp_server/
│   └── server.py
└── Makefile
```

---

## 🚀 Current Status

```text
ARCHITECTURE: Stable (Clean Code + SOLID + SDD)
MCP: Operational (FastMCP)
UNIT TESTS: 100% Passing
HARDWARE BACKEND: CPU 16-Workers (GPU pathway unstable requires fix)
EXPERIMENTAL MATURITY: Scaling Analysis (n=20, Block 1 Executed on CPU)
```

System state:
**SCIENTIFICALLY VALIDATED — INVESTIGATING HARDWARE BOTTLENECKS**

---

## 🔮 Next Research Directions

* Debug and stabilize the CUDA/AerEstimator pipeline for $n=20$ memory limits.
* Execute subsequent Blocks (`reps` 2 through 5) once parallel hardware is stabilized.
* Integrate noise models and error mitigation (ZNE).
* Expand to runtime execution on actual Qiskit hardware (IBM Quantum).

---

> 💬 *VQE Knapsack is built to bridge classical software engineering rigor with variational quantum heuristics, ensuring that every optimization step is tracked, reproducible, and ready for true quantum hardware.*
— Leonardo Maximino Bernardo, 2026
