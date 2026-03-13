.PHONY: install test lint run report notebook clean mcp

# Use the pre-existing qiskit-env which has qiskit==2.3.0 + jupyter pre-installed
PYTHON    := python3
QENV      := /home/leonardomaximinobernardo/qiskit-env
QPYTHON   := $(QENV)/bin/python
QPIP      := $(QENV)/bin/pip
PYTEST    := $(QENV)/bin/pytest
RUFF      := $(QENV)/bin/ruff
SRC       := $(shell pwd)/src
TESTS     := $(shell pwd)/tests

# PYTHONPATH ensures our src/ is importable without pip install -e
export PYTHONPATH := $(SRC)

## ── Environment ─────────────────────────────────────────────────────────────
install:
	$(QPIP) install pytest pytest-cov pytest-mock ruff -q
	@echo "✅ Tools installed into qiskit-env. Ready to use."

## ── Testing (TDD) ───────────────────────────────────────────────────────────
test:
	PYTHONPATH=$(SRC) $(PYTEST) $(TESTS)/unit/ -v --tb=short -m "not aer"

test-aer:
	PYTHONPATH=$(SRC) $(PYTEST) $(TESTS)/unit/ -v --tb=short -m "aer"

test-all:
	PYTHONPATH=$(SRC) $(PYTEST) $(TESTS)/ --ignore=$(TESTS)/integration/ -v --tb=short -m "not aer"

test-integration:
	PYTHONPATH=$(SRC) $(PYTEST) $(TESTS)/integration/ -v -s

test-unit:
	PYTHONPATH=$(SRC) $(PYTEST) $(TESTS)/unit/ -v

## ── Code Quality ────────────────────────────────────────────────────────────
lint:
	$(RUFF) check src/ tests/ scripts/
	$(RUFF) format --check src/ tests/ scripts/

format:
	$(RUFF) format src/ tests/ scripts/
	$(RUFF) check --fix src/ tests/ scripts/

## ── Run Experiments ─────────────────────────────────────────────────────────
run:
	PYTHONPATH=$(SRC) $(QPYTHON) scripts/run_experiments.py --config configs/default.json

run-fast:
	PYTHONPATH=$(SRC) $(QPYTHON) scripts/run_experiments.py --config configs/fast.json

## ── Report ──────────────────────────────────────────────────────────────────
report:
	PYTHONPATH=$(SRC) $(QPYTHON) scripts/generate_report.py

## ── Notebook ────────────────────────────────────────────────────────────────
notebook:
	PYTHONPATH=$(SRC) $(QENV)/bin/jupyter lab notebooks/analysis.ipynb

## ── MCP Server ──────────────────────────────────────────────────────────────
mcp:
	PYTHONPATH=$(SRC) $(QPYTHON) mcp_server/server.py

## ── Cleanup ─────────────────────────────────────────────────────────────────
clean:
	rm -rf outputs/*.csv outputs/*.json outputs/*.md
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean
	rm -rf .mypy_cache .ruff_cache .pytest_cache
