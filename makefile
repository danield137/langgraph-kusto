PYTHON := .venv/bin/python

.PHONY: help test test-live examples lint

help:
	@echo "Available targets:"
	@echo "  test         - run unit tests (pytest if available)"
	@echo "  test-live    - run live sanity test against Kusto"
	@echo "  examples     - run run_memory_demo.py"
	@echo "  fmt          - run black and isort formatting"

test:
	@echo "Running tests..."
	@if command -v pytest >/dev/null 2>&1; then \
		pytest; \
	else \
		$(PYTHON) -m pytest || echo "pytest not installed in this environment"; \
	fi

test-live:
	@echo "Running live Kusto sanity test..."
	$(PYTHON) -m tests.live.sanity

examples:
	@echo "Running memory demo..."
	$(PYTHON) run_memory_demo.py

fmt:
	@echo "Running black..."
	$(PYTHON) -m black --line-length=120 langgraph_kusto examples tests run_memory_demo.py
	@echo "Running isort..."
	$(PYTHON) -m isort langgraph_kusto examples tests run_memory_demo.py