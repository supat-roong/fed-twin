.PHONY: install install-dev test test-cov coverage lint fmt type-check clean build-images \
        single-cluster-setup single-cluster-teardown multi-cluster-setup multi-cluster-teardown compile-pipeline run-pipeline \
        clean-results

# ---- Deps ----
install:
	uv sync --no-dev

install-dev:
	uv sync --extra dev

# ---- Tests ----
test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -k "not test_long_running"

test-cov coverage:
	uv run pytest --cov=src --cov-report=term-missing tests/

# ---- Code Quality ----
lint:
	uv run ruff check src/ tests/

fmt:
	uv run ruff format src/ tests/

type-check:
	uv run mypy src/

# ---- Single Cluster single_cluster ----
single-cluster-setup:
	bash setup/install_single_cluster_local.sh

single-cluster-teardown:
	bash setup/teardown_single_cluster_local.sh

# ---- multi_cluster Multi-Cluster ----
multi-cluster-setup:
	bash setup/install_multi_cluster_local.sh

multi-cluster-teardown:
	bash setup/teardown_multi_cluster_local.sh

# ---- Docker ----
build-images:
	docker build -t fed-twin-app:v1 -f docker/Dockerfile.app .

load-images:
	kind load docker-image fed-twin-app:v1 --name fed-twin-cluster

# ---- Pipeline ----
compile-pipeline:
	uv run python src/pipelines/fed_twin_single_cluster_pipeline.py

run-pipeline:
	uv run python src/automate_run.py fed_twin_single_cluster
# Usage: make run-pipeline ARGS="single" or make run-pipeline ARGS="all"

# ---- Misc tools ----
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete

clean-results:
	rm -rf metrics/
