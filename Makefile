.PHONY: install install-dev test test-cov coverage lint fmt type-check clean build-images \
        k8s-setup k8s-teardown karmada-setup karmada-teardown compile-pipeline run-pipeline \
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

# ---- Single Cluster k8s ----
k8s-setup:
	bash setup/install_k8s_local.sh

k8s-teardown:
	bash setup/teardown_k8s_local.sh

# ---- Karmada Multi-Cluster ----
karmada-setup:
	bash setup/install_karmada_local.sh

karmada-teardown:
	bash setup/teardown_karmada_local.sh

# ---- Docker ----
build-images:
	docker build -t fed-twin-app:v1 -f docker/Dockerfile.app .

load-images:
	kind load docker-image fed-twin-app:v1 --name fed-twin-cluster

# ---- Pipeline ----
compile-pipeline:
	uv run python src/pipelines/fl_k8s_pipeline.py

run-pipeline:
	uv run python src/automate_run.py fl_k8s
# Usage: make run-pipeline ARGS="single" or make run-pipeline ARGS="all"

# ---- Misc tools ----
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete

clean-results:
	rm -rf metrics/
