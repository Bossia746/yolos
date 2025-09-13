# YOLOS Project Makefile
# Cross-platform development automation

.PHONY: help install install-dev test test-coverage lint format clean build deploy docs

# Default target
help:
	@echo "YOLOS Project - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install          Install the package and dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  install-gpu      Install GPU support dependencies"
	@echo "  install-raspberry Install Raspberry Pi dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  lint             Run code linting"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run mypy type checking"
	@echo ""
	@echo "Build and Deploy:"
	@echo "  clean            Clean build artifacts"
	@echo "  build            Build the package"
	@echo "  build-wheel      Build wheel distribution"
	@echo "  deploy           Deploy to production"
	@echo ""
	@echo "Documentation:"
	@echo "  docs             Generate documentation"
	@echo "  docs-serve       Serve documentation locally"
	@echo ""
	@echo "Utilities:"
	@echo "  check-deps       Check for dependency issues"
	@echo "  security-check   Run security vulnerability scan"
	@echo "  benchmark        Run performance benchmarks"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

install-gpu:
	pip install -e ".[gpu]"

install-raspberry:
	pip install -e ".[raspberry]"

install-all:
	pip install -e ".[all,dev]"

# Testing targets
test:
	pytest tests/ -v

test-unit:
	pytest tests/ -v -m "unit"

test-integration:
	pytest tests/ -v -m "integration"

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow"

# Code quality targets
lint:
	flake8 src tests scripts
	mypy src

format:
	black src tests scripts
	isort src tests scripts

type-check:
	mypy src

# Build targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

build: clean
	python -m build

build-wheel: clean
	python -m build --wheel

# Documentation targets
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Deployment targets
deploy-test:
	python -m twine upload --repository testpypi dist/*

deploy:
	python -m twine upload dist/*

# Utility targets
check-deps:
	pip-audit
	python scripts/simple_dependency_check.py

security-check:
	bandit -r src/
	safety check

benchmark:
	python tests/performance_test.py

# Development server targets
dev-server:
	python web/app.py

dev-api:
	uvicorn web.api:app --reload --host 0.0.0.0 --port 8000

# Model management
download-models:
	python scripts/download_models.py

train-model:
	python scripts/train_offline_models.py

# System setup for different platforms
setup-ubuntu:
	sudo apt-get update
	sudo apt-get install -y python3-opencv python3-dev build-essential
	pip install -e ".[dev]"

setup-raspberry:
	sudo apt-get update
	sudo apt-get install -y python3-opencv python3-dev
	pip install -e ".[raspberry,dev]"

setup-windows:
	pip install -e ".[dev]"

# Docker targets
docker-build:
	docker build -t yolos:latest .

docker-run:
	docker run -it --rm -p 8000:8000 yolos:latest

docker-dev:
	docker-compose -f docker-compose.dev.yml up

# Git hooks and pre-commit
pre-commit:
	pre-commit run --all-files

update-hooks:
	pre-commit autoupdate

# Version management
bump-patch:
	bump2version patch

bump-minor:
	bump2version minor

bump-major:
	bump2version major

# Environment setup
setup-env:
	python -m venv venv
	@echo "Activate virtual environment with:"
	@echo "  Windows: venv\\Scripts\\activate"
	@echo "  Unix/Mac: source venv/bin/activate"

# Quick development setup
quick-setup: setup-env install-dev
	@echo "Development environment ready!"
	@echo "Run 'make test' to verify installation"

# CI/CD helpers
ci-test: install-dev test-coverage lint

ci-build: clean build

ci-deploy: ci-test ci-build deploy