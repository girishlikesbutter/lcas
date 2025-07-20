# LCAS PyTorch Shadow Engine - Makefile

.PHONY: help install install-dev test lint format clean run-lightcurve validate-all security-check update-deps

# Default target
help:
	@echo "LCAS PyTorch Shadow Engine - Available commands:"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linting checks"
	@echo "  make format        - Format code with black and isort"
	@echo "  make clean         - Clean generated files and caches"
	@echo "  make run-lightcurve - Generate light curves with default settings"
	@echo "  make validate-all  - Run all validation scripts"
	@echo "  make security-check - Check for security vulnerabilities"
	@echo "  make update-deps   - Update dependencies to latest compatible versions"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Development targets
test:
	pytest tests/ -v

lint:
	flake8 src/ --max-line-length=100
	mypy src/

format:
	black src/ tests/ *.py
	isort src/ tests/ *.py

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	rm -rf lightcurve_results_pytorch/frame_*.png

# Running the application
run-lightcurve:
	python generate_lightcurves_pytorch.py --compare-shadows --points 50

validate-all: validate-orbit validate-shadows

validate-orbit:
	python validate_orbit_animation.py

validate-shadows:
	python validate_shadow_vectors_animation.py

# Security and dependency management
security-check:
	pip install safety
	safety check --json
	@echo "Checking for known PyTorch vulnerabilities..."
	@python -c "import torch; v = torch.__version__; print(f'Current PyTorch version: {v}'); import packaging.version; assert packaging.version.parse(v) >= packaging.version.parse('2.2.3'), 'WARNING: PyTorch version < 2.2.3 is vulnerable to CVE-2024-5480!'"

update-deps:
	pip install pip-tools
	pip-compile --upgrade -o requirements-updated.txt pyproject.toml
	@echo "Updated dependencies written to requirements-updated.txt"
	@echo "Review changes and update pyproject.toml accordingly"

# Docker targets (for future containerization)
docker-build:
	@echo "Docker support not yet implemented"
	# docker build -t lcas-pytorch-shadow:latest .

docker-run:
	@echo "Docker support not yet implemented"
	# docker run --gpus all -v $(PWD)/data:/app/data lcas-pytorch-shadow:latest