# Minimal Makefile for S5E7 Personality Prediction
.PHONY: install test clean help quick-test personality-prediction time-stats time-list time-demo

# Core installation
install:
	pip install -e .

dev-install:
	pip install -e .[dev,optimization,visualization]

# Setup directories
setup:
	mkdir -p data/raw data/processed outputs submissions logs
	touch data/raw/.gitkeep data/processed/.gitkeep outputs/.gitkeep submissions/.gitkeep logs/.gitkeep

# Quick test (single model)
quick-test:
	@echo "Running quick test workflow..."
	python3 scripts/quick_test_workflow.py \
		--target-col Personality \
		--problem-type classification \
		--missing-strategy model_specific \
		--output-dir outputs/quick_test

# Full personality prediction workflow
personality-prediction:
	python3 scripts/kaggle_workflow.py \
		--target-col Personality \
		--problem-type classification \
		--output-dir outputs \
		--optimize \
		--ensemble

# Code quality - unified with pre-commit hooks
lint:
	black --check src/ tests/ scripts/
	flake8 src/ tests/ scripts/
	mypy src/ tests/ scripts/

# Code formatting - apply black to all files
format:
	black src/ tests/ scripts/

# Auto-fix lint issues
lint-fix: format
	@echo "Code formatted with black"
	@echo "Note: Some lint issues may require manual fixes"

# Basic test
test:
	PYTHONPATH=. pytest tests/ -v

# Time tracking commands
time-stats:
	python3 scripts/time_tracker_cli.py --stats

time-list:
	python3 scripts/time_tracker_cli.py --list

time-demo:
	python3 scripts/demo_time_tracker.py

# Clean outputs
clean:
	rm -rf outputs/* submissions/* logs/*
	rm -rf **__pycache__** .pytest_cache .coverage htmlcov
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Help
help:
	@echo "S5E7 Personality Prediction - Minimal Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install              - Install dependencies"
	@echo "  make setup               - Create directories"
	@echo ""
	@echo "Run:"
	@echo "  make quick-test          - Quick single model test"
	@echo "  make personality-prediction - Full workflow with optimization"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint                - Check code quality (black, flake8, mypy)"
	@echo "  make format              - Format code with black"
	@echo "  make lint-fix            - Format and show results"
	@echo ""
	@echo "Time Tracking:"
	@echo "  make time-stats          - Show workflow time statistics"
	@echo "  make time-list           - List tracked workflows"
	@echo "  make time-demo           - Run time tracking demo"
	@echo ""
	@echo "Maintenance:"
	@echo "  make test                - Run tests"
	@echo "  make clean               - Clean outputs"

# Default
.DEFAULT_GOAL := help
