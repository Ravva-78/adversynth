.PHONY: help install test lint demo clean

help:
	@echo "AdverSynth Development Automation"
	@echo "=================================="
	@echo "make install    : Install package and dependencies"
	@echo "make test       : Run test suite via pytest"
	@echo "make lint       : Run flake8 syntax and style checks"
	@echo "make demo       : Run instant demo synthesis grid"
	@echo "make clean      : Remove cache and temporary artifacts"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest flake8 pytest-cov

test:
	pytest -v --durations=5

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=venv,.venv,data,outputs,checkpoints

demo:
	python generate_demo.py --out assets/demo_output.png

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
