unit_test:
	pytest

setup_test:
	pip install -e ".[dev]"

setup_normal:
	pip install -e .
