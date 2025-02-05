unit_test:
	pytest -v

setup_test:
	pip install -e ".[dev]"

setup_normal:
	pip install -e .
