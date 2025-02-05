unit_test:
	pytest -vv -s

setup_test:
	pip install -e ".[dev]"

setup_normal:
	pip install -e .
