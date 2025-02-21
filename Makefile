unit_test:
	pytest -vv -s

setup_test:
	pip install -e ".[dev]"

setup:
	pip install -e .

setup_interactive:
	pip install -e ".[interactive]"

typecheck:
	mypy src/

black:
	black --line-length 80 src/.