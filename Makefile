unit_test:
	pytest -vv -s

setup_test:
	pip install -e ".[dev]"

setup_normal:
	pip install -e .

typecheck:
	mypy src/

autoformat:
	black --line-length 80 src/.