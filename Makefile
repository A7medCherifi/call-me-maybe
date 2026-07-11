SRC = SRC

install:
    uv sync

run:
    uv run python3 $(SRC)

debug:
	uv run python -m pdb -m $(SRC)

clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache

lint:
	uv run flake8 $(SRC)
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs $(SRC)
