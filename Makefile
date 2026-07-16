SRC = src

install:
	uv sync

run:
	uv run python3 -m $(SRC)

debug:
	uv run python -m pdb -m $(SRC)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	uv run flake8 $(SRC)
	MYPYPATH=. uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs $(SRC)

moulinette_public:
	uv run -m moulinette prepare_exercises --set public

moulinette_private:
	uv run -m moulinette prepare_exercises --set private

moulinette_valid_answers:
	uv run -m moulinette grade_student_answers data/output/function_calls.json --set public