from src.parsing import parse_calling_function, parse_definition_function
from src.manager import Manager
from src.model_runner import Model

import json
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    """The main function that runs everything"""
    manager: Manager = Manager()
    prompts_calling: List[Dict[str, Any]] = parse_calling_function(
         'data/input/function_calling_tests.json')
    definition_fn: List[Dict[str, Any]] = parse_definition_function(
         'data/input/functions_definition.json')

    if not prompts_calling or not definition_fn:
        exit(1)

    manager.definition_functions = definition_fn
    manager.prompts_calling = prompts_calling

    # try:
    model: Model = Model(manager)
    data: List[Dict[str, Any]] = model.run_model()
    # except json.JSONDecodeError as e:
    #     print(f"Error, JSON failed: {e}")
    # except Exception as e:
    #     print(f"Error: {e}")
    #     exit(1)

    project_root: Path = Path(__file__).parent.parent
    output_path: Path = project_root/"data"/"output"/"function_calls.json"

    # Make sure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save it
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()


# uv run python -m src \
#     --functions_definition data/input/functions_definition.json \
#     --input data/input/function_calling_tests.json \
#     --output data/output/function_calls.json


# uv run python -m moulinette grade_student_answers
#  data/output/function_calls.json

# uv run python -m moulinette grade_student_answers
# --student_answer_path data/output/function_calls.json

# uv run python -m moulinette grade_student_answers \
#     --student_answer_path data/output/function_calls.json \
#     --set private
