from src.parsing import parse_calling_function, parse_definition_function
from src.manager import Manager
from src.model_runner import Model

import json
import argparse

from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    """The main function that runs everything"""

    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        )

    parse.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        )

    parse.add_argument(
        "--output",
        default="data/output/function_calls.json",
        )

    args = parse.parse_args()
    manager: Manager = Manager()
    prompts_calling: List[Dict[str, Any]] = parse_calling_function(
         args.input)
    definition_fn: List[Dict[str, Any]] = parse_definition_function(
         args.functions_definition)

    if not prompts_calling or not definition_fn:
        exit(1)

    manager.definition_functions = definition_fn
    manager.prompts_calling = prompts_calling

    try:
        model: Model = Model(manager)
        data: List[Dict[str, Any]] = model.run_model()

        output_path: Path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)

    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON failed: {e}")
        exit(1)
    except Exception as e:
        print(f"[ERROR]: {e}")
        exit(1)


if __name__ == "__main__":
    main()
