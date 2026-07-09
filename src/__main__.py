from src.parsing import parse_calling_function, parse_definition_function
from src.manager import Manager
from src.test import Model

import json
from pathlib import Path
# from src.template import create_template


def main():
    manager = Manager()
    prompts_calling = parse_calling_function('data/input/function_calling_tests.json')
    definition_fn = parse_definition_function('data/input/functions_definition.json')
    if not prompts_calling or not definition_fn:
        exit(1)
    manager.definition_functions = definition_fn
    manager.prompts_calling = prompts_calling
    # create_template(manager)
    # print(manager.prompts_calling[0].prompt)
    model = Model(manager)
    data = model.run_model()

    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / "output" / "function_calls.json"
    # Make sure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save it
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
	main()