from src.parsing import parse_calling_function, parse_definition_function
from src.manager import Manager
from test import test_model
# from src.template import create_template


def main():
	manager = Manager()
	prompts_calling = parse_calling_function('function_calling_tests.json')
	definition_fn = parse_definition_function('functions_definition.json')
	if not prompts_calling or not definition_fn:
		exit(1)
	manager.definition_functions = definition_fn
	manager.prompts_calling = prompts_calling
	# create_template(manager)
	# print(manager.prompts_calling[0].prompt)
	test_model(manager, "Reverse the string 'hello'")


if __name__ == "__main__":
	main()
