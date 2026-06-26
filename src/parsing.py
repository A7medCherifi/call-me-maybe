from pydantic import BaseModel, Field, ValidationError
import json

class CallingFunction(BaseModel):
	prompt: str = Field(min_length=1, str_strip_whitespace=True)


class ParameterType(BaseModel):
	type: str


class DefinitionFunction(BaseModel):
	name: str
	description: str
	parameters: dict[str, ParameterType]
	returns: dict[str, str]


def parse_calling_function(file_name: str) -> list:
	try:
		with open(file_name, 'r') as f:
			data = json.load(f)
		valid = []
		for e in data:
			try:
				prompt = CallingFunction(**e)
				valid.append(e)
			except ValidationError:
				print("Invalid prompt!")
				continue
		return valid
	except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
		print(f"Error: {e}")
		return []

def parse_definition_function(file_name: str):
	try:
		with open(file_name, 'r') as f:
			data = json.load(f)
		valid = []
		for e in data:
			try:
				prompt = DefinitionFunction(**e)
				valid.append(e)
			except ValidationError:
				print("Invalid prompt!")
				continue
		return valid
	except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
		print(f"Error: {e}")
		return []
