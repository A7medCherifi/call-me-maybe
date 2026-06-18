from pydantic import BaseModel, Field, ValidationError, model_validator
import json

class CallingFunction(BaseModel):
	prompt: str = Field(min_length=1)

	@model_validator(mode='after')
	def validation_model(self) -> 'CallingFunction':
		prompt = self.prompt.strip()
		if len(prompt) == 0:
			raise ValueError("Empty Input!")
		return self


def parse_calling_function(file_name: str):
	try:
		with open(file_name, 'r') as f:
			data2 = json.load(f)
		for e in data2:
			CallingFunction(**e)
	except FileNotFoundError:
		print("File not found stupid!")
		exit(1)
	except ValidationError:
		print("Invalid prompt!")
		exit(1)
	except Exception as e:
		print(f"Error: {e}")
		exit(1)


if __name__ == "__main__":
	parse_calling_function('function_calling_tests.json')
