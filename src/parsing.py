import json

from typing import Literal
from pydantic import BaseModel, Field, ValidationError


class CallingFunction(BaseModel):
    prompt: str = Field(min_length=1, str_strip_whitespace=True)


class ParameterType(BaseModel):
    type: str = Literal["number", "integer", "string"]


class DefinitionFunction(BaseModel):
    name: str = Field(min_length=1, str_strip_whitespace=True)
    description: str
    parameters: dict[str, ParameterType]
    returns: dict[str, str]


def parse_calling_function(file_name: str) -> list:
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
        valid = []
        for e in data:
            CallingFunction(**e)
            valid.append(e)
        return valid
    except (FileNotFoundError, json.JSONDecodeError, Exception,
            ValidationError) as e:
        print(f"Error: {e}")
        return []


def parse_definition_function(file_name: str):
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
        valid = []
        for e in data:
            DefinitionFunction(**e)
            valid.append(e)
        return valid
    except (FileNotFoundError, json.JSONDecodeError, Exception,
            ValidationError) as e:
        print(f"Error: {e}")
        return []
