import json
from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field, ValidationError, model_validator


class CallingFunction(BaseModel):
    """Pydantic class to pars Calling function"""
    prompt: str = Field(min_length=1)

    @model_validator(mode='after')
    def strip_whitespaces(self) -> 'CallingFunction':
        if isinstance(self.prompt, str):
            value = self.prompt.strip()
            if not value:
                raise ValueError("Invalid input")
        return self


class ParameterType(BaseModel):
    """Pydantic class to pars the parameter type"""
    type: Literal["number", "integer", "string", "boolean"]


class DefinitionFunction(BaseModel):
    """Pydantic class to pars Definition function"""
    name: str = Field(min_length=1)
    description: str
    parameters: Dict[str, ParameterType]
    returns: Dict[str, str]

    @model_validator(mode='after')
    def strip_whitespaces(self) -> 'DefinitionFunction':
        if isinstance(self.name, str):
            value = self.name.strip()
            if not value:
                raise ValueError("Invalid input")
        return self


def parse_calling_function(file_name: str) -> List[Dict[str, Any]]:
    """
    parse the function calling file that contains prompts

    return:
        list(): list of valid dictionaries
    """
    try:
        with open(file_name, 'r') as f:
            data: List[Dict[str, Any]] = json.load(f)
        valid: List[Dict[str, Any]] = []
        for e in data:
            CallingFunction(**e)
            valid.append(e)
        return valid
    except ValidationError as e:
        print(f"[ERROR]: {e.errors()[0].get('msg')}")
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"Error: {e}")
    return []


def parse_definition_function(file_name: str) -> List[Dict[str, Any]]:
    """
    parse the functions definition file that contains
    functions definition json

    return:
        list(): list of valid dictionaries
    """
    try:
        with open(file_name, 'r') as f:
            data: List[Dict[str, Any]] = json.load(f)
        valid: List[Dict[str, Any]] = []
        for e in data:
            DefinitionFunction(**e)
            valid.append(e)
        return valid
    except ValidationError as e:
        print(f"[ERROR]: {e.errors()[0].get('msg')}")
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"[ERROR]: {e}")
    return []
