from typing import Any, List


class Manager():
    """The manager class that has the input user data"""
    def __init__(self) -> None:
        """Initializes the Manager with necessary states."""
        self.definition_functions: List[Any] = []
        self.prompts_calling: List[Any] = []
