from abc import ABC, abstractmethod
from typing import Union, Generator, Optional, Tuple, Set
from dataclasses import dataclass, field

@dataclass
class SystemResponse:
    """
    A structured data class for passing information from the input
    processor to the output renderer.
    """
    output: Union[Generator[str, None, None], str]
    tool_names: Set[str] = field(default_factory=set)

    @property
    def is_stream(self) -> bool:
        """Flag indicating if the output is a stream."""
        return isinstance(self.output, Generator)

    @property
    def is_tool_call(self) -> bool:
        """Flag indicating if a tool is associated with the output."""
        return bool(self.tool_names)


class TerminalInterface(ABC):
    @abstractmethod
    def get_welcome_message(self) -> Tuple[str, Optional[str]]:
        """
        Returns the welcome message and an optional tool name.
        """
        pass

    @abstractmethod
    def handle_user_input(self, user_input: str) -> SystemResponse:
        """
        Processes user input and returns a structured SystemResponse object.
        This method should contain business logic, not presentation logic.
        """
        pass

    @abstractmethod
    def quit_task(self):
        """
        Handles any cleanup required when quitting.
        """
        pass
