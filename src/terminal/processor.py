import time
from typing import Tuple, Optional
from .interface import TerminalInterface, SystemResponse

class ExampleTerminalProcessor(TerminalInterface):
    def get_welcome_message(self) -> Tuple[str, Optional[str]]:
        return "Welcome! Type 'stream' for a demo or 'exit' to quit.", None

    def handle_user_input(self, user_input: str) -> SystemResponse:
        """
        Processes input and returns a SystemResponse object.
        """
        if "stream" in user_input:
            # This is a generator for streaming output
            def stream_generator():
                response = "This is a streamed response, piece by piece. "
                for char in response:
                    yield char
                    time.sleep(0.05)
            
            return SystemResponse(
                output=stream_generator(),
                tool_name="example_tool"
            )
        elif 'tool' in user_input:
            # This is a single, complete output
            return SystemResponse(
                output=f"You said: {user_input}",
                tool_name="example_tool"
            )
        else:
            return SystemResponse(
                output=f"You said: {user_input}"
            )

    def quit_task(self):
        print("\nCleanup complete. Goodbye!")
