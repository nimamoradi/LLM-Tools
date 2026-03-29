import sys
from typing import Optional, Set
from .interface import TerminalInterface, SystemResponse

class Terminal:
    def __init__(
        self,
        processor: TerminalInterface,
        tool_icons: Optional[dict] = None,
        user_input_color: str = "\033[34m",  # Blue
        system_output_color: str = "\033[32m",  # Green
    ):
        self.processor = processor
        self.tool_icons = tool_icons or {}
        self.user_input_color = user_input_color
        self.system_output_color = system_output_color
        self.reset_color = "\033[0m"

    def _get_colored_text(self, text: str, color: str) -> str:
        return f"{color}{text}{self.reset_color}"

    def _get_icons(self, tool_names: Set[str]) -> str:
        return "".join(self.tool_icons.get(name, "") for name in tool_names)

    def _handle_system_output(self, response: SystemResponse):
        """
        Handles all presentation logic based on the SystemResponse object.
        """
        icons = self._get_icons(response.tool_names)
        
        if response.is_stream:
            full_response_text = ""
            for chunk in response.output:
                full_response_text += chunk
                display_text = f"\r{icons} {full_response_text}"
                sys.stdout.write(self._get_colored_text(display_text, self.system_output_color))
                sys.stdout.flush()
            print()
        else:
            display_text = f"{icons} {response.output}"
            print(self._get_colored_text(display_text, self.system_output_color))

    def get_user_input(self) -> str:
        base_prompt = self._get_colored_text("You: ", self.user_input_color)
        line = input(base_prompt)

        if line.strip() == '"""':
            multiline_prompt = self._get_colored_text("... ", self.user_input_color)
            lines = []
            while True:
                try:
                    next_line = input(multiline_prompt)
                    if next_line.strip() == '"""':
                        break
                    lines.append(next_line)
                except (EOFError, KeyboardInterrupt):
                    print("\nMulti-line input cancelled.")
                    return ""
            return "\n".join(lines)
        else:
            return line

    def run(self):
        welcome_msg, tool = self.processor.get_welcome_message()
        info_msg = 'INFO: Type `"""` and press Enter for multi-line mode.'
        welcome_icons = self._get_icons({tool}) if tool else ""
        print(self._get_colored_text(f"{welcome_icons} {welcome_msg}", self.system_output_color))
        print(self._get_colored_text(info_msg, self.user_input_color))

        while True:
            user_input = self.get_user_input()

            if user_input.strip().lower() == "exit":
                self.processor.quit_task()
                break
            
            if not user_input.strip():
                continue

            system_response = self.processor.handle_user_input(user_input)
            self._handle_system_output(system_response)
