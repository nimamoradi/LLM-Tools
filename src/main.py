from __future__ import annotations

from agent.llm_processor import LLMTerminalProcessor
from terminal.terminal import Terminal


def main():
    tool_icons = {
        "python_calculator": "🛠️",
        "get_weather": "🌦️",
    }
    processor = LLMTerminalProcessor()
    terminal = Terminal(processor=processor, tool_icons=tool_icons)
    terminal.run()


if __name__ == "__main__":
    main()
