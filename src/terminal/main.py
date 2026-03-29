if __name__ == "__main__":
    # This allows running the terminal directly for testing purposes.
    # It adds the project root to the python path to resolve imports.
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from terminal.terminal import Terminal
    from terminal.processor import ExampleTerminalProcessor

    tool_icons = {
        "example_tool": "🛠️"
    }
    processor = ExampleTerminalProcessor()
    terminal = Terminal(processor=processor, tool_icons=tool_icons)
    terminal.run()
