import os
from dotenv import load_dotenv
from typing import cast, Tuple, Optional, Set

from agent.runner import ToolCallingAgent, AgentSettings
from providers.factory import build_langchain_model, ModelConfig, Provider
from terminal import TerminalInterface
from terminal.interface import SystemResponse
from tools.python_calculator_tool import PythonCalculatorTool
from tools.weather_tool import WeatherTool


class LLMTerminalProcessor(TerminalInterface):
    def __init__(self):
        load_dotenv()
        provider = cast(Provider, os.getenv("PROVIDER", "ollama").lower())
        cfg = ModelConfig(
            provider=provider,
            model=os.getenv("MODEL_NAME", "llama3.1:8b"),
            temperature=0.1,
            api_key=os.getenv("API_KEY"),
        )
        model = build_langchain_model(cfg)
        tools = [
            PythonCalculatorTool(),
            WeatherTool(),
        ]
        self.agent = ToolCallingAgent(
            model=model,
            tools=tools,
            settings=AgentSettings(max_steps=15),
        )

    def get_welcome_message(self) -> Tuple[str, Optional[str]]:
        return "Welcome to the LLM agent. How can I help you?", None

    def handle_user_input(self, user_input: str) -> SystemResponse:
        system_prompt = "You are a helpful assistant. You can use the available tools to answer the user's questions if needed."
        events = self.agent.run(user_input, system_prompt)

        final_output = ""
        used_tools: Set[str] = set()

        for event in events:
            if "tool_name" in event:
                used_tools.add(event["tool_name"])
            elif "output" in event:
                final_output = event["output"]

        return SystemResponse(output=final_output, tool_names=used_tools)

    def quit_task(self):
        print("\nGoodbye!")
