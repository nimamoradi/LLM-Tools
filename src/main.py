from __future__ import annotations

import os
from dotenv import load_dotenv
from typing import cast

from agent.runner import ToolCallingAgent, AgentSettings
from providers.factory import build_langchain_model, ModelConfig, Provider

from tools.python_calculator_tool import PythonCalculatorTool
from tools.weather_tool import WeatherTool


def main():
    load_dotenv()
    # Swap provider here:
    provider = cast(Provider, os.getenv("PROVIDER", "ollama").lower())
    cfg = ModelConfig(
        provider=provider,
        model=os.getenv("MODEL_NAME", "llama3.1:8b"),
        temperature=0.1,
        api_key=os.getenv("API_KEY")
    )
    # cfg = ModelConfig(provider="openai", model="gpt-5-mini", temperature=0.0, api_key="")
    # cfg = ModelConfig(provider="gemini", model="gemini-2.5-flash", temperature=0.2, api_key="")

    model = build_langchain_model(cfg)
    tools = [
        PythonCalculatorTool(),
        WeatherTool(),
    ]

    agent = ToolCallingAgent(
        model=model,
        tools=tools,
        settings=AgentSettings(max_steps=10),
    )
    #
    prompt = "compare the temperature for Montreal, New York and London on 8 feb 2026 and see which is the coldest"
    out = agent.run(
        prompt,
        system_prompt="",
    )

    print(prompt + "\n\n" + out)
    prompt = """
A ball with weight of 0.352 kg was let go from root of 3 story building, what is the speed of ball reaching the ground, each 
        floor is 2.54 metter tall and g=9.8 m/s^2.
    """
    out = agent.run(
        prompt,
        system_prompt="You are a careful assistant you can tools provided to you",
    )
    print(prompt + "\n\n" + out)


if __name__ == "__main__":
    main()
