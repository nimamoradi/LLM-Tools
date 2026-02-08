# tools/python_calculator_tool.py
from __future__ import annotations

import json
from typing import Any, Type

from pydantic import BaseModel, Field, ConfigDict
from langchain_core.tools import BaseTool

from src.tools.python_executor import SafePythonExecutor


class PythonCalcArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        ...,
        description=(
            """Python snippet for deterministic calculations. NO imports. 
            Available modules: math, cmath, statistics, fractions, numpy, sympy. 
            Set `result = ...` OR leave a final expression on the last line.
            
            Rules:
            * Never brute-force factorization (no loops to sqrt(N))
            * For factorization / number theory: use sympy.factorint
            * Always write result = ... (no return at top level)
            * Avoid huge loops; if you need loops, keep them small or use library functions
            * Return final outputs by assigning result = <python dict>. Do not use print for the final answer.
            """

        ),
        min_length=1,
    )
    timeout_s: float = Field(2.0, ge=0.1, le=31.0)


class PythonCalculatorTool(BaseTool):
    name: str = "python_calculator"
    description: str = (
        "Advanced calculator: executes a short Python snippet in a restricted sandbox "
        "to compute exact results (roots, formula evaluation, statistics, etc.)."
    )
    args_schema: Type[BaseModel] = PythonCalcArgs

    def __init__(self, executor: SafePythonExecutor | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._executor = executor or SafePythonExecutor()

    def _run(self, code: str, timeout_s: float = 2.0) -> str:
        payload = self._executor.execute(code, timeout_s=timeout_s)
        # Tool outputs should be easy for the LLM to parse: return JSON string
        return json.dumps(payload, ensure_ascii=False)

    async def _arun(self, code: str, timeout_s: float = 2.0) -> str:
        # Keep it simple; you can implement a true async version later.
        return self._run(code=code, timeout_s=timeout_s)
