from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Any

Provider = Literal["ollama", "openai", "gemini"]


@dataclass(frozen=True)
class ModelConfig:
    provider: Provider
    model: str
    temperature: float = 0.0

    # Optional knobs
    base_url: Optional[str] = None   # useful for OpenAI-compatible servers
    api_key: Optional[str] = None    # if you prefer passing keys directly


def build_langchain_model(cfg: ModelConfig) -> Any:
    if cfg.provider == "ollama":
        # pip install -U langchain-ollama :contentReference[oaicite:8]{index=8}
        from langchain_ollama import ChatOllama
        return ChatOllama(model=cfg.model, temperature=cfg.temperature)

    if cfg.provider == "openai":
        # pip install -U langchain-openai
        # uses OPENAI_API_KEY env var by default :contentReference[oaicite:9]{index=9}
        from langchain_openai import ChatOpenAI
        kwargs = {"model": cfg.model, "temperature": cfg.temperature}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        return ChatOpenAI(**kwargs)

    if cfg.provider == "gemini":
        # pip install -U langchain-google-genai :contentReference[oaicite:10]{index=10}
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = {"model": cfg.model, "temperature": cfg.temperature}
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    raise ValueError(f"Unknown provider: {cfg.provider}")
