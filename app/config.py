"""Configuration loader and LLM provider specifications.

Supports lazy configuration loading so the web server can start without an API key
and fail with a clear, informative error message on first invocation.
"""

from dataclasses import dataclass
import os
from typing import Dict
from dotenv import load_dotenv

# Load variables from .env if present
load_dotenv()


class ConfigError(Exception):
    """Raised when environment configuration or API keys are missing/invalid."""
    pass


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    default_model: str
    api_key_env: str


PROVIDERS: Dict[str, ProviderConfig] = {
    "groq": ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        default_model="openai/gpt-oss-120b",
        api_key_env="GROQ_API_KEY",
    ),
    "gemini": ProviderConfig(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-2.5-flash",
        api_key_env="GEMINI_API_KEY",
    ),
}


@dataclass
class Settings:
    """Application runtime settings."""
    provider: str
    base_url: str
    model: str
    api_key: str
    batch_size: int = 5
    cache_path: str = "cache.db"
    temperature: float = 0.2
    max_retries: int = 5


def get_settings() -> Settings:
    """Load settings from environment variables lazily.

    Raises ConfigError with guidance if the selected provider's key is missing.
    """
    provider_name = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    if provider_name not in PROVIDERS:
        allowed = ", ".join(PROVIDERS.keys())
        raise ConfigError(
            f"Unsupported LLM_PROVIDER '{provider_name}'. Must be one of: {allowed}."
        )

    provider_info = PROVIDERS[provider_name]
    api_key = os.getenv(provider_info.api_key_env, "").strip()
    if not api_key:
        raise ConfigError(
            f"Missing API key for provider '{provider_name}'. "
            f"Please set {provider_info.api_key_env} in your .env file (see .env.example for details)."
        )

    model = os.getenv("LLM_MODEL", "").strip() or provider_info.default_model

    raw_batch = os.getenv("BATCH_SIZE", "5").strip()
    try:
        batch_size = int(raw_batch)
        if batch_size < 1:
            batch_size = 5
    except ValueError:
        batch_size = 5

    cache_path = os.getenv("CACHE_PATH", "cache.db").strip() or "cache.db"

    return Settings(
        provider=provider_name,
        base_url=provider_info.base_url,
        model=model,
        api_key=api_key,
        batch_size=batch_size,
        cache_path=cache_path,
        temperature=0.2,
        max_retries=5,
    )
