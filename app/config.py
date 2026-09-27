"""
config.py
=========

Centralized, typed application configuration loaded from environment
variables. See Week 1's config.py for the full rationale (fail-fast
validation, no secrets in source, one place to reconfigure).

New this week: Redis connection settings for BOTH the checkpointer
(short-term/session memory) and the store (long-term/cross-thread
memory), plus the agent loop's safety-valve settings (see
week-4-notes.md, section 4).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Strongly-typed application settings.

    Attributes:
        openrouter_api_key: API key for OpenRouter.
        chat_model: OpenRouter model slug used for the agent's LLM calls.
            Must be a model that supports tool calling.
        redis_url: Connection string for Redis, used for BOTH the
            LangGraph checkpointer (short-term memory) and store
            (long-term memory) — see app/__init__.py for the
            distinction. Requires Redis 8.0+, or Redis Stack (bundling
            the RedisJSON and RediSearch modules) for older Redis
            versions — see README.md.
        max_iterations: Hard cap on agent loop iterations before the
            force_stop safety valve triggers (week-4-notes.md, section 4).
        request_timeout_seconds: Hard timeout for a single LLM call.
        recursion_limit: LangGraph's own graph-level safety valve,
            passed via invoke config — see week-4-notes.md, section 10.
            Set comfortably above max_iterations * 2 (each loop
            iteration is two graph steps: call_model, execute_tools) so
            our own max_iterations check is what triggers first, with
            LangGraph's limit as a final backstop.
    """

    openrouter_api_key: str
    chat_model: str = "anthropic/claude-sonnet-4.5"

    redis_url: str = "redis://localhost:6379"

    max_iterations: int = 8
    request_timeout_seconds: float = 60.0
    recursion_limit: int = 50

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return a cached, validated Settings singleton for the process."""
    return Settings()
