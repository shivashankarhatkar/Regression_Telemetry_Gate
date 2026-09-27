"""
llm.py
======

Constructs the LangChain chat model used by the agent.

Uses the dedicated `langchain-openrouter` package's `ChatOpenRouter`
class rather than the older community pattern of pointing `ChatOpenAI`
at OpenRouter's base_url. As of this writing, LangChain's own
`ChatOpenAI` documentation explicitly recommends provider-specific
packages (like this one) over the base_url-override approach for
non-OpenAI providers, since ChatOpenAI targets the official OpenAI API
spec exactly and doesn't handle provider-specific response fields.

NOTE: `langchain-openrouter` is a young, fast-moving package (its own
docs describe it as "beta, moves fast, always pull the latest"). If
its API has changed since this was written, check
https://reference.langchain.com/python/langchain-openrouter — the
base_url-override fallback (`ChatOpenAI(base_url="https://openrouter.ai/api/v1", ...)`)
still works with any LangChain version if you need a stable
alternative.
"""

from langchain_openrouter import ChatOpenRouter

from app.config import Settings
from app.tools import ALL_TOOLS


def build_chat_model(settings: Settings) -> ChatOpenRouter:
    """
    Construct the tool-calling-enabled chat model for the agent.

    Args:
        settings: Application settings containing the OpenRouter API
            key, chat model slug, and request timeout.

    Returns:
        A ChatOpenRouter instance with all of app/tools.py's tools
        bound via bind_tools() — every call through this model can
        request any of them, following the exact tool-calling
        mechanism described in week-4-notes.md, section 2.
    """
    model = ChatOpenRouter(
        model=settings.chat_model,
        api_key=settings.openrouter_api_key,
        timeout=settings.request_timeout_seconds,
        temperature=0,
    )
    return model.bind_tools(ALL_TOOLS)
