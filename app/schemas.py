"""
schemas.py
==========

Pydantic models for the HTTP API's request/response bodies. Internal
agent state (app/state.py) is a separate, LangGraph-specific schema —
see that file for why the two are kept distinct.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request body for POST /chat.

    Attributes:
        thread_id: Identifies ONE conversation. Reusing the same
            thread_id across requests continues that exact conversation
            (short-term memory, via the checkpointer). A new thread_id
            starts a fresh conversation.
        customer_id: Identifies the customer this conversation is
            about. Used to load/save LONG-TERM memory (via the store) —
            distinct from thread_id, since the same customer may have
            many separate conversation threads over time, and each new
            thread should still have access to previously learned
            customer facts.
        message: The user's message.
    """

    thread_id: str = Field(..., min_length=1)
    customer_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ToolCallSummary(BaseModel):
    """
    A single tool call made during this request's agent loop, returned
    for traceability (see week-4-notes.md's emphasis on inspectable
    ReAct-style traces).

    Attributes:
        name: Name of the tool that was called.
        arguments: Arguments the model supplied for this call.
    """

    name: str
    arguments: dict


class ChatResponse(BaseModel):
    """
    Response body for POST /chat.

    Attributes:
        reply: The agent's final text response.
        tool_calls: Every tool call made during this request's agent
            loop, in order — the inspectable trace.
        iterations_used: How many loop iterations this request took.
        stopped_reason: Why the loop stopped — "completed" (the model
            produced a final answer), "max_iterations" or
            "repeated_tool_call" (a safety valve triggered — see
            week-4-notes.md section 4).
    """

    reply: str
    tool_calls: list[ToolCallSummary]
    iterations_used: int
    stopped_reason: str
