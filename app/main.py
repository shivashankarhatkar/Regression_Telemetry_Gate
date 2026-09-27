"""
main.py
=======

FastAPI application entry point for The Regression & Telemetry Gate's
live service (the same Persistent Operator agent from Week 4, now with
Week 5's tracing wired in).

Endpoints:
    POST /chat       - send a message to the agent; thread_id continues
                       an existing conversation (short-term memory),
                       customer_id scopes long-term memory
    GET  /telemetry  - NEW in Week 5: cumulative unit-economics summary
                       (cost/tokens/latency) across every request this
                       process has served — see telemetry/usage_aggregator.py
    GET  /health     - basic liveness check

Run locally with:
    uvicorn app.main:app --reload

Requires a running Redis instance with RedisJSON + RediSearch (Redis
8.0+, or Redis Stack for older Redis) — see README.md and docker-compose.yml.
Langfuse tracing is optional here: if LANGFUSE_PUBLIC_KEY/SECRET_KEY are
not set, AgentTracer degrades gracefully to local-only recording (see
telemetry/tracing.py's AgentTracer docstring).
"""

from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Request
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.store.redis.aio import AsyncRedisStore

from app.config import Settings, get_settings
from app.graph import build_agent_graph
from app.llm import build_chat_model
from app.schemas import ChatRequest, ChatResponse, ToolCallSummary
from telemetry.tracing import AgentTracer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan hook: opens the two Redis-backed LangGraph
    persistence layers, builds the chat model, tracer, and compiled
    graph, and cleans everything up on shutdown.

    NEW in Week 5: builds one AgentTracer for the process's lifetime
    (see telemetry/tracing.py and usage_aggregator.py's TraceRecorder
    docstring for why this is process-cumulative, matching Week 1's
    UsageTracker, rather than per-request) and flushes it on shutdown
    so no pending Langfuse spans are lost — see week-5-notes.md
    section 2a's common mistake about unflushed spans in short-lived
    processes; a long-running server is less exposed to this than a
    CI script, but flushing on graceful shutdown is still good practice.
    """
    settings = get_settings()
    exit_stack = AsyncExitStack()

    checkpointer = await exit_stack.enter_async_context(
        AsyncRedisSaver.from_conn_string(settings.redis_url)
    )
    # Required once per checkpointer: creates the Redis indices
    # (RediSearch/RedisJSON) the checkpointer needs. Safe to call every
    # startup — it's a no-op if the indices already exist.
    await checkpointer.asetup()

    store = await exit_stack.enter_async_context(AsyncRedisStore.from_conn_string(settings.redis_url))
    await store.setup()

    llm = build_chat_model(settings)
    tracer = AgentTracer(model=settings.chat_model)
    workflow = build_agent_graph(llm, max_iterations=settings.max_iterations, tracer=tracer)
    compiled_graph = workflow.compile(checkpointer=checkpointer, store=store)

    app.state.compiled_graph = compiled_graph
    app.state.settings = settings
    app.state.tracer = tracer

    yield

    tracer.flush()
    await exit_stack.aclose()


app = FastAPI(
    title="The Regression & Telemetry Gate",
    description="Week 5 project: the Persistent Operator agent, instrumented with Langfuse/OTel tracing and unit-economics telemetry.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic liveness check — does not touch Redis, Langfuse, or the LLM provider."""
    return {"status": "ok"}


@app.get("/telemetry")
async def telemetry(request: Request) -> dict:
    """
    Return the cumulative unit-economics summary (see
    telemetry/usage_aggregator.py's TraceSummary) across every request
    this process has served since startup — the local, in-process
    "unit economics" view from week-5-notes.md section 3. Per-request
    detail lives in Langfuse itself if configured.
    """
    tracer: AgentTracer = request.app.state.tracer
    summary = tracer.summary()
    return {
        "total_requests_spans": len(summary.spans),
        "total_latency_ms": round(summary.total_latency_ms, 2),
        "total_input_tokens": summary.total_input_tokens,
        "total_output_tokens": summary.total_output_tokens,
        "total_cost_usd": summary.total_cost_usd,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(
    chat_request: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """
    Send a message to the agent and run its loop to completion (or
    until a safety valve triggers). Every node execution and LLM call
    made during this request is traced (see app/graph.py's tracer
    wiring and telemetry/tracing.py).

    thread_id scopes SHORT-TERM memory: reusing it continues the exact
    same conversation (the checkpointer restores prior `messages` state
    automatically — we only need to supply the new human message).
    customer_id scopes LONG-TERM memory: it's used by the graph's
    load_context node regardless of which thread_id is active, so a
    brand-new thread_id for a returning customer still has access to
    facts learned in earlier, separate conversations.
    """
    compiled_graph = request.app.state.compiled_graph

    config = {
        "configurable": {"thread_id": chat_request.thread_id},
        "recursion_limit": settings.recursion_limit,
    }

    result = await compiled_graph.ainvoke(
        {
            "messages": [HumanMessage(content=chat_request.message)],
            "customer_id": chat_request.customer_id,
            "iteration_count": 0,
            "tool_call_log": [],
            "customer_context": "",
            "stop_reason": "completed",
        },
        config=config,
    )

    final_message = result["messages"][-1]
    tool_calls = [
        ToolCallSummary(name=record["name"], arguments=record["arguments"])
        for record in result["tool_call_log"]
    ]

    return ChatResponse(
        reply=final_message.content,
        tool_calls=tool_calls,
        iterations_used=result["iteration_count"],
        stopped_reason=result["stop_reason"],
    )
