"""
graph.py
========

The agent's LangGraph — see week-4-notes.md, section 10, for the full
conceptual background on every primitive used here (StateGraph, nodes,
conditional edges, the crucial backward edge that makes this a true
agent loop rather than a fixed chain).

Graph shape:

    START -> load_context -> call_model --route_after_model-->
        "execute_tools" -> call_model   (THE LOOP — a backward edge)
        "force_stop"    -> END
        END             (model produced a final answer, no tool calls)

`load_context` runs once per HTTP request (at the very start of graph
execution), not on every loop iteration — see call_model's docstring
for how customer_context is used without re-fetching it every step.
"""

# graph.py decides which kind of telemetry context to open; tracing.py implements those contexts.

# Control flow
# load_context uses _span("load_context") in graph.py:185.
# execute_tools uses _span("execute_tools") in graph.py:268.
# call_model uses _generation("call_model") in graph.py:240.
# The local helpers in graph.py simply delegate:


# When no tracer is supplied, both yield _NullHandle, so the graph still runs without telemetry.

# Difference between span() and generation()
# In tracing.py:106:

# AgentTracer.span() records a normal operation, such as memory loading or tool execution.

# Langfuse type: "span"
# Records latency and optional output.
# Does not record token usage or cost.
# AgentTracer.generation() records an LLM call.

# Langfuse type: "generation"
# Includes the model name.
# Records latency, input tokens, output tokens, and estimated cost.
# Both return the same SpanHandle type, defined in tracing.py:32. That handle lets graph.py call:


# Example request
# A request with one model call and one tool loop may produce:
# span:       load_context
# generation: call_model
# span:       execute_tools
# generation: call_model

# The local TraceRecorder stores all of these as SpanRecord objects. A generation is therefore represented locally as a specialized span record with nonzero token and cost fields; ordinary spans normally have those fields set to zero.

# One important detail: generation() does not call span() internally. They are separate methods that create different Langfuse observation types, although both share the same SpanHandle and recording lifecycle.





import logging
from contextlib import asynccontextmanager

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from app.memory.customer_memory_store import CustomerMemoryStore
from app.safety import has_exceeded_max_iterations, is_repeated_tool_call
from app.state import AgentState
from app.tools import EXECUTABLE_TOOLS

logger = logging.getLogger(__name__)


def route_after_model(state: AgentState) -> str:
    """
    Conditional edge: decides whether to execute more tools, force-stop
    due to a detected stuck loop, or end — see week-4-notes.md section 7
    (routing) and section 10 (LangGraph conditional edges).

    Kept as a module-level, dependency-free function (rather than a
    closure inside build_agent_graph) specifically so it's directly
    unit-testable against a hand-constructed state dict with no LLM,
    graph compilation, or Redis involved — see tests/test_graph_routing.py.
    It needs no access to `llm` or `max_iterations` because call_model
    already handles the max_iterations safety valve itself, by returning
    a response with no tool_calls once the cap is hit — which this
    function then naturally routes to END, the same as any other
    tool-call-free response.
    """
    last_message = state["messages"][-1]

    if not getattr(last_message, "tool_calls", None):
        return END

    if is_repeated_tool_call(state["tool_call_log"]):
        return "force_stop"

    return "execute_tools"


def build_agent_graph(llm, max_iterations: int, tracer=None) -> StateGraph:
    """
    Construct the (uncompiled) agent StateGraph.

    Args:
        llm: A tool-bound chat model (see app/llm.py's build_chat_model).
        max_iterations: Hard cap on loop iterations — see app/safety.py.
        tracer: Optional telemetry.tracing.AgentTracer. NEW in Week 5 —
            when provided, every node execution and LLM call is wrapped
            in a Langfuse span/generation (see telemetry/tracing.py).
            Left as None by default (and every call site in this
            module guards on it) so Week 4's existing unit tests — and
            the evaluation harness's own hermetic runs, which don't
            need real tracing — keep working completely unchanged. See
            week-5-notes.md section 2's tracing mental model.

    Returns:
        An uncompiled StateGraph. The caller (main.py) is responsible
        for calling .compile() with a checkpointer and store attached —
        see week-4-notes.md's common mistake about forgetting to compile.

    Why max_iterations is a parameter here rather than read from
    Settings directly inside the node functions: it keeps the graph's
    construction (and therefore its testable behavior) independent of
    global settings state — see tests/test_graph_routing.py, which
    builds small graphs with deliberately tiny max_iterations values to
    exercise the safety valve quickly.
    """

    class _NullHandle:
        """No-op stand-in for telemetry.tracing.SpanHandle when tracer is None."""
        """
        No-op stand-in for telemetry.tracing.SpanHandle when no tracer is provided.

        The returned object intentionally exposes the same methods used by the
        real telemetry handle so callers do not need to check whether telemetry
        is enabled before recording usage or output.
        """

        def set_usage(self, *_args, **_kwargs) -> None:
            # Telemetry is disabled, so usage information is intentionally ignored.
            pass

        def set_output(self, *_args, **_kwargs) -> None:
            # Telemetry is disabled, so output information is intentionally ignored.
            pass

    @asynccontextmanager
    async def _span(name: str):
        """Delegates to tracer.span(name) if a tracer was supplied, else a no-op context."""
        """
        Create a telemetry span when a tracer is available.

        If a tracer was supplied, this delegates to tracer.span(name) and yields
        the real SpanHandle to the caller.

        If no tracer was supplied, it yields a _NullHandle instead. This preserves
        the same context-manager interface while making telemetry completely
        optional.
        """
        if tracer is not None:
        # Enter the real telemetry span context.
        #
        # `async with` ensures that the span is properly started and cleaned
        # up when the caller enters/exits the context.
            async with tracer.span(name) as handle:
            # `yield` pauses this context manager and gives `handle` to the
            # caller inside the `async with _span(...)` block.
            #
            # Execution resumes here when the caller exits that block.
                yield handle
        else:
        # No tracer is configured, so provide a no-op handle.
        #
        # The caller can still call methods such as `set_usage()` or
        # `set_output()` without needing telemetry-specific conditionals.
            yield _NullHandle()

    @asynccontextmanager
    async def _generation(name: str):
        """Delegates to tracer.generation(name) if a tracer was supplied, else a no-op context."""
        """
        Create a telemetry generation when a tracer is available.

        When telemetry is enabled, this delegates to tracer.generation(name)
        and exposes the resulting handle to the caller.

        When telemetry is disabled, a _NullHandle is yielded so the surrounding
        application code can use the same interface without special-case checks.
        """
        if tracer is not None:
        # Enter the real telemetry generation context.
        #
        # The context manager is responsible for the lifecycle of the
        # generation, including cleanup after the caller exits the block.
            async with tracer.generation(name) as handle:
            # `yield handle` hands the real telemetry handle to the code
            # inside the caller's `async with` block.
            #
            # After that block finishes, execution continues here and the
            # underlying telemetry context is exited automatically.
                yield handle
        else:
            # Telemetry is disabled, so expose the same no-op interface instead
            # of requiring callers to check `tracer is None`.
            yield _NullHandle()

    async def load_context(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> dict:
        """
        Node: fetch this customer's long-term memory facts and render
        them into a context string for call_model to use — see
        week-4-notes.md section 6's long-term memory discussion, and
        app/memory/customer_memory_store.py for the storage layer this
        reads from.

        Runs once per HTTP request execution, before the model/tool
        loop begins. Wrapped in a tracing span (see _span above) so
        this step's latency is visible independently of the LLM/tool
        steps — see week-5-notes.md section 3's latency-breakdown point.
        """
        async with _span("load_context") as handle:
            memory_store = CustomerMemoryStore(store)
            facts = await memory_store.get_facts(state["customer_id"])

            if facts:
                context = "Known context about this customer from past conversations:\n" + "\n".join(
                    f"- {fact}" for fact in facts
                )
            else:
                context = "No prior context is known about this customer yet."

            handle.set_output(context)
            return {"customer_context": context}

    async def call_model(state: AgentState) -> dict:
        """
        Node: invoke the LLM with the full conversation history plus
        the customer_context as a fresh, NOT-persisted system message
        (see app/state.py's customer_context field docstring for why
        it's kept out of the persisted `messages` list).

        Also enforces the max_iterations safety valve BEFORE making the
        LLM call at all, once the cap is reached — see
        week-4-notes.md section 4's guidance that termination
        conditions must be hard checks independent of the model's own
        judgment, not just something we hope the model respects.
        """
        next_iteration = state["iteration_count"] + 1

        if has_exceeded_max_iterations(state["iteration_count"], max_iterations=max_iterations):
            # Short-circuit: don't even call the LLM. A forced, honest
            # response is safer and cheaper than one more model call
            # that might request yet another tool.
            logger.warning("Agent loop hit max_iterations=%s; force-stopping.", max_iterations)
            forced_message = AIMessage(
                content=(
                    "I wasn't able to fully complete this within my allotted number of steps. "
                    "Here's what I found so far — let me know if you'd like me to continue."
                )
            )
            return {
                "messages": [forced_message],
                "iteration_count": next_iteration,
                "stop_reason": "max_iterations",
            }

        system_message = SystemMessage(
            content=(
                "You are a customer support operator agent. You can look up orders, issue "
                "refunds, create support tickets, do exact calculations, and remember durable "
                "facts about the customer for future conversations. Only take real actions "
                "(refunds, tickets) when the request is clear — ask for clarification otherwise.\n\n"
                f"{state['customer_context']}"
            )
        )
        async with _generation("call_model") as handle:
            response = await llm.ainvoke([system_message, *state["messages"]])
            # LangChain chat models that report usage attach a
            # usage_metadata dict to the AIMessage — guarded with
            # getattr since not every provider/model populates it.
            usage = getattr(response, "usage_metadata", None) or {}
            handle.set_usage(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            handle.set_output(response.content)
        return {"messages": [response], "iteration_count": next_iteration}

    async def execute_tools(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> dict:
        """
        Node: execute every tool call requested in the model's last
        message — see week-4-notes.md section 2's full tool-calling
        mechanism (the model requests, this node actually executes).

        Special case: `remember_customer_fact` is NOT looked up in
        EXECUTABLE_TOOLS — it's handled here directly, writing to the
        long-term memory store with this request's customer_id, which
        a plain decorated tool function has no way to access on its
        own. See app/tools.py's module docstring for the full rationale.
        """
        last_message = state["messages"][-1]
        memory_store = CustomerMemoryStore(store)

        tool_messages = []
        new_tool_call_records = []

        async with _span("execute_tools") as handle:
            for call in last_message.tool_calls:
                name, args, call_id = call["name"], call["args"], call["id"]
                new_tool_call_records.append({"name": name, "arguments": args})

                try:
                    if name == "remember_customer_fact":
                        await memory_store.add_fact(state["customer_id"], args["fact"])
                        result = f"Noted and saved for future conversations: {args['fact']}"
                    elif name in EXECUTABLE_TOOLS:
                        result = EXECUTABLE_TOOLS[name].invoke(args)
                    else:
                        # A hallucinated/unknown tool name — return this AS
                        # a tool result so the model sees the failure and
                        # can adapt, rather than crashing the whole request.
                        result = f"Error: unknown tool '{name}'."
                except Exception as exc:
                    # Any tool execution failure is returned to the model as
                    # a result too (see week-4-notes.md section 2's "Common
                    # Mistakes": never silently swallow tool errors).
                    result = f"Error executing tool '{name}': {exc}"

                tool_messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

            handle.set_output([r["name"] for r in new_tool_call_records])

        return {"messages": tool_messages, "tool_call_log": new_tool_call_records}

    def force_stop(state: AgentState) -> dict:
        """
        Node: reached only when the repeated-identical-tool-call safety
        valve fires (see route_after_model and app/safety.py) — appends
        an honest, user-facing explanation instead of silently cutting
        the conversation off.
        """
        logger.warning("Repeated identical tool call detected; force-stopping.")
        message = AIMessage(
            content=(
                "I seem to be repeating the same action without making progress, so I've "
                "stopped here rather than continue. Could you clarify what you'd like me to do?"
            )
        )
        return {"messages": [message], "stop_reason": "repeated_tool_call"}

    workflow = StateGraph(AgentState)
    workflow.add_node("load_context", load_context)
    workflow.add_node("call_model", call_model)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("force_stop", force_stop)

    workflow.add_edge(START, "load_context")
    workflow.add_edge("load_context", "call_model")
    workflow.add_conditional_edges(
        "call_model", route_after_model, ["execute_tools", "force_stop", END]
    )
    # THE CYCLE: after tools run, control returns to call_model — this
    # backward edge is what makes this graph an agent loop rather than
    # a fixed DAG. See week-4-notes.md section 10.
    workflow.add_edge("execute_tools", "call_model")
    workflow.add_edge("force_stop", END)

    return workflow
