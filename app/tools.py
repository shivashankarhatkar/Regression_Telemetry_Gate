"""
tools.py
========

Tool definitions for the Persistent Operator agent — see
week-4-notes.md, section 2, for the full tool-calling mechanism this
implements.

Each tool is defined with `@tool` from langchain_core.tools, which
generates the tool's JSON-schema definition automatically from the
function's type hints and uses the docstring as the tool's
description — the description IS what the model uses to decide when
to call it, so these are written with that audience in mind, not as
implementation comments (see week-4-notes.md's guidance on tool
description quality).

REAL ACTIONS vs. READ-ONLY LOOKUPS: `issue_refund` and
`create_support_ticket` are actions with side effects (they mutate the
simulated backend state below) — exactly the "takes real actions" part
of this week's brief. In a genuine production system these would call
real payment/ticketing APIs; here they mutate simple in-memory
dictionaries so the project is runnable without external services,
while still demonstrating the same tool-calling and safety
considerations (argument validation before execution) that a real
integration would need.

IMPORTANT — one tool defined here, `remember_customer_fact`, is
special: its @tool-decorated function body is NEVER actually called.
It exists only so the model sees its schema and can request it. The
REAL execution happens in graph.py's execute_tools node, which
special-cases this tool name to call the customer memory store with
the current customer_id — something a plain decorated function has no
way to know, since it only receives the arguments the model supplies.
See graph.py's execute_tools docstring for the full explanation.
"""

from langchain_core.tools import tool

# --------------------------------------------------------------------------
# Simulated backend state
# --------------------------------------------------------------------------
# In-memory stand-ins for what would be real database/API calls in
# production. Mutating these dicts is what makes issue_refund and
# create_support_ticket "real actions" rather than pure lookups — their
# effects are visible to subsequent tool calls within the same process.

_ORDERS: dict[str, dict] = {
    "ORD-1001": {"status": "delivered", "total": 89.99, "refunded": False},
    "ORD-1002": {"status": "processing", "total": 34.50, "refunded": False},
    "ORD-1003": {"status": "delivered", "total": 210.00, "refunded": False},
}

_SUPPORT_TICKETS: list[dict] = []


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression (numbers, +, -, *, /,
    parentheses only) and return the result. Use this for any exact
    numeric calculation the user needs — refund totals, discounts,
    quantity math — rather than computing it yourself, since exact
    arithmetic should come from a real calculation, not a guess.
    """
    import ast
    import operator as op

    # A restricted expression evaluator — deliberately does NOT use
    # Python's eval(), which would execute arbitrary code. Only the
    # specific arithmetic AST node types below are permitted; anything
    # else raises, rather than silently doing something unintended.
    _ALLOWED_OPERATORS = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.USub: op.neg,
    }

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
        raise ValueError(f"Unsupported expression element: {ast.dump(node)}")

    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_node(parsed.body)
        return str(result)
    except Exception as exc:
        # Returned as a normal tool result (not raised) so the model
        # sees the failure and can react — e.g. by asking the user to
        # clarify the expression — rather than the whole request erroring.
        return f"Could not evaluate '{expression}': {exc}"


@tool
def get_order_status(order_id: str) -> str:
    """
    Look up the current status, total amount, and refund status of a
    customer's order, given its order ID (e.g. 'ORD-1001'). Use this
    whenever the user asks about an order's status, cost, or whether
    it's been refunded.
    """
    order = _ORDERS.get(order_id)
    if order is None:
        return f"No order found with ID '{order_id}'."
    return (
        f"Order {order_id}: status={order['status']}, total=${order['total']:.2f}, "
        f"refunded={order['refunded']}"
    )


@tool
def issue_refund(order_id: str, reason: str) -> str:
    """
    Issue a refund for a customer's order. This is a REAL ACTION with a
    side effect (it will actually mark the order as refunded) — only
    call this when the user has clearly requested a refund and you have
    confirmed which order via get_order_status first. Do not call this
    speculatively or without a clear reason.
    """
    order = _ORDERS.get(order_id)
    if order is None:
        return f"Cannot issue refund: no order found with ID '{order_id}'."
    if order["refunded"]:
        return f"Order {order_id} has already been refunded."
    order["refunded"] = True
    return f"Refund of ${order['total']:.2f} issued for order {order_id}. Reason recorded: {reason}"


@tool
def create_support_ticket(summary: str, priority: str = "normal") -> str:
    """
    Create a support ticket for an issue that needs human follow-up.
    Use this for problems you cannot resolve directly with the other
    tools available (e.g. a complaint that needs escalation). priority
    should be one of: 'low', 'normal', 'high'.
    """
    ticket_id = f"TCK-{len(_SUPPORT_TICKETS) + 1:04d}"
    _SUPPORT_TICKETS.append({"id": ticket_id, "summary": summary, "priority": priority})
    return f"Support ticket {ticket_id} created (priority={priority})."


@tool
def remember_customer_fact(fact: str) -> str:
    """
    Save a durable fact about this customer for future conversations —
    e.g. a stated preference ('prefers email over phone'), a recurring
    issue, or context worth remembering long after this conversation
    ends. Use this when the customer shares something worth recalling
    next time, not for routine transactional details already captured
    by other tools.

    NOTE FOR MAINTAINERS: this function body is intentionally never
    executed — see this module's docstring and graph.py's execute_tools
    node, which special-cases this tool name to write to the customer
    memory store with the current request's customer_id.
    """
    raise NotImplementedError(
        "remember_customer_fact must be handled specially in graph.py's execute_tools node, "
        "not called directly — see this function's docstring."
    )


# The full toolset exposed to the model via bind_tools(). Order doesn't
# matter functionally, but keeping read-only lookups before
# side-effecting actions before the memory tool is a reasonable,
# readable convention.
ALL_TOOLS = [calculate, get_order_status, issue_refund, create_support_ticket, remember_customer_fact]

# A name -> callable registry for the execute_tools node (graph.py) to
# dispatch against for every tool EXCEPT remember_customer_fact, which
# is special-cased there instead of looked up here.
EXECUTABLE_TOOLS = {
    "calculate": calculate,
    "get_order_status": get_order_status,
    "issue_refund": issue_refund,
    "create_support_ticket": create_support_ticket,
}
