"""
customer_memory_store.py
==========================

Long-term, cross-thread customer memory — see week-4-notes.md, section
6, for the short-term vs. long-term memory distinction this
implements structurally: this store is keyed by customer_id and is
deliberately independent of any single conversation's thread_id, so
facts learned in one conversation are available in a completely
different, later conversation with the same customer.

Built on LangGraph's `BaseStore` interface (satisfied by
`AsyncRedisStore` in production — see main.py's lifespan — or any
other BaseStore implementation, including a fake one in tests). This
module depends only on the interface, not the Redis-specific class,
following the same swappable-component pattern as Week 2's VectorStore
interface.

Scope note: this implementation stores and retrieves facts by
customer_id via plain get/list operations — it does NOT use
AsyncRedisStore's semantic (embedding-indexed) search capability. Every
fact for a customer is loaded on each conversation start (see
graph.py's load_context node); at the scale of "durable facts about
one customer," this is simpler and sufficient. Retrieving only the
most SEMANTICALLY RELEVANT facts (rather than all of them) via
embedding search is a natural extension once a customer's fact count
grows large — see week-4-notes.md's note that long-term memory
retrieval is structurally a RAG problem, directly reusing Weeks 2-3's
embedding + similarity search machinery.
"""

import uuid

from langgraph.store.base import BaseStore

_NAMESPACE_PREFIX = "customer_memory"


class CustomerMemoryStore:
    """Reads and writes durable, cross-conversation facts about a customer."""

    def __init__(self, store: BaseStore) -> None:
        """
        Args:
            store: Any BaseStore implementation — AsyncRedisStore in
                production, or a fake/in-memory double in tests.
        """
        self._store = store

    def _namespace(self, customer_id: str) -> tuple[str, str]:
        """
        Build the namespace tuple facts are stored under for a given
        customer. Namespacing by customer_id (not by thread_id) is
        exactly what makes this cross-thread — see module docstring.
        """
        return (_NAMESPACE_PREFIX, customer_id)

    async def get_facts(self, customer_id: str) -> list[str]:
        """
        Retrieve every durable fact stored for this customer, across
        all past conversations.

        Args:
            customer_id: The customer to retrieve facts for.

        Returns:
            A list of fact strings, in no particular guaranteed order.
            Returns an empty list for a customer with no stored facts
            yet (e.g. a brand-new customer) — this is the expected,
            non-error case for a first-ever conversation.
        """
        items = await self._store.asearch(self._namespace(customer_id))
        return [item.value["fact"] for item in items]

    async def add_fact(self, customer_id: str, fact: str) -> None:
        """
        Persist a new durable fact about a customer.

        Args:
            customer_id: The customer this fact is about.
            fact: The fact text to store, e.g. "Prefers email contact
                over phone calls."

        Note: each call stores a NEW entry (keyed by a fresh UUID)
        rather than overwriting a single value — a customer can have
        many independent facts accumulated over many conversations,
        the same way section 6's "episodic memory" accumulates
        discrete records rather than replacing a single summary.
        """
        key = str(uuid.uuid4())
        await self._store.aput(self._namespace(customer_id), key, {"fact": fact})
