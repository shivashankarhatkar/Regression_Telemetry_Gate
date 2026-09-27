"""
test_customer_memory_store.py
================================

Unit tests for CustomerMemoryStore, using a fake in-memory BaseStore
double instead of a real AsyncRedisStore/Redis instance — verifies our
wrapper's namespacing and data-shape logic without needing Redis
running. See app/memory/customer_memory_store.py's module docstring
for the store's role (long-term, cross-thread memory).
"""

import pytest

from app.memory.customer_memory_store import CustomerMemoryStore


class _FakeItem:
    """Minimal stand-in for a langgraph.store.base.Item — only needs .value."""

    def __init__(self, value: dict):
        self.value = value


class _FakeStore:
    """
    Minimal in-memory double for langgraph.store.base.BaseStore,
    implementing only the two async methods CustomerMemoryStore uses.
    """

    def __init__(self):
        self._data: dict[tuple, dict[str, dict]] = {}

    async def aput(self, namespace: tuple, key: str, value: dict) -> None:
        self._data.setdefault(namespace, {})[key] = value

    async def asearch(self, namespace: tuple, **kwargs) -> list[_FakeItem]:
        return [_FakeItem(value) for value in self._data.get(namespace, {}).values()]


@pytest.mark.asyncio
async def test_get_facts_empty_for_new_customer() -> None:
    """A customer with no stored facts should get an empty list, not an error."""
    store = CustomerMemoryStore(_FakeStore())

    facts = await store.get_facts("customer-1")

    assert facts == []


@pytest.mark.asyncio
async def test_add_fact_then_get_facts_returns_it() -> None:
    """A fact added for a customer should be retrievable afterward."""
    store = CustomerMemoryStore(_FakeStore())

    await store.add_fact("customer-1", "Prefers email contact over phone calls.")
    facts = await store.get_facts("customer-1")

    assert facts == ["Prefers email contact over phone calls."]


@pytest.mark.asyncio
async def test_facts_are_isolated_per_customer() -> None:
    """Facts stored for one customer must not leak into another customer's results."""
    store = CustomerMemoryStore(_FakeStore())

    await store.add_fact("customer-1", "Fact about customer 1")
    await store.add_fact("customer-2", "Fact about customer 2")

    assert await store.get_facts("customer-1") == ["Fact about customer 1"]
    assert await store.get_facts("customer-2") == ["Fact about customer 2"]


@pytest.mark.asyncio
async def test_multiple_facts_accumulate_for_same_customer() -> None:
    """Adding several facts for one customer should accumulate, not overwrite."""
    store = CustomerMemoryStore(_FakeStore())

    await store.add_fact("customer-1", "First fact")
    await store.add_fact("customer-1", "Second fact")

    facts = await store.get_facts("customer-1")
    assert set(facts) == {"First fact", "Second fact"}
