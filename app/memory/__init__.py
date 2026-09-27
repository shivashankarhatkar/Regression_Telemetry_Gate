"""
Memory subpackage — long-term, cross-thread customer memory built on
LangGraph's Redis-backed BaseStore. See customer_memory_store.py.

This is distinct from the checkpointer (short-term, thread-scoped
memory) configured directly in main.py.
"""
