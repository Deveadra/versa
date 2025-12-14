# base/core/context.py

"""
Global session context for cross-plugin memory.
Plugins can store/retrieve temporary state here
to allow multi-turn conversations.
"""

session_context = {}


def set_context(key, value):
    """Store a value in context under key."""
    session_context[key] = value


def get_context(key, default=None):
    """Retrieve a value from context."""
    return session_context.get(key, default)


def clear_context(key=None):
    """Clear one key or all context."""
    if key:
        session_context.pop(key, None)
    else:
        session_context.clear()
