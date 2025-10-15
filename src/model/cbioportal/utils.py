"""Utility functions for cBioPortal data handling."""

from typing import Any


def safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """
    Safely get attribute from dict or object.

    Args:
        obj: Dictionary or object
        attr: Attribute/key name
        default: Default value if not found

    Returns:
        Value or default
    """
    if isinstance(obj, dict):
        return obj.get(attr, default)
    else:
        return getattr(obj, attr, default)
