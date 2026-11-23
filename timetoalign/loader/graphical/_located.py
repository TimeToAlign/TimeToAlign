"""The refusal of a shape read from a file, prefixed by where it stands.

Shared by the readers of areas drawn on page images (DMOS, MEI facsimile),
whose messages name the page, system, measure or zone a refused shape
belongs to.
"""

from __future__ import annotations

from pydantic import ValidationError


def located(where: str, error: Exception) -> ValueError:
    """The refusal *error* of a shape, prefixed by *where* it stands.

    A refusal raised by a model validator arrives wrapped in pydantic's
    report; its own message is taken from there.

    Args:
        where: Where the shape stands in its file, e.g. ``"MEI zone 'z1'"``.
        error: The shape's refusal.

    Returns:
        ``"{where}: {message}"`` as a ``ValueError``.
    """
    reasons = (
        [
            str(entry.get("ctx", {}).get("error", entry["msg"]))
            for entry in error.errors()
        ]
        if isinstance(error, ValidationError)
        else [str(error)]
    )
    return ValueError(f"{where}: {' '.join(reasons)}")
