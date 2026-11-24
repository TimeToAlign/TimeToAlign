"""Warning hygiene for third-party backends."""

from __future__ import annotations

import warnings
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def suppressed_backend_warnings() -> Generator[None, None, None]:
    """Suppress non-actionable warnings emitted by third-party backends.

    Partitura reports internal parsing progress and third-party deprecations
    through the warnings module. These messages describe backend internals and
    are not actionable by a TimeToAlign! user, so backend imports and parsing
    calls use this scoped context while leaving all other warnings untouched.

    Yields:
        Control to the backend operation with targeted warning filters active.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"^partitura(?:\.|$)",
        )
        warnings.filterwarnings(
            "ignore",
            category=ImportWarning,
            module=r"^partitura(?:\.|$)",
        )
        yield
