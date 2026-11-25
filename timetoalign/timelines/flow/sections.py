"""Read interval-like descriptors as the played spans of a flow."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction

from timetoalign.core.events import Gap

# region Interval coercion


def _interval_value(x: object) -> Fraction:
    """Coerce a single coordinate-like value to an exact ``Fraction``.

    Args:
        x: A coordinate-like object exposing a ``value`` attribute
            (``Coordinate`` / ``IdCoordinate``) or a raw ``int`` / ``float`` /
            ``Fraction``.

    Returns:
        The position as an exact ``Fraction``.
    """
    return Fraction(x.value if hasattr(x, "value") else x)


def _as_interval(
    obj: object,
    *,
    resolve: Callable[[str], object] | None = None,
) -> tuple[Fraction, Fraction, str | None]:
    """Coerce a single interval-like descriptor to a ``(start, end, label)`` triple.

    Descriptors are recognised in strict priority order, and each carries a
    best-effort identity ``label`` (``None`` when the descriptor names nothing):

    1. ``str`` — a region id resolved through *resolve* (typically a
       timeline's ``get_region``). The string itself is the label. A ``str``
       with no resolver is an error; a string is never iterated character by
       character.
    2. Any object exposing ``start`` and ``end`` — a ``Region``, an interval
       event (``Note`` / ``Measure``), or a ``TimeIntervalStamp``. The label is
       ``getattr(obj, "id", None) or getattr(obj, "name", None)`` (a
       ``Region`` is labelled by its ``id``).
    3. Any object exposing ``origin`` and ``length`` — a ``Timeline`` is
       itself an interval running from its origin to its length. The label is
       ``getattr(obj, "id", None) or getattr(obj, "name", None)``.
    4. A two-element sequence — an explicit ``(start, end)`` coordinate pair,
       which names nothing, so the label is ``None``.

    Args:
        obj: The descriptor to coerce.
        resolve: Callable mapping a region id to an interval-like object.
            Required only when *obj* is a ``str``.

    Returns:
        The ``(start, end, label)`` triple: the bounds as exact ``Fraction``
        values, and the identity label (a ``str`` or ``None``).

    Raises:
        ValueError: If *obj* is a ``str`` and no *resolve* is given, or if a
            sequence does not hold exactly two elements.
        TypeError: If *obj* matches none of the recognised descriptor shapes.
        UnknownLengthError: If *obj* is a timeline of unknown length.
    """
    if isinstance(obj, str):
        if resolve is None:
            raise ValueError(f"Cannot resolve interval name {obj!r} without a resolver")
        start, end, _ = _as_interval(resolve(obj), resolve=resolve)
        return (start, end, obj)
    if hasattr(obj, "start") and hasattr(obj, "end"):
        label = getattr(obj, "id", None) or getattr(obj, "name", None)
        from ..base import Timeline

        if isinstance(obj, Timeline):
            obj._require_length("read the timeline as an interval")
        return (_interval_value(obj.start), _interval_value(obj.end), label)
    if hasattr(obj, "origin") and hasattr(obj, "length"):
        label = getattr(obj, "id", None) or getattr(obj, "name", None)
        return (_interval_value(obj.origin), _interval_value(obj.length), label)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        if len(obj) != 2:
            raise ValueError(
                f"A coordinate-pair sequence must hold exactly two elements, "
                f"got {len(obj)}: {obj!r}"
            )
        start, end = obj
        return (_interval_value(start), _interval_value(end), None)
    raise TypeError(f"Cannot interpret {obj!r} as an interval")


def _coerce_intervals(
    spec: object,
    *,
    resolve: Callable[[str], object] | None = None,
) -> list[tuple[Fraction, Fraction, str | None]]:
    """Coerce a singleton-or-collection spec to ``(start, end, label)`` triples.

    One positional argument carries both a single interval-like descriptor and
    a collection of them. Disambiguation tries the whole *spec* as a single
    interval first; only if that coercion fails is *spec* iterated as a
    collection. A ``str`` is always a singleton — it is never iterated
    character by character.

    Args:
        spec: One interval-like descriptor (see :func:`_as_interval`) or an
            iterable of them.
        resolve: Callable mapping a region id to an interval-like object,
            forwarded to :func:`_as_interval`.

    Returns:
        The list of ``(start, end, label)`` triples, one per interval.

    Raises:
        ValueError: If any resulting interval has ``end < start``.
    """
    if isinstance(spec, str):
        intervals = [_as_interval(spec, resolve=resolve)]
    else:
        try:
            intervals = [_as_interval(spec, resolve=resolve)]
        except (TypeError, ValueError):
            intervals = [_as_interval(x, resolve=resolve) for x in spec]

    for start, end, _label in intervals:
        if end < start:
            raise ValueError(f"Interval end ({end}) cannot be before start ({start})")
    return intervals


# endregion


def _coerce_flow_entries(
    spec: object,
    *,
    resolve: Callable[[str], object] | None = None,
) -> list["tuple[Fraction, Fraction, str | None] | Gap"]:
    """Coerce a flow spec that may contain Gaps into an ordered entry list.

    Extends :func:`_coerce_intervals` with :class:`Gap` entries. Every
    non-``Gap`` element is coerced by :func:`_as_interval` exactly as before,
    so a spec holding no gaps yields the same ``(start, end, label)`` triples
    that :func:`_coerce_intervals` produces.

    A spec that *is* a single ``Gap`` — or that holds nothing but gaps —
    describes no played material and is rejected: a flow has to play something.

    Args:
        spec: One interval-like descriptor (see :func:`_as_interval`), a
            ``Gap``, or an iterable mixing the two.
        resolve: Callable mapping a region id to an interval-like object,
            forwarded to :func:`_as_interval`.

    Returns:
        The entries in target order: ``(start, end, label)`` triples for played
        spans and :class:`Gap` objects for holes.

    Raises:
        ValueError: If any played interval has ``end < start``, or if the spec
            holds no played span at all.
    """

    def _played(item: object) -> tuple[Fraction, Fraction, str | None]:
        start, end, label = _as_interval(item, resolve=resolve)
        if end < start:
            raise ValueError(f"Interval end ({end}) cannot be before start ({start})")
        return (start, end, label)

    entries: list["tuple[Fraction, Fraction, str | None] | Gap"] = []
    if isinstance(spec, Gap):
        entries.append(spec)
    elif isinstance(spec, str):
        entries.append(_played(spec))
    else:
        # Same singleton-or-collection disambiguation as `_coerce_intervals`:
        # try the whole spec as one interval, and only iterate if that fails.
        try:
            entries.append(_played(spec))
        except (TypeError, ValueError):
            for item in spec:  # type: ignore[attr-defined]
                entries.append(item if isinstance(item, Gap) else _played(item))

    if not any(not isinstance(entry, Gap) for entry in entries):
        raise ValueError(
            "A flow spec must contain at least one played span; "
            "gaps alone describe no material to play"
        )
    return entries


# endregion
