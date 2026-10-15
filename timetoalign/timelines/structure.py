"""Immutable measure maps and the structural hierarchies built over them."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd, isfinite, lcm
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, NamedTuple, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from timetoalign.core import (
    BeatPolicy,
    CadenzaMeasure,
    Coordinate,
    Domain,
    Duration,
    Interval,
    Measure,
    MeasureConstituent,
    RegularMeasure,
    SplitIrregularMeasure,
    SplitRegularMeasure,
    Tempo,
    TimeUnit,
)
from timetoalign.core.retrieval import coordinate_from_dict, coordinate_to_dict
from timetoalign.core.time import number_from_json, number_to_json, struct_to_rational
from timetoalign.timelines.flow import AtomicSection

if TYPE_CHECKING:
    from timetoalign.loader.score.stores.measures import MeasureData


_MemoValue = TypeVar("_MemoValue")


_ROW_KEYS = (
    "ID",
    "count",
    "qstamp",
    "number",
    "name",
    "time_signature",
    "nominal_length",
    "actual_length",
    "start_repeat",
    "end_repeat",
    "next",
    "volta",
    "offset_within_measure",
)
"""Measure Map standard keys plus the ``volta`` and ``offset_within_measure`` extensions."""

_QUARTER_KEYS = frozenset(
    {"qstamp", "nominal_length", "actual_length", "offset_within_measure"}
)


def _row_quarters(row: Mapping[str, Any], key: str) -> Fraction | None:
    """Read one quarter-note value of a document row exactly.

    Rows carry the TimeSkeleton JSON number forms only, read by
    :func:`~timetoalign.core.time.number_from_json`; a JSON float becomes
    its exact binary fraction, never a guessed ratio.
    """
    value = row.get(key)
    return None if value is None else Fraction(number_from_json(value))


def _is_cadenza(signature: str | None) -> bool:
    return signature is not None and str(signature).lower() == "cadenza"


def _signature_length(signature: str | None) -> Fraction | None:
    """Return the nominal length a metered signature implies, else ``None``."""
    if signature is None or _is_cadenza(signature):
        return None
    return BeatPolicy.from_time_signature(str(signature)).span


def _reconciled_qstamp(
    stated: Fraction | None, derived: Fraction | None, label: object
) -> Fraction | None:
    """Return the qstamp a measure keeps, warning when the stated one disagrees.

    The position derived from printed order and preceding actual lengths
    wins; a stated qstamp survives only where no position can be derived.
    """
    if stated is None or derived is None:
        return derived if stated is None else stated
    if stated != derived:
        warnings.warn(
            f"Measure {label!r} supplies qstamp {stated}, but prefix-summing "
            f"actual lengths computes {derived}",
            stacklevel=3,
        )
    return derived


def _table_successors(value: Any) -> list[int] | None:
    """Read successor counts from a measure table's comma-separated text.

    Measure tables spell the end of the piece ``-1``; it names no measure and
    is dropped, so ``"-1"`` reads as no successor at all.
    """
    if value is None or value == "":
        return None
    text = str(value).strip().strip("()[]")
    parts = [int(part.strip()) for part in text.split(",") if part.strip()]
    return [part for part in parts if part != -1]


class MeasureMap:
    """An immutable sequence of measures in printed order.

    A measure's ``id`` is its Measure Map row ``ID``: ``str(count)`` unless a
    source states another, and ``next`` names successors by these IDs. A
    map states every successor explicitly: a measure built without ``next``
    is followed by the next measure in printed order, and the last one by
    none.

    A map read from a manifest keeps the keys of each row that the row form
    does not define, by measure, and writes them back in that measure's row;
    they take no part in equality.
    """

    _unknown_row_keys: Mapping[str, Mapping[str, Any]] = MappingProxyType({})

    @classmethod
    def from_measure_data(cls, data: MeasureData) -> MeasureMap:
        """Build the canonical measure structure described by loaded rows.

        Args:
            data: Normalized measure-event rows from a score loader.

        Returns:
            The immutable measure map represented by the rows.

        Raises:
            ValueError: If the data is empty or lacks required exact positions
                and durations.
        """
        rows = list(data)
        if not rows:
            raise ValueError("Cannot build a MeasureMap from empty measure data")
        expanded: list[dict[str, Any]] = []
        signature: str | None = None
        for index, row in enumerate(rows, start=1):
            if row.get("timesig"):
                signature = str(row["timesig"])
            if row.get("duration") is None:
                raise ValueError(f"Measure row {index} states no actual length")
            actual = Fraction(struct_to_rational(row["duration"]))
            count = int(row.get("mc") or index)
            number = row.get("mn_int")
            offset = row.get("mc_offset")
            expanded.append(
                {
                    "ID": row.get("mm_id") or str(count),
                    "count": count,
                    "qstamp": (
                        Fraction(struct_to_rational(row["start"]))
                        if row.get("start") is not None
                        else None
                    ),
                    "number": int(number) if number is not None else count,
                    "name": str(row["mn"]) if row.get("mn") is not None else str(count),
                    "time_signature": signature,
                    "nominal_length": (
                        actual
                        if _is_cadenza(signature)
                        else _signature_length(signature)
                    ),
                    "actual_length": actual,
                    "start_repeat": bool(row.get("start_repeat", False)),
                    "end_repeat": bool(row.get("end_repeat", False)),
                    "next": _table_successors(row.get("next")),
                    "volta": row.get("volta"),
                    # Measure tables state the offset in whole notes.
                    "offset_within_measure": (
                        Fraction(str(offset)) * 4
                        if offset not in (None, "")
                        else Fraction(0)
                    ),
                }
            )
        return cls.from_expanded_rows(expanded)

    @classmethod
    def from_dict(cls, data: list[dict[str, Any]]) -> MeasureMap:
        """Restore a map from compressed Measure Map rows.

        This is a manifest's ``measure_map``: a non-empty list of rows, each
        carrying ``count`` or ``ID``, with a non-negative ``qstamp`` and a
        strictly positive ``actual_length`` and ``nominal_length`` where it
        states them. A row's keys outside the row form (a ``tempo`` among
        them, which a document cannot hold as a :class:`Tempo`) are kept and
        written back in that measure's row.

        Args:
            data: JSON-safe list of compressed Measure Map rows.

        Returns:
            The restored immutable measure map.

        Raises:
            ValueError: If *data* is not a non-empty row list, a row carries
                neither ``count`` nor ``ID``, states a negative ``qstamp`` or
                a length that is not positive, or breaks a rule of
                :meth:`expand_rows`.
        """
        if not isinstance(data, list) or not data:
            raise ValueError(
                "MeasureMap document form must be a non-empty list of Measure Map rows"
            )
        for position, row in enumerate(data, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"Measure Map row {position} is not an object")
            if "count" not in row and "ID" not in row:
                raise ValueError(
                    f"Measure Map row {position} carries neither count nor ID"
                )
            qstamp = _row_quarters(row, "qstamp")
            if qstamp is not None and qstamp < 0:
                raise ValueError(
                    f"Measure Map row {position} states a negative qstamp {qstamp}"
                )
            for key in ("actual_length", "nominal_length"):
                length = _row_quarters(row, key)
                if length is not None and length <= 0:
                    raise ValueError(
                        f"Measure Map row {position} states {key} {length}, which "
                        "is not positive"
                    )
        return cls.from_rows(data, keep_unknown_keys=True)

    @classmethod
    def expand_rows(
        cls, rows: Iterable[Mapping[str, Any]], *, keep_unknown_keys: bool = False
    ) -> list[dict[str, Any]]:
        """Expand compressed Measure Map rows into canonical full rows.

        Rows are in their document form: quarter-note values are JSON numbers
        or Fraction pairs (``number_from_json``), ``next`` is a list of row
        ``ID`` strings or ``count`` integers. Omitted rows are restored as
        default successors of the row before them (count, number and qstamp
        advance, actual length becomes the nominal length, the name is the
        number, repeats clear, ``volta`` and ``offset_within_measure`` are
        unset, ``next`` is the following row, none for the last row).

        Args:
            rows: Compressed or full rows in Measure Map order, optionally
                carrying the in-memory ``tempo`` extension.
            keep_unknown_keys: Keep every key outside the row form, ``tempo``
                included, on its expanded row under ``"unknown_keys"``
                instead of refusing it: the document form of a manifest.

        Returns:
            Full rows with exact quarter-note values, ``ID`` strings and
            ``next`` as row counts (``[]`` for the last row by default).

        Raises:
            ValueError: If a row uses an unknown key (unless kept), a number in another
                form, a count that does not increase, an ``ID`` stated twice,
                a ``next`` naming no row, a non-positive offset, or contradicts
                derived signatures.

        Warns:
            UserWarning: If a stated qstamp disagrees with the derived one;
                the derived qstamp is kept.
        """
        allowed = set(_ROW_KEYS) if keep_unknown_keys else {*_ROW_KEYS, "tempo"}
        source = [dict(row) for row in rows]
        for row in source:
            unknown = sorted(set(row) - allowed)
            if unknown and not keep_unknown_keys:
                raise ValueError(f"Unknown Measure Map row key {unknown[0]!r}")
            if (
                "tempo" in allowed
                and "tempo" in row
                and not isinstance(row["tempo"], Tempo)
            ):
                raise ValueError("Measure Map tempo must be a Tempo instance")
            if "ID" in row and not isinstance(row["ID"], str):
                raise ValueError(f"Measure Map ID {row['ID']!r} is not a string")
            if "next" in row and not isinstance(row["next"], list):
                raise ValueError(f"Measure Map next {row['next']!r} is not a list")
        if not source:
            return []

        fill_key = "count" if any("count" in row for row in source) else "number"
        sparse: list[dict[str, Any]] = []
        previous_fill: int | None = None
        previous_number: int | None = None
        inherited_signature: str | None = None
        inherited_nominal: Fraction | None = None

        for stated in source:
            row = dict(stated)
            raw_fill = row.get(fill_key)
            if raw_fill is None:
                if previous_fill is None:
                    nominal_probe = _row_quarters(row, "nominal_length")
                    if nominal_probe is None:
                        nominal_probe = _signature_length(row.get("time_signature"))
                    actual_probe = _row_quarters(row, "actual_length")
                    fill = 1
                    if (
                        fill_key == "number"
                        and actual_probe is not None
                        and nominal_probe is not None
                        and actual_probe < nominal_probe
                    ):
                        fill = 0
                else:
                    fill = previous_fill + 1
            else:
                fill = int(raw_fill)
                if (
                    fill_key == "count"
                    and previous_fill is not None
                    and fill <= previous_fill
                ):
                    raise ValueError(
                        f"Measure Map count {fill} does not increase after count "
                        f"{previous_fill}"
                    )

            if previous_fill is not None and fill > previous_fill + 1:
                for missing in range(previous_fill + 1, fill):
                    inserted: dict[str, Any] = {fill_key: missing}
                    if previous_number is not None:
                        inserted["number"] = previous_number + (missing - previous_fill)
                    if inherited_signature is not None:
                        inserted["time_signature"] = inherited_signature
                    if inherited_nominal is not None:
                        inserted["nominal_length"] = number_to_json(inherited_nominal)
                        inserted["actual_length"] = number_to_json(inherited_nominal)
                    sparse.append(inserted)
            row.setdefault(fill_key, fill)
            sparse.append(row)
            previous_fill = fill
            if row.get("number") is not None:
                previous_number = int(row["number"])

            if "time_signature" in row:
                inherited_signature = row["time_signature"]
            stated_nominal = _row_quarters(row, "nominal_length")
            if stated_nominal is not None:
                inherited_nominal = stated_nominal
            elif inherited_signature is not None:
                inherited_nominal = _signature_length(inherited_signature)

        expanded: list[dict[str, Any]] = []
        current_signature: str | None = None
        qstamp: Fraction | None = Fraction(0)
        for position, stated in enumerate(sparse, start=1):
            if "count" in stated and int(stated["count"]) != position:
                raise ValueError(
                    f"Measure Map count {stated['count']} disagrees with "
                    f"expanded position {position}"
                )
            if "time_signature" in stated:
                current_signature = stated["time_signature"]
            signature = current_signature
            cadenza = _is_cadenza(signature)

            nominal = _row_quarters(stated, "nominal_length")
            actual = _row_quarters(stated, "actual_length")
            if nominal is None and cadenza:
                if actual is None:
                    raise ValueError("A cadenza measure must state actual_length")
                nominal = actual
            elif nominal is None:
                nominal = _signature_length(signature)
            elif signature is not None and not cadenza:
                signature_length = _signature_length(signature)
                if nominal != signature_length:
                    row_number = stated.get("number", stated.get(fill_key, position))
                    raise ValueError(
                        f"Measure Map row number {row_number} states signature "
                        f"{signature!r} with length {signature_length}, but states "
                        f"nominal_length {nominal}"
                    )
            if actual is None:
                actual = nominal
            offset = _row_quarters(stated, "offset_within_measure")
            if offset is not None and offset <= 0:
                raise ValueError(
                    f"Measure Map offset_within_measure {offset} at count {position} "
                    "is not positive; a measure at offset 0 omits the key"
                )

            if "number" in stated:
                number = int(stated["number"])
            elif (
                fill_key == "count"
                and position == 1
                and actual is not None
                and nominal is not None
                and actual < nominal
            ):
                number = 0
            else:
                number = int(stated.get(fill_key, position))
            row = {
                "ID": stated.get("ID", str(position)),
                "count": position,
                "qstamp": _reconciled_qstamp(
                    _row_quarters(stated, "qstamp"),
                    qstamp,
                    stated.get("ID", str(position)),
                ),
                "number": number,
                "name": str(stated.get("name", number)),
                "time_signature": signature,
                "nominal_length": nominal,
                "actual_length": actual,
                "start_repeat": bool(stated.get("start_repeat", False)),
                "end_repeat": bool(stated.get("end_repeat", False)),
                "next": stated.get("next"),
                "volta": stated.get("volta"),
                "offset_within_measure": offset or Fraction(0),
            }
            if "tempo" in allowed and "tempo" in stated:
                row["tempo"] = stated["tempo"]
            unknown_keys = {
                key: deepcopy(value)
                for key, value in stated.items()
                if key not in allowed
            }
            if unknown_keys:
                row["unknown_keys"] = unknown_keys
            expanded.append(row)
            qstamp = None if qstamp is None or actual is None else qstamp + actual

        counts_by_id: dict[str, int] = {}
        for row in expanded:
            if row["ID"] in counts_by_id:
                raise ValueError(f"Measure Map ID {row['ID']!r} is stated twice")
            counts_by_id[row["ID"]] = row["count"]
        for index, row in enumerate(expanded):
            if row["next"] is None:
                row["next"] = [index + 2] if index + 1 < len(expanded) else []
                continue
            successors: list[int] = []
            for entry in row["next"]:
                if isinstance(entry, str):
                    if entry not in counts_by_id:
                        raise ValueError(
                            f"Measure Map next {entry!r} at count {row['count']} "
                            "names no row ID"
                        )
                    successors.append(counts_by_id[entry])
                elif isinstance(entry, int) and not isinstance(entry, bool):
                    if not 1 <= entry <= len(expanded):
                        raise ValueError(
                            f"Measure Map next {entry} at count {row['count']} lies "
                            f"beyond the last row (count {len(expanded)})"
                        )
                    successors.append(entry)
                else:
                    raise ValueError(
                        f"Measure Map next {entry!r} is neither an ID nor a count"
                    )
            row["next"] = successors
        return expanded

    @classmethod
    def from_rows(
        cls, rows: Iterable[Mapping[str, Any]], *, keep_unknown_keys: bool = False
    ) -> MeasureMap:
        """Build a measure map from Measure Map authoring rows.

        Args:
            rows: Compressed or full rows in their document form, without
                tempo extensions.
            keep_unknown_keys: Keep each row's keys outside the row form,
                written back in that measure's row, instead of refusing them.

        Returns:
            The exact immutable measure map represented by the rows.

        Raises:
            ValueError: If rows are invalid or contain tempo extensions, which
                require :meth:`MetricHierarchy.from_rows`.
        """
        expanded = cls.expand_rows(rows, keep_unknown_keys=keep_unknown_keys)
        if any("tempo" in row for row in expanded):
            raise ValueError(
                "Rows with tempo must be read by MetricHierarchy.from_rows"
            )
        return cls.from_expanded_rows(expanded)

    @classmethod
    def from_expanded_rows(cls, rows: Iterable[Mapping[str, Any]]) -> MeasureMap:
        """Build measures from full rows whose ``next`` holds row counts.

        Every lane ends here: rows from :meth:`expand_rows`, which a loader
        may complete first, and rows read from a score loader's measure
        table. Each measure is classified by :meth:`Measure.from_values`;
        ``next`` counts become row IDs. A row's ``"unknown_keys"`` are kept
        for that measure's row.

        Args:
            rows: Full rows with the keys :meth:`expand_rows` returns.

        Returns:
            The measure map the rows describe.

        Raises:
            ValueError: If a ``next`` count names no row.
        """
        source = list(rows)
        ids_by_count = {row["count"]: row["ID"] for row in source}
        measures: list[Measure] = []
        for row in source:
            successors: tuple[str, ...] | None = None
            if row["next"] is not None:
                unknown = [count for count in row["next"] if count not in ids_by_count]
                if unknown:
                    raise ValueError(
                        f"Measure {row['ID']!r} names successor count {unknown[0]}, "
                        "which no measure has"
                    )
                successors = tuple(ids_by_count[count] for count in row["next"])
            measures.append(
                Measure.from_values(
                    id=row["ID"],
                    count=row["count"],
                    qstamp=row["qstamp"],
                    number=row["number"],
                    name=row["name"],
                    time_signature=row["time_signature"],
                    nominal_length=row["nominal_length"],
                    actual_length=row["actual_length"],
                    start_repeat=row["start_repeat"],
                    end_repeat=row["end_repeat"],
                    next=successors,
                    volta=row["volta"],
                    offset_within_measure=row["offset_within_measure"],
                )
            )
        measure_map = cls(measures)
        measure_map._unknown_row_keys = {
            row["ID"]: row["unknown_keys"] for row in source if row.get("unknown_keys")
        }
        return measure_map

    @classmethod
    def _from_normalized(cls, measures: Iterable[Measure]) -> MeasureMap:
        """Build a view over measures already normalized by a parent map."""
        instance = cls.__new__(cls)
        instance._measures = tuple(measures)
        return instance

    def __init__(self, measures: Iterable[Measure]) -> None:
        """Normalize an immutable sequence of measures.

        Args:
            measures: Measures in printed order.

        Raises:
            ValueError: If measure identifiers are not unique, or a ``next``
                names no measure of the map.

        Warns:
            UserWarning: If a stated count or qstamp disagrees with the value
                derived from printed order and preceding actual lengths.
        """
        source = tuple(measures)
        ids = [
            measure.id or str(count) for count, measure in enumerate(source, start=1)
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("Measure IDs must be unique within a MeasureMap")
        known = set(ids)
        normalized: list[Measure] = []
        running: Fraction | None = Fraction(0)
        for count, (measure_id, measure) in enumerate(zip(ids, source), start=1):
            if measure.count is not None and measure.count != count:
                warnings.warn(
                    f"Measure {measure_id!r} supplies count {measure.count}, "
                    f"but printed order computes {count}",
                    stacklevel=2,
                )
            successors = measure.next
            if successors is None:
                successors = (ids[count],) if count < len(ids) else ()
            unknown = [successor for successor in successors if successor not in known]
            if unknown:
                raise ValueError(
                    f"Measure {measure_id!r} names successor {unknown[0]!r}, "
                    "which no measure of the map has"
                )
            normalized.append(
                measure.model_copy(
                    update={
                        "id": measure_id,
                        "count": count,
                        "qstamp": _reconciled_qstamp(
                            measure.qstamp, running, measure_id
                        ),
                        "next": successors,
                    }
                )
            )
            if running is not None:
                running = (
                    running + measure.actual_length
                    if measure.actual_length is not None
                    else None
                )
        self._measures = tuple(normalized)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MeasureMap):
            return NotImplemented
        return len(self) == len(other) and all(
            type(left) is type(right) and left == right
            for left, right in zip(self, other, strict=True)
        )

    def __repr__(self) -> str:
        total = self.total_actual_length
        length = "abstract" if total is None else str(total)
        signatures = tuple(
            dict.fromkeys(
                measure.time_signature
                for measure in self._measures
                if measure.time_signature is not None
            )
        )
        return (
            f"MeasureMap(measures={len(self)}, actual_length={length}, "
            f"signatures={signatures!r})"
        )

    def __iter__(self) -> Iterator[Measure]:
        return iter(self._measures)

    def __len__(self) -> int:
        return len(self._measures)

    def __getitem__(self, index: int | slice) -> Measure | tuple[Measure, ...]:
        return self._measures[index]

    @property
    def measures(self) -> tuple[Measure, ...]:
        """Measures in printed order."""
        return self._measures

    @property
    def total_actual_length(self) -> Fraction | None:
        """Sum of actual lengths, or ``None`` when any length is abstract."""
        if any(measure.actual_length is None for measure in self._measures):
            return None
        return sum(
            (
                measure.actual_length
                for measure in self._measures
                if measure.actual_length is not None
            ),
            Fraction(0),
        )

    def by_id(self, measure_id: str) -> Measure:
        """Return the uniquely identified measure.

        Args:
            measure_id: Identifier to locate.

        Returns:
            The measure with that identifier.

        Raises:
            KeyError: If no measure has the requested identifier.
        """
        for measure in self._measures:
            if measure.id == measure_id:
                return measure
        raise KeyError(measure_id)

    def to_rows(self, *, compress: bool = True) -> list[dict[str, Any]]:
        """Return Measure Map rows, omitting entries equal to their default successor.

        A row holds every value its measure states, ``number`` (and ``name``)
        falling back to the count where a measure states none: ``ID`` only when
        it is not ``str(count)``, ``next`` as row IDs (on the last row only when it names
        a successor), ``volta`` when set, and ``offset_within_measure`` only
        when it is non-zero; then the keys outside the row form that the
        measure's row was read with. The first and the last rows are always
        retained, and so is every row read with such keys. Every other row is
        omitted when it equals the default successor of the preceding
        measure: count increments, the ID is ``str(count)``, qstamp
        advances by actual_length, actual_length becomes nominal_length, number
        increments, name becomes that number, signatures and nominal lengths
        persist, repeats clear, ``volta`` and ``offset_within_measure`` are
        unset, and ``next`` names the following measure.

        Args:
            compress: Whether to omit default-successor rows.

        Returns:
            JSON-safe compressed or full Measure Map rows.

        Raises:
            ValueError: If the map holds a ``SplitRegularMeasure`` or
                ``SplitIrregularMeasure``, which no row can express; nothing
                is returned.
        """
        measures = self._measures
        for measure in measures:
            if isinstance(measure, (SplitRegularMeasure, SplitIrregularMeasure)):
                raise ValueError(
                    f"Measure {measure.id!r} is a {type(measure).__name__}, which "
                    "the Measure Map row form cannot express"
                )
        last = len(measures) - 1

        def encoded(values: dict[str, Any]) -> dict[str, Any]:
            return {
                key: (
                    number_to_json(values[key]) if key in _QUARTER_KEYS else values[key]
                )
                for key in _ROW_KEYS
                if values.get(key) is not None
            }

        def row_for(index: int) -> dict[str, Any]:
            measure = measures[index]
            number = measure.number if measure.number is not None else measure.count
            offset = getattr(measure, "offset_within_measure", Fraction(0))
            successors = list(measure.next or ())
            return encoded(
                {
                    "ID": None if measure.id == str(measure.count) else measure.id,
                    "count": measure.count,
                    "qstamp": measure.qstamp,
                    "number": number,
                    "name": measure.name if measure.name is not None else str(number),
                    "time_signature": measure.time_signature,
                    "nominal_length": measure.nominal_length,
                    "actual_length": measure.actual_length,
                    "start_repeat": measure.start_repeat,
                    "end_repeat": measure.end_repeat,
                    "next": successors if index < last or successors else None,
                    "volta": measure.volta,
                    "offset_within_measure": offset or None,
                }
            )

        def default_successor(index: int) -> dict[str, Any]:
            """Return the default successor row of the measure at *index*."""
            measure = measures[index]
            successor = row_for(index)
            for key in ("ID", "volta", "offset_within_measure"):
                successor.pop(key, None)
            if measure.count is not None:
                successor["count"] = measure.count + 1
            if measure.qstamp is not None and measure.actual_length is not None:
                successor["qstamp"] = number_to_json(
                    measure.qstamp + measure.actual_length
                )
            if measure.actual_length is not None:
                successor.pop("actual_length")
                if measure.nominal_length is not None:
                    successor["actual_length"] = number_to_json(measure.nominal_length)
            if measure.number is not None:
                successor["number"] = measure.number + 1
            if measure.name is not None and measure.number is not None:
                successor["name"] = str(measure.number + 1)
            successor["start_repeat"] = False
            successor["end_repeat"] = False
            successor["next"] = [measures[index + 2].id]
            return successor

        unknown = self._unknown_row_keys
        return [
            {**row_for(index), **deepcopy(dict(unknown.get(measures[index].id, {})))}
            for index in range(len(measures))
            if not compress
            or index in (0, last)
            or measures[index].id in unknown
            or row_for(index) != default_successor(index - 1)
        ]

    def to_dict(self) -> list[dict[str, Any]]:
        """Return this map as compressed JSON-safe Measure Map rows.

        Returns:
            The compressed Measure Map row list.
        """
        return self.to_rows()


class SectionHierarchy:
    """A section partition whose leaves carry immutable measure-map segments."""

    @classmethod
    def from_measures(
        cls,
        source: (
            Iterable[Measure]
            | Iterable[Iterable[Measure]]
            | Mapping[str, Any]
            | MeasureMap
        ),
    ) -> SectionHierarchy:
        """Build sections from flat, nested, mapped, or already-mapped measures.

        Args:
            source: Measures, groups of measures, named groups, or an existing
                measure map.

        Returns:
            A hierarchy whose leaves retain the requested grouping.
        """
        if isinstance(source, MeasureMap):
            return cls.from_measure_map(source)
        elif isinstance(source, Mapping):
            named_groups = [(str(name), tuple(group)) for name, group in source.items()]
        else:
            values = tuple(source)
            if not values or isinstance(values[0], Measure):
                named_groups = [(None, values)]
            else:
                named_groups = [(None, tuple(group)) for group in values]
        return cls._from_groups(named_groups)

    @classmethod
    def from_measure_map(
        cls,
        measure_map: MeasureMap,
        *,
        counts: Iterable[int] | None = None,
    ) -> SectionHierarchy:
        """Partition an existing map without rebuilding its measures.

        Args:
            measure_map: The map retained as the hierarchy's whole-work map.
            counts: Consecutive leaf sizes. ``None`` creates one leaf.

        Returns:
            A hierarchy whose ``measure_map`` is the supplied object.

        Raises:
            ValueError: If the counts do not sum to the map length.
        """
        section_counts = (len(measure_map),) if counts is None else tuple(counts)
        if any(count < 0 for count in section_counts) or sum(section_counts) != len(
            measure_map
        ):
            raise ValueError(
                f"Section counts must sum to MeasureMap length {len(measure_map)}, "
                f"got {section_counts}"
            )
        leaves: list[AtomicSection] = []
        start = 0
        for index, count in enumerate(section_counts, start=1):
            end = start + count
            leaves.append(
                AtomicSection(
                    id=f"sec{index}",
                    mc_start=start + 1,
                    mc_end=end + 1,
                    measure_map=MeasureMap._from_normalized(
                        measure_map.measures[start:end]
                    ),
                )
            )
            start = end
        return cls(measure_map, leaves)

    @classmethod
    def from_measure_counts(
        cls,
        source: int | Iterable[int] | Mapping[str, int] | Iterable[tuple[str, int]],
    ) -> SectionHierarchy:
        """Build sections containing the requested counts of abstract measures.

        Args:
            source: One count, ordered counts, named counts, or name-count pairs.

        Returns:
            A hierarchy of abstract measures with those consecutive leaf sizes.
        """
        if isinstance(source, int):
            entries: list[tuple[str | None, int]] = [(None, source)]
        elif isinstance(source, Mapping):
            entries = [(str(name), int(count)) for name, count in source.items()]
        else:
            consumed = tuple(source)
            entries = []
            for entry in consumed:
                if isinstance(entry, tuple):
                    name, count = entry
                    entries.append((str(name), int(count)))
                else:
                    entries.append((None, int(entry)))
        return cls._from_groups(
            [(name, tuple(Measure() for _ in range(count))) for name, count in entries]
        )

    @classmethod
    def _from_groups(
        cls, groups: Iterable[tuple[str | None, Iterable[Measure]]]
    ) -> SectionHierarchy:
        materialized = [(name, tuple(measures)) for name, measures in groups]
        flat = MeasureMap(
            measure for _, measures in materialized for measure in measures
        )
        leaves: list[AtomicSection] = []
        start = 0
        for index, (name, measures) in enumerate(materialized, start=1):
            end = start + len(measures)
            segment = MeasureMap._from_normalized(flat.measures[start:end])
            leaves.append(
                AtomicSection(
                    id=f"sec{index}",
                    mc_start=start + 1,
                    mc_end=end + 1,
                    measure_map=segment,
                    name=name,
                )
            )
            start = end
        return cls(flat, leaves)

    def __init__(
        self, measure_map: MeasureMap, sections: Iterable[AtomicSection]
    ) -> None:
        """Store a whole-work map and its ordered section leaves.

        Args:
            measure_map: Whole-work measure map.
            sections: Consecutive atomic section leaves.
        """
        self._measure_map = measure_map
        self._sections = tuple(sections)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SectionHierarchy):
            return NotImplemented
        section_shape_matches = [section.n_measures for section in self._sections] == [
            section.n_measures for section in other._sections
        ]
        return (
            section_shape_matches
            and self._measure_map.measures == other._measure_map.measures
        )

    @property
    def measure_map(self) -> MeasureMap:
        """The whole work's measure map."""
        return self._measure_map

    @property
    def n_sections(self) -> int:
        """Number of leaf sections."""
        return len(self._sections)

    @property
    def n_measures(self) -> int:
        """Total number of measures."""
        return len(self._measure_map)

    @property
    def sections(self) -> list[AtomicSection]:
        """Leaf sections in printed order."""
        return list(self._sections)


def _coordinate_to_dict(value: Coordinate) -> dict[str, Any]:
    """Encode a coordinate through the shared typed dictionary contract."""
    return dict(coordinate_to_dict(value))


def _coordinate_from_dict(value: Mapping[str, Any]) -> Coordinate:
    """Restore a coordinate through the shared typed dictionary contract."""
    return coordinate_from_dict(value)


def _unsupported_json_type(value: Any, seen: set[int] | None = None) -> type | None:
    """Return the first type that the standard JSON encoder cannot carry."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return None
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return type(value)
    seen.add(identity)
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                return type(key)
            unsupported = _unsupported_json_type(child, seen)
            if unsupported is not None:
                return unsupported
        return None
    if isinstance(value, (list, tuple)):
        for child in value:
            unsupported = _unsupported_json_type(child, seen)
            if unsupported is not None:
                return unsupported
        return None
    return type(value)


def _require_physical_unit(unit: TimeUnit, member: str) -> None:
    """Refuse an observation coordinate that is not on the physical axis.

    Args:
        unit: Unit of the coordinate given for that member.
        member: Name of the ``TempoEntry`` member being validated.

    Raises:
        ValueError: If *unit* does not belong to the physical domain.
    """
    if unit.domain is Domain.physical:
        return
    allowed = ", ".join(
        sorted(
            candidate.value
            for candidate in TimeUnit
            if candidate.domain is Domain.physical
        )
    )
    raise ValueError(
        f"TempoEntry {member} unit {unit.value!r} is not physical; "
        f"allowed units: {allowed}"
    )


def _freeze_json(value: Any) -> Any:
    """Return an immutable copy of a JSON-compatible value."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Return a JSON-encodable copy of an immutable provenance value."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


class TempoEntry(BaseModel):
    """One positioned symbolic-axis row in a tempo map.

    Attributes:
        at: Exact symbolic position stated by the source.
        tempo: Optional rate law stated at that position.
        observed: Optional physical point or support interval.
        observation_undefined: Whether the source explicitly states no onset.
        alternatives: Weighted alternative physical positions.
        is_interpolated: Whether the symbolic position was derived.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    at: Coordinate
    tempo: Tempo | None = None
    observed: Coordinate | Interval | None = None
    observation_undefined: bool = False
    alternatives: tuple[tuple[Coordinate, Fraction], ...] = ()
    is_interpolated: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TempoEntry:
        """Restore a tempo entry from its JSON dictionary representation.

        Args:
            data: JSON-safe tempo-entry dictionary.

        Returns:
            The restored tempo entry.

        Raises:
            ValueError: If an observed payload has an unknown type.
        """
        observed_data = data.get("observed")
        observed: Coordinate | Interval | None
        if observed_data is None:
            observed = None
        elif observed_data["type"] == "coordinate":
            observed = _coordinate_from_dict(observed_data["value"])
        elif observed_data["type"] == "interval":
            observed = Interval(
                start=_coordinate_from_dict(observed_data["start"]),
                end=_coordinate_from_dict(observed_data["end"]),
            )
        else:
            raise ValueError(
                f"Unknown tempo observation type {observed_data['type']!r}"
            )
        return cls(
            at=_coordinate_from_dict(data["at"]),
            tempo=(
                None if data.get("tempo") is None else Tempo.from_dict(data["tempo"])
            ),
            observed=observed,
            observation_undefined=bool(data.get("observation_undefined", False)),
            alternatives=tuple(
                (
                    _coordinate_from_dict(item["at"]),
                    Fraction(number_from_json(item["weight"])),
                )
                for item in data.get("alternatives", [])
            ),
            is_interpolated=bool(data.get("is_interpolated", False)),
        )

    @field_validator("at")
    @classmethod
    def _validate_symbolic_position(cls, value: Coordinate) -> Coordinate:
        if value.unit.domain is Domain.logical:
            return value
        allowed = ", ".join(
            sorted(unit.value for unit in TimeUnit if unit.domain is Domain.logical)
        )
        raise ValueError(
            f"TempoEntry at unit {value.unit.value!r} is not symbolic; "
            f"allowed units: {allowed}"
        )

    @field_validator("observed")
    @classmethod
    def _validate_physical_observation(
        cls, value: Coordinate | Interval | None
    ) -> Coordinate | Interval | None:
        if isinstance(value, Interval):
            _require_physical_unit(value.start.unit, "observed")
            _require_physical_unit(value.end.unit, "observed")
        elif value is not None:
            _require_physical_unit(value.unit, "observed")
        return value

    @field_validator("alternatives", mode="after")
    @classmethod
    def _validate_physical_alternatives(
        cls, value: tuple[tuple[Coordinate, Fraction], ...]
    ) -> tuple[tuple[Coordinate, Fraction], ...]:
        for at, _weight in value:
            _require_physical_unit(at.unit, "alternative")
        return value

    @field_validator("alternatives", mode="before")
    @classmethod
    def _coerce_and_validate_weights(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        normalized: list[tuple[object, Fraction]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                return value
            at, weight = item
            if (
                isinstance(weight, bool)
                or not isinstance(weight, Real)
                or weight <= 0
                or weight > 1
                or not isfinite(float(weight))
            ):
                raise ValueError(
                    "TempoEntry alternative weight must be a finite real number "
                    "greater than zero and at most one"
                )
            normalized.append((at, Fraction(weight)))
        return tuple(normalized)

    @model_validator(mode="after")
    def _validate_observation_state(self) -> TempoEntry:
        if self.observation_undefined and self.observed is not None:
            raise ValueError(
                "TempoEntry observation_undefined cannot be true when observed is set"
            )
        return self

    def __repr__(self) -> str:
        parts: list[str] = []
        parts.append(f"at={self.at!r}")
        if self.tempo is not None:
            parts.append(f"tempo={self.tempo!r}")
        if self.observed is not None:
            parts.append(f"observed={self.observed!r}")
        if self.observation_undefined:
            parts.append("observation_undefined=True")
        if self.alternatives:
            parts.append(f"alternatives={self.alternatives!r}")
        if self.is_interpolated:
            parts.append("is_interpolated=True")
        return f"TempoEntry({', '.join(parts)})"

    def to_dict(self) -> dict[str, Any]:
        """Return this entry as a JSON-safe dictionary.

        Returns:
            A dictionary retaining typed coordinates and exact tempo values.
        """
        observed: dict[str, Any] | None
        if self.observed is None:
            observed = None
        elif isinstance(self.observed, Interval):
            observed = {
                "type": "interval",
                "start": _coordinate_to_dict(self.observed.start),
                "end": _coordinate_to_dict(self.observed.end),
            }
        else:
            observed = {
                "type": "coordinate",
                "value": _coordinate_to_dict(self.observed),
            }
        return {
            "at": _coordinate_to_dict(self.at),
            "tempo": None if self.tempo is None else self.tempo.to_dict(),
            "observed": observed,
            "observation_undefined": self.observation_undefined,
            "alternatives": [
                {"at": _coordinate_to_dict(at), "weight": number_to_json(weight)}
                for at, weight in self.alternatives
            ],
            "is_interpolated": self.is_interpolated,
        }


class TempoMap(BaseModel):
    """An independent sequence of tempo facts from one source.

    The ``steady`` kind declares that entries are anchor-and-rate generators
    which abut; consumers use the kind rather than the producing format's
    name to select that rule.

    Attributes:
        kind: Behaviour of the map's entries.
        entries: Tempo facts in source order.
        provenance: Immutable JSON-compatible source metadata.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    kind: Literal["tempo_indication", "steady", "observed", "modelled"]
    entries: tuple[TempoEntry, ...]
    provenance: Mapping[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TempoMap:
        """Restore a tempo map from its JSON dictionary representation.

        Args:
            data: JSON-safe tempo-map dictionary.

        Returns:
            The restored tempo map.
        """
        return cls(
            kind=data["kind"],
            entries=tuple(TempoEntry.from_dict(entry) for entry in data["entries"]),
            provenance=dict(data["provenance"]),
        )

    @field_validator("provenance", mode="after")
    @classmethod
    def _validate_and_freeze_provenance(
        cls, value: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        for key, item in value.items():
            unsupported = _unsupported_json_type(item)
            try:
                json.dumps(item, allow_nan=False)
            except (TypeError, ValueError):
                value_type = unsupported or type(item)
                raise ValueError(
                    f"TempoMap provenance key {key!r} has a non-JSON-serializable "
                    f"value of type {value_type.__name__}"
                ) from None
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )

    def __repr__(self) -> str:
        parts = [f"kind={self.kind!r}", f"entries={len(self.entries)}"]
        if self.provenance:
            parts.append(f"provenance={_thaw_json(self.provenance)!r}")
        return f"TempoMap({', '.join(parts)})"

    def to_dict(self) -> dict[str, Any]:
        """Return this map as a JSON-safe dictionary.

        Returns:
            A dictionary retaining entries and provenance unchanged.
        """
        return {
            "kind": self.kind,
            "entries": [entry.to_dict() for entry in self.entries],
            "provenance": _thaw_json(self.provenance),
        }


class MetricNode(BaseModel):
    """A metrical timespan with alternative attributed partitions.

    ``Coordinate`` canonicalizes logical values as exact fractions, so the
    interval endpoints are exact by construction.

    Attributes:
        level: Required metrical level, with pulse at zero and groove below.
        span: Exact half-open span on one logical axis.
        expansions: Alternative partitions of this span.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    level: int
    span: Interval
    expansions: tuple[Expansion, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricNode:
        """Restore a metric node from its JSON dictionary representation.

        Args:
            data: JSON-safe metric-node dictionary.

        Returns:
            The restored node and all descendant expansions.
        """
        return cls(
            level=data["level"],
            span=Interval(
                start=_coordinate_from_dict(data["span"]["start"]),
                end=_coordinate_from_dict(data["span"]["end"]),
            ),
            expansions=tuple(
                Expansion.from_dict(expansion) for expansion in data["expansions"]
            ),
        )

    @model_validator(mode="after")
    def _validate_span_and_expansions(self) -> MetricNode:
        if self.span.unit.domain is not Domain.logical:
            raise ValueError("MetricNode span must use a logical unit")
        if self.span.start.value >= self.span.end.value:
            raise ValueError(f"MetricNode span {self.span} must have positive length")
        for expansion in self.expansions:
            cursor = self.span.start
            for child in expansion.children:
                if child.level >= self.level:
                    raise ValueError(
                        f"Child {child.span} level {child.level} is not lower than "
                        f"parent {self.span} level {self.level}"
                    )
                if child.span.start != cursor:
                    raise ValueError(
                        f"Parent span {self.span} is not covered at child {child.span}"
                    )
                cursor = child.span.end
            if not expansion.children or cursor != self.span.end:
                offending = (
                    expansion.children[-1].span
                    if expansion.children
                    else "empty expansion"
                )
                raise ValueError(
                    f"Parent span {self.span} is not covered by child {offending}"
                )
        return self

    def __repr__(self) -> str:
        return (
            f"MetricNode(level={self.level}, span={self.span!s}, "
            f"expansions={len(self.expansions)})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self) -> dict[str, Any]:
        """Return this node as a JSON-safe recursive dictionary.

        Returns:
            A dictionary retaining its exact typed span and all alternatives.
        """
        return {
            "level": self.level,
            "span": {
                "start": _coordinate_to_dict(self.span.start),
                "end": _coordinate_to_dict(self.span.end),
            },
            "expansions": [expansion.to_dict() for expansion in self.expansions],
        }


class Expansion(BaseModel):
    """One partition of a metric node, attributed to readings.

    Attributes:
        children: Ordered child timespans forming the partition.
        readings: Reading identifiers which assert the partition.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    children: tuple[MetricNode, ...]
    readings: frozenset[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Expansion:
        """Restore an expansion from its JSON dictionary representation.

        Args:
            data: JSON-safe expansion dictionary.

        Returns:
            The restored attributed partition.
        """
        return cls(
            children=tuple(MetricNode.from_dict(child) for child in data["children"]),
            readings=frozenset(data["readings"]),
        )

    def __repr__(self) -> str:
        return (
            f"Expansion(children={len(self.children)}, "
            f"readings={sorted(self.readings)!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return this expansion as a JSON-safe dictionary.

        Returns:
            A dictionary retaining children and sorted reading identifiers.
        """
        return {
            "children": [child.to_dict() for child in self.children],
            "readings": sorted(self.readings),
        }


class Reading(BaseModel):
    """Provenance and vocabulary for one consistent forest reading.

    Attributes:
        id: Identifier used by attributed expansions.
        source: Epistemic source of the reading.
        by: Optional author or system name.
        reinterprets: Optional identifier of a prior reading.
        level_names: Reading-local names for metrical levels.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    id: str
    source: Literal["notation", "performance", "analysis", "algorithm"]
    by: str | None = None
    reinterprets: str | None = None
    level_names: Mapping[int, str] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Reading:
        """Restore reading provenance from its JSON dictionary representation.

        Args:
            data: JSON-safe reading dictionary.

        Returns:
            The restored reading.
        """
        return cls(
            id=data["id"],
            source=data["source"],
            by=data.get("by"),
            reinterprets=data.get("reinterprets"),
            level_names={
                int(level): name for level, name in data["level_names"].items()
            },
        )

    @field_validator("level_names", mode="after")
    @classmethod
    def _freeze_level_names(cls, value: Mapping[int, str]) -> Mapping[int, str]:
        return MappingProxyType(dict(value))

    def __repr__(self) -> str:
        parts = [f"id={self.id!r}", f"source={self.source!r}"]
        if self.by is not None:
            parts.append(f"by={self.by!r}")
        if self.reinterprets is not None:
            parts.append(f"reinterprets={self.reinterprets!r}")
        if self.level_names:
            parts.append(f"level_names={dict(self.level_names)!r}")
        return f"Reading({', '.join(parts)})"

    def to_dict(self) -> dict[str, Any]:
        """Return this reading as a JSON-safe dictionary.

        Returns:
            A dictionary retaining provenance and level vocabulary.
        """
        return {
            "id": self.id,
            "source": self.source,
            "by": self.by,
            "reinterprets": self.reinterprets,
            "level_names": {
                str(level): name for level, name in self.level_names.items()
            },
        }


MetricNode.model_rebuild()


def _metric_interval(start: Fraction, end: Fraction) -> Interval:
    """Construct one exact interval on the hierarchy's quarter-note axis."""
    return Interval(
        Coordinate(Fraction(start), TimeUnit.quarters),
        Coordinate(Fraction(end), TimeUnit.quarters),
    )


def _node_identity(node: MetricNode) -> tuple[int, Interval]:
    """Return the identity used to pack metric nodes."""
    return node.level, node.span


def _rational_gcd(values: Iterable[Fraction]) -> Fraction:
    """Return the greatest common divisor of positive rational values."""
    exact = tuple(Fraction(value) for value in values)
    denominator = lcm(*(value.denominator for value in exact))
    return Fraction(
        gcd(*(value.numerator * (denominator // value.denominator) for value in exact)),
        denominator,
    )


def _rational_lcm(first: Fraction, second: Fraction) -> Fraction:
    """Return the least common multiple of two positive rationals."""
    first = Fraction(first)
    second = Fraction(second)
    return Fraction(
        lcm(first.numerator, second.numerator),
        gcd(first.denominator, second.denominator),
    )


def _beat_quarters(tempo: Tempo | None) -> Fraction | None:
    """Return a tempo beat unit in quarters, when the tempo states one."""
    if tempo is None or tempo.beat is None:
        return None
    if tempo.beat.unit is TimeUnit.whole_note:
        return Fraction(tempo.beat.value) * 4
    if tempo.beat.unit is TimeUnit.quarters:
        return Fraction(tempo.beat.value)
    raise ValueError("A tempo indication beat must use whole_note or quarters")


def _generated_reading_id(beat: Fraction) -> str:
    """Return the stable ``generated_q_<numerator>_<denominator>`` id."""
    exact = Fraction(beat)
    return f"generated_q_{exact.numerator}_{exact.denominator}"


def _sort_nodes(
    nodes: Iterable[MetricNode], *, include_level: bool = False
) -> tuple[MetricNode, ...]:
    """Sort nodes in symbolic-axis order."""
    return tuple(
        sorted(
            nodes,
            key=lambda node: (
                node.span.start.value,
                -(node.span.end.value - node.span.start.value),
                *((node.level,) if include_level else ()),
            ),
        )
    )


def _flatten_nodes(roots: Iterable[MetricNode]) -> tuple[MetricNode, ...]:
    """Return each node identity once from a reading's trees."""
    found: dict[tuple[int, Interval], MetricNode] = {}

    def visit(node: MetricNode) -> None:
        found.setdefault(_node_identity(node), node)
        for expansion in node.expansions:
            for child in expansion.children:
                visit(child)

    for root in roots:
        visit(root)
    return _sort_nodes(found.values())


def _pack_trees(trees: Iterable[Iterable[MetricNode]]) -> tuple[MetricNode, ...]:
    """Pack per-reading trees by node identity and partition identity."""
    canonical: dict[tuple[int, Interval], MetricNode] = {}
    root_ids: set[tuple[int, Interval]] = set()
    child_ids: set[tuple[int, Interval]] = set()

    def merge(node: MetricNode, *, is_root: bool) -> None:
        identity = _node_identity(node)
        if is_root:
            root_ids.add(identity)
        for expansion in node.expansions:
            for child in expansion.children:
                child_ids.add(_node_identity(child))
                merge(child, is_root=False)
        existing = canonical.get(identity)
        if existing is None:
            canonical[identity] = node
            return
        alternatives = list(existing.expansions)
        for incoming in node.expansions:
            incoming_partition = tuple(
                _node_identity(child) for child in incoming.children
            )
            for index, present in enumerate(alternatives):
                present_partition = tuple(
                    _node_identity(child) for child in present.children
                )
                if present_partition == incoming_partition:
                    alternatives[index] = Expansion(
                        children=present.children,
                        readings=present.readings | incoming.readings,
                    )
                    break
            else:
                alternatives.append(incoming)
        canonical[identity] = MetricNode(
            level=existing.level,
            span=existing.span,
            expansions=tuple(alternatives),
        )

    for roots in trees:
        for root in roots:
            merge(root, is_root=True)

    def hydrate(identity: tuple[int, Interval]) -> MetricNode:
        node = canonical[identity]
        return MetricNode(
            level=node.level,
            span=node.span,
            expansions=tuple(
                Expansion(
                    children=tuple(
                        hydrate(_node_identity(child)) for child in expansion.children
                    ),
                    readings=expansion.readings,
                )
                for expansion in node.expansions
            ),
        )

    top = root_ids - child_ids
    return _sort_nodes((hydrate(identity) for identity in top), include_level=True)


class _MetricLevels(NamedTuple):
    """Levels assigned to one signature and tempo beat combination."""

    bar_level: int
    beat_level: int
    division_level: int | None
    subpulse: Fraction | None
    defines_own_reading: bool


@dataclass(frozen=True, eq=False, kw_only=True, slots=True)
class MetricHierarchy:
    """Metrical readings anchored on one exact measure map.

    Structural equality compares the derived packed forest. Tempo-map rates,
    text, and provenance remain outside equality, while a tempo indication's
    beat unit participates through the pulse level it identifies.

    Attributes:
        measure_map: Exact map whose quarter-note extents anchor every node.
        authored_readings: Provenance for explicitly supplied readings.
        authored_forest: Packed top-level nodes of explicitly supplied readings.
        tempo_maps: Independent tempo facts and physical observations.
    """

    __hash__: ClassVar[None] = None

    measure_map: MeasureMap
    authored_readings: tuple[Reading, ...] = ()
    authored_forest: tuple[MetricNode, ...] = ()
    tempo_maps: tuple[TempoMap, ...] = ()
    _memoized: dict[str, Any] = field(init=False, repr=False, compare=False)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, measure_map: MeasureMap
    ) -> MetricHierarchy:
        """Restore authored hierarchy data against its timeline-owned map.

        Args:
            data: JSON-safe authored hierarchy payload.
            measure_map: The owning timeline's restored measure map.

        Returns:
            The validated hierarchy with derived readings rebuilt.

        Raises:
            ValueError: If authored data contradicts the measure map.
        """
        return cls(
            measure_map=measure_map,
            authored_readings=tuple(
                Reading.from_dict(item) for item in data["authored_readings"]
            ),
            authored_forest=tuple(
                MetricNode.from_dict(item) for item in data["authored_forest"]
            ),
            tempo_maps=tuple(TempoMap.from_dict(item) for item in data["tempo_maps"]),
        )

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> MetricHierarchy:
        """Build a measure map and its tempo indications from authoring rows.

        Args:
            rows: Compressed or full Measure Map rows, optionally with tempo.
            provenance: JSON-compatible provenance for the tempo map.

        Returns:
            A hierarchy with no authored readings and one tempo map when needed.

        Raises:
            ValueError: If row expansion or metric derivation fails.
        """
        expanded = MeasureMap.expand_rows(rows)
        measure_map = MeasureMap.from_expanded_rows(expanded)
        entries = tuple(
            TempoEntry(
                at=Coordinate(Fraction(row["qstamp"]), TimeUnit.quarters),
                tempo=row["tempo"],
            )
            for row in expanded
            if "tempo" in row
        )
        tempo_maps = (
            (
                TempoMap(
                    kind="tempo_indication",
                    entries=entries,
                    provenance=(
                        {"source": "measure_map_rows"}
                        if provenance is None
                        else provenance
                    ),
                ),
            )
            if entries
            else ()
        )
        return cls(measure_map=measure_map, tempo_maps=tempo_maps)

    def __post_init__(self) -> None:
        object.__setattr__(self, "authored_readings", tuple(self.authored_readings))
        object.__setattr__(self, "authored_forest", tuple(self.authored_forest))
        object.__setattr__(self, "tempo_maps", tuple(self.tempo_maps))
        object.__setattr__(self, "_memoized", {})
        total = self.measure_map.total_actual_length
        if total is None:
            raise ValueError(
                "A metric hierarchy cannot use an abstract measure map; "
                "an abstract measure map has no symbolic extents"
            )
        authored_ids = [reading.id for reading in self.authored_readings]
        if len(authored_ids) != len(set(authored_ids)):
            raise ValueError("Authored metric reading ids must be unique")
        reserved = {
            "default",
            *(_generated_reading_id(beat) for beat in self._generated_beats()),
        }
        overlap = reserved.intersection(authored_ids)
        if overlap:
            raise ValueError(f"Reserved metric reading id {sorted(overlap)[0]!r}")

        def validate(node: MetricNode) -> None:
            if node.span.unit is not TimeUnit.quarters:
                raise ValueError("Metric hierarchy spans must use quarters")
            if node.span.start.value < 0 or node.span.end.value > total:
                raise ValueError(
                    f"Metric node span {node.span} lies outside allowed range [0, {total})"
                )
            for expansion in node.expansions:
                invalid = expansion.readings - set(authored_ids)
                if invalid:
                    raise ValueError(
                        f"Authored expansion names reserved or unknown reading id "
                        f"{sorted(invalid)[0]!r}"
                    )
                for child in expansion.children:
                    validate(child)

        for root in self.authored_forest:
            validate(root)
        _ = self.tempo_indications
        self._validate_generated_stretches()

    def __eq__(self, other: object) -> bool:
        """Compare only the complete derived and authored packed forest."""
        if not isinstance(other, MetricHierarchy):
            return NotImplemented
        return self.forest == other.forest

    def __repr__(self) -> str:
        kinds = [tempo_map.kind for tempo_map in self.tempo_maps]
        summary = (
            f"MetricHierarchy(measures={len(self.measure_map)}, "
            f"readings={[reading.id for reading in self.readings]!r}, "
            f"authored_roots={len(self.authored_forest)}, "
            f"tempo_maps={len(self.tempo_maps)}:{kinds!r}"
        )
        if "forest" in self._memoized:
            summary += f", forest_roots={len(self._memoized['forest'])}"
        return summary + ")"

    @property
    def default_reading(self) -> Reading:
        """The notation reading derived from the measure map.

        Returns:
            The reserved default reading descriptor.
        """
        return self._memo(
            "default_reading", lambda: Reading(id="default", source="notation")
        )

    @property
    def generated_readings(self) -> tuple[Reading, ...]:
        """Notation readings generated by off-grid tempo beat units.

        Returns:
            Deterministically identified readings in first-use order.

        Raises:
            ValueError: If a generated stretch cannot be tiled by full groups.
        """

        def build() -> tuple[Reading, ...]:
            readings: list[Reading] = []
            seen: set[str] = set()
            for beat in self._generated_beats():
                reading_id = _generated_reading_id(beat)
                if reading_id not in seen:
                    seen.add(reading_id)
                    readings.append(
                        Reading(
                            id=reading_id,
                            source="notation",
                            reinterprets="default",
                        )
                    )
            return tuple(readings)

        return self._memo("generated_readings", build)

    @property
    def readings(self) -> tuple[Reading, ...]:
        """Default, generated, and authored readings in that order.

        Returns:
            Every reading visible through the hierarchy.
        """
        return self._memo(
            "readings",
            lambda: (
                self.default_reading,
                *self.generated_readings,
                *self.authored_readings,
            ),
        )

    @property
    def tempo_indications(self) -> tuple[TempoEntry, ...]:
        """All tempo indication entries sorted by symbolic position.

        Returns:
            Tempo indication entries in stable axis order.

        Raises:
            ValueError: If simultaneous entries state different beat units.
        """

        def build() -> tuple[TempoEntry, ...]:
            entries = sorted(
                (
                    entry
                    for tempo_map in self.tempo_maps
                    if tempo_map.kind == "tempo_indication"
                    for entry in tempo_map.entries
                ),
                key=lambda entry: entry.at.value,
            )
            by_position: dict[Fraction, set[Fraction | None]] = {}
            total = self.measure_map.total_actual_length
            assert total is not None
            for entry in entries:
                if entry.at.unit is not TimeUnit.quarters:
                    raise ValueError("Tempo indication positions must use quarters")
                position = Fraction(entry.at.value)
                if position < 0 or position >= total:
                    raise ValueError(
                        f"Tempo indication position {position} lies outside allowed "
                        f"range [0, {total})"
                    )
                by_position.setdefault(position, set()).add(_beat_quarters(entry.tempo))
            for position, beat_units in by_position.items():
                if len(beat_units) > 1:
                    raise ValueError(
                        f"Tempo indications at {position} state different beat units"
                    )
            return tuple(entries)

        return self._memo("tempo_indications", build)

    @property
    def forest(self) -> tuple[MetricNode, ...]:
        """Packed forest of default, generated, and authored readings.

        Returns:
            Top-level nodes in axis order.
        """

        def build() -> tuple[MetricNode, ...]:
            trees = [self._default_trees(), *self._generated_trees().values()]
            trees.extend(self._authored_trees().values())
            return _pack_trees(trees)

        return self._memo("forest", build)

    def _memo(self, name: str, build: Callable[[], _MemoValue]) -> _MemoValue:
        """Return one derived value, constructing and retaining it once."""
        if name not in self._memoized:
            self._memoized[name] = build()
        return self._memoized[name]

    def _tempo_in_force(self, at: Fraction) -> Tempo | None:
        current: Tempo | None = None
        for entry in self.tempo_indications:
            if entry.at.value > at:
                break
            current = entry.tempo
        return current

    def _levels_for(
        self, policy: BeatPolicy, beat_unit: Fraction | None
    ) -> _MetricLevels:
        rods = policy.rods
        uniform_beat = rods[0] if all(rod == rods[0] for rod in rods) else None
        division = Fraction(policy.division)
        bar = policy.span
        distinct_division = uniform_beat is not None and division != uniform_beat
        if beat_unit == bar:
            return _MetricLevels(
                0, -1, (-2 if distinct_division else None), None, False
            )
        if (
            beat_unit is not None
            and uniform_beat is not None
            and beat_unit == uniform_beat
        ):
            return _MetricLevels(1, 0, (-1 if distinct_division else None), None, False)
        if beat_unit == division:
            return _MetricLevels(
                2 if distinct_division else 1,
                1 if distinct_division else 0,
                0 if distinct_division else None,
                None,
                False,
            )
        if beat_unit is not None and beat_unit < division and division % beat_unit == 0:
            return _MetricLevels(
                3 if distinct_division else 2,
                2 if distinct_division else 1,
                1 if distinct_division else None,
                beat_unit,
                False,
            )
        return _MetricLevels(
            1,
            0,
            (-1 if distinct_division else None),
            None,
            beat_unit is not None,
        )

    def _build_default_trees(self) -> tuple[MetricNode, ...]:
        roots: list[MetricNode] = []
        for measure in self.measure_map:
            assert measure.qstamp is not None and measure.actual_length is not None
            start = measure.qstamp
            end = start + measure.actual_length
            signature = measure.time_signature
            if signature is None:
                roots.append(MetricNode(level=0, span=_metric_interval(start, end)))
                continue
            if signature.lower() == "cadenza":
                beat = _beat_quarters(self._tempo_in_force(start))
                level = 0 if beat == measure.actual_length else 1
                roots.append(MetricNode(level=level, span=_metric_interval(start, end)))
                continue
            policy = BeatPolicy.from_time_signature(signature)
            beat_unit = _beat_quarters(self._tempo_in_force(start))
            levels = self._levels_for(policy, beat_unit)
            bar_level = levels.bar_level
            beat_level = levels.beat_level
            division_level = levels.division_level
            subpulse = levels.subpulse
            nominal = (
                measure.nominal_length
                if measure.nominal_length is not None
                else policy.span
            )
            expandable = isinstance(measure, (RegularMeasure, MeasureConstituent)) or (
                type(measure) is Measure
                and measure.actual_length == policy.span
                and nominal == policy.span
            )
            if not expandable:
                roots.append(
                    MetricNode(level=bar_level, span=_metric_interval(start, end))
                )
                continue
            window_start = (
                measure.offset_within_measure
                if isinstance(measure, MeasureConstituent)
                else Fraction(0)
            )
            window_end = window_start + measure.actual_length
            if isinstance(measure, MeasureConstituent) and window_end > nominal:
                raise ValueError(
                    f"Measure {measure.id!r} has offset {window_start} plus actual "
                    f"length {measure.actual_length} beyond nominal length {nominal}"
                )
            beat_nodes: list[MetricNode] = []
            cursor = Fraction(0)
            for rod, group_size in zip(policy.rods, policy.grouping, strict=True):
                beat_start, beat_end = cursor, cursor + rod
                cursor = beat_end
                if isinstance(measure, MeasureConstituent):
                    clipped_start = max(beat_start, window_start)
                    clipped_end = min(beat_end, window_end)
                else:
                    clipped_start, clipped_end = beat_start, beat_end
                if clipped_start >= clipped_end:
                    continue
                axis_start = start + clipped_start - window_start
                axis_end = start + clipped_end - window_start
                division_nodes: list[MetricNode] = []
                division = Fraction(policy.division)
                if group_size > 1:
                    for index in range(group_size):
                        div_start = beat_start + index * division
                        div_end = div_start + division
                        if isinstance(measure, MeasureConstituent):
                            part_start = max(div_start, window_start)
                            part_end = min(div_end, window_end)
                        else:
                            part_start, part_end = div_start, div_end
                        if part_start >= part_end:
                            continue
                        div_axis_start = start + part_start - window_start
                        div_axis_end = start + part_end - window_start
                        division_nodes.append(
                            self._division_node(
                                div_axis_start,
                                div_axis_end,
                                division_level,
                                subpulse,
                                div_start,
                                window_start,
                                start,
                                measure.id,
                            )
                        )
                elif subpulse is not None:
                    division_nodes = [
                        self._division_node(
                            axis_start,
                            axis_end,
                            beat_level,
                            subpulse,
                            beat_start,
                            window_start,
                            start,
                            measure.id,
                        )
                    ]
                if division_nodes and group_size > 1:
                    beat_nodes.append(
                        MetricNode(
                            level=beat_level,
                            span=_metric_interval(axis_start, axis_end),
                            expansions=(
                                Expansion(
                                    children=tuple(division_nodes),
                                    readings=frozenset({"default"}),
                                ),
                            ),
                        )
                    )
                elif division_nodes:
                    beat_nodes.append(division_nodes[0])
                else:
                    beat_nodes.append(
                        MetricNode(
                            level=beat_level,
                            span=_metric_interval(axis_start, axis_end),
                        )
                    )
            roots.append(
                MetricNode(
                    level=bar_level,
                    span=_metric_interval(start, end),
                    expansions=(
                        Expansion(
                            children=tuple(beat_nodes),
                            readings=frozenset({"default"}),
                        ),
                    ),
                )
            )
        return tuple(roots)

    def _default_trees(self) -> tuple[MetricNode, ...]:
        return self._memo("default_trees", self._build_default_trees)

    def _division_node(
        self,
        axis_start: Fraction,
        axis_end: Fraction,
        level: int | None,
        subpulse: Fraction | None,
        nominal_start: Fraction,
        window_start: Fraction,
        measure_start: Fraction,
        measure_id: str | None,
    ) -> MetricNode:
        actual_level = level if level is not None else 0
        if subpulse is None:
            return MetricNode(
                level=actual_level, span=_metric_interval(axis_start, axis_end)
            )
        children: list[MetricNode] = []
        nominal_end = nominal_start + (axis_end - axis_start)
        cursor = nominal_start
        while cursor < nominal_end:
            part_start = max(cursor, window_start)
            part_end = min(cursor + subpulse, window_start + (axis_end - measure_start))
            if part_start < part_end:
                start = measure_start + part_start - window_start
                end = measure_start + part_end - window_start
                if axis_start <= start < end <= axis_end:
                    children.append(
                        MetricNode(level=0, span=_metric_interval(start, end))
                    )
            cursor += subpulse
        if (
            not children
            or children[0].span.start != Coordinate(axis_start, TimeUnit.quarters)
            or children[-1].span.end != Coordinate(axis_end, TimeUnit.quarters)
        ):
            raise ValueError(
                f"Measure {measure_id!r} has clipped divisions that do not cover "
                f"axis span [{axis_start}, {axis_end})"
            )
        return MetricNode(
            level=actual_level,
            span=_metric_interval(axis_start, axis_end),
            expansions=(
                Expansion(children=tuple(children), readings=frozenset({"default"})),
            ),
        )

    def _generated_beats(self) -> tuple[Fraction, ...]:
        beats: list[Fraction] = []
        for entry in self.tempo_indications:
            beat = _beat_quarters(entry.tempo)
            if beat is None:
                continue
            measure = self._measure_at(Fraction(entry.at.value))
            if (
                measure.time_signature is None
                or measure.time_signature.lower() == "cadenza"
            ):
                continue
            policy = BeatPolicy.from_time_signature(measure.time_signature)
            if self._levels_for(policy, beat).defines_own_reading:
                beats.append(beat)
        return tuple(beats)

    def _generated_stretches(
        self,
    ) -> tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...]:
        """Return generated-reading stretches as start, end, beat, and group."""
        stretches: list[tuple[Fraction, Fraction, Fraction, Fraction]] = []
        entries = self.tempo_indications
        total = self.measure_map.total_actual_length
        assert total is not None
        for index, entry in enumerate(entries):
            beat = _beat_quarters(entry.tempo)
            if beat is None:
                continue
            start = Fraction(entry.at.value)
            measure = self._measure_at(start)
            signature = measure.time_signature
            if signature is None or signature.lower() == "cadenza":
                continue
            policy = BeatPolicy.from_time_signature(signature)
            if not self._levels_for(policy, beat).defines_own_reading:
                continue
            boundaries = [total]
            if index + 1 < len(entries):
                boundaries.append(Fraction(entries[index + 1].at.value))
            for later in self.measure_map:
                assert later.qstamp is not None
                if later.qstamp > start and later.time_signature != signature:
                    boundaries.append(later.qstamp)
                    break
            end = min(boundaries)
            group = _rational_lcm(policy.span, beat)
            stretches.append((start, end, beat, group))
        return tuple(stretches)

    def _validate_generated_stretches(self) -> None:
        """Validate exact group tiling without constructing metric nodes."""
        for start, end, _, group in self._generated_stretches():
            remainder = (end - start) % group
            if remainder:
                raise ValueError(
                    f"Generated reading stretch [{start}, {end}) is not tiled by "
                    f"group {group}; remainder {remainder}"
                )

    def _build_generated_trees(self) -> dict[str, tuple[MetricNode, ...]]:
        result: dict[str, list[MetricNode]] = {}
        for start, end, beat, group in self._generated_stretches():
            reading_id = _generated_reading_id(beat)
            roots = result.setdefault(reading_id, [])
            cursor = start
            while cursor < end:
                pulses = tuple(
                    MetricNode(
                        level=0,
                        span=_metric_interval(
                            cursor + pulse * beat,
                            cursor + (pulse + 1) * beat,
                        ),
                    )
                    for pulse in range(int(group / beat))
                )
                roots.append(
                    MetricNode(
                        level=1,
                        span=_metric_interval(cursor, cursor + group),
                        expansions=(
                            Expansion(
                                children=pulses,
                                readings=frozenset({reading_id}),
                            ),
                        ),
                    )
                )
                cursor += group
        return {key: tuple(value) for key, value in result.items()}

    def _generated_trees(self) -> dict[str, tuple[MetricNode, ...]]:
        return self._memo("generated_trees", self._build_generated_trees)

    def _build_authored_trees(self) -> dict[str, tuple[MetricNode, ...]]:
        result: dict[str, tuple[MetricNode, ...]] = {}

        def select(
            node: MetricNode, reading_id: str, inherited: bool = False
        ) -> MetricNode | None:
            expansions = tuple(
                Expansion(
                    children=tuple(
                        child_selected
                        for child in expansion.children
                        if (child_selected := select(child, reading_id, True))
                        is not None
                    ),
                    readings=frozenset({reading_id}),
                )
                for expansion in node.expansions
                if reading_id in expansion.readings
            )
            if inherited or expansions:
                return MetricNode(
                    level=node.level, span=node.span, expansions=expansions
                )
            return None

        for reading in self.authored_readings:
            result[reading.id] = tuple(
                selected
                for root in self.authored_forest
                if (selected := select(root, reading.id)) is not None
            )
        return result

    def _authored_trees(self) -> dict[str, tuple[MetricNode, ...]]:
        return self._memo("authored_trees", self._build_authored_trees)

    def _trees_for(self, reading_id: str) -> tuple[MetricNode, ...]:
        if reading_id == "default":
            return self._default_trees()
        generated = self._generated_trees()
        if reading_id in generated:
            return generated[reading_id]
        authored = self._authored_trees()
        if reading_id in authored:
            return authored[reading_id]
        known = ", ".join(reading.id for reading in self.readings)
        raise ValueError(f"Unknown metric reading {reading_id!r}; known ids: {known}")

    def _position(self, at: Coordinate | Fraction | int | float) -> Fraction:
        if isinstance(at, Coordinate):
            if at.unit is not TimeUnit.quarters:
                raise ValueError(
                    f"Metric position unit {at.unit.value!r} is not quarters; "
                    "the hierarchy unit is quarters"
                )
            position = Fraction(at.value)
        else:
            position = Fraction(at)
        total = self.measure_map.total_actual_length
        assert total is not None
        if position < 0 or position >= total:
            raise ValueError(
                f"Metric position {position} lies outside allowed range [0, {total})"
            )
        return position

    def _measure_at(self, at: Fraction) -> Measure:
        for measure in self.measure_map:
            assert measure.qstamp is not None and measure.actual_length is not None
            if measure.qstamp <= at < measure.qstamp + measure.actual_length:
                return measure
        raise ValueError(f"No measure contains metric position {at}")

    def nodes(
        self, *, reading: Reading | str = "default", level: int | None = None
    ) -> tuple[MetricNode, ...]:
        """Return one reading's nodes in axis order.

        Args:
            reading: Reading object or known reading id.
            level: Optional exact level restriction.

        Returns:
            Nodes ordered by start and then by decreasing span length.

        Raises:
            ValueError: If the reading id is unknown.
        """
        reading_id = reading.id if isinstance(reading, Reading) else reading
        found = self._memo(
            f"nodes:{reading_id}",
            lambda: _flatten_nodes(self._trees_for(reading_id)),
        )
        return tuple(node for node in found if level is None or node.level == level)

    def node_at(
        self,
        at: Coordinate | Fraction | int | float,
        *,
        reading: Reading | str = "default",
        level: int = 0,
    ) -> MetricNode:
        """Return the selected reading's level node containing a position.

        Args:
            at: Quarter-note position or quarter-note coordinate.
            reading: Reading object or known reading id.
            level: Required metrical level.

        Returns:
            The unique containing node.

        Raises:
            ValueError: If the position, reading, or requested node is absent.
        """
        position = self._position(at)
        for node in self.nodes(reading=reading, level=level):
            if node.span.start.value <= position < node.span.end.value:
                return node
        reading_id = reading.id if isinstance(reading, Reading) else reading
        raise ValueError(
            f"Reading {reading_id!r} has no level {level} node at {position}"
        )

    def pulse_at(
        self,
        at: Coordinate | Fraction | int | float,
        *,
        reading: Reading | str = "default",
    ) -> Duration:
        """Return the represented length of the containing pulse node.

        Args:
            at: Quarter-note position or quarter-note coordinate.
            reading: Reading object or known reading id.

        Returns:
            A quarter-note :class:`Duration`.

        Raises:
            ValueError: If no pulse node contains the position.
        """
        position = self._position(at)
        measure = self._measure_at(position)
        if isinstance(measure, CadenzaMeasure):
            raise ValueError(f"Measure {measure.id!r} is a cadenza and states no pulse")
        node = self.node_at(position, reading=reading, level=0)
        return Duration(
            node.span.end.value - node.span.start.value,
            TimeUnit.quarters,
        )

    def policy_at(
        self,
        at: Coordinate | Fraction | int | float,
        *,
        reading: Reading | str = "default",
        level: int = 0,
    ) -> BeatPolicy:
        """Derive one measure's counting at a reading level.

        Args:
            at: Quarter-note position or quarter-note coordinate.
            reading: Reading object or known reading id.
            level: Level whose nodes form the policy rods.

        Returns:
            Exact grouping and division named by the containing signature.

        Raises:
            ValueError: If nodes are absent or cross the measure boundary.
        """
        position = self._position(at)
        measure = self._measure_at(position)
        if isinstance(measure, CadenzaMeasure):
            raise ValueError(f"Measure {measure.id!r} is a cadenza and states no pulse")
        assert measure.qstamp is not None and measure.actual_length is not None
        start, end = measure.qstamp, measure.qstamp + measure.actual_length
        level_nodes = []
        for node in self.nodes(reading=reading, level=level):
            overlaps = node.span.start.value < end and node.span.end.value > start
            inside = start <= node.span.start.value and node.span.end.value <= end
            if overlaps and not inside:
                raise ValueError(
                    f"Measure {measure.id!r} has a crossing level {level} node"
                )
            if inside:
                level_nodes.append(node)
        if not level_nodes:
            raise ValueError(
                f"Measure {measure.id!r} has no level {level} nodes in the reading"
            )
        rods = tuple(
            node.span.end.value - node.span.start.value for node in level_nodes
        )
        lower = [
            node.span.end.value - node.span.start.value
            for node in self.nodes(reading=reading, level=level - 1)
            if start <= node.span.start.value and node.span.end.value <= end
        ]
        division = _rational_gcd(lower if lower else rods)
        return BeatPolicy(
            grouping=tuple(int(rod / division) for rod in rods),
            division=division,
            name=measure.time_signature,
        )

    def with_reading(
        self, reading: Reading, nodes: Iterable[MetricNode]
    ) -> MetricHierarchy:
        """Return a copy with one explicitly authored reading merged in.

        Args:
            reading: New provenance record with a unique, non-reserved id.
            nodes: The reading's top-level trees.

        Returns:
            A new hierarchy with packed authored nodes.

        Raises:
            ValueError: If the id is used or an expansion has invalid tags.
        """
        known = {item.id for item in self.readings}
        if (
            reading.id in known
            or reading.id == "default"
            or reading.id.startswith("generated_q_")
        ):
            raise ValueError(f"Reading id {reading.id!r} is already used or reserved")

        def tag(node: MetricNode) -> MetricNode:
            expansions: list[Expansion] = []
            for expansion in node.expansions:
                if expansion.readings not in (frozenset(), frozenset({reading.id})):
                    raise ValueError(
                        f"Expansion for {reading.id!r} carries invalid reading ids "
                        f"{sorted(expansion.readings)!r}"
                    )
                expansions.append(
                    Expansion(
                        children=tuple(tag(child) for child in expansion.children),
                        readings=frozenset({reading.id}),
                    )
                )
            return MetricNode(
                level=node.level, span=node.span, expansions=tuple(expansions)
            )

        tagged = tuple(tag(node) for node in nodes)
        authored = _pack_trees((self.authored_forest, tagged))
        return type(self)(
            measure_map=self.measure_map,
            authored_readings=(*self.authored_readings, reading),
            authored_forest=authored,
            tempo_maps=self.tempo_maps,
        )

    def with_tempo_map(self, tempo_map: TempoMap) -> MetricHierarchy:
        """Return a copy with one independent tempo map appended.

        Args:
            tempo_map: Map to append.

        Returns:
            A new hierarchy sharing its measure map and authored forest.
        """
        return type(self)(
            measure_map=self.measure_map,
            authored_readings=self.authored_readings,
            authored_forest=self.authored_forest,
            tempo_maps=(*self.tempo_maps, tempo_map),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return only authored data and independent tempo maps.

        Returns:
            The JSON-safe hierarchy payload owned by a timeline.
        """
        return {
            "authored_readings": [
                reading.to_dict() for reading in self.authored_readings
            ],
            "authored_forest": [node.to_dict() for node in self.authored_forest],
            "tempo_maps": [tempo_map.to_dict() for tempo_map in self.tempo_maps],
        }
