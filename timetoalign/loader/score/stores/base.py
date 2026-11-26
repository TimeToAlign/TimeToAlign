"""ScoreEventData: the base of a score loader's category tables."""

from __future__ import annotations

from typing import Any, ClassVar

import pyarrow as pa
from typing_extensions import Self

from timetoalign.core import IntervalPolicy, NumberType, TimeUnit
from timetoalign.core.ids import IdMinter
from timetoalign.storage.events import EventData
from timetoalign.storage.schema import make_table_metadata


class ScoreEventData(EventData):
    """EventData of one category of score events, in quarters by default.

    A score loader writes its rows in its source's vocabulary
    (``quarterbeats``, ``duration_qb``, …); :meth:`from_dicts` maps them onto
    the class's schema (:meth:`_source_row`) and then builds the table the
    way every :class:`~timetoalign.storage.EventData` does, so a row
    without an ``id`` gets one minted after the table is built.
    """

    #: Source keys a row may carry that the table does not hold.
    _dropped_keys: ClassVar[tuple[str, ...]] = (
        "quarterbeats_float",
        "duration_qb_float",
    )

    @classmethod
    def empty(
        cls,
        unit: TimeUnit = TimeUnit.quarters,
        number_type: NumberType | None = None,
    ) -> Self:
        """Create an empty table of this class."""
        return super().empty(unit, number_type)

    @classmethod
    def from_dicts(
        cls,
        rows: list[dict[str, Any]],
        unit: TimeUnit = TimeUnit.quarters,
        number_type: NumberType | None = None,
        *,
        interval_policy: IntervalPolicy = IntervalPolicy.warn,
        seed: int | None = None,
        minter: IdMinter | None = None,
    ) -> Self:
        """Create from a score loader's rows.

        A row without an ``id`` gets one minted from *minter* (a loader's),
        else from ``IdMinter(seed)``, else from the content minter of the
        table (every column but ``id``), as
        :meth:`EventData.from_dicts <timetoalign.storage.EventData.from_dicts>`
        mints them; a stated ``id`` is kept.

        Args:
            rows: Row dictionaries in the source's vocabulary.
            unit: Time unit (default: quarters).
            number_type: Number type for coordinates.
            interval_policy: How to handle end/duration inconsistencies.
            seed: Mint the missing IDs from ``IdMinter(seed)``.
            minter: Mint the missing IDs from this minter (a loader's).

        Returns:
            An instance of the class holding the rows.

        Raises:
            TypeError: If both *seed* and *minter* are given.
        """
        return super().from_dicts(
            rows,
            unit,
            number_type,
            interval_policy=interval_policy,
            seed=seed,
            minter=minter,
        )

    @classmethod
    def _rows_table(
        cls,
        rows: Any,
        unit: TimeUnit,
        number_type: NumberType,
        *,
        interval_policy: IntervalPolicy = IntervalPolicy.warn,
    ) -> pa.Table:
        """The table of *rows* on the class's schema; a row without ID holds a null ``id``."""
        if not rows:
            return cls._empty_table(unit, number_type)
        schema = cls.get_schema(unit, number_type=number_type).with_metadata(
            make_table_metadata(unit, number_type, loader_class=cls.__name__)
        )
        processed_rows = []
        for row in rows:
            processed = cls._source_row(dict(row))
            processed["id"] = processed.get("id")
            processed.setdefault("name", None)
            for key in cls._dropped_keys:
                processed.pop(key, None)
            # Converts coordinate fields to structs, fills a missing end or
            # duration and infers the temporal type.
            cls._normalize_intervals_row(
                processed, policy=interval_policy, number_type=number_type
            )
            processed_rows.append(processed)
        return pa.Table.from_pylist(processed_rows, schema=schema)

    @classmethod
    def _source_row(cls, row: dict[str, Any]) -> dict[str, Any]:
        """*row* with the source's keys renamed to the schema's, in place.

        ``quarterbeats`` becomes ``start`` and ``duration_qb`` ``duration``.
        """
        if "quarterbeats" in row:
            row["start"] = row.pop("quarterbeats")
        if "duration_qb" in row:
            row["duration"] = row.pop("duration_qb")
        return row
