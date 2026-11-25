"""ScoreLoader: Base loader for symbolic scores."""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Mapping
from datetime import datetime, timezone
from fractions import Fraction
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from typing_extensions import Self

from timetoalign.core import TimeUnit
from timetoalign.core.enums import FlowManifestation
from timetoalign.loader.base import EventLoader
from timetoalign.storage import EventData

from .store import ScoreStore
from .stores.notes import NoteEventData

if TYPE_CHECKING:
    from timetoalign.timelines import MeasureMap, StatedFlow, Timeline
    from timetoalign.timelines.flow import FlowControlDraft
    from timetoalign.timelines.flowcontrol import FlowControlElement

module_logger = logging.getLogger(__name__)


def _coordinate_struct(value: Fraction | float) -> dict[str, Any]:
    """Return a coordinate struct without inventing a ratio for a float."""
    if isinstance(value, Fraction):
        return {
            "value": float(value),
            "numerator": value.numerator,
            "denominator": value.denominator,
        }
    return {"value": float(value), "numerator": None, "denominator": None}


def _struct_coordinate_value(struct: dict[str, Any]) -> Fraction | float:
    """Read a coordinate pair, falling back to its value-only projection."""
    numerator = struct.get("numerator")
    denominator = struct.get("denominator")
    if (
        isinstance(numerator, Integral)
        and not isinstance(numerator, bool)
        and isinstance(denominator, Integral)
        and not isinstance(denominator, bool)
    ):
        return Fraction(int(numerator), int(denominator))
    return float(struct["value"])


_DA_CAPO = frozenset({"start", "firstMeasure"})
"""The values of ``jump_bwd`` that name the beginning instead of a marker."""


def _names(cell: Any) -> list[str]:
    """Return the names one cell of a measure table holds, joined by ``&``."""
    if not cell:
        return []
    return [name.strip() for name in str(cell).split("&") if name.strip()]


def _leap_manifestation(da_capo: bool, until: str | None) -> FlowManifestation:
    """Return what a D.C. or D.S. looks like, from where it plays until."""
    kind = "da_capo" if da_capo else "dal_segno"
    if until == "fine":
        kind += "_al_fine"
    elif until is not None and until.startswith("coda"):
        kind += "_al_coda"
    return FlowManifestation(kind)


def _sign_manifestation(name: str) -> FlowManifestation | None:
    """Return what a named marker looks like, where its name says so."""
    for kind in (FlowManifestation.segno, FlowManifestation.coda):
        if name.startswith(kind.value):
            return kind
    return None


class ScoreLoader(EventLoader):
    """Base loader for symbolic music scores (MusicXML, MIDI-Score, TSV).

    All score loaders normalise coordinates to quarter-note positions using
    ``Fraction`` precision.  The reported metadata reflects this:
    ``unit='quarters'``, ``number_type='fraction'``.
    """

    _default_unit = TimeUnit.quarters
    _event_data_class = NoteEventData
    _barline_manifestations: ClassVar[Mapping[str, FlowManifestation]] = {
        "double": FlowManifestation.double_barline,
        "end": FlowManifestation.final_barline,
    }
    """The values of the measure table's ``barline`` that are written as a
    :class:`~timetoalign.timelines.flowcontrol.Barline`."""

    @staticmethod
    def _normalise_anacrusis(
        note_rows: list[dict[str, Any]], *other_rows: list[dict[str, Any]]
    ) -> Fraction | float:
        """Shift every row so that the earliest note onset is 0; return the shift.

        A timeline starts at 0, while a parser may place the notes of an
        anacrusis at negative quarter positions. The shift is minus the
        earliest ``quarterbeats`` of *note_rows* when that is negative, else
        0, and is added to the ``quarterbeats`` of every row of *note_rows*
        and *other_rows* (measures, controls, annotations), in place. Exact
        positions stay exact: the shift is a ``Fraction`` unless the earliest
        onset is stated only as a float, and a shifted cell keeps its ratio
        whenever the sum has one. The loader exposes the shift as its
        ``anacrusis_offset`` and stores it in the ``ScoreStore`` metadata.
        """
        onsets = [
            _struct_coordinate_value(row["quarterbeats"])
            for row in note_rows
            if row.get("quarterbeats") is not None
        ]
        earliest: Fraction | float = min(onsets, default=Fraction(0))
        offset: Fraction | float = -earliest if earliest < 0 else Fraction(0)
        if offset:
            for rows in (note_rows, *other_rows):
                for row in rows:
                    position = _struct_coordinate_value(row["quarterbeats"])
                    row["quarterbeats"] = _coordinate_struct(position + offset)
        return offset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._store = ScoreStore.empty()

    @property
    def store(self) -> ScoreStore:
        """The ScoreStore containing all loaded score elements."""
        return self._store

    @property
    def events(self) -> NoteEventData:
        """The primary event data (notes), for compatibility."""
        return self._store.notes

    def get_events(self, properties: bool | str | tuple[str, ...] = True) -> EventData:
        """Return note events with field control.

        Overrides the base ``Loader.get_events`` because ``ScoreLoader``
        stores events in a ``ScoreStore``, not in ``self._events``.

        Args:
            properties: Controls which non-field-spec fields to include.
                ``True``: all fields.  ``False``: spec + core fields only.
                Single string: shorthand for a one-element tuple.
                Tuple of strings: named property fields only.
        """
        if isinstance(properties, str):
            properties = (properties,)
        self._events = self._store.notes
        return self._assemble_events_table(properties)

    @abstractmethod
    def _load_source(self, source: Path) -> ScoreStore:
        """Load a single source file into a ScoreStore.

        Args:
            source: Path to the source file.

        Returns:
            A ScoreStore containing the loaded data.
        """
        ...

    def load(self, *sources: Path | str) -> Self:
        """Load sources into the store.

        Args:
            *sources: Paths to source files.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If the loader reads one source and is given none or
                more than one, or any while it holds one.
        """
        self._refuse_further_source(*sources)
        for source in sources:
            path = Path(source)

            # Load store from source
            loaded_store = self._load_source(path)

            # Update metadata
            meta = loaded_store.metadata.copy()
            meta["path"] = str(path)
            meta["loaded_at"] = datetime.now(timezone.utc).isoformat()

            self._sources.append(path)
            self._source_metadata.append(meta)

            # Extend internal store
            self._store.extend(loaded_store)

        return self

    def clear(self) -> None:
        """Clear all loaded sources and store data."""
        super().clear()
        self._store = ScoreStore.empty()

    def create_timeline(self, uid: str | None = None, **kwargs: Any) -> Timeline:
        """Create a score timeline with measure and flow-control structure."""
        timeline = super().create_timeline(uid=uid, **kwargs)
        self._populate_timeline_structure(timeline)
        return timeline

    def _measure_map(self) -> MeasureMap:
        """Return the measure map the loaded measure table describes."""
        from timetoalign.timelines import MeasureMap

        return MeasureMap.from_measure_data(self._store.measures)

    def _populate_timeline_structure(self, timeline: Timeline) -> None:
        """Attach structure derivable from a non-empty measure store.

        The measure store must describe one measure map in printed order. A
        store whose measures cannot form that map is a defect of the source,
        not a lossy reading of it: an unfolded measures table repeats printed
        measure counts and therefore measure ids. Such a store raises instead
        of producing a timeline without its measure structure.

        Measures whose actual lengths are all stated state the score's
        length: the timeline spans at least their sum (the notated extent,
        not just the last event's end), and is then locked, so that content
        added later never lengthens it silently. A score without measures,
        or whose measures leave a length unstated, keeps the length its last
        event gives and stays unlocked.

        Raises:
            ValueError: If the stored measures cannot form a ``MeasureMap``.
        """
        if len(self._store.measures) == 0:
            return

        measure_map = self._measure_map()
        timeline.add_measure_map(measure_map)
        elements = self._flow_control_elements(measure_map, timeline.id)

        notated_length = measure_map.total_actual_length
        if notated_length is not None:
            timeline.length = max(timeline.length.value, notated_length)
            timeline.lock()

        # After the length: flow control lies within it, the end included.
        timeline.flow_control.add_all(elements)

    # region Flow Control

    def _flow_control_elements(
        self, measure_map: MeasureMap, timeline_id: str
    ) -> list[FlowControlElement]:
        """Return the flow control the loaded score states, as elements.

        Repeat signs and brackets of alternative endings are what the
        measure map states, read by the one translation of measures into
        flow control
        (:func:`~timetoalign.timelines.flow.measures.flow_control_from_measures`).
        Every other sign comes from the columns of the measure table that
        the loader fills from its source (:meth:`_write_signs`). Signs
        falling on one coordinate are the destinations of one Jump.

        Args:
            measure_map: The measure map of the loaded score.
            timeline_id: The ID of the timeline the elements are written
                for, which a D.C. names where it must (:meth:`_write_signs`).
        """
        from timetoalign.timelines.flow import FlowControlDraft

        draft = FlowControlDraft()
        draft.add_measures(measure_map)
        self._write_signs(draft, measure_map, timeline_id)
        return draft.elements()

    def _write_signs(
        self, draft: FlowControlDraft, measure_map: MeasureMap, timeline_id: str
    ) -> None:
        """Write the signs the measure table states beside repeats and brackets.

        Read from the table's columns, in the meaning every score loader
        fills them with:

        - ``markers``: the named signs of a measure, joined by ``&``. A
          marker a ``jump_bwd`` or ``jump_fwd`` names, and one no row names,
          is a jump target at its measure's start, its name kept in
          ``name`` (``segno`` and ``coda`` by their manifestation too). A
          marker named only by ``play_until`` is the place where that
          instruction acts, its measure's end, and no jump target. A
          ``fine`` no row plays until is a break at its measure's end that
          ends the flow from its second arrival on (``ends: {passes:
          {from: 2}}``): the leap that leads back to it is not stated, so
          the count of arrivals stands in for it.
        - ``jump_bwd`` (a D.S., or with ``start`` a D.C.): a destination
          ``repeat_count: 1, reset: false`` of the Jump at the row's end,
          ``to`` the marker's jump target, or without place for a D.C. A
          D.C. whose Jump already holds a destination without place (a
          repeat end without a printed start on the same barline) names
          the timeline instead (``to`` its ID, its start), so that the two
          are distinct places.
        - ``play_until`` on that row: at the end of the measure carrying the
          marker of that name, either a destination ``to`` the jump target
          of the row's ``jump_fwd`` marker, ``after`` the D.S. or D.C., or,
          without ``jump_fwd``, a break that ends the flow ``after`` it.
          Several rows playing until one marker give one break each.
        - ``breaks`` naming ``section``: a section break at the row's end.
        - ``barline``: a barline at the row's end, for the values of
          :attr:`_barline_manifestations`.

        What cannot be written is left out with a logged warning: a D.S.
        or a ``play_until`` whose marker no measure carries, a jump forward
        going on at a place another sign of its Jump goes on at (a To Coda
        shared by two leaps to one coda: one ``after`` names one leap), a
        ``play_until`` marker in the measure of its own D.S. or D.C., and a
        ``jump_fwd`` on a row without ``jump_bwd``.

        Raises:
            ValueError: If a measure states no count, no ``qstamp`` or no
                ``actual_length``.
        """
        from timetoalign.timelines.flowcontrol import (
            Barline,
            Break,
            DestinationReference,
            FlowCondition,
            JumpDestination,
            JumpTarget,
            PassesFrom,
            SectionBreak,
        )

        bounds: dict[int, tuple[Fraction, Fraction]] = {}
        for measure in measure_map:
            if (
                measure.count is None
                or measure.qstamp is None
                or measure.actual_length is None
            ):
                raise ValueError(
                    f"measure {measure.count!r} states no count, no qstamp or no "
                    "actual_length: flow control lies at positions"
                )
            bounds[measure.count] = (
                measure.qstamp,
                measure.qstamp + measure.actual_length,
            )
        rows = [row for row in self._store.measures if row.get("mc") in bounds]

        marked: dict[str, int] = {}
        for row in rows:
            for name in _names(row.get("markers")):
                marked.setdefault(name, row["mc"])
        leaps = [row for row in rows if row.get("jump_bwd")]
        gone_to = {row["jump_bwd"] for row in leaps} | {
            row["jump_fwd"] for row in rows if row.get("jump_fwd")
        }
        played_until = {row["play_until"] for row in leaps if row.get("play_until")}

        signs: dict[str, str] = {}
        for row in rows:
            for name in _names(row.get("markers")):
                if name not in gone_to and name in played_until:
                    continue
                if name not in gone_to and name == FlowManifestation.fine.value:
                    # No leap of the source says when this Fine ends the
                    # flow: it does from its second arrival on.
                    draft.add(
                        Break,
                        bounds[row["mc"]][1],
                        name=name,
                        manifestation=FlowManifestation.fine,
                        ends=FlowCondition(passes=PassesFrom(2)),
                    )
                    continue
                sign = draft.add(
                    JumpTarget,
                    bounds[row["mc"]][0],
                    name=name,
                    manifestation=_sign_manifestation(name),
                )
                signs.setdefault(name, sign.id)

        written: list[tuple[dict[str, Any], Fraction, DestinationReference]] = []
        for row in rows:
            name = row.get("jump_bwd")
            if not name:
                if row.get("jump_fwd"):
                    module_logger.warning(
                        "MC %s goes on at %r without a D.C. or D.S. it follows; "
                        "the jump is not written.",
                        row["mc"],
                        row["jump_fwd"],
                    )
                continue
            da_capo = name in _DA_CAPO
            if not da_capo and name not in signs:
                module_logger.warning(
                    "The D.S. of MC %s goes back to %r, which no measure carries; "
                    "it is not written.",
                    row["mc"],
                    name,
                )
                continue
            at = bounds[row["mc"]][1]
            to = None if da_capo else signs[name]
            if da_capo and any(
                held.place is None for held in draft.destinations_at(at)
            ):
                # A repeat end without a printed start lies on this barline
                # and has no place either: the D.C. names the timeline.
                to = timeline_id
            leap = JumpDestination(
                to=to,
                repeat_count=1,
                reset=False,
                manifestation=_leap_manifestation(da_capo, row.get("play_until")),
            )
            reference = DestinationReference(
                jump=draft.add_destination(at, leap), to=to
            )
            written.append((row, at, reference))

        for row, leap_at, reference in written:
            until, onward = row.get("play_until"), row.get("jump_fwd")
            if not until:
                if onward:
                    module_logger.warning(
                        "MC %s goes on at %r without saying from where; the jump "
                        "is not written.",
                        row["mc"],
                        onward,
                    )
                continue
            if until not in marked:
                module_logger.warning(
                    "MC %s plays until %r, which no measure carries; it is not "
                    "written.",
                    row["mc"],
                    until,
                )
                continue
            at = bounds[marked[until]][1]
            if not onward:
                draft.add(
                    Break,
                    at,
                    name=until,
                    manifestation=(
                        FlowManifestation.fine
                        if until == FlowManifestation.fine.value
                        else None
                    ),
                    ends=FlowCondition(after=reference),
                )
                continue
            if onward not in signs:
                module_logger.warning(
                    "MC %s goes on at %r, which no measure carries; the jump is "
                    "not written.",
                    row["mc"],
                    onward,
                )
                continue
            destination = JumpDestination(
                to=signs[onward],
                after=reference,
                manifestation=(
                    FlowManifestation.to_coda
                    if _sign_manifestation(onward) is FlowManifestation.coda
                    else None
                ),
                label=until,
            )
            if at == leap_at or any(
                held.place == destination.place for held in draft.destinations_at(at)
            ):
                module_logger.warning(
                    "The jump from %r to %r that MC %s states cannot be told from "
                    "another sign of its barline; it is not written.",
                    until,
                    onward,
                    row["mc"],
                )
                continue
            draft.add_destination(at, destination)

        for row in rows:
            end = bounds[row["mc"]][1]
            if "section" in _names(row.get("breaks")):
                draft.add(SectionBreak, end)
            barline = row.get("barline")
            manifestation = (
                self._barline_manifestations.get(barline)
                if barline is not None
                else None
            )
            if manifestation is not None:
                draft.add(Barline, end, manifestation=manifestation)

    def create_stated_flow_map(
        self,
        timeline: Timeline,
        *,
        id: str | None = None,
        name: str | None = None,
    ) -> StatedFlow | None:
        """Write the order of measures the source states as a FlowMap of *timeline*.

        Some sources state the order in which their measures are played
        beside the signs that give it: an ms3 measures table and a Measure
        Map file list the successors of each measure (``next``). That order
        is written as a FlowMap of the timeline over Regions of consecutive
        measures, its provenance saying that the source states it
        (:meth:`Timeline.create_flow_map_from_measure_order
        <timetoalign.timelines.Timeline.create_flow_map_from_measure_order>`).
        It is compared with the default flow of the timeline's flow control:
        where the two differ, a warning is logged and the result says from
        where.

        Args:
            timeline: The root timeline of this loader's score, holding its
                measure map (:meth:`create_timeline`).
            id: The FlowMap's ID; minted when None.
            name: Optional display name of the FlowMap.

        Returns:
            The Regions and the FlowMap written and where the default flow
            differs, or None where the source states no order (nothing is
            written then).

        Raises:
            ValueError: As
                :meth:`Timeline.create_flow_map_from_measure_order
                <timetoalign.timelines.Timeline.create_flow_map_from_measure_order>`,
                or if the successors the source states never end.
        """
        order = self._stated_measure_order()
        if order is None:
            return None
        return timeline.create_flow_map_from_measure_order(order, id=id, name=name)

    def _stated_measure_order(self) -> tuple[int, ...] | None:
        """Return the measure counts in the order the source states, if it does.

        Read from the ``next`` column of the measure table, which keeps the
        successors as the source lists them, the ``-1`` that ends the piece
        included. None where no measure states a successor.
        """
        from timetoalign.timelines.flow import measure_order_from_successors

        measures = self._store.measures
        if len(measures) == 0 or "next" not in measures._table.column_names:
            return None
        counts = measures._table.column("mc").to_pylist()
        stated = measures._table.column("next").to_pylist()
        if all(cell is None or str(cell).strip() == "" for cell in stated):
            return None
        successors = {
            count: (
                []
                if cell is None
                else [int(entry) for entry in str(cell).split(",") if entry.strip()]
            )
            for count, cell in zip(counts, stated, strict=True)
        }
        return measure_order_from_successors(successors, counts[0])

    # endregion

    # region Serialization

    def to_parquet(self, path: Path | str) -> None:
        """Save the loaded score data to Parquet files.

        For ``ScoreLoader`` subclasses, data lives in a multi-facet
        ``ScoreStore`` (notes, measures, controls, annotations).  Each
        facet is written as a separate ``.parquet`` file inside a directory
        at *path*, together with a ``metadata.json`` for store-level
        metadata.

        Args:
            path: Directory to write into.  Created if it does not exist.
        """
        self._store.to_parquet(path)

    @classmethod
    def from_parquet(cls, path: Path | str) -> Self:
        """Load a ScoreLoader from a directory of Parquet files.

        Reconstructs the internal ``ScoreStore`` from the facet files
        produced by :meth:`to_parquet`.

        Args:
            path: Directory containing the facet Parquet files.

        Returns:
            A new ScoreLoader with events loaded from the files.
        """
        store = ScoreStore.from_parquet(path)
        loader = cls.__new__(cls)
        # Set attributes that __init__ would normally create
        loader._unit = cls._default_unit
        loader._number_type = loader._unit.default_number_type
        loader._sources = []
        loader._source_metadata = []
        loader._events = cls._event_data_class.empty(loader._unit, loader._number_type)
        loader._store = store
        return loader

    # endregion
