"""ScoreLoader: Base loader for symbolic scores."""

from __future__ import annotations

import logging
from abc import abstractmethod
from datetime import datetime, timezone
from fractions import Fraction
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from typing_extensions import Self

from timetoalign.core import TimeUnit
from timetoalign.loader.base import EventLoader

from .store import ScoreStore
from .stores.measures import MeasureData
from .stores.notes import NoteEventData

if TYPE_CHECKING:
    from timetoalign.timelines import MeasureMap, Timeline
    from timetoalign.timelines.flow import ScoreFlowController, SegmentNameGenerator
    from timetoalign.timelines.flowcontrol import FlowControlRegistry

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


_LEAPS = (
    "dal_segno_al_coda",
    "dal_segno_al_fine",
    "da_capo_al_coda",
    "da_capo_al_fine",
    "dal_segno",
    "da_capo",
)
"""The manifestations of a D.S. or D.C., the most specific first."""


def _flow_control_of(controller: "ScoreFlowController") -> "FlowControlRegistry":
    """Return the flow-control elements a score's measures state.

    A repeat start, a segno and a coda are jump targets at the start of
    their measure; everything else lies at the end of its measure. The signs
    at one measure's end are the destinations of one jump:

    - a repeat end: ``repeat_count: 1``, ``to`` its repeat start where the
      measures state one, without place where it goes back to the start;
    - a D.C. or D.S.: ``repeat_count: 1`` and ``reset: false``, a D.S. ``to``
      its segno;
    - a To Coda: ``to`` its coda, ``after`` the score's D.C. or D.S. where
      it has exactly one on another jump; otherwise ``determined: false``.

    A Fine is a break that ends the flow ``after`` the score's one D.C. or
    D.S., and without condition where the score has none or several; a
    section break is a section break. A repeat end that resolves to no
    measure, and a D.S. or To Coda whose sign no measure carries, give no
    destination. Nothing is added that the measures do not state.
    """
    from timetoalign.core import Coordinate
    from timetoalign.core.enums import FlowManifestation
    from timetoalign.timelines.flowcontrol import (
        Break,
        DestinationReference,
        FlowCondition,
        FlowControlRegistry,
        Jump,
        JumpDestination,
        JumpTarget,
        SectionBreak,
    )

    def start_of(mc: int) -> Coordinate:
        return Coordinate(controller._qb_at_mc(mc), TimeUnit.quarters)

    def end_of(mc: int) -> Coordinate:
        return Coordinate(controller._qb_at_mc_end(mc), TimeUnit.quarters)

    registry = FlowControlRegistry()
    units = list(controller.iter_units())
    repeat_starts: dict[int, str] = {}
    signs: dict[str, str] = {}
    for unit in units:
        if unit.start_repeat:
            repeat_starts[unit.mc] = registry.add(
                JumpTarget(
                    start_of(unit.mc), manifestation=FlowManifestation.repeat_start
                )
            ).id
        for name, manifestation in (
            (unit.segno, FlowManifestation.segno),
            (unit.coda, FlowManifestation.coda),
        ):
            if name:
                sign = registry.add(
                    JumpTarget(
                        start_of(unit.mc), name=name, manifestation=manifestation
                    )
                )
                # The first sign of a name is the one a jump goes on at.
                signs.setdefault(name, sign.id)
        if unit.section_break:
            registry.add(
                SectionBreak(end_of(unit.mc), label=f"section break after MC {unit.mc}")
            )

    repeat_targets, _ = controller._repeat_end_resolution()
    # The destinations of each measure's end, a To Coda flagged: its
    # condition names a jump that has to be stored first.
    drafts: dict[int, list[tuple[dict[str, Any], bool]]] = {}
    leaps: list[tuple[int, dict[str, Any]]] = []
    for unit in units:
        found: list[tuple[dict[str, Any], bool]] = []
        if unit.end_repeat and unit.mc in repeat_targets:
            target_mc = repeat_targets[unit.mc]
            found.append(
                (
                    {
                        "to": repeat_starts.get(target_mc),
                        "repeat_count": 1,
                        "manifestation": FlowManifestation.repeat_end,
                        "label": f"MC {unit.mc} → MC {target_mc}",
                    },
                    False,
                )
            )
        types = unit.flow_control_types
        leap = next((kind for kind in _LEAPS if kind in types), None)
        if leap is not None:
            sign_name = None if leap.startswith("da_capo") else unit.jump_bwd or "segno"
            if sign_name is None or sign_name in signs:
                destination = {
                    "to": None if sign_name is None else signs[sign_name],
                    "repeat_count": 1,
                    "reset": False,
                    "manifestation": FlowManifestation(leap),
                    "label": f"MC {unit.mc} → {sign_name or 'start'}",
                }
                found.append((destination, False))
                leaps.append((unit.mc, destination))
        elif FlowManifestation.to_coda.value in types:
            sign_name = unit.jump_fwd or "coda"
            if sign_name in signs:
                found.append(
                    (
                        {
                            "to": signs[sign_name],
                            "manifestation": FlowManifestation.to_coda,
                            "label": f"MC {unit.mc} → {sign_name}",
                        },
                        True,
                    )
                )
        kept: list[tuple[dict[str, Any], bool]] = []
        for destination, is_to_coda in found:
            if any(destination["to"] == other["to"] for other, _ in kept):
                module_logger.warning(
                    "The signs at the end of MC %s go on at one place; %s is left out.",
                    unit.mc,
                    destination["manifestation"].value,
                )
                leaps[:] = [entry for entry in leaps if entry[1] is not destination]
                continue
            kept.append((destination, is_to_coda))
        if kept:
            drafts[unit.mc] = kept

    # A To Coda and a Fine act after the score's leap, which they can name
    # only where the score has exactly one.
    leap_mc = leaps[0][0] if len(leaps) == 1 else None
    reference: DestinationReference | None = None

    def add_jump(mc: int) -> Jump:
        destinations: list[JumpDestination] = []
        for destination, is_to_coda in drafts[mc]:
            if not is_to_coda:
                destinations.append(JumpDestination(**destination))
            elif reference is not None and mc != leap_mc:
                destinations.append(JumpDestination(**destination, after=reference))
            else:
                module_logger.warning(
                    "The To Coda at the end of MC %s cannot name the D.C. or D.S. "
                    "it follows (the score has %s); it is written as not "
                    "determined.",
                    mc,
                    "it on the same barline" if mc == leap_mc else f"{len(leaps)}",
                )
                destinations.append(JumpDestination(**destination, determined=False))
        return registry.add(Jump(end_of(mc), destinations))

    if leap_mc is not None:
        reference = DestinationReference(
            jump=add_jump(leap_mc).id, to=leaps[0][1]["to"]
        )
    for mc in drafts:
        if mc != leap_mc:
            add_jump(mc)

    for unit in units:
        if not unit.fine:
            continue
        if reference is None:
            module_logger.warning(
                "The Fine at the end of MC %s cannot name the D.C. or D.S. it "
                "follows (the score has %s); it is written as a break without "
                "condition.",
                unit.mc,
                len(leaps),
            )
        registry.add(
            Break(
                end_of(unit.mc),
                manifestation=FlowManifestation.fine,
                name="fine",
                label=f"fine at MC {unit.mc}",
                ends=None if reference is None else FlowCondition(after=reference),
            )
        )
    return registry


class ScoreLoader(EventLoader):
    """Base loader for symbolic music scores (MusicXML, MIDI-Score, TSV).

    All score loaders normalise coordinates to quarter-note positions using
    ``Fraction`` precision.  The reported metadata reflects this:
    ``unit='quarters'``, ``number_type='fraction'``.
    """

    _default_unit = TimeUnit.quarters
    _event_data_class = NoteEventData

    @classmethod
    def _with_successors(cls, measures: MeasureData) -> MeasureData:
        """Return *measures* with ``next`` derived where the source states none.

        A measure table read from a format that lists successors (an ms3
        TSV, a Measure Map) keeps them. Otherwise each part's successors are
        derived from its repeat, volta and navigation columns by
        :meth:`_successors`.
        """
        if len(measures) == 0 or any(
            value is not None for value in measures.column_values("next")
        ):
            return measures
        rows = list(measures)
        parts: dict[Any, list[int]] = {}
        for index, row in enumerate(rows):
            parts.setdefault(row.get("part_id"), []).append(index)
        successors: list[str | None] = [None] * len(rows)
        for indices in parts.values():
            for index, following in zip(
                indices, cls._successors([rows[i] for i in indices])
            ):
                if following is not None:
                    successors[index] = ", ".join(str(mc) for mc in following)
        table = measures.table
        position = table.schema.get_field_index("next")
        table = table.set_column(
            position, table.schema.field(position), pa.array(successors, pa.string())
        )
        return type(measures)(table, measures.unit, measures.number_type)

    @staticmethod
    def _successors(rows: list[dict[str, Any]]) -> list[list[int] | None]:
        """Derive each measure's successor MCs from its flow-control columns.

        Reads ``mc``, ``start_repeat``, ``end_repeat``, ``volta``, ``markers``
        and ``jump_bwd`` of one part's rows in MC order; ``None`` means the
        following measure. The rules, later ones taking precedence:

        * A repeat end goes back to the nearest open repeat start; without
          one, to the measure after the last ending or repeat end before it,
          else to the first measure. Outside an ending it also continues to
          the following measure (``-1`` after the last).
        * An ending is a run of measures with a volta number (a lower number
          than the previous one opens a new run). The measure before it
          continues to the first measure of each volta; with one volta only,
          also past the run. The last measure of every volta but the highest
          continues past the run unless it ends a repeat, which goes back to
          the run's repeat start.
        * A D.S. or D.C. (``jump_bwd``) continues to the measure carrying the
          named marker, ``start`` naming the first measure, in place of the
          measure's onward successors; a repeat end keeps its way back.
        * A Fine (a ``fine`` marker or flag, or ``jump_fwd`` naming ``fine``)
          also ends the piece (``-1``).

        Args:
            rows: One part's measure rows in MC order.

        Returns:
            One successor list (or ``None``) per row.
        """
        count = len(rows)
        mcs = [row["mc"] for row in rows]
        voltas = [row.get("volta") for row in rows]
        successors: list[list[int] | None] = [None] * count

        def following(index: int) -> int:
            return mcs[index + 1] if index + 1 < count else -1

        endings: list[dict[int, list[int]]] = []
        ending_of: dict[int, int] = {}
        for index, volta in enumerate(voltas):
            if volta is None:
                continue
            previous = voltas[index - 1] if index > 0 else None
            if previous is None or volta < previous:
                endings.append({})
            endings[-1].setdefault(volta, []).append(index)
            ending_of[index] = len(endings) - 1

        open_starts: list[int] = []
        ending_starts: dict[int, int] = {}
        back: dict[int, int] = {}
        boundary = 0
        for index, row in enumerate(rows):
            if row.get("start_repeat"):
                open_starts.append(index)
            ending = ending_of.get(index)
            if row.get("end_repeat"):
                if ending in ending_starts:
                    start = ending_starts[ending]
                else:
                    start = open_starts.pop() if open_starts else boundary
                if ending is None:
                    successors[index] = [mcs[start], following(index)]
                else:
                    ending_starts[ending] = start
                    successors[index] = [mcs[start]]
                back[index] = mcs[start]
                boundary = index + 1
            elif ending is not None and ending_of.get(index + 1) != ending:
                boundary = index + 1

        for ending in endings:
            numbers = sorted(ending)
            first = ending[numbers[0]][0]
            last = max(index for indices in ending.values() for index in indices)
            exit_mc = following(last)
            if first > 0:
                fork = [mcs[ending[number][0]] for number in numbers]
                successors[first - 1] = fork + [exit_mc] if len(numbers) == 1 else fork
            for number in numbers[:-1]:
                end = ending[number][-1]
                if not rows[end].get("end_repeat"):
                    successors[end] = [exit_mc]

        names = [
            {name.strip() for name in (row.get("markers") or "").split("&")} - {""}
            for row in rows
        ]
        marked: dict[str, int] = {}
        for row, row_names in zip(rows, names):
            for name in row_names:
                marked.setdefault(name, row["mc"])
        for index, row in enumerate(rows):
            target = row.get("jump_bwd")
            if target in ("start", "firstMeasure"):
                target_mc = mcs[0]
            elif target in marked:
                target_mc = marked[target]
            else:
                continue
            successors[index] = [*([back[index]] if index in back else []), target_mc]
        for index, row in enumerate(rows):
            if (
                row.get("fine")
                or row.get("jump_fwd") == "fine"
                or "fine" in names[index]
            ):
                edges = successors[index] or [following(index)]
                if -1 not in edges:
                    successors[index] = [*edges, -1]
        return successors

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

    def get_events(
        self, properties: bool | str | tuple[str, ...] = True
    ) -> NoteEventData:
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
            loaded_store.measures = self._with_successors(loaded_store.measures)

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

        timeline.flow_control._absorb(_flow_control_of(self.create_flow_controller()))

        notated_length = measure_map.total_actual_length
        if notated_length is not None:
            timeline.length = max(timeline.length.value, notated_length)
            timeline.lock()

    # region Flow Control

    def create_flow_controller(
        self,
        *,
        name_generator: "SegmentNameGenerator | None" = None,
    ) -> "ScoreFlowController":
        """Create a `timetoalign.ScoreFlowController` from the loaded measure data.

        The flow controller derives the repeat structure (repeats, voltas,
        jumps, D.S., D.C.) from the measure data and computes the
        traversal order.  Requires measures to have been loaded; raises
        ``ValueError`` if no measure data is available.

        Args:
            name_generator: Strategy for labelling atomic sections, forwarded
                to the ``ScoreFlowController``.  ``None`` uses the default
                naming (standard alphabet with the volta-suffix rule).

        Returns:
            A configured ``ScoreFlowController`` ready to compute flows.

        Raises:
            ValueError: If no measure data has been loaded.

        Examples:
            >>> loader = Ms3Loader.from_file("notes.tsv", "measures.tsv")
            >>> controller = loader.create_flow_controller()
            >>> flow = controller.compute_flow(FlowMode.default)
        """
        from timetoalign.timelines.flow import ScoreFlowController

        if self._store.measures is None or len(self._store.measures) == 0:
            raise ValueError(
                "Cannot create a ScoreFlowController: no measure data has been "
                "loaded.  Load a measures file first (e.g. *.measures.tsv)."
            )
        return ScoreFlowController(self._store.measures, name_generator=name_generator)

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
