"""Ms3Loader: Load scores from TSV using ms3."""

from __future__ import annotations

import logging
import math
from fractions import Fraction
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from timetoalign.core import NumberType, TimeUnit
from timetoalign.core.time import (
    coordinate_to_struct,
    rational_to_struct,
)

from .base import ScoreLoader
from .store import ScoreStore
from .stores import (
    AnnotationEventData,
    ControlEventData,
    MeasureData,
    NoteEventData,
)

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from timetoalign.timelines import ContinuousLogicalTimeline, MeasureMap


class Ms3Loader(ScoreLoader):
    """Load symbolic scores from DCML-style TSV files.

    Wraps ms3.load_tsv to load standard tabular data.
    Returns a ScoreStore with category-specific data.

    Unlike single-file loaders (PartituraLoader, Music21Loader), TSV corpora
    store each facet (notes, measures, chords, harmonies) in a separate file.
    By default the loader loads exactly the files it is given. Pass
    ``auto_discover=True`` to have it locate companion facet files
    automatically.

    Auto-discovery uses two strategies:

    1. **Flat siblings** (ms3 convention): for ``name.notes.tsv``, look for
       ``name.measures.tsv``, ``name.chords.tsv``, ``name.harmonies.tsv``
       in the same directory.
    2. **Facet directories** (DCML corpus convention): if the file sits in a
       directory whose name matches a facet (e.g., ``notes/``), look for
       sibling directories named after the other facets and find a file
       with the same stem there.

    TSV columns (notes.tsv gold standard):
    - mc, mn: Measure count/number
    - quarterbeats: Continuous logical time (Fraction string)
    - duration_qb: Duration in quarter beats (float)
    - mc_onset, mn_onset: Measure-relative offsets (Fraction string)
    - timesig: Time signature context
    - staff, voice: Part context
    - duration, nominal_duration, scalar: Symbolic duration
    - tpc, midi, name, octave: Pitch info
    - tied, gracenote, chord_id: Note attributes
    """

    # Known DCML/ms3 facet names (used for discovery).
    _FACETS: tuple[str, ...] = ("notes", "measures", "chords", "harmonies")

    def __init__(self, *args, auto_discover: bool = False, **kwargs) -> None:
        """Initialize the Ms3Loader.

        Args:
            *args: Passed to ``ScoreLoader.__init__``.
            auto_discover: If True, automatically locate companion facet
                files (measures, chords, harmonies, etc.) for every source
                file passed to `load`. Defaults to False.
            **kwargs: Passed to ``ScoreLoader.__init__``.
        """
        super().__init__(*args, **kwargs)
        self._auto_discover = auto_discover
        self._section_counts: list[int] = []

    @classmethod
    def from_file(cls, *paths: Path | str, auto_discover: bool = False) -> "Ms3Loader":
        """Create a loader and load TSV files in one step.

        Convenience constructor for loading one or more TSV files.

        Args:
            *paths: Paths to TSV files (notes, measures, harmonies, chords).
            auto_discover: If True, automatically locate companion facet
                files for each source.

        Returns:
            A new Ms3Loader instance with the files already loaded.

        Examples:
            >>> loader = Ms3Loader.from_file("score.notes.tsv", "score.measures.tsv")
            >>> loader.store.summary()

            >>> # Auto-discover all facet files from a single notes file:
            >>> loader = Ms3Loader.from_file("score.notes.tsv", auto_discover=True)
        """
        loader = cls(auto_discover=auto_discover)
        loader.load(*paths)
        return loader

    # region Auto-discovery

    @staticmethod
    def _parse_facet(path: Path) -> str | None:
        """Extract the facet name from a TSV filename.

        Recognises two naming conventions:
        - ms3 flat: ``name.notes.tsv`` -> ``"notes"``
        - plain:    ``notes.tsv``      -> ``"notes"``

        Args:
            path: Path to a TSV file.

        Returns:
            The facet name if recognised, else None.
        """
        parts = path.name.lower().rsplit(".", maxsplit=2)
        # "name.notes.tsv" -> ["name", "notes", "tsv"]
        if len(parts) == 3 and parts[2] == "tsv" and parts[1] in Ms3Loader._FACETS:
            return parts[1]
        # "notes.tsv" -> ["notes", "tsv"]
        if len(parts) == 2 and parts[1] == "tsv" and parts[0] in Ms3Loader._FACETS:
            return parts[0]
        return None

    @staticmethod
    def _discover_companions(source: Path) -> list[Path]:
        """Locate companion facet files for *source*.

        Uses two strategies in order:

        1. **Flat siblings** — for ``name.<facet>.tsv``, glob
           ``name.*.<other_facet>.tsv`` in the same directory.
        2. **Facet directories** — if the parent directory name equals the
           current facet (e.g. ``notes/``), look for sibling directories
           named after the remaining facets and find a file whose stem
           matches.

        Files with an ``_unfolded`` suffix are matched only to other
        ``_unfolded`` files and vice-versa, to avoid mixing folded and
        unfolded variants.

        Args:
            source: Resolved path to a TSV file.

        Returns:
            Companion paths (may be empty). Never includes *source* itself.
        """
        source = source.resolve()
        facet = Ms3Loader._parse_facet(source)
        if facet is None:
            return []

        companions: list[Path] = []
        other_facets = [f for f in Ms3Loader._FACETS if f != facet]
        name_lower = source.name.lower()

        # Strategy 1: flat siblings (name.facet.tsv convention)
        # Extract the piece stem: everything before ".<facet>.tsv"
        dot_facet = f".{facet}."
        idx = name_lower.find(dot_facet)
        if idx >= 0:
            piece_stem = source.name[:idx]  # preserve original case
            for other in other_facets:
                candidate = source.parent / f"{piece_stem}.{other}.tsv"
                if candidate.resolve() != source and candidate.is_file():
                    companions.append(candidate.resolve())
            if companions:
                return companions

        # Strategy 2: facet directories
        parent_name = source.parent.name.lower()
        if parent_name == facet:
            grandparent = source.parent.parent
            # Derive the expected filename in sibling facet directories.
            # Replace the facet directory name in the stem if present,
            # otherwise keep the original name.
            file_stem = source.name
            for other in other_facets:
                sibling_dir = grandparent / other
                if sibling_dir.is_dir():
                    candidate = sibling_dir / file_stem
                    if candidate.is_file():
                        companions.append(candidate.resolve())

        return companions

    def load(self, *sources: Path | str) -> Self:
        """Load one or more TSV source files.

        When ``auto_discover`` is enabled, each source file is expanded to
        include its companion facet files (measures, chords, harmonies,
        etc.) before loading. Files that have already been loaded (or that
        appear more than once in the expanded set) are silently skipped.

        Args:
            *sources: Paths to TSV files.

        Returns:
            Self, for method chaining.
        """
        if not self._auto_discover:
            return super().load(*sources)

        # Expand sources with discovered companions, preserving order and
        # deduplicating by resolved path.
        seen: set[Path] = {p.resolve() for p in self._sources}
        expanded: list[Path] = []
        for source in sources:
            path = Path(source).resolve()
            if path in seen:
                continue
            seen.add(path)
            expanded.append(path)
            for companion in self._discover_companions(path):
                if companion not in seen:
                    seen.add(companion)
                    expanded.append(companion)
                    logger.debug("Auto-discovered companion: %s", companion)

        return super().load(*expanded)

    # endregion

    def _load_source(self, source: Path) -> ScoreStore:
        """Load TSV file(s) and return ScoreStore.

        Args:
            source: Path to TSV file or directory.

        Returns:
            ScoreStore with populated data.
        """
        fname = source.name.lower()

        if "measures" in fname:
            if pd is None:
                raise ImportError(
                    "Ms3Loader requires pandas to read measures TSV files"
                )
            df = pd.read_csv(source, sep="\t")
        else:
            try:
                import ms3
            except ImportError:
                raise ImportError(
                    "Ms3Loader requires 'ms3' for non-measure facets. "
                    "Install with pip install ms3"
                )
            df = ms3.load_tsv(str(source))

        # Determine category from filename
        if "measures" in fname:
            return self._load_measures(df, source)
        elif "notes" in fname:
            return self._load_notes(df, source)
        elif "harmonies" in fname:
            return self._load_annotations(df, source, subtype="Harmony")
        elif "chords" in fname:
            return self._load_controls(df, source, subtype="Chord")
        else:
            # Default to notes if has pitch columns
            if "midi" in df.columns or "tpc" in df.columns:
                return self._load_notes(df, source)
            return ScoreStore.empty()

    def _parse_fraction(self, val: Any) -> Fraction | None:
        """Parse a value when its exact rational representation is trusted.

        Decimal strings are source values and therefore remain exact. Native
        floats are treated as binary rationals only when their denominator is
        compact enough to be a faithful symbolic value rather than a derived
        decimal artifact.

        Args:
            val: A TSV value to parse.

        Returns:
            An exact fraction, or ``None`` when the value is missing or
            should remain value-only.
        """
        if val is None:
            return None
        if isinstance(val, Real):
            numeric = float(val)
            if not math.isfinite(numeric):
                return None
        elif pd is not None:
            try:
                if bool(pd.isna(val)):
                    return None
            except (TypeError, ValueError):
                pass

        if isinstance(val, Fraction):
            return val
        if isinstance(val, str):
            if "/" in val:
                parts = val.split("/")
                try:
                    return Fraction(int(parts[0]), int(parts[1]))
                except (ValueError, ZeroDivisionError):
                    return None
            try:
                return Fraction(val)
            except (ValueError, ZeroDivisionError):
                return None
        if isinstance(val, Integral):
            return Fraction(int(val))
        if isinstance(val, Real):
            if numeric.is_integer():
                return Fraction(int(numeric))
            numerator, denominator = numeric.as_integer_ratio()
            if denominator <= 4096:
                return Fraction(numerator, denominator)
        return None

    def _coordinate_value(self, val: Any) -> Fraction | float | None:
        """Parse a coordinate exactly, retaining finite inexact floats.

        Args:
            val: A coordinate value from a TSV row.

        Returns:
            An exact fraction, a finite float without an exact pair, or None.
        """
        exact = self._parse_fraction(val)
        if exact is not None:
            return exact
        if isinstance(val, Real):
            numeric = float(val)
            if math.isfinite(numeric):
                return numeric
        return None

    @staticmethod
    def _has_value(val: Any) -> bool:
        """Return whether a scalar TSV value is present and finite."""
        if val is None:
            return False
        if isinstance(val, Real):
            return math.isfinite(float(val))
        if pd is not None:
            try:
                return not bool(pd.isna(val))
            except (TypeError, ValueError):
                return True
        return True

    def _resolve_duration(
        self, row: "pd.Series", exact_column: str
    ) -> Fraction | float | None:
        """Resolve a quarter-beat duration from symbolic and derived columns.

        Args:
            row: A pandas row from an MS3 facet.
            exact_column: The whole-note symbolic duration column.

        Returns:
            A quarter-beat duration. Exact values are fractions; inexact
            values are finite floats; missing values are None unless the
            derived column itself is absent, preserving the old zero default.
        """
        exact_source = self._parse_fraction(row.get(exact_column))
        has_duration_qb = "duration_qb" in row
        duration_qb = row.get("duration_qb")

        if exact_source is not None:
            exact_duration = exact_source * 4
            if has_duration_qb and self._has_value(duration_qb):
                derived_float = float(duration_qb)
                if abs(float(exact_duration) - derived_float) > 1e-6:
                    return self._coordinate_value(duration_qb)
            return exact_duration

        if has_duration_qb and self._has_value(duration_qb):
            return self._coordinate_value(duration_qb)
        if not has_duration_qb:
            return Fraction(0)
        return None

    def _piece_end_from_measures(self, source: Path) -> Fraction | None:
        """Exact end of the piece, read from the companion measures facet.

        A label's span runs to the next label; the final label runs to the end
        of the piece, which a harmonies facet never states. The measures facet
        states it exactly, as the furthest measure position plus that
        measure's actual duration.

        Args:
            source: The harmonies file being loaded.

        Returns:
            The exact end position in quarters, or None when no measures
            companion is available or its cells are not exact.
        """
        try:
            import ms3
        except ImportError:  # pragma: no cover - guarded by the caller's import
            return None

        for companion in self._discover_companions(source):
            if self._parse_facet(companion) != "measures":
                continue
            try:
                measures = ms3.load_tsv(str(companion))
            except Exception:
                logger.debug("Could not read measures companion: %s", companion)
                return None

            end: Fraction | None = None
            for _, row in measures.iterrows():
                start = self._resolve_quarterbeats(row)
                act_dur = self._parse_fraction(row.get("act_dur"))
                if not isinstance(start, Fraction) or act_dur is None:
                    continue
                candidate = start + act_dur * 4
                if end is None or candidate > end:
                    end = candidate
            return end

        return None

    def _derive_label_spans(
        self, starts: list[Fraction | float | None], source: Path
    ) -> list[Fraction | None]:
        """Span of each label: exactly up to the label that follows it.

        A DCML harmonies facet carries no symbolic duration column. It states
        ``duration_qb``, which is this very subtraction carried out in float —
        and the float keeps the residue: a sixth of a quarter arrives as
        ``0.16666666666668561``, which is not even the nearest double to 1/6.
        Redoing the subtraction on the exact ``quarterbeats`` cells recovers
        the value the source already encodes. That is the source's own
        derivation without its rounding, not a semantics this loader invented,
        and it is what keeps a label position exact instead of turning into a
        56-bit dyadic that no longer fits the coordinate struct once added to
        a position of any musical size.

        Args:
            starts: Resolved label positions, in file order.
            source: The harmonies file, used to find the measures companion.

        Returns:
            One span per label, or None where the pair is not exact and the
            caller should fall back to the derived column.
        """
        piece_end = self._piece_end_from_measures(source)
        spans: list[Fraction | None] = []
        for position, start in enumerate(starts):
            following = (
                starts[position + 1] if position + 1 < len(starts) else piece_end
            )
            if (
                isinstance(start, Fraction)
                and isinstance(following, Fraction)
                and following >= start
            ):
                spans.append(following - start)
            else:
                # Out of order, missing, or inexact: say nothing rather than
                # assert a span the source does not support.
                spans.append(None)
        return spans

    def _resolve_quarterbeats(self, row: "pd.Series") -> Fraction | float | None:
        """Resolve the quarterbeats coordinate from a TSV row.

        Prefers ``quarterbeats_all_endings`` when the column is present
        in the row, falling back to ``quarterbeats`` otherwise.  This is
        required for scores with volta brackets: the ``quarterbeats``
        column uses a coordinate space that *skips* first-ending measures
        (leaving them as NaN), whereas ``quarterbeats_all_endings``
        assigns every measure a unique position in a continuous coordinate
        space that includes all volta endings.

        Using ``quarterbeats_all_endings`` as the primary source ensures
        that the resulting `timetoalign.Timeline` has a coordinate range
        that spans the full folded score, matching the cumulative
        MeasureUnit durations used by `compute_qb_sections`.

        Args:
            row: A pandas Series representing one row of the TSV.

        Returns:
            Parsed Fraction coordinate, or *None* if neither column
            provides a usable value.
        """
        # Prefer quarterbeats_all_endings (includes all volta endings)
        qb_all = self._coordinate_value(row.get("quarterbeats_all_endings"))
        if qb_all is not None:
            return qb_all
        # Fallback: plain quarterbeats
        return self._coordinate_value(row.get("quarterbeats"))

    def _load_notes(self, df: pd.DataFrame, source: Path) -> ScoreStore:
        """Load notes TSV into NoteEventData."""
        import pandas as pd

        if df.empty:
            return ScoreStore.empty()

        note_rows = []
        has_rests = False

        for _, row in df.iterrows():
            # Temporal - Primary (with quarterbeats_all_endings fallback)
            qb = self._resolve_quarterbeats(row)

            dur_qb = self._resolve_duration(row, "duration")

            # Temporal - Measure context
            mc = int(row["mc"]) if pd.notna(row.get("mc")) else None
            mn = str(row.get("mn", "")) if pd.notna(row.get("mn")) else None
            mc_onset = self._parse_fraction(row.get("mc_onset"))
            mn_onset = self._parse_fraction(row.get("mn_onset"))
            timesig = (
                str(row.get("timesig", "")) if pd.notna(row.get("timesig")) else None
            )

            # Temporal - Symbolic duration
            duration = self._parse_fraction(row.get("duration"))
            nominal_duration = self._parse_fraction(row.get("nominal_duration"))
            scalar = self._parse_fraction(row.get("scalar"))

            # Pitch - source MIDI number (raw int; NON-DEFAULT — affords an
            # EnharmonicPitch view on request, redundant with specific_pitch).
            midi = int(row["midi"]) if pd.notna(row.get("midi")) else None

            # Pitch - Specific (canonical pydantic-derived shape)
            specific_pitch = None
            if pd.notna(row.get("name")):
                name = str(row["name"])
                step = name[0].upper() if name else "C"
                alter = 0
                if len(name) > 1:
                    alter = (
                        name[1:].count("#") - name[1:].count("b") - name[1:].count("-")
                    )

                octave = int(row["octave"]) if pd.notna(row.get("octave")) else 4

                specific_pitch = {
                    "step": step,
                    "alter": alter,
                    "octave": octave,
                    "cents": 0.0,
                }

            octave_val = int(row["octave"]) if pd.notna(row.get("octave")) else None
            tpc_val = int(row["tpc"]) if pd.notna(row.get("tpc")) else None

            # Attributes
            tied = int(row["tied"]) if pd.notna(row.get("tied")) else 0
            gracenote = (
                str(row["gracenote"]) if pd.notna(row.get("gracenote")) else None
            )
            chord_id = (
                str(int(row["chord_id"])) if pd.notna(row.get("chord_id")) else None
            )

            voice = int(row["voice"]) if pd.notna(row.get("voice")) else None
            staff = int(row["staff"]) if pd.notna(row.get("staff")) else None
            part_id = str(row.get("part", "P1"))

            # Check if rest (no pitch)
            if midi is None:
                has_rests = True

            note_rows.append(
                {
                    "name": str(row.get("name", "")),
                    "temporal_type": (
                        "interval" if (dur_qb is not None and dur_qb > 0) else "instant"
                    ),
                    "event_type": "Note" if midi is not None else "Rest",
                    # Temporal - core coordinates use coordinate_to_struct
                    # (value/numerator/denominator format for EventData)
                    "quarterbeats": (
                        coordinate_to_struct(qb)
                        if qb is not None
                        else coordinate_to_struct(0)
                    ),
                    "duration_qb": (
                        coordinate_to_struct(dur_qb) if dur_qb is not None else None
                    ),
                    "mc": mc,
                    "mn": mn,
                    # Extra fraction fields use the canonical rational struct
                    "mc_onset": (
                        rational_to_struct(mc_onset) if mc_onset is not None else None
                    ),
                    "mn_onset": (
                        rational_to_struct(mn_onset) if mn_onset is not None else None
                    ),
                    "timesig": timesig,
                    "duration": (
                        rational_to_struct(duration) if duration is not None else None
                    ),
                    "nominal_duration": (
                        rational_to_struct(nominal_duration)
                        if nominal_duration is not None
                        else None
                    ),
                    "scalar": (
                        rational_to_struct(scalar) if scalar is not None else None
                    ),
                    # Pitch — specific_pitch is the sole default semantic
                    # pitch field; ``midi`` is the redundant raw number.
                    "specific_pitch": specific_pitch,
                    "midi": midi,
                    "tpc": tpc_val,
                    "octave": octave_val,
                    # Attributes
                    "tied": tied,
                    "gracenote": gracenote,
                    "chord_id": chord_id,
                    "voice": voice,
                    "staff": staff,
                    "part_id": part_id,
                }
            )

        notes_data = getattr(NoteEventData, "from_" + "dicts")(
            note_rows,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
            has_rests=has_rests,
        )

        return ScoreStore(
            notes=notes_data,
            measures=MeasureData.empty(),
            controls=ControlEventData.empty(),
            annotations=AnnotationEventData.empty(),
            metadata={
                "format": "tsv",
                "parser": "ms3",
                "source": str(source),
                "has_rests": has_rests,
            },
        )

    @property
    def measure_map(self) -> MeasureMap:
        """The immutable measure structure parsed from a measures TSV.

        Each access constructs a new ``MeasureMap`` from the measure store.
        """
        if len(self.store.measures) == 0:
            raise ValueError("No measures TSV has been loaded")
        from timetoalign.timelines import MeasureMap

        return MeasureMap.from_measure_data(self.store.measures)

    def create_timeline(
        self,
        uid: str | None = None,
        name: str | None = None,
        *,
        flatten: bool = False,
        **kwargs: Any,
    ) -> ContinuousLogicalTimeline:
        """Create a score timeline and, by default, attach parsed structure.

        Known limitation: a score whose measures are **split** — two
        records sharing one notated downbeat, as when a bar is divided
        across a repeat sign — has no floating-measure lattice, because
        the two halves cannot both anchor the same ordinal. The timeline
        then carries no ``floating_measures`` conversion; every other
        conversion, and the measure structure itself, is unaffected. No
        approximate lattice is built in its place, since an fm reading
        those measures do not support is worse than no fm reading.
        """
        from timetoalign.alignment import TimeSkeleton
        from timetoalign.timelines import ContinuousLogicalTimeline, SectionHierarchy

        has_measures = len(self.store.measures) > 0
        if has_measures and all(
            len(store) == 0
            for store in (self.store.notes, self.store.controls, self.store.annotations)
        ):
            timeline = ContinuousLogicalTimeline(
                length=0,
                unit=TimeUnit.quarters,
                number_type=NumberType.fraction,
                uid=uid,
                name=name,
            )
            self._populate_timeline_structure(timeline)
            for cmap in self.store.get_cmaps().values():
                timeline.add_conversion_map(cmap)
        else:
            excluded = list(kwargs.pop("exclude_stores", []) or [])
            if "measures" not in excluded:
                excluded.append("measures")
            timeline = super().create_timeline(
                uid=uid,
                exclude_stores=excluded,
                flatten=flatten,
                **kwargs,
            )
            if name is not None:
                timeline.name = name

        if timeline.measure_map is not None:
            notated_length = timeline.measure_map.total_actual_length or Fraction(0)
            timeline.length = max(timeline.length.value, notated_length)
            if not flatten:
                counts = self._section_counts or [len(timeline.measure_map)]
                hierarchy = SectionHierarchy.from_measure_map(
                    timeline.measure_map, counts=counts
                )
                TimeSkeleton(hierarchy).attach(timeline)
        return timeline

    def _load_measures(self, df: pd.DataFrame, source: Path) -> ScoreStore:
        """Parse a measures TSV as an immutable measure map.

        Parses ms3/DCML measures.tsv format with columns:
        - mc, mn: Measure count/number
        - quarterbeats: Start position in quarter notes
        - quarterbeats_all_endings: Alternative qstamp including all endings
        - duration_qb: Duration in quarter beats
        - keysig: Key signature (fifths value or string)
        - timesig: Time signature string
        - act_dur: Actual duration as fraction string
        - mc_offset: Offset for split bars
        - volta: Ending number (1, 2, ...)
        - numbering_offset: MN offset
        - dont_count: Skip in MN counting
        - barline: Barline type
        - breaks: Section boundary marker
        - repeats: "start", "end", "firstMeasure"
        - next: Comma-separated list of possible next MCs

        Args:
            df: pandas DataFrame from ms3.load_tsv.
            source: Path to source file.

        Returns:
            ScoreStore retaining the source facts needed by legacy flow readers.
        """
        import pandas as pd

        if df.empty:
            return ScoreStore.empty()

        measure_rows = []

        for idx, row in df.iterrows():
            # ===== Core Identity =====
            mc = int(row["mc"]) if pd.notna(row.get("mc")) else idx + 1
            mn = str(row.get("mn", mc)) if pd.notna(row.get("mn")) else str(mc)

            # Parse MN as integer (strip suffix if present)
            try:
                mn_int = int("".join(c for c in mn if c.isdigit() or c == "-"))
            except ValueError:
                mn_int = mc

            # ===== Temporal (with quarterbeats_all_endings fallback) =====
            qb = self._resolve_quarterbeats(row)

            dur_qb = self._resolve_duration(row, "act_dur")

            # act_dur is often a fraction string like "1/2" for half a bar
            act_dur = row.get("act_dur")
            if self._has_value(act_dur):
                actual_length = float(dur_qb) if dur_qb is not None else None
            else:
                actual_length = float(dur_qb) if dur_qb is not None else 0.0

            # mc_offset for split bars
            mc_offset = (
                str(row.get("mc_offset"))
                if pd.notna(row.get("mc_offset"))
                and row.get("mc_offset") not in (0, "0")
                else None
            )

            # quarterbeats_all_endings
            qb_all = (
                str(row.get("quarterbeats_all_endings"))
                if pd.notna(row.get("quarterbeats_all_endings"))
                else None
            )

            # ===== Signatures =====
            timesig = (
                str(row.get("timesig", "")) if pd.notna(row.get("timesig")) else None
            )

            # Parse timesig components
            timesig_num = None
            timesig_den = None
            nominal_length = None
            if timesig and "/" in timesig:
                try:
                    parts = timesig.split("/")
                    timesig_num = int(parts[0])
                    timesig_den = int(parts[1])
                    # nominal_length in quarters = (num / den) * 4
                    nominal_length = (timesig_num / timesig_den) * 4
                except (ValueError, IndexError):
                    pass

            # keysig: can be int (fifths) or string
            keysig = row.get("keysig")
            keysig_fifths = None
            if pd.notna(keysig):
                try:
                    keysig_fifths = int(keysig)
                    keysig = None  # Will use fifths value
                except (ValueError, TypeError):
                    keysig = str(keysig)

            # ===== Flow Control =====
            # repeats column: "start", "end", "firstMeasure"
            repeats_val = (
                str(row.get("repeats", "")) if pd.notna(row.get("repeats")) else None
            )

            start_repeat = repeats_val in ("start",) if repeats_val else False
            end_repeat = repeats_val in ("end",) if repeats_val else False

            # volta (ending number)
            volta = int(row["volta"]) if pd.notna(row.get("volta")) else None

            # breaks (section boundary)
            breaks_val = (
                str(row.get("breaks", "")) if pd.notna(row.get("breaks")) else None
            )

            # ms3 jump/marker columns: markers, jump_bwd, jump_fwd, play_until
            def _str_or_none(name: str) -> str | None:
                val = row.get(name)
                if val is None or not pd.notna(val):
                    return None
                s = str(val).strip()
                return s or None

            markers_val = _str_or_none("markers")
            jump_bwd_val = _str_or_none("jump_bwd")
            jump_fwd_val = _str_or_none("jump_fwd")
            play_until_val = _str_or_none("play_until")

            # dont_count
            dont_count = (
                bool(row.get("dont_count")) if pd.notna(row.get("dont_count")) else None
            )

            # numbering_offset
            numbering_offset = (
                int(row["numbering_offset"])
                if pd.notna(row.get("numbering_offset"))
                else None
            )

            # barline
            barline = (
                str(row.get("barline", "")) if pd.notna(row.get("barline")) else None
            )

            # next: comma-separated list of possible next MCs
            next_val = row.get("next")
            if pd.notna(next_val):
                next_str = str(next_val).strip()
            else:
                next_str = None

            # ===== Build row =====
            measure_rows.append(
                {
                    # Identity - mc:NNNNN format based on measure count
                    "id": f"mc:{mc:05d}",
                    "name": f"M{mn}",
                    "temporal_type": "interval",
                    "event_type": "Measure",
                    # Core fields
                    "mc": mc,
                    "mn": mn,
                    "mn_int": mn_int,
                    # Temporal
                    "start": (
                        coordinate_to_struct(qb)
                        if qb is not None
                        else coordinate_to_struct(0)
                    ),
                    "duration": (
                        coordinate_to_struct(dur_qb)
                        if dur_qb is not None
                        else coordinate_to_struct(0)
                    ),
                    "end": coordinate_to_struct(
                        (qb if qb is not None else Fraction(0))
                        + (dur_qb if dur_qb is not None else Fraction(0))
                    ),
                    "nominal_length": nominal_length,
                    "actual_length": actual_length,
                    "mc_offset": mc_offset,
                    "quarterbeats_all_endings": qb_all,
                    # Signatures
                    "timesig": timesig,
                    "timesig_num": timesig_num,
                    "timesig_den": timesig_den,
                    "keysig": keysig,
                    "keysig_fifths": keysig_fifths,
                    # Flow control
                    "start_repeat": start_repeat,
                    "end_repeat": end_repeat,
                    "next": next_str,
                    "volta": volta,
                    "repeats": repeats_val,
                    "breaks": breaks_val,
                    "markers": markers_val,
                    "jump_bwd": jump_bwd_val,
                    "jump_fwd": jump_fwd_val,
                    "play_until": play_until_val,
                    "dont_count": dont_count,
                    "numbering_offset": numbering_offset,
                    # Context
                    "barline": barline,
                }
            )

        measures_data = MeasureData.from_dicts(
            measure_rows,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )
        section_ends = [
            int(row["mc"])
            for row in measure_rows
            if "section" in str(row.get("breaks") or "").lower()
        ]
        previous = 0
        self._section_counts = []
        for end in [*section_ends, len(measure_rows)]:
            self._section_counts.append(end - previous)
            previous = end

        return ScoreStore(
            notes=NoteEventData.empty(),
            measures=measures_data,
            controls=ControlEventData.empty(),
            annotations=AnnotationEventData.empty(),
            metadata={
                "format": "tsv",
                "parser": "ms3",
                "source": str(source),
                "n_measures": len(measure_rows),
                "flow_control": measures_data.get_flow_control_summary(),
            },
        )

    def _load_annotations(
        self, df: "pd.DataFrame", source: Path, subtype: str
    ) -> ScoreStore:
        """Load annotations from TSV (harmonies).

        Parses ms3/DCML harmonies.tsv format with columns:
        - mc, mn: Measure context
        - quarterbeats: Position in quarter notes
        - label: The annotation text (Roman numeral harmony label)
        - numeral, form, figbass, relativeroot, etc.: Parsed components

        Args:
            df: pandas DataFrame from ms3.load_tsv.
            source: Path to source file.
            subtype: The annotation subtype ("Harmony").

        Returns:
            ScoreStore with populated AnnotationEventData.
        """
        import pandas as pd

        if df.empty:
            return ScoreStore.empty()

        annotation_rows = []

        # Temporal (with quarterbeats_all_endings fallback). Resolved up
        # front because a label's span is defined by the label after it.
        indexed_rows = list(df.iterrows())
        starts = [self._resolve_quarterbeats(row) for _, row in indexed_rows]
        label_spans = self._derive_label_spans(starts, source)

        for position, (idx, row) in enumerate(indexed_rows):
            qb = starts[position]

            # Duration: the exact span to the next label, falling back to the
            # source's own derived column where that span is not available.
            dur_qb = label_spans[position]
            if dur_qb is None:
                dur_qb = self._resolve_duration(row, "duration")

            # Measure context
            mc = int(row["mc"]) if pd.notna(row.get("mc")) else None
            mn = str(row.get("mn", "")) if pd.notna(row.get("mn")) else None
            mc_onset = self._parse_fraction(row.get("mc_onset"))
            mn_onset = self._parse_fraction(row.get("mn_onset"))

            # Label - try various column names used in DCML
            label = None
            for label_col in ["label", "chord", "numeral", "globalkey_is_minor"]:
                if pd.notna(row.get(label_col)):
                    label = str(row[label_col])
                    break

            if label is None:
                # Construct label from components if available
                parts = []
                for col in ["numeral", "form", "figbass", "relativeroot"]:
                    if pd.notna(row.get(col)) and row.get(col):
                        parts.append(str(row[col]))
                label = "".join(parts) if parts else f"{subtype}_{idx}"

            # Staff context
            staff = int(row["staff"]) if pd.notna(row.get("staff")) else None

            annotation_rows.append(
                {
                    # ID auto-generated from event_type (subtype is used as event_type)
                    "name": label,
                    "temporal_type": "interval" if dur_qb else "instant",
                    "event_type": subtype,  # Use subtype (e.g., "Harmony") as event_type
                    "subtype": subtype,
                    "text": label,
                    # Temporal
                    "quarterbeats": (
                        coordinate_to_struct(qb)
                        if qb is not None
                        else coordinate_to_struct(0)
                    ),
                    "duration_qb": (
                        coordinate_to_struct(dur_qb) if dur_qb is not None else None
                    ),
                    # Measure context
                    "mc": mc,
                    "mn": mn,
                    "mc_onset": (
                        rational_to_struct(mc_onset) if mc_onset is not None else None
                    ),
                    "mn_onset": (
                        rational_to_struct(mn_onset) if mn_onset is not None else None
                    ),
                    # Context
                    "staff": staff,
                }
            )

        annotations_data = getattr(AnnotationEventData, "from_" + "dicts")(
            annotation_rows,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        return ScoreStore(
            notes=NoteEventData.empty(),
            measures=MeasureData.empty(),
            controls=ControlEventData.empty(),
            annotations=annotations_data,
            metadata={
                "format": "tsv",
                "parser": "ms3",
                "source": str(source),
                "annotation_type": subtype,
                "n_annotations": len(annotation_rows),
            },
        )

    def _load_controls(
        self, df: "pd.DataFrame", source: Path, subtype: str
    ) -> ScoreStore:
        """Load control events from TSV (chords).

        Parses ms3/DCML chords.tsv format with columns:
        - mc, mn: Measure context
        - quarterbeats: Position in quarter notes
        - chord: The chord symbol text
        - pedal, numeral, form, figbass, etc.: Parsed components

        Args:
            df: pandas DataFrame from ms3.load_tsv.
            source: Path to source file.
            subtype: The control subtype ("Chord").

        Returns:
            ScoreStore with populated ControlEventData.
        """
        import pandas as pd

        if df.empty:
            return ScoreStore.empty()

        control_rows = []

        for idx, row in df.iterrows():
            # Temporal (with quarterbeats_all_endings fallback)
            qb = self._resolve_quarterbeats(row)

            # Duration (if available)
            dur_qb = self._resolve_duration(row, "duration")

            # Measure context
            mc = int(row["mc"]) if pd.notna(row.get("mc")) else None
            mn = str(row.get("mn", "")) if pd.notna(row.get("mn")) else None
            mc_onset = self._parse_fraction(row.get("mc_onset"))
            mn_onset = self._parse_fraction(row.get("mn_onset"))

            # Label - try various column names used in DCML
            label = None
            for label_col in ["chord", "label", "numeral"]:
                if pd.notna(row.get(label_col)):
                    label = str(row[label_col])
                    break

            if label is None:
                # Construct label from components if available
                parts = []
                for col in ["numeral", "form", "figbass", "relativeroot"]:
                    if pd.notna(row.get(col)) and row.get(col):
                        parts.append(str(row[col]))
                label = "".join(parts) if parts else f"{subtype}_{idx}"

            # Staff/voice context
            staff = int(row["staff"]) if pd.notna(row.get("staff")) else None
            voice = int(row["voice"]) if pd.notna(row.get("voice")) else None

            control_rows.append(
                {
                    # ID auto-generated from event_type (subtype is used as event_type)
                    "name": label,
                    "temporal_type": "interval" if dur_qb else "instant",
                    "event_type": subtype,  # Use subtype (e.g., "Chord") as event_type
                    "subtype": subtype,
                    "text": label,
                    "value": None,  # Chord symbols don't have numeric values
                    # Temporal
                    "quarterbeats": (
                        coordinate_to_struct(qb)
                        if qb is not None
                        else coordinate_to_struct(0)
                    ),
                    "duration_qb": (
                        coordinate_to_struct(dur_qb) if dur_qb is not None else None
                    ),
                    # Measure context
                    "mc": mc,
                    "mn": mn,
                    "mc_onset": (
                        rational_to_struct(mc_onset) if mc_onset is not None else None
                    ),
                    "mn_onset": (
                        rational_to_struct(mn_onset) if mn_onset is not None else None
                    ),
                    # Context
                    "staff": staff,
                    "voice": voice,
                }
            )

        controls_data = getattr(ControlEventData, "from_" + "dicts")(
            control_rows,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        return ScoreStore(
            notes=NoteEventData.empty(),
            measures=MeasureData.empty(),
            controls=controls_data,
            annotations=AnnotationEventData.empty(),
            metadata={
                "format": "tsv",
                "parser": "ms3",
                "source": str(source),
                "control_type": subtype,
                "n_controls": len(control_rows),
            },
        )
