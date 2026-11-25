"""Ms3Loader: Load scores from TSV using ms3."""

from __future__ import annotations

import logging
import math
import warnings
from fractions import Fraction
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from timetoalign.core import NumberType, TimeUnit
from timetoalign.core.events import canonical_pitch_name
from timetoalign.core.time import (
    coordinate_to_struct,
    rational_to_struct,
)
from timetoalign.loader._optional import missing_dependency
from timetoalign.loader._sources import (
    Source,
    SourceLike,
    as_source,
    describe_source,
    readable,
    source_parent,
    source_record,
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
    import pandas

    from timetoalign.timelines import MeasureMap, Timeline


def _holds_no_table(source: Source) -> ValueError:
    """The refusal of an ms3 TSV file that holds no table, naming it."""
    return ValueError(
        f"Ms3Loader cannot read '{describe_source(source)}': the file holds no table (an ms3 TSV "
        "file states its columns in its first line)."
    )


def _states_a_table(path: Source) -> bool:
    """Whether *path* holds anything but whitespace (an ms3 TSV states its columns)."""
    with path.open("rb") as file:
        return any(line.strip() for line in file)


def _source_key(source: Source) -> Path | tuple[str, str]:
    """What tells two sources apart: a resolved path, or archive and member."""
    if isinstance(source, Path):
        return source.resolve()
    provenance = source_record(source)
    return provenance["path"], provenance["member"]


class Ms3Loader(ScoreLoader):
    """Load symbolic scores from DCML-style TSV files.

    Wraps ms3.load_tsv to load standard tabular data.
    Returns a ScoreStore with category-specific data.

    Unlike single-file loaders (PartituraLoader, Music21Loader), TSV corpora
    store each facet (notes, measures, chords, harmonies) in a separate file.
    By default the loader loads exactly the files it is given. Pass
    ``auto_discover=True`` to have it locate companion facet files
    automatically.

    Auto-discovery keeps the piece name and swaps the facet, looking in two
    places for each other facet:

    1. **Flat siblings** (ms3 convention): for ``name.notes.tsv``, look for
       ``name.measures.tsv``, ``name.chords.tsv``, ``name.harmonies.tsv``
       in the same directory.
    2. **Facet directories** (DCML corpus convention): if the file sits in a
       directory named after its facet (``notes/name.notes.tsv``), look in
       the sibling directories named after the other facets
       (``measures/name.measures.tsv``, ``chords/name.chords.tsv``,
       ``harmonies/name.harmonies.tsv``).

    A facet the piece lacks is skipped; a file for which no companion is
    found at all raises a ``UserWarning`` and is loaded alone.

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

    # region Auto-discovery

    @staticmethod
    def _parse_facet(path: Source) -> str | None:
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
    def _companion_candidates(source: Source) -> dict[str, list[Source]]:
        """Paths where each other facet of *source*'s piece may be stored.

        A facet file is named ``<piece>.<facet>.tsv`` (or ``<facet>.tsv``
        when the piece has no name of its own). Each other facet keeps the
        piece name and swaps the facet: ``<piece>.<other>.tsv``. It is
        looked for in two places, in this order:

        1. **Flat siblings** (ms3 convention): the directory of *source*.
        2. **Facet directories** (DCML corpus convention): when *source*
           lies in a directory named after its facet (``notes/``), the
           sibling directory named after the other facet (``measures/``).

        Because the piece name is kept whole, an ``_unfolded`` file is
        matched only to ``_unfolded`` files and a folded file only to folded
        ones.

        The candidates of an archive member are members of the same archive.

        Args:
            source: Resolved path to a TSV file, or a member of a ZIP archive.

        Returns:
            For each facet other than *source*'s, the candidate paths in
            search order; empty when *source* names no facet.
        """
        facet = Ms3Loader._parse_facet(source)
        if facet is None:
            return {}
        # "<piece>.<facet>.tsv" -> "<piece>"; a bare "<facet>.tsv" -> "".
        piece = source.name[: max(len(source.name) - len(f".{facet}.tsv"), 0)]
        directory = source_parent(source)
        in_facet_directory = directory.name.lower() == facet
        candidates: dict[str, list[Source]] = {}
        for other in Ms3Loader._FACETS:
            if other == facet:
                continue
            file_name = f"{piece}.{other}.tsv" if piece else f"{other}.tsv"
            candidates[other] = [directory / file_name]
            if in_facet_directory:
                candidates[other].append(source_parent(directory) / other / file_name)
        return candidates

    @staticmethod
    def _discover_companions(source: Source) -> list[Source]:
        """Locate companion facet files for *source*.

        For each other facet, the first existing candidate of
        `_companion_candidates` is taken, so a piece may keep some facets
        beside its notes and others in facet directories.

        Args:
            source: Path to a TSV file, or a member of a ZIP archive.

        Returns:
            Companion paths, resolved (members of *source*'s archive for a
            member), in facet order (may be empty). Never includes *source*
            itself.
        """
        if isinstance(source, Path):
            source = source.resolve()
        key = _source_key(source)
        companions: list[Source] = []
        for candidates in Ms3Loader._companion_candidates(source).values():
            for candidate in candidates:
                if candidate.is_file() and _source_key(candidate) != key:
                    companions.append(
                        candidate.resolve()
                        if isinstance(candidate, Path)
                        else candidate
                    )
                    break
        return companions

    def load(self, *sources: SourceLike) -> Self:
        """Load one or more TSV source files.

        When ``auto_discover`` is enabled, each source file is expanded to
        include its companion facet files (measures, chords, harmonies,
        etc.) before loading. Files that have already been loaded (or that
        appear more than once in the expanded set) are silently skipped, and
        so is a discovered companion that holds no table (an empty file); a
        source the caller names that holds no table is refused.
        A facet the piece does not have is not looked for further; a source
        for which no companion at all is found raises a ``UserWarning``
        naming the paths that were looked at, and is loaded alone.

        A member of a ZIP archive (``zipfile.Path``) finds its companions
        among the members of the same archive.

        Args:
            *sources: Paths to TSV files, or members of ZIP archives.

        Returns:
            Self, for method chaining.
        """
        if not self._auto_discover:
            return super().load(*sources)

        # Expand sources with discovered companions, preserving order and
        # deduplicating by resolved path (by archive and member for a member).
        seen: set[Path | tuple[str, str]] = {_source_key(p) for p in self._sources}
        expanded: list[Source] = []
        for source in sources:
            path = as_source(source)
            if isinstance(path, Path):
                path = path.resolve()
            key = _source_key(path)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(path)
            companions = self._discover_companions(path)
            if not companions:
                self._warn_no_companion(path)
            for companion in companions:
                companion_key = _source_key(companion)
                if companion_key not in seen:
                    seen.add(companion_key)
                    if not _states_a_table(companion):
                        logger.debug(
                            "Companion holds no table, skipped: %s",
                            describe_source(companion),
                        )
                        continue
                    expanded.append(companion)
                    logger.debug(
                        "Auto-discovered companion: %s", describe_source(companion)
                    )

        return super().load(*expanded)

    @staticmethod
    def _warn_no_companion(source: Source) -> None:
        """Warn that auto-discovery found no companion facet for *source*."""
        candidates = Ms3Loader._companion_candidates(source)
        if not candidates:
            detail = (
                "its file name states no facet (expected '<piece>.<facet>.tsv' "
                f"with a facet among {list(Ms3Loader._FACETS)})"
            )
        else:
            looked_at = [
                describe_source(path) for paths in candidates.values() for path in paths
            ]
            detail = f"none of these files exists: {looked_at}"
        warnings.warn(
            f"auto_discover found no companion facet file for '{describe_source(source)}': {detail}. "
            "It is loaded alone.",
            UserWarning,
            stacklevel=3,
        )

    # endregion

    def _load_source(self, source: Source) -> ScoreStore:
        """Load TSV file(s) and return ScoreStore.

        Args:
            source: Path to a TSV file, or a member of a ZIP archive (read
                in place).

        Returns:
            ScoreStore with populated data.

        Raises:
            ValueError: If the file holds no table (it is empty).
        """
        fname = source.name.lower()

        if "measures" in fname:
            if pd is None:
                raise ImportError(
                    "Ms3Loader requires pandas to read measures TSV files"
                )
            try:
                with readable(source) as table:
                    df = pd.read_csv(table, sep="\t")
            except pd.errors.EmptyDataError as error:
                raise _holds_no_table(source) from error
        else:
            try:
                import ms3
            except ModuleNotFoundError as error:
                raise ImportError(
                    missing_dependency("Ms3Loader", error.name, "ms3")
                ) from error
            with readable(source) as table:
                df = ms3.load_tsv(table)
            if df is None:
                raise _holds_no_table(source)

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
            numeric = float(val)
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
        self, row: pandas.Series, exact_column: str
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
            if (
                has_duration_qb
                and duration_qb is not None
                and self._has_value(duration_qb)
            ):
                derived_float = float(duration_qb)
                if abs(float(exact_duration) - derived_float) > 1e-6:
                    return self._coordinate_value(duration_qb)
            return exact_duration

        if has_duration_qb and self._has_value(duration_qb):
            return self._coordinate_value(duration_qb)
        if not has_duration_qb:
            return Fraction(0)
        return None

    def _piece_end_from_measures(self, source: Source) -> Fraction | None:
        """Exact end of the piece, read from the companion measures facet.

        A label's span runs to the next label; the final label runs to the end
        of the piece, which a harmonies facet never states. The measures facet
        states it exactly, as the furthest measure position plus that
        measure's actual duration.

        Args:
            source: The harmonies file being loaded.

        Returns:
            The exact end position in quarters, or None when no measures
            companion is available, cannot be read, holds no table (it is
            empty) or its cells are not exact.
        """
        try:
            import ms3
        except ImportError:  # pragma: no cover - guarded by the caller's import
            return None

        for companion in self._discover_companions(source):
            if self._parse_facet(companion) != "measures":
                continue
            try:
                with readable(companion) as table:
                    measures = ms3.load_tsv(table)
            except Exception:
                logger.debug(
                    "Could not read measures companion: %s", describe_source(companion)
                )
                return None
            if measures is None:
                logger.debug(
                    "Measures companion holds no table: %s", describe_source(companion)
                )
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
        self, starts: list[Fraction | float | None], source: Source
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

    def _resolve_quarterbeats(self, row: pandas.Series) -> Fraction | float | None:
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
        that spans the full folded score, as the measures' lengths sum up.

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

    def _load_notes(self, df: pandas.DataFrame, source: Source) -> ScoreStore:
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

            # Pitch - source MIDI pitch (raw int; NON-DEFAULT — affords an
            # EnharmonicPitch view on request, redundant with specific_pitch).
            midi = int(row["midi"]) if pd.notna(row.get("midi")) else None

            octave_val = int(row["octave"]) if pd.notna(row.get("octave")) else None
            tpc_val = int(row["tpc"]) if pd.notna(row.get("tpc")) else None

            # Pitch - Specific (canonical pydantic-derived shape). A spelling
            # without an octave states no specific pitch.
            specific_pitch = None
            if pd.notna(row.get("name")) and octave_val is not None:
                name = str(row["name"])
                step = name[0].upper() if name else "C"
                alter = 0
                if len(name) > 1:
                    # A "-" after the step is a negative octave (C-1), never a flat.
                    alter = name[1:].count("#") - name[1:].count("b")

                specific_pitch = {
                    "step": step,
                    "alter": alter,
                    "octave": octave_val,
                    "cents": 0.0,
                }

            # Attributes
            # ms3: 1 tie start, 0 tie middle, -1 tie end; an empty cell is
            # no tie and stays null.
            tied = int(row["tied"]) if pd.notna(row.get("tied")) else None
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
                    # An empty name cell states no name: null, never "nan".
                    "name": (
                        canonical_pitch_name(str(row["name"]))
                        if pd.notna(row.get("name"))
                        else None
                    ),
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

        notes_data = NoteEventData.from_dicts(
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
                **source_record(source, "source"),
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
    ) -> Timeline:
        """Create a score timeline carrying the parsed measure structure.

        Known limitation: a score whose measures are **split** — two
        records sharing one notated downbeat, as when a bar is divided
        across a repeat sign — has no floating-measure lattice, because
        the two halves cannot both anchor the same ordinal. The timeline
        then carries no ``floating_measures`` conversion; every other
        conversion, and the measure structure itself, is unaffected. No
        approximate lattice is built in its place, since an fm reading
        those measures do not support is worse than no fm reading.
        """
        from timetoalign.timelines import ContinuousLogicalTimeline

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

        return timeline

    def _load_measures(self, df: pandas.DataFrame, source: Source) -> ScoreStore:
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
        - repeats: "start", "end", "startend", "firstMeasure", "lastMeasure"
        - next: Comma-separated list of possible next MCs

        Args:
            df: pandas DataFrame from ms3.load_tsv.
            source: The source file: a path or a ZIP archive member.

        Returns:
            ScoreStore holding the measure table with the source's columns.

        Raises:
            ValueError: If a row states no ``mc``, or neither ``act_dur`` nor
                ``duration_qb`` (the measure's length). Neither is invented.
        """
        import pandas as pd

        if df.empty:
            return ScoreStore.empty()

        measure_rows = []
        table = f"The measures table {source.name!r}"

        for position, (_, row) in enumerate(df.iterrows(), start=1):
            # ===== Core Identity =====
            if not self._has_value(row.get("mc")):
                raise ValueError(f"{table} states no mc in row {position}")
            mc = int(row["mc"])
            mn = str(row.get("mn", mc)) if pd.notna(row.get("mn")) else str(mc)

            # Parse MN as integer (strip suffix if present)
            try:
                mn_int = int("".join(c for c in mn if c.isdigit() or c == "-"))
            except ValueError:
                mn_int = mc

            # ===== Temporal (with quarterbeats_all_endings fallback) =====
            qb = self._resolve_quarterbeats(row)

            # act_dur (whole notes, exact) or duration_qb (quarters, derived)
            if self._parse_fraction(row.get("act_dur")) is None and not (
                self._has_value(row.get("duration_qb"))
            ):
                raise ValueError(
                    f"{table} states no length of mc {mc} (row {position}): "
                    "neither act_dur nor duration_qb holds a value"
                )
            dur_qb = self._resolve_duration(row, "act_dur")
            if dur_qb is None:
                raise ValueError(
                    f"{table} states no readable length of mc {mc} (row {position})"
                )
            actual_length = float(dur_qb)

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
            # repeats column: "start", "end", "startend" (both signs on one
            # measure), "firstMeasure", "lastMeasure"
            repeats_val = (
                str(row.get("repeats", "")) if pd.notna(row.get("repeats")) else None
            )

            start_repeat = repeats_val in ("start", "startend")
            end_repeat = repeats_val in ("end", "startend")

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
                    "duration": coordinate_to_struct(dur_qb),
                    "end": coordinate_to_struct(
                        (qb if qb is not None else Fraction(0)) + dur_qb
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
        return ScoreStore(
            notes=NoteEventData.empty(),
            measures=measures_data,
            controls=ControlEventData.empty(),
            annotations=AnnotationEventData.empty(),
            metadata={
                "format": "tsv",
                "parser": "ms3",
                **source_record(source, "source"),
                "n_measures": len(measure_rows),
                "flow_control": measures_data.get_flow_control_summary(),
            },
        )

    def _load_annotations(
        self, df: pandas.DataFrame, source: Source, subtype: str
    ) -> ScoreStore:
        """Load annotations from TSV (harmonies).

        Parses ms3/DCML harmonies.tsv format with columns:
        - mc, mn: Measure context
        - quarterbeats: Position in quarter notes
        - label: The annotation text (Roman numeral harmony label)
        - numeral, form, figbass, relativeroot, etc.: Parsed components

        Args:
            df: pandas DataFrame from ms3.load_tsv.
            source: The source file: a path or a ZIP archive member.
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

        annotations_data = AnnotationEventData.from_dicts(
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
                **source_record(source, "source"),
                "annotation_type": subtype,
                "n_annotations": len(annotation_rows),
            },
        )

    def _load_controls(
        self, df: pandas.DataFrame, source: Source, subtype: str
    ) -> ScoreStore:
        """Load control events from TSV (chords).

        Parses ms3/DCML chords.tsv format with columns:
        - mc, mn: Measure context
        - quarterbeats: Position in quarter notes
        - chord: The chord symbol text
        - pedal, numeral, form, figbass, etc.: Parsed components

        Args:
            df: pandas DataFrame from ms3.load_tsv.
            source: The source file: a path or a ZIP archive member.
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

        controls_data = ControlEventData.from_dicts(
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
                **source_record(source, "source"),
                "control_type": subtype,
                "n_controls": len(control_rows),
            },
        )
