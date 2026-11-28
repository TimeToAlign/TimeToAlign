"""ATONLoader for loading ATON format files.

ATON (Artistic Text-based Object Notation) is a structured text format
used by the Stanford SUPRA project for piano roll analysis data.

Format specification: http://aton.sapp.org

The format uses:
- Lines starting with `@@` for comments
- Lines starting with `@KEY:` for key-value metadata
- `@@BEGIN: SECTION` / `@@END: SECTION` for block structures
- Nested blocks for structured data (e.g., ROLLINFO containing HOLES)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from timetoalign.core import TimeUnit
from timetoalign.core.events import BoundingBox
from timetoalign.loader._ids import give_event_ids
from timetoalign.loader._sources import Source, describe_source
from timetoalign.loader.base import Loader
from timetoalign.timelines import DiscreteGraphicalTimeline

if TYPE_CHECKING:
    from timetoalign.core import Coordinate

module_logger = logging.getLogger(__name__)


@dataclass
class ATONHole:
    """Information about a single piano roll hole punch.

    Attributes:
        id: Hole identifier (e.g., "K38_N1").
        origin_row: Pixel row of hole's leading edge (y-coordinate, time axis).
        origin_col: Pixel column of hole's leading edge (x-coordinate, pitch axis).
        width_row: Height of bounding box in pixels.
        width_col: Width of bounding box in pixels.
        centroid_row: Center of mass row.
        centroid_col: Center of mass column.
        area: Area of hole in pixels.
        perimeter: Perimeter in pixels.
        circularity: Circularity measure (0-1, 1=circular).
        tracker_hole: Which tracker bar hole (0-99 for Welte-Mignon).
        midi_key: MIDI key number (-1 if not a note).
        note_attack: Pixel row of note attack (if applicable).
        off_time: Pixel row of note release (if applicable).
        hpixcor: Horizontal pixel correction.
        major_axis: Major axis angle in degrees.
        raw_data: All raw key-value pairs from the ATON file.
    """

    id: str
    origin_row: int
    origin_col: int
    width_row: int
    width_col: int
    centroid_row: float
    centroid_col: float
    area: int
    perimeter: float
    circularity: float
    tracker_hole: int
    midi_key: int = -1
    note_attack: int | None = None
    off_time: int | None = None
    hpixcor: float = 0.0
    major_axis: int = 0
    raw_data: dict[str, str] = field(default_factory=dict)

    @property
    def bounding_box(self) -> BoundingBox:
        """Return this hole's axis-aligned bounding box in image pixels.

        ATON stores an upper-left origin plus row/column extents. The
        lower-right corner is therefore the origin plus the extent; image y
        coordinates grow downward.
        """
        return BoundingBox.from_corners(
            self.origin_col,
            self.origin_row,
            self.origin_col + self.width_col,
            self.origin_row + self.width_row,
            unit="pixels",
        )


class ATONLoader(Loader[list[ATONHole]]):
    """Load ATON format files (piano roll analysis data).

    ATON (Artistic Text-based Object Notation) is a structured text format
    used by the Stanford SUPRA project for storing piano roll analysis data
    including hole punch positions and metadata.

    The loader parses ROLLINFO metadata and HOLE blocks, providing access
    to both individual hole data and aggregate statistics.

    Examples:
        >>> from timetoalign.testdata import ensure_data
        >>> roll = ensure_data("supra") / "image" / "fd660zf8362_analysis.txt"
        >>> loader = ATONLoader().load(roll)
        >>> loader.rollinfo['MUSICAL_HOLES']
        30092
        >>> len(loader.holes)
        30092

        Hole positions are pixel coordinates on the scanned roll:

        >>> loader.first_hole
        Coordinate(15343, pixels)
        >>> loader.last_hole
        Coordinate(293119, pixels)

    Every hole event names its file's parse and, as its ``locator``,
    its place, ``hole:<k>`` (the k-th HOLE block of the file, from 0),
    followed by the ID its HOLE block states (``hole:0 key:K0_N1``). The
    event keeps that ID where no other hole loaded states it, and is given
    a minted one otherwise.

    Attributes:
        rollinfo: Dictionary of ROLLINFO metadata.
        holes: List of parsed ATONHole objects.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        """Initialize the loader.

        Args:
            seed: The seed of the loader's ID minter (see ``Loader.load``).
        """
        super().__init__(seed=seed)
        self._rollinfo: dict[str, Any] = {}
        self._holes: list[ATONHole] = []
        # Per hole, the file stating it and its place there (``hole:<k>``).
        self._hole_places: list[tuple[Source, str]] = []
        self._source_path: Source | None = None
        self._logger = module_logger.getChild("ATONLoader")

    # region Loading

    def _load_source(self, source: Source) -> list[ATONHole]:
        """Read and parse one ATON file.

        Args:
            source: The ATON file: a path or a ZIP archive member.

        Returns:
            The holes parsed so far.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is invalid.
        """
        if not source.exists():
            raise FileNotFoundError(f"ATON file not found: {describe_source(source)}")

        held = len(self._holes)
        self._parse_content(source.read_text(encoding="utf-8"))
        self._hole_places += [
            (source, f"hole:{k}") for k in range(len(self._holes) - held)
        ]

        self._logger.debug(
            f"Loaded ATON file from {describe_source(source)}: "
            f"rollinfo keys={len(self._rollinfo)}, holes={len(self._holes)}"
        )
        return self._holes

    def _accept_source(
        self, path: Source, source_meta: dict[str, Any], payload: list[ATONHole]
    ) -> None:
        """Record the ATON file read."""
        super()._accept_source(path, source_meta, payload)
        self._source_path = path

    def clear(self) -> None:
        """Forget the files read: their metadata and holes."""
        super().clear()
        self._rollinfo = {}
        self._holes = []
        self._hole_places = []
        self._source_path = None

    def _parse_content(self, content: str) -> None:
        """Parse ATON file content.

        Args:
            content: Raw file content.
        """
        # Split into lines
        lines = content.split("\n")

        # State machine for parsing
        in_rollinfo = False
        in_holes = False
        in_hole = False
        current_hole_data: dict[str, str] = {}

        for line in lines:
            line = line.strip()

            # Skip empty lines and pure comments
            if not line or line.startswith("@@ "):
                continue

            # Block start/end markers
            if line == "@@BEGIN: ROLLINFO":
                in_rollinfo = True
                continue
            elif line == "@@END: ROLLINFO":
                in_rollinfo = False
                continue
            elif line == "@@BEGIN: HOLES":
                in_holes = True
                continue
            elif line == "@@END: HOLES":
                in_holes = False
                continue
            elif line == "@@BEGIN: HOLE":
                in_hole = True
                current_hole_data = {}
                continue
            elif line == "@@END: HOLE":
                in_hole = False
                if current_hole_data:
                    hole = self._parse_hole(current_hole_data)
                    self._holes.append(hole)
                continue

            # Skip comment lines
            if line.startswith("@@"):
                continue

            # Parse key-value pairs
            if line.startswith("@") and ":" in line:
                # Extract key and value
                match = re.match(r"@([A-Z_0-9]+):\s*(.*)$", line, re.IGNORECASE)
                if match:
                    key = match.group(1).upper()
                    value = match.group(2).strip()

                    if in_rollinfo and not in_holes:
                        self._rollinfo[key] = self._parse_value(value)
                    elif in_hole:
                        current_hole_data[key] = value

    def _parse_value(self, value: str) -> Any:
        """Parse a value string into appropriate type.

        Args:
            value: Raw value string (may have units like "px", "ppi", etc.)

        Returns:
            Parsed value (int, float, or string).
        """
        # Strip common units
        value = value.strip()
        for unit in ["px", "ppi", "sec", "deg", "ppm"]:
            if value.endswith(unit):
                value = value[: -len(unit)].strip()
                break

        # Try to parse as number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _parse_hole(self, data: dict[str, str]) -> ATONHole:
        """Parse hole data dictionary into ATONHole object.

        Args:
            data: Dictionary of hole key-value pairs.

        Returns:
            Parsed ATONHole object.
        """

        def get_int(key: str, default: int = 0) -> int:
            """Read *key* as an integer, stripping a ``px`` or ``deg`` unit.

            A missing or unparsable value, an infinite one included, gives *default*;
            a decimal is truncated.
            """
            val = data.get(key, str(default))
            # Strip units
            for unit in ["px", "deg"]:
                if val.endswith(unit):
                    val = val[: -len(unit)]
            try:
                return int(float(val))
            except (ValueError, TypeError, OverflowError):
                return default

        def get_float(key: str, default: float = 0.0) -> float:
            """Read *key* as a float, stripping a ``px`` or ``deg`` unit.

            A missing or unparsable value gives *default*.
            """
            val = data.get(key, str(default))
            # Strip units
            for unit in ["px", "deg"]:
                if val.endswith(unit):
                    val = val[: -len(unit)]
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        def get_optional_int(key: str) -> int | None:
            """Read *key* as an integer, stripping a ``px`` unit, or ``None``.

            A missing or unparsable value, an infinite one included, gives ``None``;
            a decimal is truncated.
            """
            val = data.get(key)
            if val is None:
                return None
            for unit in ["px"]:
                if val.endswith(unit):
                    val = val[: -len(unit)]
            try:
                return int(float(val))
            except (ValueError, TypeError, OverflowError):
                return None

        return ATONHole(
            id=data.get("ID", ""),
            origin_row=get_int("ORIGIN_ROW"),
            origin_col=get_int("ORIGIN_COL"),
            width_row=get_int("WIDTH_ROW"),
            width_col=get_int("WIDTH_COL"),
            centroid_row=get_float("CENTROID_ROW"),
            centroid_col=get_float("CENTROID_COL"),
            area=get_int("AREA"),
            perimeter=get_float("PERIMETER"),
            circularity=get_float("CIRCULARITY"),
            tracker_hole=get_int("TRACKER_HOLE"),
            midi_key=get_int("MIDI_KEY", -1),
            note_attack=get_optional_int("NOTE_ATTACK"),
            off_time=get_optional_int("OFF_TIME"),
            hpixcor=get_float("HPIXCOR"),
            major_axis=get_int("MAJOR_AXIS"),
            raw_data=dict(data),
        )

    # endregion

    # region Properties

    @property
    def rollinfo(self) -> dict[str, Any]:
        """Return ROLLINFO metadata dictionary."""
        return dict(self._rollinfo)

    @property
    def holes(self) -> list[ATONHole]:
        """Return list of parsed hole objects."""
        return list(self._holes)

    @property
    def n_holes(self) -> int:
        """Number of hole blocks parsed."""
        return len(self._holes)

    @property
    def first_hole(self) -> "Coordinate":
        """Pixel coordinate of first musical hole (from ROLLINFO).

        Returns:
            Coordinate with pixel unit, usable directly as the offset for an
            inherited DiscreteGraphicalTimeline child.
        """
        from timetoalign.core import Coordinate, TimeUnit

        return Coordinate(int(self._rollinfo.get("FIRST_HOLE", 0)), TimeUnit.pixels)

    @property
    def last_hole(self) -> "Coordinate":
        """Pixel coordinate of last musical hole (from ROLLINFO).

        Returns:
            Coordinate with pixel unit.
        """
        from timetoalign.core import Coordinate, TimeUnit

        return Coordinate(int(self._rollinfo.get("LAST_HOLE", 0)), TimeUnit.pixels)

    @property
    def musical_length(self) -> "Coordinate":
        """Pixel length from first to last hole (from ROLLINFO).

        Returns:
            Coordinate with pixel unit (representing a length/distance).
        """
        from timetoalign.core import Coordinate, TimeUnit

        return Coordinate(int(self._rollinfo.get("MUSICAL_LENGTH", 0)), TimeUnit.pixels)

    @property
    def musical_holes(self) -> int:
        """Number of musical holes declared in ROLLINFO."""
        return int(self._rollinfo.get("MUSICAL_HOLES", 0))

    @property
    def musical_notes(self) -> int:
        """Number of musical notes (merged holes) declared in ROLLINFO."""
        return int(self._rollinfo.get("MUSICAL_NOTES", 0))

    @property
    def image_dimensions(self) -> dict[str, int]:
        """Image dimensions from ROLLINFO.

        Returns:
            Dictionary with 'width' and 'height' keys.
        """
        return {
            "width": int(self._rollinfo.get("IMAGE_WIDTH", 0)),
            "height": int(self._rollinfo.get("IMAGE_LENGTH", 0)),
        }

    @property
    def dpi(self) -> float:
        """Scan resolution in DPI (dots per inch)."""
        return float(self._rollinfo.get("LENGTH_DPI", 0))

    @property
    def source_path(self) -> Source | None:
        """Path to the loaded ATON file."""
        return self._source_path

    # endregion

    # region Query Methods

    def get_holes_by_tracker(self, tracker_hole: int) -> list[ATONHole]:
        """Get all holes for a specific tracker bar position.

        Args:
            tracker_hole: Tracker bar hole number (0-99).

        Returns:
            List of holes at that tracker position.
        """
        return [h for h in self._holes if h.tracker_hole == tracker_hole]

    def get_holes_in_range(self, start_row: int, end_row: int) -> list[ATONHole]:
        """Get holes within a pixel row range.

        Args:
            start_row: Starting pixel row (inclusive).
            end_row: Ending pixel row (inclusive).

        Returns:
            List of holes with origin_row in range.
        """
        return [h for h in self._holes if start_row <= h.origin_row <= end_row]

    def get_note_holes(self) -> list[ATONHole]:
        """Get holes that represent note attacks.

        Returns:
            List of holes with note_attack set.
        """
        return [h for h in self._holes if h.note_attack is not None]

    # endregion

    # region Timeline Creation

    @property
    def name(self) -> str:
        """Human-readable name for the loaded data.

        Returns the source filename stem, or 'ATON Data' if not loaded.
        """
        if self._source_path:
            return self._source_path.stem
        return "ATON Data"

    def create_timeline(
        self,
        uid: str | None = None,
        name: str | None = None,
    ) -> DiscreteGraphicalTimeline:
        """Create a timeline populated with hole events.

        Creates a graphical timeline (in pixels) spanning the full image height,
        with all parsed holes as InstantEvents at their absolute pixel coordinates.

        This is the primary timeline for the piano roll image. To get a relative
        coordinate view (first hole = 0), create an inherited
        DiscreteGraphicalTimeline child at ``first_hole``.

        Each event is named by its hole's ``@ID``, which is also its ID. The
        SUPRA analysis software gives every hole that is no note the key
        ``0`` and counts per tracker-bar hole, so holes on different tracker
        holes can share one ``@ID`` (``K0_N1``). Such an ``@ID`` names no one
        hole: each hole holding it keeps it as its name and is given an ID
        minted from the loader's minter (``hole_…``). Two holes of one
        ``@ID`` on one tracker hole are refused: the loader does not decide
        that a repeated block is one hole.

        Args:
            uid: Unique identifier for the timeline; minted ``dgt_…`` from
                the loader's minter if None.
            name: Human-readable name. Uses source filename if None.

        Returns:
            A DiscreteGraphicalTimeline with hole events at absolute coordinates.

        Raises:
            RuntimeError: If no data has been loaded.
            ValueError: If two holes of one ``@ID`` lie on one tracker hole.

        Examples:
            >>> loader = ATONLoader()
            >>> loader.load("analysis.txt")
            >>> dgt_1 = loader.create_timeline(uid="dgt_1")
            >>> dgt_1.n_events
            30092
            >>> # Child inherits DiscreteGraphicalTimeline; its origin is the first hole
            >>> dgt_holes = dgt_1.create_child(
            ...     length=loader.musical_length,
            ...     offset=loader.first_hole,
            ...     uid="dgt_holes",
            ... )
        """
        if not self._holes:
            raise RuntimeError("No data loaded. Call load() first.")

        from timetoalign.timelines.factory import mint_timeline_id

        # Create timeline spanning full image
        timeline = DiscreteGraphicalTimeline(
            length=self.image_dimensions["height"],
            unit=TimeUnit.pixels,
            uid=(
                mint_timeline_id(TimeUnit.pixels, self._id_minter)
                if uid is None
                else uid
            ),
            name=name or self.name,
        )

        # Add holes as InstantEvents at absolute pixel coordinates
        stated = Counter((hole.id, hole.tracker_hole) for hole in self._holes)
        twice = [key for key, count in stated.items() if count > 1]
        if twice:
            hole_id, tracker_hole = twice[0]
            raise ValueError(
                f"ATON hole {hole_id!r} is stated twice on tracker hole "
                f"{tracker_hole}: the file names one hole twice, or two holes alike."
            )
        id_counts = Counter(hole.id for hole in self._holes)
        hole_events = [
            {
                "id": hole.id if id_counts[hole.id] == 1 else None,
                "temporal_type": "instant",
                "event_type": "Hole",
                "instant": hole.origin_row,  # Absolute coordinate
                **self._stamp(source, place, hole.id),
            }
            for hole, (source, place) in zip(
                self._holes, self._hole_places, strict=True
            )
        ]
        give_event_ids(hole_events, self._id_minter)
        timeline.add_events(hole_events, parses=self.parses)

        return timeline

    # endregion

    def __repr__(self) -> str:
        if not self._rollinfo:
            return "ATONLoader(not loaded)"
        return (
            f"ATONLoader("
            f"holes={len(self._holes)}, "
            f"musical_holes={self.musical_holes}, "
            f"musical_notes={self.musical_notes})"
        )
