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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from timetoalign.core import TimeUnit
from timetoalign.core.events import BoundingBox
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
        """Return this hole's axis-aligned image-coordinate bounding box.

        ATON stores an upper-left origin plus row/column extents. The
        lower-right corner is therefore the origin plus the extent; image y
        coordinates grow downward.
        """
        return BoundingBox.from_corners(
            self.origin_col,
            self.origin_row,
            self.origin_col + self.width_col,
            self.origin_row + self.width_row,
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

    Attributes:
        rollinfo: Dictionary of ROLLINFO metadata.
        holes: List of parsed ATONHole objects.
    """

    def __init__(self) -> None:
        """Initialize the loader."""
        super().__init__()
        self._rollinfo: dict[str, Any] = {}
        self._holes: list[ATONHole] = []
        self._source_path: Path | None = None
        self._logger = module_logger.getChild("ATONLoader")

    def _load_source(self, source: Path) -> list[ATONHole]:
        """Reject the event-oriented root lifecycle for ATON parsing."""
        raise NotImplementedError("ATONLoader uses its text-specific load() method.")

    @classmethod
    def from_file(cls, path: Path | str) -> Self:
        """Create a loader and load an ATON file in one step.

        Convenience constructor for loading a single file.

        Args:
            path: Path to the ATON file.

        Returns:
            A new ATONLoader instance with the file already loaded.

        Examples:
            >>> loader = ATONLoader.from_file("analysis.txt")
            >>> loader.musical_holes
            30092
        """
        return cls().load(path)

    # region Loading

    def load(self, path: Path | str) -> Self:
        """Load and parse an ATON file.

        Args:
            path: Path to the ATON file.

        Returns:
            Self, for method chaining.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ATON file not found: {path}")

        with open(path, encoding="utf-8") as f:
            content = f.read()

        self._parse_content(content)
        self._source_path = path

        self._logger.debug(
            f"Loaded ATON file from {path}: "
            f"rollinfo keys={len(self._rollinfo)}, holes={len(self._holes)}"
        )

        return self

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
            val = data.get(key, str(default))
            # Strip units
            for unit in ["px", "deg"]:
                if val.endswith(unit):
                    val = val[: -len(unit)]
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return default

        def get_float(key: str, default: float = 0.0) -> float:
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
            val = data.get(key)
            if val is None:
                return None
            for unit in ["px"]:
                if val.endswith(unit):
                    val = val[: -len(unit)]
            try:
                return int(float(val))
            except (ValueError, TypeError):
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
    def source_path(self) -> Path | None:
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

        Args:
            uid: Unique identifier for the timeline. Auto-generated if None.
            name: Human-readable name. Uses source filename if None.

        Returns:
            A DiscreteGraphicalTimeline with hole events at absolute coordinates.

        Raises:
            RuntimeError: If no data has been loaded.

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

        # Create timeline spanning full image
        timeline = DiscreteGraphicalTimeline(
            length=self.image_dimensions["height"],
            unit=TimeUnit.pixels,
            uid=uid,
            name=name or self.name,
        )

        # Add holes as InstantEvents at absolute pixel coordinates
        hole_events = [
            {
                "id": f"hole_{hole.id}",
                "name": hole.id,
                "temporal_type": "instant",
                "event_type": "Hole",
                "instant": hole.origin_row,  # Absolute coordinate
            }
            for hole in self._holes
        ]
        timeline.add_events(hole_events)

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
