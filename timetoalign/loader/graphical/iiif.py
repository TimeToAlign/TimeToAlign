"""IIIFManifestLoader for loading image metadata from IIIF manifests.

IIIF (International Image Interoperability Framework) manifests contain
structured metadata about images, including dimensions, that can be used
to construct graphical timelines without loading the full image data. An
image has no time axis of its own: the caller gives each canvas the time
path that runs through it (:mod:`timetoalign.loader.graphical.paths`).

This loader supports IIIF Presentation API 2.x and 3.x manifests. A
Presentation 3 manifest may also hold time-based canvases (sound or video
with a ``duration``); they are read, and have no pixel extent of their own
unless the canvas states a width and height.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from timetoalign.loader.base import Loader
from timetoalign.loader.graphical._pages import Page, build_pages, build_volume
from timetoalign.loader.graphical.paths import (
    HorizontalLinePath,
    PiecewisePath,
    TimeAxisPath,
    VerticalLinePath,
)
from timetoalign.timelines import DiscreteGraphicalTimeline, SegmentLine

module_logger = logging.getLogger(__name__)


SOURCE_PROPERTY = "dcterms:source"
"""The property by which a timeline built from a manifest names what it measures.

Provisional: the property may change.
"""


@dataclass
class IIIFCanvasInfo:
    """Information about a single IIIF canvas.

    A canvas is spatial (an image: ``width`` and ``height``), time-based (a
    sound: ``duration``), or both (a video).

    Attributes:
        id: Canvas identifier (URL or local ID).
        width: Canvas width in pixels, or None when the canvas states none.
        height: Canvas height in pixels, or None when the canvas states none.
        label: Human-readable label for the canvas.
        metadata: Additional canvas metadata.
        duration: Canvas duration in seconds (Presentation 3 ``duration``,
            kept as the JSON number given), or None when the canvas states
            none.
    """

    id: str
    width: int | None
    height: int | None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    duration: float | None = None

    @property
    def is_spatial(self) -> bool:
        """Whether the canvas states both a width and a height."""
        return self.width is not None and self.height is not None


@dataclass
class IIIFManifestInfo:
    """Parsed information from an IIIF manifest.

    Attributes:
        id: Manifest identifier (URL).
        label: Human-readable manifest label.
        canvases: List of canvas information.
        metadata: Manifest-level metadata.
    """

    id: str
    label: str | None
    canvases: list[IIIFCanvasInfo]
    metadata: dict[str, Any] = field(default_factory=dict)


class IIIFManifestLoader(Loader[IIIFManifestInfo]):
    """Load image metadata from IIIF Presentation API manifests.

    IIIF manifests contain canvas dimensions that represent image sizes.
    This loader extracts width/height for use with graphical timelines
    without needing to download or load the actual image files. A timeline
    runs along the time path the caller gives for a canvas:
    :meth:`create_timelines` answers one timeline per canvas,
    :meth:`create_timeline` the whole volume, its pages laid end to end.

    Supports both IIIF Presentation API 2.x and 3.x formats. Time-based
    Presentation 3 canvases (``duration`` without ``width``/``height``) are
    read and expose their duration; they have no pixels, so no time path
    runs through them and the single-image accessors refuse a manifest whose
    first canvas is one.

    Examples:
        >>> from timetoalign.testdata import ensure_data
        >>> manifest = ensure_data("supra") / "image" / "ifff_manifest.json"
        >>> loader = IIIFManifestLoader().load(manifest)
        >>> loader.dimensions
        {'width': 4096, 'height': 299400}

        Individual canvases carry their own label and dimensions:

        >>> for canvas in loader.manifest_info.canvases:
        ...     print(f"{canvas.label}: {canvas.width}x{canvas.height}")
        display image: 4096x299400

        A piano roll is read from top to bottom, so its time path is a
        vertical line down the middle of the image:

        >>> from timetoalign.loader.graphical.paths import VerticalLinePath
        >>> path = VerticalLinePath(x=2048, y0=0, y1=299400, tolerance=2048)
        >>> (roll,) = loader.create_timelines(path)
        >>> roll.length.value
        299400

    Attributes:
        manifest_info: Parsed manifest information (after loading).
    """

    def __init__(self) -> None:
        """Initialize the loader."""
        super().__init__()
        self._manifest_info: IIIFManifestInfo | None = None
        self._source_path: Path | None = None
        self._logger = module_logger.getChild("IIIFManifestLoader")

    def _load_source(self, source: Path) -> IIIFManifestInfo:
        """Load and parse one IIIF manifest file.

        Args:
            source: Path to the IIIF manifest JSON file.

        Returns:
            Parsed IIIF manifest information.

        Raises:
            FileNotFoundError: If the manifest file doesn't exist.
            ValueError: If the manifest is invalid or has no canvases.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        if not source.exists():
            raise FileNotFoundError(f"Manifest file not found: {source}")

        with open(source, encoding="utf-8") as f:
            manifest_data = json.load(f)
        return self._parse_manifest(manifest_data)

    def _accept_source(
        self,
        path: Path,
        source_meta: dict[str, Any],
        payload: IIIFManifestInfo,
    ) -> None:
        """Retain parsed manifest information from the shared lifecycle."""
        super()._accept_source(path, source_meta, payload)
        self._manifest_info = payload
        self._source_path = path
        self._logger.debug(
            f"Loaded manifest from {path}: "
            f"{len(self._manifest_info.canvases)} canvas(es)"
        )

    def _parse_manifest(self, data: dict[str, Any]) -> IIIFManifestInfo:
        """Parse manifest data into structured info.

        Handles both IIIF Presentation API 2.x and 3.x formats.

        Args:
            data: Raw manifest dictionary from JSON.

        Returns:
            Parsed IIIFManifestInfo.

        Raises:
            ValueError: If manifest structure is invalid.
        """
        # Detect API version
        context = data.get("@context", "")
        if isinstance(context, list):
            # IIIF 3.x uses a list of contexts
            is_v3 = any("presentation/3" in str(c) for c in context)
        else:
            is_v3 = "presentation/3" in str(context)

        if is_v3:
            return self._parse_v3_manifest(data)
        else:
            return self._parse_v2_manifest(data)

    def _parse_v2_manifest(self, data: dict[str, Any]) -> IIIFManifestInfo:
        """Parse IIIF Presentation API 2.x manifest.

        In 2.x, canvases are nested in sequences[].canvases[].
        """
        manifest_id = data.get("@id", "")
        label = data.get("label", None)

        # Extract metadata
        metadata = {}
        if "metadata" in data:
            for item in data["metadata"]:
                key = item.get("label", "")
                value = item.get("value", "")
                metadata[key] = value

        # Extract canvases from sequences
        canvases = []
        sequences = data.get("sequences", [])
        for sequence in sequences:
            for canvas_data in sequence.get("canvases", []):
                canvas = self._parse_v2_canvas(canvas_data)
                canvases.append(canvas)

        if not canvases:
            raise ValueError("Manifest contains no canvases")

        return IIIFManifestInfo(
            id=manifest_id,
            label=label,
            canvases=canvases,
            metadata=metadata,
        )

    def _parse_v2_canvas(self, data: dict[str, Any]) -> IIIFCanvasInfo:
        """Parse a single canvas from IIIF 2.x format."""
        canvas_id = data.get("@id", "")
        width = data.get("width")
        height = data.get("height")
        label = data.get("label")

        if width is None or height is None:
            raise ValueError(f"Canvas {canvas_id} missing width/height")

        return IIIFCanvasInfo(
            id=canvas_id,
            width=int(width),
            height=int(height),
            label=label,
        )

    def _parse_v3_manifest(self, data: dict[str, Any]) -> IIIFManifestInfo:
        """Parse IIIF Presentation API 3.x manifest.

        In 3.x, canvases are directly in items[].
        """
        manifest_id = data.get("id", "")
        label = self._extract_v3_label(data.get("label"))

        # Extract metadata
        metadata = {}
        if "metadata" in data:
            for item in data["metadata"]:
                key = self._extract_v3_label(item.get("label", {}))
                value = self._extract_v3_label(item.get("value", {}))
                if key:
                    metadata[key] = value

        # Extract canvases from items
        canvases = []
        for item in data.get("items", []):
            if item.get("type") == "Canvas":
                canvas = self._parse_v3_canvas(item)
                canvases.append(canvas)

        if not canvases:
            raise ValueError("Manifest contains no canvases")

        return IIIFManifestInfo(
            id=manifest_id,
            label=label,
            canvases=canvases,
            metadata=metadata,
        )

    def _parse_v3_canvas(self, data: dict[str, Any]) -> IIIFCanvasInfo:
        """Parse a single canvas from IIIF 3.x format.

        A canvas needs a width and a height, or a duration.

        Raises:
            ValueError: If the canvas states neither.
        """
        canvas_id = data.get("id", "")
        width = data.get("width")
        height = data.get("height")
        duration = data.get("duration")
        label = self._extract_v3_label(data.get("label"))

        spatial = width is not None and height is not None
        if not spatial and duration is None:
            raise ValueError(
                f"Canvas {canvas_id} states neither width and height nor a duration"
            )

        return IIIFCanvasInfo(
            id=canvas_id,
            width=None if width is None else int(width),
            height=None if height is None else int(height),
            label=label,
            duration=duration,
        )

    def _extract_v3_label(self, label_data: Any) -> str | None:
        """Extract label string from IIIF 3.x language map.

        In 3.x, labels are language maps like {"en": ["Label text"]}.
        """
        if label_data is None:
            return None
        if isinstance(label_data, str):
            return label_data
        if isinstance(label_data, dict):
            # Try common language codes
            for lang in ["en", "none", "und"]:
                if lang in label_data:
                    values = label_data[lang]
                    if isinstance(values, list) and values:
                        return str(values[0])
                    return str(values)
            # Return first available
            for values in label_data.values():
                if isinstance(values, list) and values:
                    return str(values[0])
                return str(values)
        return None

    # endregion

    # region Properties

    @property
    def manifest_info(self) -> IIIFManifestInfo:
        """Return parsed manifest information.

        Raises:
            RuntimeError: If no manifest has been loaded.
        """
        if self._manifest_info is None:
            raise RuntimeError("No manifest loaded. Call load() first.")
        return self._manifest_info

    @property
    def dimensions(self) -> dict[str, int]:
        """Return dimensions of the first (or only) canvas.

        Returns:
            Dictionary with 'width' and 'height' keys.

        Raises:
            RuntimeError: If no manifest has been loaded.
            ValueError: If the first canvas is time-based and states no
                width and height.
        """
        canvas = self.manifest_info.canvases[0]
        if canvas.width is None or canvas.height is None:
            raise ValueError(
                f"Canvas {canvas.id} is time-based (duration {canvas.duration}) and "
                "states no width and height, so it has no extent in pixels"
            )
        return {"width": canvas.width, "height": canvas.height}

    @property
    def width(self) -> int:
        """Width of the first canvas in pixels.

        Raises:
            ValueError: If the first canvas is time-based (see
                :attr:`dimensions`).
        """
        return self.dimensions["width"]

    @property
    def height(self) -> int:
        """Height of the first canvas in pixels.

        Raises:
            ValueError: If the first canvas is time-based (see
                :attr:`dimensions`).
        """
        return self.dimensions["height"]

    @property
    def n_canvases(self) -> int:
        """Number of canvases in the manifest."""
        return len(self.manifest_info.canvases)

    @property
    def label(self) -> str | None:
        """Human-readable label from the manifest."""
        return self.manifest_info.label

    @property
    def source_path(self) -> Path | None:
        """Path to the loaded manifest file."""
        return self._source_path

    # endregion

    # region Metadata Access

    def get_metadata(self, key: str) -> str | None:
        """Get a metadata value by key.

        Args:
            key: Metadata label to look up.

        Returns:
            Metadata value, or None if not found.
        """
        return self.manifest_info.metadata.get(key)

    def get_canvas(self, index: int = 0) -> IIIFCanvasInfo:
        """Get canvas information by index.

        Args:
            index: Zero-based canvas index.

        Returns:
            IIIFCanvasInfo for the requested canvas.

        Raises:
            IndexError: If index is out of range.
        """
        return self.manifest_info.canvases[index]

    # endregion

    # region Timeline Creation

    def create_timeline(
        self,
        paths: Mapping[int | str, TimeAxisPath] | TimeAxisPath,
        *,
        canvases: Sequence[int | str] | None = None,
        uid: str | None = None,
        name: str | None = None,
        page_ids: Sequence[str] | str | None = None,
        accolade_ids: Sequence[str] | str | None = None,
    ) -> SegmentLine:
        """Create the volume: the selected canvases laid end to end, in pixels.

        Each selected canvas becomes one page along its time path, in
        manifest order, and the volume is the segment line of the pages, as
        long as all of them together. A page whose path is a
        :class:`~timetoalign.loader.graphical.paths.PiecewisePath` is itself
        a segment line of accolades, one per piece; any other path gives one
        plain timeline of the path's length. See :meth:`create_timelines`
        for what a page and an accolade state.

        The volume states the manifest it was read from as the property
        ``dcterms:source`` (``{"id": <manifest id>}``; a manifest without an
        id states nothing). This property is provisional: it may change.

        Args:
            paths: The time path of each selected canvas: a mapping from a
                canvas, named by its position from 0 (as :meth:`get_canvas`)
                or by its id, to its path; or one path, where exactly one
                canvas is selected.
            canvases: The canvases to read, by position or id; they are read
                in manifest order whatever order they are named in. None
                selects every canvas that states a width and a height.
            uid: The volume's id. Minted (``dgt_N``) if None.
            name: The volume's name. The manifest's label if None.
            page_ids: The pages' ids: a sequence, one per page, or a
                template filled with ``page`` and ``n`` (``"p{page}"``).
                Minted if None.
            accolade_ids: The accolades' ids: a sequence over the whole
                call, or a template filled with ``page``, ``accolade`` and
                ``n`` (``"p{page}_s{accolade}"``). Minted if None.

        Returns:
            The locked volume: ``SegmentLine[SegmentLine[DiscreteGraphical
            Timeline]]`` where the pages are built of accolades, else
            ``SegmentLine[DiscreteGraphicalTimeline]``.

        Raises:
            RuntimeError: If no manifest has been loaded.
            TypeError: If *canvases* is a bare string, or *paths* is neither
                a mapping nor a time path.
            KeyError: If *canvases* names a canvas the manifest does not
                hold, or *paths* names a canvas that is unknown, time-based
                or not selected.
            ValueError: If a selected canvas has no path, one path is given
                for several canvases, a line leaves its canvas, no canvas is
                selected, some pages are built of accolades and others are
                not, or an ids argument does not fit the call. Nothing is
                built then.

        Examples:
            Two pages of a printed score, each with two systems::

                from timetoalign.loader.graphical.paths import (
                    HorizontalLinePath, PiecewisePath,
                )

                def systems(*lines):
                    return PiecewisePath(tuple(
                        HorizontalLinePath(x0=x0, x1=x1, y=y, tolerance=300)
                        for x0, x1, y in lines
                    ))

                loader = IIIFManifestLoader().load("manifest.json")
                volume = loader.create_timeline(
                    {
                        1: systems((1221, 4101, 1900), (971, 4112, 3300)),
                        2: systems((519, 3661, 900), (522, 3659, 2300)),
                    },
                    canvases=[1, 2],
                    uid="scan",
                )
                for _, offset, page in volume.iter_segments():
                    print(page.name, offset.value, page.length.value,
                          page.statements["dcterms:source"])
        """
        info = self.manifest_info
        return build_volume(
            self,
            self._pages(paths, canvases),
            uid=uid,
            name=info.label if name is None else name,
            statements=_source_statements(info.id),
            page_ids=page_ids,
            accolade_ids=accolade_ids,
        )

    def create_timelines(
        self,
        paths: Mapping[int | str, TimeAxisPath] | TimeAxisPath,
        *,
        canvases: Sequence[int | str] | None = None,
        page_ids: Sequence[str] | str | None = None,
        accolade_ids: Sequence[str] | str | None = None,
    ) -> list[DiscreteGraphicalTimeline]:
        """Create one timeline in pixels per selected canvas, along its time path.

        A canvas whose path is a
        :class:`~timetoalign.loader.graphical.paths.PiecewisePath` becomes a
        segment line of accolades, one per piece in path order: an accolade
        along a horizontal piece is as long as the piece (``x1 - x0``) and
        states the strip it stands for in ``meta["box"]``. Any other path
        gives one plain timeline of the path's length, which must be a whole
        number of pixels: a piano roll is read along a ``VerticalLinePath``.

        Each page is named with its canvas's label and states the canvas it
        measures: its id in ``meta["image"]`` (the accolades too) and as the
        property ``dcterms:source`` (``{"id": <canvas id>}``; Presentation 2
        ``@id``, Presentation 3 ``id``). A canvas without an id states
        nothing. This property is provisional: it may change.

        A horizontal or vertical line must lie on its canvas: ``0 <= x0``,
        ``x1 <= width`` and ``0 <= y <= height`` (for a vertical line ``0 <=
        x <= width``, ``0 <= y0`` and ``y1 <= height``).

        Args:
            paths: As :meth:`create_timeline`.
            canvases: As :meth:`create_timeline`.
            page_ids: As :meth:`create_timeline`.
            accolade_ids: As :meth:`create_timeline`.

        Returns:
            The pages in manifest order, each locked; an empty list where no
            canvas is selected.

        Raises:
            RuntimeError: If no manifest has been loaded.
            TypeError: As :meth:`create_timeline`.
            KeyError: As :meth:`create_timeline`.
            ValueError: As :meth:`create_timeline`, except that no canvas
                need be selected and the pages need not be of one kind.

        Examples:
            A piano roll, read from top to bottom::

                from timetoalign.loader.graphical.paths import VerticalLinePath

                loader = IIIFManifestLoader().load("manifest.json")
                (roll,) = loader.create_timelines(
                    VerticalLinePath(x=2048, y0=0, y1=299400, tolerance=2048)
                )
        """
        return build_pages(
            self,
            self._pages(paths, canvases),
            page_ids=page_ids,
            accolade_ids=accolade_ids,
        )

    def _pages(
        self,
        paths: Mapping[int | str, TimeAxisPath] | TimeAxisPath,
        canvases: Sequence[int | str] | None,
    ) -> list[Page]:
        """Describe each selected canvas with its time path, in manifest order.

        Raises:
            TypeError, KeyError, ValueError: Every refusal of *paths* and
                *canvases* (see :meth:`create_timeline`).
        """
        info = self.manifest_info
        positions = _positions_by_id(info.canvases)
        listing = _listed(info.canvases, range(len(info.canvases)))

        if canvases is None:
            selected = [
                i for i, canvas in enumerate(info.canvases) if canvas.is_spatial
            ]
        elif isinstance(canvases, str):
            raise TypeError(
                f"canvases takes a sequence of positions or ids, not the bare "
                f"string {canvases!r}: write canvases=[{canvases!r}]."
            )
        else:
            resolved = [
                _position(key, positions, len(info.canvases)) for key in canvases
            ]
            unknown = [key for key, at in zip(canvases, resolved) if at is None]
            if unknown:
                raise KeyError(
                    f"canvases names {_spelled(unknown)}, which the manifest does "
                    f"not hold. Available canvases: {listing}"
                )
            selected = sorted(set(resolved))

        if isinstance(paths, TimeAxisPath):
            if len(selected) != 1:
                raise ValueError(
                    f"One time path was given and {len(selected)} canvases are "
                    f"selected ({_listed(info.canvases, selected)}): give a mapping "
                    "from each canvas to its path, or select one canvas with "
                    "canvases=."
                )
            path_of = {selected[0]: paths}
        elif isinstance(paths, Mapping):
            path_of = {}
            named_as: dict[int, int | str] = {}
            unknown, time_based, not_selected = [], [], []
            for key, path in paths.items():
                at = _position(key, positions, len(info.canvases))
                if at is None:
                    unknown.append(key)
                elif not info.canvases[at].is_spatial:
                    time_based.append(key)
                elif at not in selected:
                    not_selected.append(key)
                elif at in named_as:
                    raise ValueError(
                        f"paths gives canvas {at} two time paths, as "
                        f"{named_as[at]!r} and as {key!r}."
                    )
                else:
                    named_as[at] = key
                    path_of[at] = path
            refused = [
                f"{_spelled(keys)} ({reason})"
                for keys, reason in (
                    (unknown, "not in the manifest"),
                    (time_based, "time-based: no width and height"),
                    (not_selected, "not selected by canvases="),
                )
                if keys
            ]
            if refused:
                raise KeyError(
                    f"paths gives a time path to {'; '.join(refused)}. Selected "
                    f"canvases: {_listed(info.canvases, selected)}"
                )
        else:
            raise TypeError(
                "paths takes a time path or a mapping from canvases to time "
                f"paths, not {type(paths).__name__}."
            )

        without = [at for at in selected if at not in path_of]
        if len(without) == 1:
            raise ValueError(
                f"canvas {without[0]} has no time path: give it one or leave it "
                "out with canvases="
            )
        if without:
            raise ValueError(
                f"canvases {', '.join(map(str, without))} have no time path: give "
                "each one or leave them out with canvases="
            )

        pages = []
        for at in selected:
            canvas = info.canvases[at]
            _refuse_off_canvas(path_of[at], at, canvas)
            pages.append(
                Page(
                    name=canvas.label,
                    image=canvas.id or None,
                    path=path_of[at],
                    statements=_source_statements(canvas.id),
                )
            )
        return pages

    # endregion

    def __repr__(self) -> str:
        if self._manifest_info is None:
            return "IIIFManifestLoader(not loaded)"
        first = self._manifest_info.canvases[0]
        shown = (
            f"dimensions={first.width}x{first.height}"
            if first.is_spatial
            else f"duration={first.duration}"
        )
        return (
            f"IIIFManifestLoader("
            f"canvases={len(self._manifest_info.canvases)}, {shown})"
        )


def _source_statements(iri: str) -> list[tuple[str, dict[str, str]]]:
    """The statement that a timeline measures the resource *iri*; none if empty."""
    return [(SOURCE_PROPERTY, {"id": iri})] if iri else []


def _positions_by_id(canvases: Sequence[IIIFCanvasInfo]) -> dict[str, int]:
    """Each canvas id's position; of canvases sharing an id, the first."""
    positions: dict[str, int] = {}
    for at, canvas in enumerate(canvases):
        positions.setdefault(canvas.id, at)
    return positions


def _position(key: object, positions: Mapping[str, int], count: int) -> int | None:
    """The position of the canvas *key* names (a position or an id), or None."""
    if isinstance(key, str):
        return positions.get(key)
    if isinstance(key, int) and not isinstance(key, bool) and 0 <= key < count:
        return key
    return None


def _spelled(keys: Sequence[object]) -> str:
    """Canvas names as the caller wrote them, for a message."""
    return ", ".join(repr(key) for key in keys)


def _listed(canvases: Sequence[IIIFCanvasInfo], positions: Sequence[int]) -> str:
    """The canvases at *positions*, each as ``position: 'id'``, for a message."""
    return ", ".join(f"{at}: {canvases[at].id!r}" for at in positions) or "none"


def _refuse_off_canvas(path: TimeAxisPath, at: int, canvas: IIIFCanvasInfo) -> None:
    """Refuse a horizontal or vertical line of *path* that leaves *canvas*.

    Raises:
        ValueError: Naming the canvas, the piece and the canvas's size.
    """
    pieces = path.pieces if isinstance(path, PiecewisePath) else (path,)
    for k, piece in enumerate(pieces, start=1):
        if isinstance(piece, HorizontalLinePath):
            on_canvas = (
                0 <= piece.x0
                and piece.x1 <= canvas.width
                and 0 <= piece.y <= canvas.height
            )
            needed = f"0 <= x0, x1 <= {canvas.width} and 0 <= y <= {canvas.height}"
        elif isinstance(piece, VerticalLinePath):
            on_canvas = (
                0 <= piece.y0
                and piece.y1 <= canvas.height
                and 0 <= piece.x <= canvas.width
            )
            needed = f"0 <= y0, y1 <= {canvas.height} and 0 <= x <= {canvas.width}"
        else:
            continue
        if not on_canvas:
            which = (
                f"piece {k} of its time path" if path is not piece else "its time path"
            )
            raise ValueError(
                f"Canvas {at} ({canvas.id!r}, {canvas.width} x {canvas.height} "
                f"pixels): {which}, {piece!r}, leaves the canvas; a line on it "
                f"needs {needed}."
            )
