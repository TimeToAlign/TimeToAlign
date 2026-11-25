"""MeiFacsimileLoader for the facsimile of an MEI file: surfaces and zones.

An MEI ``facsimile`` (inside ``music``) holds one ``surface`` per page or
image, each with a ``graphic`` naming the image (``@target``, ``@width``,
``@height``) and ``zone`` elements, areas of the surface given by their
corners ``@ulx``, ``@uly``, ``@lrx``, ``@lry`` or by ``@points``. Elements of
the encoding point to their zones with ``@facs``.

A page image has no time axis of its own: its width is not one. The axis is
the page's time path, the line a reader follows through the image, one
horizontal piece per accolade (system). MEI zones name no accolade, so the
path is made from the zones themselves
(:meth:`~timetoalign.loader.graphical.paths.TimeAxisPath.from_boxes`: in
document order, a zone whose left edge falls back by more than half the
span of all left edges begins the next accolade), or given by the caller.

Each surface becomes a page: a segment line of its accolades, all in
``pixels``. Each accolade is a :class:`DiscreteGraphicalTimeline` as long as
its piece of the path, and each zone a :class:`Box` event of the accolade it
lies on (a :class:`BoundingBox`, or a :class:`Quadrilateral` where the zone
states ``@points``). The ``@facs`` links are kept as a mapping from element
to zones.

MEI corners are points, not pixel indices: a zone's width is ``lrx - ulx``.
Coordinates are read exactly as written; a value that is no whole number of
pixels is refused, never rounded. A zone's ``@points`` are read as a
quadrilateral when they are four points bounding a convex area, listed in
either direction from any corner (they are held from the top-left corner,
clockwise); any other ``@points`` (another number of points, a concave or
crossed outline) refuses the file, naming the zone. Nothing is put in such a
zone's place, not even its bounding box.

Zone coordinates are taken as pixels of the surface's first graphic. The
graphic's size is on no timeline; where it states its width in pixels, a
zone ending past that width refuses the file. The surface's own ``@ulx`` ..
``@lry`` are not read.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar

from timetoalign.core import Shape, TimeUnit
from timetoalign.loader._xml import XML_ID
from timetoalign.loader.base import Loader
from timetoalign.loader.graphical._located import located
from timetoalign.loader.graphical._pages import (
    Ids,
    Page,
    PageAccolade,
    PageBox,
    build_pages,
    build_volume,
)
from timetoalign.loader.graphical.paths import (
    HorizontalLinePath,
    PiecewisePath,
    TimeAxisPath,
    group_into_accolades,
)
from timetoalign.loader.metadata.mei_head import MEI_NAMESPACE
from timetoalign.timelines import DiscreteGraphicalTimeline, SegmentLine

module_logger = logging.getLogger(__name__)

_MEI = f"{{{MEI_NAMESPACE}}}"
# The attributes that place a zone on its image.
_CORNERS = ("ulx", "uly", "lrx", "lry")
# A graphic's size in pixels: a number, unitless or in px.
_PIXEL_SIZE = re.compile(r"\s*(\d+(?:\.\d+)?)\s*(?:px)?\s*")

#: The time paths a caller gives: one path, or one per surface named.
Paths = TimeAxisPath | Mapping[str | int, TimeAxisPath] | None


@dataclass(frozen=True)
class _MeiZone:
    """One zone with coordinates: its ``xml:id`` and ``@type`` (if any), its shape."""

    id: str | None
    type: str | None
    shape: Shape


@dataclass(frozen=True)
class _MeiSurface:
    """One surface, as the page it becomes needs it.

    Attributes:
        id: The surface's ``xml:id``, or None.
        name: Its ``@n``, or None.
        image: The first graphic's ``@target``, or None.
        zones: Its zones with coordinates, in document order.
    """

    id: str | None
    name: str | None
    image: str | None
    zones: tuple[_MeiZone, ...]

    def taking_part(self, types: Collection[str] | None) -> list[_MeiZone]:
        """The zones whose ``@type`` is one of *types*; every zone for None."""
        return [zone for zone in self.zones if types is None or zone.type in types]

    def grouped(
        self, position: int, zones: list[_MeiZone]
    ) -> tuple[PiecewisePath, list[int]]:
        """Group *zones* of the surface at *position* into accolades.

        Returns:
            The time path made from the zones, one piece per accolade, and
            per zone the index of its accolade (from 0).

        Raises:
            ValueError: If no path can be made from them, naming the surface.
        """
        accolades = group_into_accolades([zone.shape.x_min.value for zone in zones])
        try:
            path = TimeAxisPath.from_boxes(
                [zone.shape for zone in zones], accolades=accolades
            )
        except ValueError as error:
            raise located(f"MEI {self.label(position)}", error) from error
        return path, accolades

    def label(self, position: int) -> str:
        """How a message names the surface standing at *position* (from 0)."""
        return f"surface {position}" if self.id is None else f"surface {self.id!r}"


@dataclass(frozen=True)
class _MeiFacsimile:
    """What the loader reads of one MEI file."""

    surfaces: tuple[_MeiSurface, ...]
    zones_without_coordinates: tuple[str, ...]
    facs: dict[str, tuple[str, ...]]


def _zone_types(zone_types: str | Collection[str] | None) -> Collection[str] | None:
    """The ``@type`` values asked for, one given as a bare string included."""
    return {zone_types} if isinstance(zone_types, str) else zone_types


def _pixel_size(text: str | None) -> Fraction | None:
    """Read a graphic size given in pixels; None for none or another unit."""
    match = None if text is None else _PIXEL_SIZE.fullmatch(text)
    return None if match is None else Fraction(match.group(1))


def _surface(surface: ET.Element, missing: list[str]) -> _MeiSurface:
    """Read one surface, its graphic and its zones.

    Args:
        surface: The ``surface`` element.
        missing: Receives the ids of zones stating no coordinates.

    Raises:
        ValueError: If a zone is refused, naming the zone.
    """
    surface_id = surface.get(XML_ID)
    graphic = surface.find(f"{_MEI}graphic")
    # The image's right edge, where the graphic states its width in pixels.
    right = None if graphic is None else _pixel_size(graphic.get("width"))
    zones = []
    label = "an MEI surface" if surface_id is None else f"MEI surface {surface_id!r}"
    for k, zone in enumerate(surface.findall(f"{_MEI}zone"), start=1):
        zone_id = zone.get(XML_ID)
        attributes = {
            name: value
            for name in (*_CORNERS, "points")
            if (value := zone.get(name)) is not None
        }
        if not attributes:
            if zone_id is not None:
                missing.append(zone_id)
            continue
        where = f"zone {k} of {label}" if zone_id is None else f"MEI zone {zone_id!r}"
        try:
            shape = Shape.from_mei_zone(attributes, unit=TimeUnit.pixels)
        except ValueError as error:
            raise located(where, error) from error
        if right is not None and shape.x_max.value > right:
            raise ValueError(
                f"{where} ends at x {shape.x_max.value}, past the right edge of its "
                f"image at x {right}."
            )
        zones.append(_MeiZone(id=zone_id, type=zone.get("type"), shape=shape))
    return _MeiSurface(
        id=surface_id,
        name=surface.get("n"),
        image=None if graphic is None else graphic.get("target"),
        zones=tuple(zones),
    )


def _read_facsimile(source: Path) -> _MeiFacsimile:
    """Parse the surfaces, zones and ``@facs`` links of the MEI file *source*.

    Raises:
        ValueError: As :meth:`MeiFacsimileLoader._load_source` states.
    """
    root = ET.parse(source).getroot()
    if not root.tag.startswith(_MEI):
        raise ValueError(
            f"{source} is no MEI document: its root <{root.tag}> is not in the MEI "
            f"namespace {MEI_NAMESPACE}."
        )
    missing: list[str] = []
    surfaces = tuple(
        _surface(surface, missing)
        for facsimile in root.iter(f"{_MEI}facsimile")
        for surface in facsimile.findall(f"{_MEI}surface")
    )
    facs = {
        element_id: tuple(reference.removeprefix("#") for reference in links.split())
        for element in root.iter()
        if (links := element.get("facs")) is not None
        and (element_id := element.get(XML_ID)) is not None
    }
    return _MeiFacsimile(
        surfaces=surfaces, zones_without_coordinates=tuple(missing), facs=facs
    )


class MeiFacsimileLoader(Loader[_MeiFacsimile]):
    """Load the surfaces and zones of an MEI file's facsimile.

    Each ``surface`` holding zones becomes a page: a segment line of its
    accolades, in ``pixels`` (``int``). An accolade is one horizontal piece
    of the page's time path, a :class:`DiscreteGraphicalTimeline` as long as
    the piece, stating the area it stands for in ``meta["box"]`` and its
    image in ``meta["image"]``. Each zone taking part is a
    :class:`~timetoalign.core.Box` event of the accolade it lies on, holding
    the zone's shape in the image's pixels. :attr:`facs` keeps which zones
    each element of the encoding points to.

    For a Box on an accolade, ``start`` is the zone's least x minus the x of
    the accolade's ``meta["box"]`` and ``duration`` is the zone's x extent,
    so the box, the shape and the interval agree.

    :meth:`create_timeline` gives the whole facsimile as one volume (a
    segment line of the pages), :meth:`create_timelines` one root per page,
    :meth:`create_paths` the time paths made from the zones.

    Known limits: only the first ``graphic`` of a surface is read, and zone
    coordinates are taken as its pixels (the surface's own ``@ulx`` ..
    ``@lry`` are not read, so a surface whose corners measure another space
    than the graphic's pixels has its zones placed as if they were the
    graphic's pixels); a graphic size in a unit other than pixels (``cm``,
    ``in``, ...) states no pixel size and is not read; MEI states no
    accolades, so they are found from the zones' left edges unless paths
    are given, and a page whose zones are not listed accolade by accolade
    needs a path; a zone without ``xml:id`` is minted ``box_N`` on its
    accolade and cannot be pointed to; an element without ``xml:id`` is
    left out of :attr:`facs`, and a zone stating no coordinates out of the
    timelines (:attr:`zones_without_coordinates` lists those with an id). A
    zone whose ``@points`` are not four points bounding a convex area (five
    points, a concave or crossed outline) cannot be a
    :class:`~timetoalign.core.Quadrilateral` and refuses the file, the
    message naming the zone and the reason. The volume takes no name from
    the file's title.

    Examples:
        The pages of an MEI file as timelines, the measures of the first
        accolade, and the zones a measure points to::

            loader = MeiFacsimileLoader.from_file("chorale_facsimile.mei")
            (page,) = loader.create_timelines(zone_types="measure")
            accolade = page.get_child(page.list_children()[0])
            boxes = accolade.get_events().table.to_pylist()
            zones = loader.facs["m-43"]
    """

    #: One loader reads one MEI file.
    _single_source: ClassVar[bool] = True

    def __init__(self) -> None:
        """Initialize the loader."""
        super().__init__()
        self._facsimile: _MeiFacsimile | None = None
        self._logger = module_logger.getChild(self.__class__.__name__)

    def __repr__(self) -> str:
        """Return a concise representation."""
        if self._facsimile is None:
            return "MeiFacsimileLoader(not loaded)"
        surfaces = self._facsimile.surfaces
        zones = sum(len(surface.zones) for surface in surfaces)
        return f"MeiFacsimileLoader(surfaces={len(surfaces)}, zones={zones})"

    @property
    def facs(self) -> dict[str, tuple[str, ...]]:
        """Per element ``xml:id``, the zone ids its ``@facs`` names, in order.

        Each reference is given as written, without its leading ``#``. A
        zone's ``xml:id`` is the id of its Box unless ``box_ids`` is given to
        :meth:`create_timeline` or :meth:`create_timelines`: the Boxes then
        carry the given ids, and this mapping, which still names the zones
        of the file, no longer names Box ids.
        """
        return dict(self._require_facsimile().facs)

    @property
    def zones_without_coordinates(self) -> tuple[str, ...]:
        """The ids of the zones stating neither corners nor points."""
        return self._require_facsimile().zones_without_coordinates

    def _load_source(self, source: Path) -> _MeiFacsimile:
        """Read the facsimile of one MEI file.

        Args:
            source: Path to the MEI file.

        Returns:
            Its surfaces, zones and ``@facs`` links.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the root is not in the MEI namespace, or a zone
                states incomplete corners, ``@points`` other than four
                points bounding a convex area, or points and corners naming
                two areas, or ends past the width its graphic states in
                pixels; a zone's message names it (by ``xml:id``, else by
                its position on its surface).
        """
        return _read_facsimile(source)

    def _accept_source(
        self, path: Path, source_meta: dict[str, Any], payload: _MeiFacsimile
    ) -> None:
        """Keep the facsimile of the one source."""
        super()._accept_source(path, source_meta, payload)
        self._facsimile = payload
        self._logger.debug(f"Loaded {path}: {self!r}")

    def clear(self) -> None:
        """Forget the loaded file."""
        super().clear()
        self._facsimile = None

    def _require_facsimile(self) -> _MeiFacsimile:
        """Return the loaded facsimile, or raise when nothing is loaded."""
        if self._facsimile is None:
            raise RuntimeError("No MEI file loaded. Call load() first.")
        return self._facsimile

    def _pages(
        self, paths: Paths, zone_types: str | Collection[str] | None
    ) -> list[Page]:
        """The pages of a call: each surface with a time path, in document order.

        Args:
            paths: The time paths given by the caller.
            zone_types: The ``@type`` values of the zones taking part, or
                None for every zone with coordinates.

        Raises:
            RuntimeError: If no file has been loaded.
            KeyError: If *paths* names a surface the file does not hold.
            ValueError: If one path is given for a file of several surfaces,
                a surface is given two paths, zones take part on a path that
                is neither a horizontal line nor a ``PiecewisePath``, or no
                path can be made from a surface's zones.
        """
        surfaces = self._require_facsimile().surfaces
        types = _zone_types(zone_types)
        given = self._given_paths(paths)
        pages = []
        for position, surface in enumerate(surfaces):
            zones = surface.taking_part(types)
            boxes = [PageBox(zone.shape, source_id=zone.id) for zone in zones]
            path = given.get(position)
            accolades = None
            if path is None:
                if not zones:
                    continue
                # Each zone stays in the accolade it was grouped into.
                path, indices = surface.grouped(position, zones)
                accolades = [
                    PageAccolade(
                        boxes=[box for box, index in zip(boxes, indices) if index == k]
                    )
                    for k in range(len(path.pieces))
                ]
                boxes = []
            elif zones and not isinstance(path, HorizontalLinePath | PiecewisePath):
                raise ValueError(
                    f"The time path given for MEI {surface.label(position)} is a "
                    f"{type(path).__name__}; its {len(zones)} zones are placed "
                    "along a HorizontalLinePath or the pieces of a PiecewisePath."
                )
            pages.append(
                Page(
                    name=surface.name,
                    image=surface.image,
                    path=path,
                    source_id=surface.id,
                    accolades=accolades,
                    boxes=boxes,
                )
            )
        return pages

    def _given_paths(self, paths: Paths) -> dict[int, TimeAxisPath]:
        """The given *paths* by the position of the surface each one names."""
        surfaces = self._require_facsimile().surfaces
        if paths is None:
            return {}
        if isinstance(paths, TimeAxisPath):
            if len(surfaces) != 1:
                raise ValueError(
                    f"One time path names one surface; this file holds "
                    f"{len(surfaces)}: give a mapping from each surface's xml:id "
                    "or position to its path."
                )
            return {0: paths}
        positions = {
            surface.id: position
            for position, surface in enumerate(surfaces)
            if surface.id is not None
        }
        given: dict[int, TimeAxisPath] = {}
        for key, path in paths.items():
            position = positions.get(key) if isinstance(key, str) else key
            if position is None or position not in range(len(surfaces)):
                raise KeyError(
                    f"paths names the surface {key!r}, which this file does not "
                    f"hold. Available surfaces: "
                    f"{[surface.id for surface in surfaces]} (or their positions "
                    "from 0)."
                )
            if position in given:
                raise ValueError(
                    f"paths gives MEI {surfaces[position].label(position)} two "
                    "paths, one by its xml:id and one by its position."
                )
            given[position] = path
        return given

    def create_paths(
        self, *, zone_types: str | Collection[str] | None = None
    ) -> dict[str | int, PiecewisePath]:
        """Make each surface's time path from its zones.

        The zones taking part are grouped into accolades in document order
        (:meth:`~timetoalign.loader.graphical.paths.TimeAxisPath.from_boxes`):
        a zone whose left edge lies left of the previous zone's by more than
        half the span of all left edges begins the next accolade. Each
        accolade is one horizontal piece from its zones' least x to their
        greatest, its strip the zones' y extent. These are the paths
        :meth:`create_timeline` and :meth:`create_timelines` use where none
        is given; a path that needs correcting can be rewritten and passed
        to them as ``paths``.

        Args:
            zone_types: The ``@type`` of the zones taking part (one, or
                several); None for every zone with coordinates.

        Returns:
            Per surface with a zone taking part, in document order, its
            path; a surface is named by its ``xml:id``, or by its position
            in the file (from 0) where it has none.

        Raises:
            RuntimeError: If no file has been loaded.
            ValueError: If no path can be made from a surface's zones (an
                accolade without x extent).
        """
        types = _zone_types(zone_types)
        paths: dict[str | int, PiecewisePath] = {}
        for position, surface in enumerate(self._require_facsimile().surfaces):
            zones = surface.taking_part(types)
            if zones:
                key = position if surface.id is None else surface.id
                paths[key], _ = surface.grouped(position, zones)
        return paths

    def create_timeline(
        self,
        *,
        uid: str | None = None,
        name: str | None = None,
        paths: Paths = None,
        zone_types: str | Collection[str] | None = None,
        page_ids: Ids = None,
        accolade_ids: Ids = None,
        box_ids: Ids = None,
    ) -> SegmentLine:
        """Create the facsimile as one volume: a segment line of its pages.

        The pages are those of :meth:`create_timelines`, in document order,
        each a segment of the volume; the volume's length is the sum of
        theirs. The volume is locked.

        Args:
            uid: The volume's id; minted ``dgt_N`` when None.
            name: The volume's name.
            paths: As :meth:`create_timelines`.
            zone_types: As :meth:`create_timelines`.
            page_ids: As :meth:`create_timelines`.
            accolade_ids: As :meth:`create_timelines`.
            box_ids: As :meth:`create_timelines`.

        Returns:
            The volume, a ``SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]``
            (a ``SegmentLine[DiscreteGraphicalTimeline]`` where every page is
            one line given in ``paths``).

        Raises:
            RuntimeError: If no file has been loaded.
            KeyError: As :meth:`create_timelines`.
            ValueError: As :meth:`create_timelines`; also if no surface has
                a time path, or some pages are built of accolades and
                others are one line.
        """
        return build_volume(
            self,
            self._pages(paths, zone_types),
            uid=uid,
            name=name,
            page_ids=page_ids,
            accolade_ids=accolade_ids,
            box_ids=box_ids,
        )

    def create_timelines(
        self,
        *,
        paths: Paths = None,
        zone_types: str | Collection[str] | None = None,
        page_ids: Ids = None,
        accolade_ids: Ids = None,
        box_ids: Ids = None,
    ) -> list[DiscreteGraphicalTimeline | SegmentLine[DiscreteGraphicalTimeline]]:
        """Create one root timeline per surface, in document order.

        A surface's page is a locked segment line of its accolades, in
        ``pixels`` (``int``): its id the surface's ``xml:id`` (minted
        ``dgt_N`` where it has none), its name the surface's ``@n``
        (``page_N`` by its position in the call, from 1, where it has
        none), its length the sum of its accolades', ``meta["image"]`` the
        graphic's ``@target``. It states no ``meta["box"]``: its axis is a
        concatenation of accolades, not an area, and the graphic's size is
        on no timeline.

        The accolades are the pieces of the surface's time path, made from
        the zones taking part (:meth:`create_paths`) unless ``paths`` gives
        one. Each is a :class:`DiscreteGraphicalTimeline` named
        ``accolade_K`` (K from 1 on its page) whose length is its piece's
        ``x1 - x0``, with ``meta["box"]`` the piece's strip (``x0``, ``y -
        tolerance``, ``x1 - x0``, ``2 * tolerance``) and ``meta["image"]``.

        Each zone taking part is a Box event of its accolade: the one it was
        grouped into or, on a given path, the piece it lies on (the piece
        whose x extent holds the zone's and whose strip holds the zone's
        vertical centre; of several the nearest). Its id is the zone's
        ``xml:id`` (minted ``box_N`` on its accolade where it has none); its
        shape is the zone's coordinates as written; its start is the shape's
        least x less the accolade's ``x0``.

        A surface without a zone taking part and without a given path is
        left out.

        Args:
            paths: Time paths replacing those made from the zones: one path
                where the file holds one surface, or a mapping from a
                surface's ``xml:id`` or position in the file (from 0) to its
                path. Where zones take part, a path is a
                ``HorizontalLinePath`` (the page is then one plain timeline
                along it) or a ``PiecewisePath``.
            zone_types: The ``@type`` of the zones taking part (one, or
                several); None for every zone with coordinates.
            page_ids: The pages' ids, replacing the surfaces' ``xml:id``:
                one id per page in document order, or a template filled
                with ``str.format`` from ``page`` (the page's position in
                the call, from 1) and ``n`` (the running count).
            accolade_ids: The accolades' ids (minted ``dgt_N`` when None):
                one id per accolade in document order, or a template filled
                from ``page``, ``accolade`` (its position on its page) and
                ``n``.
            box_ids: The Boxes' ids, replacing the zones' ``xml:id`` (see
                :attr:`facs`): one id per zone taking part in document
                order, or a template filled from ``page``, ``accolade``,
                ``box`` (its position in its accolade) and ``n`` (the zone's
                position among those taking part in the call):
                ``box_ids="m{n}"``.

        Returns:
            The pages' timelines.

        Raises:
            RuntimeError: If no file has been loaded.
            KeyError: If ``paths`` names a surface the file does not hold.
            ValueError: If one path is given for a file of several surfaces
                or a surface is given two; zones take part on a path that is
                neither a horizontal line nor a ``PiecewisePath``; a zone
                lies on no piece of its surface's path; an ids argument does
                not fit the call or gives one id to two timelines; or a
                path holds a number the pixel axis cannot hold exactly.
                Nothing is built then.
        """
        return build_pages(
            self,
            self._pages(paths, zone_types),
            page_ids=page_ids,
            accolade_ids=accolade_ids,
            box_ids=box_ids,
        )
