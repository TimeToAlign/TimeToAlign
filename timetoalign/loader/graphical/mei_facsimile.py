"""MeiFacsimileLoader for the facsimile of an MEI file: surfaces and zones.

An MEI ``facsimile`` (inside ``music``) holds one ``surface`` per page or
image, each with a ``graphic`` naming the image (``@target``, ``@width``,
``@height``) and ``zone`` elements, areas of the surface given by their
corners ``@ulx``, ``@uly``, ``@lrx``, ``@lry`` or by ``@points``. Elements of
the encoding point to their zones with ``@facs``.

Each surface becomes a root :class:`DiscreteGraphicalTimeline` in ``pixels``
along the image's x axis, each zone a :class:`Box` event of its surface (a
:class:`BoundingBox`, or a :class:`Quadrilateral` where the zone states
``@points``), and the ``@facs`` links a mapping from element to zones.

MEI corners are points, not pixel indices: a zone's width is ``lrx - ulx``.
Coordinates are read exactly as written; a value that is no whole number of
pixels is refused, never rounded.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar

from timetoalign.core import BoundingBox, Box, Coordinate, Shape, TimeUnit
from timetoalign.loader._xml import XML_ID
from timetoalign.loader.base import Loader
from timetoalign.loader.graphical.dmos import _located
from timetoalign.loader.metadata.mei_head import MEI_NAMESPACE
from timetoalign.timelines import DiscreteGraphicalTimeline

module_logger = logging.getLogger(__name__)

_MEI = f"{{{MEI_NAMESPACE}}}"
# The attributes that place a zone (and a surface) on its image.
_CORNERS = ("ulx", "uly", "lrx", "lry")
# A graphic's size in pixels: a number, unitless or in px.
_PIXEL_SIZE = re.compile(r"\s*(\d+(?:\.\d+)?)\s*(?:px)?\s*")


@dataclass(frozen=True)
class _MeiZone:
    """One zone with coordinates: its ``xml:id`` (if any) and its shape."""

    id: str | None
    shape: Shape


@dataclass(frozen=True)
class _MeiSurface:
    """One surface, as the timeline it becomes needs it.

    Attributes:
        id: The surface's ``xml:id``, or None.
        name: Its ``@n``, or None.
        image: The first graphic's ``@target``, or None.
        length: The width of the image in pixels, or None where unknown.
        box: The area of the image the surface stands for, or None.
        zones: Its zones with coordinates, in document order.
    """

    id: str | None
    name: str | None
    image: str | None
    length: Coordinate | None
    box: BoundingBox | None
    zones: tuple[_MeiZone, ...]


@dataclass(frozen=True)
class _MeiFacsimile:
    """What the loader reads of one MEI file."""

    surfaces: tuple[_MeiSurface, ...]
    zones_without_coordinates: tuple[str, ...]
    facs: dict[str, tuple[str, ...]]


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
        ValueError: If a size or a zone is refused, naming the zone.
    """
    surface_id = surface.get(XML_ID)
    graphic = surface.find(f"{_MEI}graphic")
    width = height = None
    if graphic is not None:
        width = _pixel_size(graphic.get("width"))
        height = _pixel_size(graphic.get("height"))
    stated = {name: surface.get(name) for name in _CORNERS}
    corners = (
        Shape.from_mei_zone(stated, unit=TimeUnit.pixels)
        if None not in stated.values()
        else None
    )
    if width is not None:
        length = Coordinate(width, TimeUnit.pixels)
        box = (
            None
            if height is None
            else BoundingBox.from_xywh(0, 0, width, height, unit=TimeUnit.pixels)
        )
    elif corners is not None:
        length, box = Coordinate(corners.width.value, TimeUnit.pixels), corners
    else:
        length = box = None

    # The image's right edge in its own space, where its width is known.
    right = (
        None
        if length is None
        else length.value + (0 if box is None else box.x_min.value)
    )
    zones = []
    label = "an MEI surface" if surface_id is None else f"MEI surface {surface_id!r}"
    for k, zone in enumerate(surface.findall(f"{_MEI}zone"), start=1):
        zone_id = zone.get(XML_ID)
        attributes = {
            name: zone.get(name)
            for name in (*_CORNERS, "points")
            if zone.get(name) is not None
        }
        if not attributes:
            if zone_id is not None:
                missing.append(zone_id)
            continue
        where = f"zone {k} of {label}" if zone_id is None else f"MEI zone {zone_id!r}"
        try:
            shape = Shape.from_mei_zone(attributes, unit=TimeUnit.pixels)
        except ValueError as error:
            raise _located(where, error) from error
        if right is not None and shape.x_max.value > right:
            raise ValueError(
                f"{where} ends at x {shape.x_max.value}, past the right edge of its "
                f"image at x {right}."
            )
        zones.append(_MeiZone(id=zone_id, shape=shape))
    return _MeiSurface(
        id=surface_id,
        name=surface.get("n"),
        image=None if graphic is None else graphic.get("target"),
        length=length,
        box=box,
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
        element.get(XML_ID): tuple(
            reference.removeprefix("#") for reference in element.get("facs").split()
        )
        for element in root.iter()
        if element.get("facs") is not None and element.get(XML_ID) is not None
    }
    return _MeiFacsimile(
        surfaces=surfaces, zones_without_coordinates=tuple(missing), facs=facs
    )


class MeiFacsimileLoader(Loader[_MeiFacsimile]):
    """Load the surfaces and zones of an MEI file's facsimile.

    Each ``surface`` becomes a root :class:`DiscreteGraphicalTimeline` in
    ``pixels`` (``int``) along its image's x axis, and each zone with
    coordinates a :class:`~timetoalign.core.Box` event of it, holding the
    zone's shape in the image's pixels. :attr:`facs` keeps which zones each
    element of the encoding points to.

    For a Box on a surface, ``start`` is the zone's least x minus the x of
    the surface's ``meta["box"]`` (0 where it has none) and ``duration`` is
    the zone's x extent, so the box, the shape and the interval agree.

    Known limits: only the first ``graphic`` of a surface is read; a graphic
    size in a unit other than pixels (``cm``, ``in``, ...) states no pixel
    size and is not read; a zone without ``xml:id`` is minted ``box_N`` and
    cannot be pointed to; an element without ``xml:id`` is left out of
    :attr:`facs`, and a zone stating no coordinates out of the timelines
    (:attr:`zones_without_coordinates` lists those with an id).

    Examples:
        >>> loader = MeiFacsimileLoader.from_file("chorale_facsimile.mei")
        >>> (surface,) = loader.create_timelines()
        >>> boxes = surface.get_events().table.to_pylist()
        >>> zones = loader.facs["m-43"]
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

        Each reference is given as written, without its leading ``#``.
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
            ValueError: If the root is not in the MEI namespace, a graphic's
                pixel size is no whole number, or a zone states incomplete
                corners, points that do not run around a convex outline, or
                points and corners naming two areas, or ends past the right
                edge of its image; a zone's message names it.
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

    def create_timelines(self) -> list[DiscreteGraphicalTimeline]:
        """Create one root timeline per surface, in document order.

        A surface's timeline is in ``pixels`` (``int``), its id the
        surface's ``xml:id`` (minted ``dgt_N`` where it has none), its name
        the surface's ``@n`` (where it has none, the timeline's id). Its length is the graphic's ``@width``, else
        ``lrx - ulx`` of the surface, else unknown; a timeline of known
        length is locked, since the width is the image's.
        ``meta["image"]`` is the graphic's ``@target``; ``meta["box"]`` is
        ``{"x": 0, "y": 0, "width": w, "height": h}`` from the graphic's
        size, else the surface's corners where the graphic states no width.

        Each zone with coordinates is a Box event, its id the zone's
        ``xml:id`` (minted ``box_N`` where it has none), in document order.

        Returns:
            The surfaces' timelines.

        Raises:
            RuntimeError: If no file has been loaded.
            ValueError: If a zone's id repeats on its surface.
        """
        from timetoalign.timelines.factory import _flag_minted, _root_timeline_id

        timelines = []
        for surface in self._require_facsimile().surfaces:
            meta: dict[str, Any] = {}
            if surface.image is not None:
                meta["image"] = surface.image
            if surface.box is not None:
                meta["box"] = surface.box.to_dict()
            origin = 0 if surface.box is None else surface.box.x_min
            uid, minted = _root_timeline_id(self, TimeUnit.pixels, surface.id)
            timeline = _flag_minted(
                DiscreteGraphicalTimeline(
                    length=surface.length,
                    unit=TimeUnit.pixels,
                    uid=uid,
                    name=surface.name,
                    locked=surface.length is not None,
                    meta=meta,
                ),
                minted,
            )
            timeline.add_events(
                [
                    Box.from_shape(zone.shape, origin=origin, id=zone.id)
                    for zone in surface.zones
                ]
            )
            timelines.append(timeline)
        return timelines
