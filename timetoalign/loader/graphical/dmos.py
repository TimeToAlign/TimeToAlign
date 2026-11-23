"""DmosLoader for the page layout that DMOS writes as JSON.

DMOS, an optical music recognition system, writes for each page of a score
image the systems it found and, per system, its measures, each as an area
given by four corners in the pixels of the full page image (x to the right,
y down, origin at the top left corner). The file is one JSON object whose
``pages`` list holds, per page, ``no_page``, ``page_url`` (the page image)
and ``systems``; a system holds ``id``, ``region`` and ``measures``; a
measure holds ``region``.

The loader reads exactly that: pages, systems and measures. Each page is a
:class:`DiscreteGraphicalTimeline` in ``pixels`` along the image's x axis,
each system a child of its page at its least x, each measure a :class:`Box`
event of its system holding the measure's :class:`Quadrilateral`. Not read:
the symbols (the measures' ``voices`` and their elements), the staff headers
of systems and measures, ``header_systems``, and the document's ``id``,
``date`` and ``score_image_url``.

DMOS writes corners in two orders, neither of which draws the outline when
followed: measures top left, top right, bottom left, bottom right
(:data:`MEASURE_CORNER_ORDER`); systems top left, bottom left, top right,
bottom right (:data:`SYSTEM_CORNER_ORDER`). Every quadrilateral is read in
its order and held in the canonical one (top left, top right, bottom right,
bottom left), so its outline runs around the area.

DMOS states no page size and no measure identity: the size of a page image
is unknown unless given to :meth:`DmosLoader.create_timelines`, and measures
are known by their position in the document only. A measure must lie within
its system's x extent; one leaving it on either side is refused.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from timetoalign.core import BoundingBox, Box, Quadrilateral, TimeUnit
from timetoalign.loader.base import Loader
from timetoalign.loader.graphical._located import located
from timetoalign.timelines import DiscreteGraphicalTimeline

module_logger = logging.getLogger(__name__)

#: The roles of the four corners of a DMOS measure region, in the order DMOS
#: writes them.
MEASURE_CORNER_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")

#: The roles of the four corners of a DMOS system region, in the order DMOS
#: writes them.
SYSTEM_CORNER_ORDER = ("top_left", "bottom_left", "top_right", "bottom_right")


@dataclass(frozen=True)
class _DmosSystem:
    """One system as DMOS states it.

    Attributes:
        name: DMOS's system id, as a string (ids repeat on some pages).
        area: The system's region.
        measures: The regions of its measures, in document order.
    """

    name: str
    area: Quadrilateral
    measures: tuple[Quadrilateral, ...]


@dataclass(frozen=True)
class _DmosPage:
    """One page as DMOS states it.

    Attributes:
        number: The page's ``no_page``.
        image: The page's ``page_url``, as written.
        systems: Its systems, in document order.
    """

    number: int
    image: str
    systems: tuple[_DmosSystem, ...]


def _member(mapping: Any, key: str, where: str) -> Any:
    """Return *mapping*'s *key*, refusing a mapping that lacks it."""
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f"{where} has no {key!r}.")
    return mapping[key]


def _area(region: Any, order: tuple[str, ...], where: str) -> Quadrilateral:
    """Read a DMOS region whose corners are written in *order*.

    Raises:
        ValueError: The quadrilateral's refusal, prefixed by *where*.
    """
    try:
        return Quadrilateral.from_points(region, order=order, unit=TimeUnit.pixels)
    except (TypeError, ValueError) as error:
        raise located(where, error) from error


def _within(measure: Quadrilateral, system: Quadrilateral, where: str) -> Quadrilateral:
    """Return *measure*, refusing it where its x extent leaves its *system*'s.

    A measure is a Box of its system, placed at its least x less the
    system's: one starting left of its system would start before the
    system's timeline, one ending right of it past its end. Both edges are
    inside.

    Raises:
        ValueError: Naming *where* the measure stands and both x extents.
    """
    if measure.x_min.value < system.x_min.value or (
        measure.x_max.value > system.x_max.value
    ):
        raise ValueError(
            f"{where} spans x {measure.x_min.value} to {measure.x_max.value}, "
            f"which leaves its system's region, spanning x {system.x_min.value} "
            f"to {system.x_max.value}."
        )
    return measure


def _read_pages(source: Path) -> tuple[_DmosPage, ...]:
    """Parse the pages, systems and measures of the DMOS file at *source*.

    Raises:
        ValueError: As :meth:`DmosLoader._load_source` states.
    """
    document = json.loads(source.read_text(encoding="utf-8"))
    pages = []
    for page in _member(document, "pages", f"The DMOS file {source}"):
        number = _member(page, "no_page", f"A page of {source}")
        where = f"DMOS page {number}"
        systems = []
        for k, system in enumerate(_member(page, "systems", where), start=1):
            at = f"{where}, system {k}"
            area = _area(_member(system, "region", at), SYSTEM_CORNER_ORDER, at)
            measures = []
            for j, measure in enumerate(_member(system, "measures", at), start=1):
                measure_at = f"{at}, measure {j}"
                measures.append(
                    _within(
                        _area(
                            _member(measure, "region", measure_at),
                            MEASURE_CORNER_ORDER,
                            measure_at,
                        ),
                        area,
                        measure_at,
                    )
                )
            systems.append(
                _DmosSystem(
                    name=str(_member(system, "id", at)),
                    area=area,
                    measures=tuple(measures),
                )
            )
        pages.append(
            _DmosPage(
                number=number,
                image=_member(page, "page_url", where),
                systems=tuple(systems),
            )
        )
    return tuple(pages)


class DmosLoader(Loader[tuple[_DmosPage, ...]]):
    """Load the pages, systems and measures of a DMOS JSON file.

    Each page becomes a root :class:`DiscreteGraphicalTimeline` in ``pixels``
    (``int``) along its image's x axis, each system a child of its page and
    each measure a :class:`~timetoalign.core.Box` event of its system,
    holding the measure's quadrilateral in the page image's pixels (see the
    module docstring for what is and is not read).

    For a Box on a system, ``start`` is the measure's least x minus the
    system's least x and ``duration`` is the measure's x extent, so the
    system's ``meta["box"]`` and the Box's shape, both in image space, agree
    with the interval.

    Known limits: DMOS states no page size, so a page's length is unknown
    unless :meth:`create_timelines` is given it; measures have no id and no
    number in DMOS, so their Boxes are minted ``box_N`` per system in
    document order; DMOS system ids repeat on some pages, so a system's name
    (its DMOS id) need not be unique, while its timeline id is. A measure
    whose x extent leaves its system's region, on the left or on the right,
    cannot be a Box of that system's timeline and refuses the file (the
    system's timeline spans exactly its region's x extent; neither is
    widened to hold the measure).

    Examples:
        The systems and measures of a DMOS file whose first page image is
        4512 by 6174 pixels::

            loader = DmosLoader.from_file("dmos.json")
            pages = loader.create_timelines(page_sizes={1: (4512, 6174)})
            page = pages[0]
            system = page.get_child(page.list_children()[0])
            boxes = system.get_events().table.to_pylist()
    """

    #: One loader reads one DMOS file.
    _single_source: ClassVar[bool] = True

    def __init__(self) -> None:
        """Initialize the loader."""
        super().__init__()
        self._pages: tuple[_DmosPage, ...] | None = None
        self._logger = module_logger.getChild(self.__class__.__name__)

    def __repr__(self) -> str:
        """Return a concise representation."""
        if self._pages is None:
            return "DmosLoader(not loaded)"
        systems = [system for page in self._pages for system in page.systems]
        measures = sum(len(system.measures) for system in systems)
        return (
            f"DmosLoader(pages={len(self._pages)}, systems={len(systems)}, "
            f"measures={measures})"
        )

    def _load_source(self, source: Path) -> tuple[_DmosPage, ...]:
        """Read the pages, systems and measures of one DMOS JSON file.

        Args:
            source: Path to the JSON file.

        Returns:
            Its pages, in document order.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is no JSON, a page, system or measure
                lacks a member read, a region is not four corners running
                around a convex outline in the order DMOS writes them, or a
                measure's x extent leaves its system's region on either
                side; the message names the page by ``no_page`` and the
                system and measure by their position, counted from 1.
        """
        return _read_pages(source)

    def _accept_source(
        self, path: Path, source_meta: dict[str, Any], payload: tuple[_DmosPage, ...]
    ) -> None:
        """Keep the pages of the one source."""
        super()._accept_source(path, source_meta, payload)
        self._pages = payload
        self._logger.debug(f"Loaded {path}: {self!r}")

    def clear(self) -> None:
        """Forget the loaded file."""
        super().clear()
        self._pages = None

    def create_timelines(
        self, *, page_sizes: Mapping[int, tuple[int, int]] | None = None
    ) -> list[DiscreteGraphicalTimeline]:
        """Create one root timeline per page, in document order.

        A page is in ``pixels`` (``int``), minted ``dgt_N``, named by its
        ``no_page`` and states its image in ``meta["image"]`` (the
        ``page_url``). Where *page_sizes* gives its image's width and height,
        its length is the width, ``meta["box"]`` is ``{"x": 0, "y": 0,
        "width": w, "height": h}`` and the page is locked, since the width
        is the image's; else its length is unknown and it has no box.

        Each system is a child of its page at its least x, of length its x
        extent, minted ``dgt_N``, named by its DMOS id, with ``meta["box"]``
        its quadrilateral's dict. Its measures are its Box events in
        document order, minted ``box_N``.

        Args:
            page_sizes: The ``(width, height)`` in pixels of each page image
                whose size is known, keyed by ``no_page``.

        Returns:
            The pages' timelines.

        Raises:
            RuntimeError: If no file has been loaded.
            KeyError: If *page_sizes* names a page the file does not hold.
            ValueError: If a size is not a whole, non-negative number of
                pixels, or a system ends past the right edge of its page's
                image; the message names the page by ``no_page`` and the
                system by its position on the page, counted from 1.
        """
        from timetoalign.timelines.factory import (
            _flag_minted,
            _mint_timeline_id,
            _root_timeline_id,
        )

        if self._pages is None:
            raise RuntimeError("No DMOS file loaded. Call load() first.")
        sizes = dict(page_sizes or {})
        numbers = [page.number for page in self._pages]
        for key in sizes:
            if key not in numbers:
                raise KeyError(
                    f"page_sizes names page {key!r}, which the DMOS file does not "
                    f"hold; its pages are {', '.join(map(str, numbers))}."
                )

        timelines = []
        for page in self._pages:
            meta: dict[str, Any] = {"image": page.image}
            size = sizes.get(page.number)
            if size is not None:
                width, height = size
                meta["box"] = BoundingBox.from_xywh(
                    0, 0, width, height, unit=TimeUnit.pixels
                ).to_dict()
            uid, minted = _root_timeline_id(self, TimeUnit.pixels, None)
            timeline = _flag_minted(
                DiscreteGraphicalTimeline(
                    length=None if size is None else width,
                    unit=TimeUnit.pixels,
                    uid=uid,
                    name=str(page.number),
                    locked=size is not None,
                    meta=meta,
                ),
                minted,
            )
            for k, system in enumerate(page.systems, start=1):
                area = system.area
                if size is not None and area.x_max.value > width:
                    raise ValueError(
                        f"DMOS page {page.number}, system {k} ends at x "
                        f"{area.x_max.value}, past the right edge of its image, "
                        f"{width} pixels wide."
                    )
                child = _flag_minted(
                    DiscreteGraphicalTimeline(
                        length=area.x_max.value - area.x_min.value,
                        unit=TimeUnit.pixels,
                        uid=_mint_timeline_id(self, TimeUnit.pixels),
                        name=system.name,
                        meta={"box": area.to_dict()},
                    ),
                    True,
                )
                timeline.add_child(child, area.x_min)
                child.add_events(
                    [
                        Box.from_shape(measure, origin=area.x_min)
                        for measure in system.measures
                    ]
                )
            timelines.append(timeline)
        return timelines
