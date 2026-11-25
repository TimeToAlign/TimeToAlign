"""DmosLoader for the page layout that DMOS writes as JSON.

DMOS, an optical music recognition system, writes for each page of a score
image the systems it found and, per system, its measures, each as an area
given by four corners in the pixels of the full page image (x to the right,
y down, origin at the top left corner). The file is one JSON object whose
``pages`` list holds, per page, ``no_page``, ``page_url`` (the page image)
and ``systems``; a system holds ``id``, ``region`` and ``measures``; a
measure holds ``region``.

The loader reads exactly that: pages, systems and measures. Nothing in a
page image is a time axis until a path through it is given, and DMOS gives
one: its systems, read one after the other. Each system is one horizontal
piece of its page's time path, over the x extent of its region; a page is
the :class:`~timetoalign.loader.graphical.paths.PiecewisePath` of its
systems. The timelines mirror the path, all in ``pixels``: a system is an
accolade as long as its region is wide, a page the segment line of its
accolades, the file the segment line of its pages. Each measure is a
:class:`Box` event of its system's accolade holding the measure's
:class:`Quadrilateral`. Not read: the symbols (the measures' ``voices`` and
their elements), the staff headers of systems and measures,
``header_systems``, and the document's ``id`` and ``date``.

DMOS writes corners in two orders, neither of which draws the outline when
followed: measures top left, top right, bottom left, bottom right
(:data:`MEASURE_CORNER_ORDER`); systems top left, bottom left, top right,
bottom right (:data:`SYSTEM_CORNER_ORDER`). Every quadrilateral is read in
its order and held in the canonical one (top left, top right, bottom right,
bottom left), so its outline runs around the area.

DMOS states no measure identity: measures are known by their position in
the document only, and are named where the timelines are created. A measure
must lie within its system's x extent; one leaving it on either side is
refused.

DMOS output holds, on some pages, a system whose region is a vertical line
(no x extent) and which holds no measures. Such a system is no part of the
time path: it is left out of the paths and of the timelines, and loading
says so once per file in a warning naming each page and system. A system
without x extent that holds measures is refused where paths or timelines
are created. A page without systems, or all of whose systems are left out,
has no time path and becomes no timeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from timetoalign.core import Quadrilateral, TimeUnit
from timetoalign.loader.base import Loader, Source
from timetoalign.loader.graphical._located import located
from timetoalign.loader.graphical._pages import (
    Ids,
    Page,
    PageAccolade,
    PageBox,
    build_pages,
    build_volume,
)
from timetoalign.loader.graphical.paths import PiecewisePath, TimeAxisPath
from timetoalign.timelines import DiscreteGraphicalTimeline, SegmentLine

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
        position: Its position on its page, counted from 1.
        name: DMOS's system id, as a string (ids repeat on some pages).
        area: The system's region.
        measures: The regions of its measures, in document order.
    """

    position: int
    name: str
    area: Quadrilateral
    measures: tuple[Quadrilateral, ...]

    @property
    def has_x_extent(self) -> bool:
        """Whether the region is wider than a vertical line."""
        return self.area.x_max.value > self.area.x_min.value

    @property
    def is_left_out(self) -> bool:
        """Whether the system is no part of the time path.

        True for a region without x extent holding no measures: nothing of
        it lies along a time axis.
        """
        return not self.has_x_extent and not self.measures


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

    @property
    def path_systems(self) -> tuple[_DmosSystem, ...]:
        """The systems that are pieces of the page's time path, in document order."""
        return tuple(system for system in self.systems if not system.is_left_out)

    @property
    def path(self) -> PiecewisePath:
        """The page's time path: its path systems, one horizontal piece each.

        Raises:
            ValueError: If a system holding measures has a region without x
                extent, naming the page and the system's position.
        """
        systems = self.path_systems
        for system in systems:
            if not system.has_x_extent:
                raise ValueError(
                    f"DMOS page {self.number}, system {system.position} has no x "
                    f"extent (its region spans x {system.area.x_min.value} to "
                    f"{system.area.x_max.value}) and holds {len(system.measures)} "
                    "measure(s): it cannot be a piece of a time path."
                )
        try:
            return TimeAxisPath.from_boxes(
                [system.area for system in systems], accolades=range(len(systems))
            )
        except ValueError as error:
            raise located(f"DMOS page {self.number}", error) from error


@dataclass(frozen=True)
class _DmosDocument:
    """One DMOS file.

    Attributes:
        image: The file's ``score_image_url``, as written; None where it
            states none.
        pages: Its pages, in document order.
    """

    image: str | None
    pages: tuple[_DmosPage, ...]


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


def _read_document(source: Source) -> _DmosDocument:
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
                    position=k,
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
    return _DmosDocument(image=document.get("score_image_url"), pages=tuple(pages))


class DmosLoader(Loader[_DmosDocument]):
    """Load the pages, systems and measures of a DMOS JSON file.

    A page's time path is its systems read one after the other
    (:meth:`create_paths`), and the timelines mirror it, all in ``pixels``
    (``int``):

    - a **system** is an accolade: a
      :class:`~timetoalign.timelines.DiscreteGraphicalTimeline` as long as
      its region is wide, named by its DMOS id, stating the region in
      ``meta["box"]`` and the page image in ``meta["image"]``;
    - a **page** is the segment line of its accolades in document order,
      named by its ``no_page``, stating ``meta["image"]`` (its ``page_url``)
      and no ``meta["box"]``: its axis is a concatenation, not an area;
    - the **volume** (:meth:`create_timeline`) is the segment line of the
      pages.

    Each measure is a :class:`~timetoalign.core.Box` event of its system's
    accolade, holding the measure's quadrilateral in the page image's
    pixels: ``start`` is the measure's least x minus the system's least x
    and ``duration`` the measure's x extent, so the accolade's
    ``meta["box"]`` and the Box's shape, both in image space, agree with the
    interval.

    Known limits: measures have no id and no number in DMOS, so their Boxes
    are minted ``box_N`` per system unless ``box_ids`` names them; DMOS
    system ids repeat on some pages, so an accolade's name (its DMOS id)
    need not be unique, while its timeline id is. A measure whose x extent
    leaves its system's region, on the left or on the right, cannot be a Box
    of that system's accolade and refuses the file (the accolade spans
    exactly its region's x extent; neither is widened to hold the measure).
    A system whose region has no x extent (a vertical line) and which holds
    no measures, as DMOS output holds some, is no part of the time path: it
    is left out of :meth:`create_paths`, :meth:`create_timelines` and
    :meth:`create_timeline` alike, and :meth:`load` logs one warning per
    file naming each such system by its page and its position on it. A
    system without x extent that holds measures loads, but creating paths
    or timelines from the file is refused, naming the page and the system's
    position. A page without systems, or all of whose systems are left out,
    has no time path and is left out.

    Examples:
        The pages of a DMOS file as one timeline, its measures numbered
        through the score::

            loader = DmosLoader.from_file("dmos.json")
            volume = loader.create_timeline(uid="scan", box_ids="m{n}")
            page = volume.get_child(volume.list_children()[0])
            accolade = page.get_child(page.list_children()[0])
            boxes = accolade.get_events().table.to_pylist()
    """

    #: One loader reads one DMOS file.
    _single_source: ClassVar[bool] = True

    def __init__(self) -> None:
        """Initialize the loader."""
        super().__init__()
        self._document: _DmosDocument | None = None
        self._logger = module_logger.getChild(self.__class__.__name__)

    def __repr__(self) -> str:
        """Return a concise representation."""
        if self._document is None:
            return "DmosLoader(not loaded)"
        pages = self._document.pages
        systems = [system for page in pages for system in page.systems]
        measures = sum(len(system.measures) for system in systems)
        return (
            f"DmosLoader(pages={len(pages)}, systems={len(systems)}, "
            f"measures={measures})"
        )

    def _load_source(self, source: Source) -> _DmosDocument:
        """Read the pages, systems and measures of one DMOS JSON file.

        Args:
            source: The JSON file: a path or a ZIP archive member.

        Returns:
            The document: its pages, in document order.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is no JSON, a page, system or measure
                lacks a member read, a region is not four corners running
                around a convex outline in the order DMOS writes them, or a
                measure's x extent leaves its system's region on either
                side; the message names the page by ``no_page`` and the
                system and measure by their position, counted from 1.
        """
        return _read_document(source)

    def _accept_source(
        self, path: Source, source_meta: dict[str, Any], payload: _DmosDocument
    ) -> None:
        """Keep the document of the one source."""
        super()._accept_source(path, source_meta, payload)
        self._document = payload
        self._logger.debug(f"Loaded {path}: {self!r}")
        left_out = [
            f"page {page.number}, system {system.position}"
            for page in payload.pages
            for system in page.systems
            if system.is_left_out
        ]
        if left_out:
            self._logger.warning(
                f"{path}: {len(left_out)} system(s) without x extent and without "
                "measures are no part of the time path and are left out: "
                f"{'; '.join(left_out)}."
            )

    def clear(self) -> None:
        """Forget the loaded file."""
        super().clear()
        self._document = None

    def _loaded(self) -> _DmosDocument:
        """The loaded document.

        Raises:
            RuntimeError: If no file has been loaded.
        """
        if self._document is None:
            raise RuntimeError("No DMOS file loaded. Call load() first.")
        return self._document

    def _pages_with_paths(self) -> list[tuple[_DmosPage, PiecewisePath]]:
        """Each page whose time path has a system, with that path, in document order.

        A page without systems, or all of whose systems are left out of the
        path, is not listed.
        """
        return [(page, page.path) for page in self._loaded().pages if page.path_systems]

    def _described_pages(self) -> list[Page]:
        """The pages as the page builder takes them: a path, its accolades, their Boxes."""
        return [
            Page(
                name=str(page.number),
                image=page.image,
                path=path,
                accolades=[
                    PageAccolade(
                        name=system.name,
                        area=system.area,
                        boxes=[PageBox(measure) for measure in system.measures],
                    )
                    for system in page.path_systems
                ],
            )
            for page, path in self._pages_with_paths()
        ]

    def create_paths(self) -> dict[int, PiecewisePath]:
        """Return each page's time path, keyed by its ``no_page``.

        A page's path is its systems in document order, each one horizontal
        piece: from its region's least x to its greatest, at the middle of
        its y extent, with half that extent as ``tolerance``, so the piece's
        strip is the system's y extent. Nothing is grouped or guessed: every
        stated system is one piece, also where it starts right of the one
        before. A system without x extent and without measures is no piece
        of the path and is left out; a page without systems, or all of whose
        systems are left out, has no path and no entry.

        Returns:
            The paths, in document order.

        Raises:
            RuntimeError: If no file has been loaded.
            ValueError: If a system holding measures has a region without x
                extent; the message names the page by ``no_page`` and the
                system by its position on the page, counted from 1.
        """
        return {page.number: path for page, path in self._pages_with_paths()}

    def create_timeline(
        self,
        *,
        uid: str | None = None,
        name: str | None = None,
        page_ids: Ids = None,
        accolade_ids: Ids = None,
        box_ids: Ids = None,
    ) -> SegmentLine:
        """Create the volume: one timeline of all pages, in document order.

        The volume is a locked segment line of the pages, its length the sum
        of theirs; each page is a segment line of its systems' accolades, as
        :meth:`create_timelines` builds it. The volume states the file's
        ``score_image_url`` in ``meta["image"]`` where the file has one.

        Each of *page_ids*, *accolade_ids* and *box_ids* names every element
        of its kind: a sequence of ids in document order, or a template
        filled with :meth:`str.format` from ``page`` (the page's position
        among the pages created, from 1), ``accolade`` (its position on its
        page), ``box`` (its position in its accolade) and ``n`` (the running
        count of the kind over the call). ``box_ids="m{n}"`` numbers the
        measures through the score. Left out, timelines are minted ``dgt_N``
        (the volume, then each page followed by its accolades) and Boxes
        ``box_N`` per accolade.

        Args:
            uid: The volume's id; minted ``dgt_N`` when None.
            name: The volume's name.
            page_ids: The pages' ids.
            accolade_ids: The accolades' ids.
            box_ids: The Boxes' ids.

        Returns:
            The volume, a ``SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]``.

        Raises:
            RuntimeError: If no file has been loaded.
            ValueError: If no page of the file has a system on its time
                path, a system holding measures has a region without x
                extent, or an ids argument does not fit the
                call (a sequence of another length, an unknown template
                field, an invalid id, one id given to two timelines).
                Nothing is built then.
        """
        document = self._loaded()
        return build_volume(
            self,
            self._described_pages(),
            uid=uid,
            name=name,
            image=document.image,
            page_ids=page_ids,
            accolade_ids=accolade_ids,
            box_ids=box_ids,
        )

    def create_timelines(
        self,
        *,
        page_ids: Ids = None,
        accolade_ids: Ids = None,
        box_ids: Ids = None,
    ) -> list[DiscreteGraphicalTimeline | SegmentLine[DiscreteGraphicalTimeline]]:
        """Create one root timeline per page, in document order.

        A page is a locked segment line of its systems' accolades in
        ``pixels`` (``int``), its length the sum of theirs, named by its
        ``no_page``; it states its image in ``meta["image"]`` (the
        ``page_url``) and no ``meta["box"]``. Each system is an accolade as
        long as its region's x extent, named by its DMOS id, with
        ``meta["box"]`` its quadrilateral's dict and ``meta["image"]`` the
        page's. Its measures are its Box events in document order. A system
        without x extent and without measures is left out, and so is a page
        without systems or all of whose systems are left out.

        Args:
            page_ids: The pages' ids (see :meth:`create_timeline`); minted
                ``dgt_N`` when None.
            accolade_ids: The accolades' ids; minted ``dgt_N`` when None.
            box_ids: The Boxes' ids; minted ``box_N`` per accolade when None.

        Returns:
            The pages' timelines, each a
            ``SegmentLine[DiscreteGraphicalTimeline]``.

        Raises:
            RuntimeError: If no file has been loaded.
            ValueError: If a system holding measures has a region without x
                extent, or an ids argument does not fit the call (see :meth:`create_timeline`).
                Nothing is built then.
        """
        return build_pages(
            self,
            self._described_pages(),
            page_ids=page_ids,
            accolade_ids=accolade_ids,
            box_ids=box_ids,
        )
