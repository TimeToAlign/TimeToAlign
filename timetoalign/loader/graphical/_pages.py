"""The one builder of page timelines from time paths.

Every loader of page images (scans with drawn areas, facsimiles, IIIF
canvases) ends in the same shape, which this module builds for all of them
so that a page is the same thing whatever file it was read from. Everything
is in ``pixels``, ``int``.

- An **accolade** is one piece of a page's time path: a
  :class:`~timetoalign.timelines.DiscreteGraphicalTimeline` as long as the
  piece. For a horizontal piece it states the area it stands for in
  ``meta["box"]`` (the area the source states, else the piece's strip,
  ``x0, y - tolerance, x1 - x0, 2 * tolerance``) and its image in
  ``meta["image"]``. Its Boxes are its events, each at ``start = x_min -
  x0``.
- A **page** is one image. A
  :class:`~timetoalign.loader.graphical.paths.PiecewisePath` gives a segment
  line of its accolades in path order; any other path gives one plain
  timeline of the path's length (with ``meta["box"]`` and Boxes only for a
  ``HorizontalLinePath``). A page states ``meta["image"]``, and no
  ``meta["box"]`` when it is built of accolades.
- A **volume** is a segment line of its pages in the order given.

No length is unknown and nothing is rounded: a number the pixel axis cannot
hold exactly is refused.

Interface
---------

A loader describes each page with three small records and calls one of two
functions:

- :class:`PageBox` ``(shape, source_id=None, locator=None)``: one area that
  becomes a Box.
- :class:`PageAccolade` ``(name=None, area=None, boxes=())``: what the
  source states about ONE piece of the page's path: its name, its area and
  the Boxes it holds.
- :class:`Page` ``(name, image, path, source_id=None, accolades=None,
  boxes=(), statements=(), source=None)``: one page. ``accolades`` (one per piece, in
  path order) is given when the source states accolades; ``boxes`` are the
  areas the source assigns to no accolade, each placed on the piece it lies
  on (:meth:`PiecewisePath.piece_of`).
- :func:`build_volume` ``(owner, pages, *, uid=None, name=None, image=None,
  statements=(), page_ids=None, accolade_ids=None, box_ids=None)``: the
  volume, a locked segment line of the pages.
- :func:`build_pages` ``(owner, pages, *, page_ids=None, accolade_ids=None,
  box_ids=None)``: one locked root per page.

``owner`` is the loader: every id the call mints (timelines ``dgt_…``,
Boxes ``box_…``) is drawn from its minter in document order (the volume,
then each page followed by its accolades, each accolade followed by its
Boxes); a timeline or Box the call names takes none. A page for which the
loader can make no path is simply not handed over. The Boxes of a page read
from a file (``Page.source``) name that file's extraction and their
``PageBox.locator``, the loader's place of the area in the file
(:attr:`Loader.extractions <timetoalign.loader.base.Loader.extractions>`).

Identity
--------

``page_ids``, ``accolade_ids`` and ``box_ids`` each name every element of
their kind in the call and replace the ids the source states. Each is

- ``None``: a page and a Box take their ``source_id`` where the source has
  one; everything else is minted from the loader's minter (timelines
  ``dgt_…``, Boxes ``box_…``);
- a sequence of ids in document order over the whole call; or
- a template string filled with :meth:`str.format` from ``page`` (the
  page's position in the call, from 1), ``accolade`` (its position on its
  page), ``box`` (its position in its accolade) and ``n`` (the running count
  of the kind being named): ``box_ids="m{n}"``,
  ``accolade_ids="p{page}_s{accolade}"``. A kind is filled from its own
  position and those above it (a page knows no ``accolade``).

Document order of Boxes is: page by page, the Boxes of the stated accolades
in their order, then the unassigned ones in the order given.

Everything is checked before anything is built or minted: a sequence of
another length, a template field that does not exist, an invalid id, one id
given to two timelines of the call, to two Boxes of one timeline tree (a
page of :func:`build_pages`, the volume of :func:`build_volume`: event IDs
are unique across a tree, so ``box_ids="m{box}"``, numbering each
accolade's Boxes from 1, is refused on a page of two accolades, and
``"m{n}"`` is not), or to a Box and a timeline the tree holds as a child, a
Box outside its accolade, an accolade area that is not its piece's x
extent, pages of one volume of two kinds.

Examples:
    One scanned page with two systems, the first holding two measures::

        from timetoalign.core import BoundingBox
        from timetoalign.loader.graphical._pages import (
            Page, PageAccolade, PageBox, build_volume,
        )
        from timetoalign.loader.graphical.paths import TimeAxisPath

        systems = [
            BoundingBox.from_corners(100, 50, 900, 250, unit="pixels"),
            BoundingBox.from_corners(80, 300, 900, 500, unit="pixels"),
        ]
        measures = [
            BoundingBox.from_corners(100, 50, 480, 250, unit="pixels"),
            BoundingBox.from_corners(480, 50, 900, 250, unit="pixels"),
        ]
        page = Page(
            name="1",
            image="https://example.org/scan/f1",
            path=TimeAxisPath.from_boxes(systems, accolades=range(2)),
            accolades=[
                PageAccolade(area=systems[0], boxes=[PageBox(m) for m in measures]),
                PageAccolade(area=systems[1]),
            ],
        )
        volume = build_volume(loader, [page], uid="scan", box_ids="m{n}")
        # volume 'scan' (length 1620) > page '1' (dgt_…, length 1620)
        #   > accolade_1 (dgt_…, length 800, Boxes m1 at 0 and m2 at 380)
        #   > accolade_2 (dgt_…, offset 800, length 820)

    A facsimile page whose areas name no accolade::

        zones = [PageBox(shape, source_id=zone_id) for zone_id, shape in read]
        page = Page(
            name="15",
            image="page15.jpg",
            path=TimeAxisPath.from_boxes([zone.shape for zone in zones]),
            source_id="surface_15",
            boxes=zones,
        )
        (root,) = build_pages(loader, [page])
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from string import Formatter
from typing import TYPE_CHECKING, Any, TypeVar

from timetoalign.core import (
    BoundingBox,
    Box,
    NumberType,
    Shape,
    TimeScalarValue,
    TimeUnit,
)
from timetoalign.core.events import _held_exactly
from timetoalign.core.ids import validate_id
from timetoalign.loader.graphical._located import located
from timetoalign.loader.graphical.paths import (
    HorizontalLinePath,
    PiecewisePath,
    TimeAxisPath,
)
from timetoalign.storage import mint_event_ids
from timetoalign.timelines import DiscreteGraphicalTimeline, SegmentLine, Timeline

if TYPE_CHECKING:
    from timetoalign.loader._sources import Source
    from timetoalign.loader.base import Loader

#: The axis of every timeline built here.
_UNIT = TimeUnit.pixels

_TimelineT = TypeVar("_TimelineT", bound=Timeline)

#: An ids argument: nothing, one id per element in document order, or a template.
Ids = Sequence[str] | str | None

#: The template fields, and those each kind is filled from.
_TEMPLATE_FIELDS = ("page", "accolade", "box", "n")
_KIND_FIELDS = {
    "page": ("page", "n"),
    "accolade": ("page", "accolade", "n"),
    "Box": ("page", "accolade", "box", "n"),
}
_KIND_PLURALS = {"page": "pages", "accolade": "accolades", "Box": "Boxes"}
_KIND_ARTICLES = {"page": "a", "accolade": "an", "Box": "a"}


# region What a loader hands over


@dataclass(frozen=True)
class PageBox:
    """One area of a page image that becomes a Box event.

    Attributes:
        shape: The area, in the pixels of the page's image.
        source_id: The id the source gives the area (an MEI zone's
            ``xml:id``); it becomes the Box's id unless ``box_ids`` is given.
        locator: Where the source file states the area, by the loader's
            rule: the Box's ``locator``.
    """

    shape: Shape
    source_id: str | None = None
    locator: str | None = None


@dataclass(frozen=True)
class PageAccolade:
    """What the source states about one piece of a page's time path.

    Attributes:
        name: The accolade's name as the source gives it (a system's id);
            ``accolade_K`` by its position on the page, from 1, when None.
        area: The area the source states for the accolade; its x extent must
            be exactly its piece's ``x0`` to ``x1``. When None, the area is
            the piece's strip.
        boxes: The areas the source places in this accolade, in order.
    """

    name: str | None = None
    area: Shape | None = None
    boxes: Sequence[PageBox] = ()


@dataclass(frozen=True)
class Page:
    """One page image and its time path.

    Attributes:
        name: The page's name as the source gives it; ``page_N`` by its
            position in the call, from 1, when None.
        image: The reference to the page's image as the source writes it
            (``meta["image"]`` of the page and of its accolades); None where
            the source names none.
        path: The page's time path.
        source_id: The id the source gives the page; it becomes the
            timeline's id unless ``page_ids`` is given.
        accolades: What the source states per piece of a ``PiecewisePath``,
            one entry per piece in path order; None where it states nothing.
        boxes: The areas the source assigns to no accolade, in document
            order; each is placed on the piece it lies on.
        statements: ``(property, value)`` pairs the page's timeline states.
        source: The file the page's areas were read from: its Boxes name
            that file's extraction. None for areas made in code.
    """

    name: str | None
    image: str | None
    path: TimeAxisPath
    source_id: str | None = None
    accolades: Sequence[PageAccolade] | None = None
    boxes: Sequence[PageBox] = ()
    statements: Sequence[tuple[str, Any]] = ()
    source: Source | None = None


# endregion


# region The plan: everything decided and checked before anything is built


@dataclass
class _AccoladePlan:
    """One timeline along one piece: an accolade, or a page of one line.

    ``order`` holds, per Box, its position among the Boxes of its page in
    document order, from 1.
    """

    name: str | None
    length: TimeScalarValue
    area: Shape | None
    boxes: list[PageBox] = field(default_factory=list)
    order: list[int] = field(default_factory=list)
    uid: str | None = None
    box_ids: list[str | None] = field(default_factory=list)


@dataclass
class _PagePlan:
    """One page: its accolades, or the one line it is."""

    page: Page
    name: str
    accolades: list[_AccoladePlan] | None
    line: _AccoladePlan | None
    uid: str | None = None

    @property
    def holders(self) -> list[_AccoladePlan]:
        """The timelines of the page that can hold Boxes, in path order."""
        if self.accolades is not None:
            return self.accolades
        assert self.line is not None
        return [self.line]


def _line_length(piece: TimeAxisPath, where: str) -> int:
    """The length of *piece* as a whole number of pixels, never rounded."""
    return _held_exactly(
        piece.length, _UNIT, NumberType.int, f"{where}: the length of its time path"
    )


def _line_plan(
    piece: TimeAxisPath, stated: PageAccolade | None, name: str | None, where: str
) -> _AccoladePlan:
    """Plan the timeline along *piece*, with the area it stands for.

    Raises:
        ValueError: If a number of the piece is one the pixel axis cannot
            hold exactly, the stated area is not the piece's x extent, or an
            area is stated for a piece that is no horizontal line.
    """
    area = None if stated is None else stated.area
    if not isinstance(piece, HorizontalLinePath):
        if area is not None:
            raise ValueError(
                f"{where} states an area, which only a horizontal line of the "
                f"time path takes; its piece is a {type(piece).__name__}."
            )
        return _AccoladePlan(name=name, length=_line_length(piece, where), area=None)
    if area is None:
        try:
            area = BoundingBox.from_xywh(
                piece.x0,
                piece.y - piece.tolerance,
                piece.x1 - piece.x0,
                2 * piece.tolerance,
                unit=_UNIT,
            )
        except (TypeError, ValueError) as error:
            raise located(where, error) from error
    else:
        _on_pixels(area, f"{where}: its area")
        if area.x_min.value != piece.x0 or area.x_max.value != piece.x1:
            raise ValueError(
                f"{where} states an area spanning x {area.x_min.value} to "
                f"{area.x_max.value}, while its piece of the time path spans x "
                f"{piece.x0} to {piece.x1}: the two must agree."
            )
    return _AccoladePlan(
        name=name, length=area.x_max.value - area.x_min.value, area=area
    )


def _on_pixels(shape: Shape, what: str) -> None:
    """Refuse a *shape* that is not in whole pixels."""
    if shape.unit is not _UNIT or shape.number_type is not NumberType.int:
        raise ValueError(
            f"{what} is in {shape.unit} as {shape.number_type.name}; a page "
            "image is measured in pixels as int."
        )


def _plan_page(page: Page, position: int) -> _PagePlan:
    """Plan one page: its timelines, and each Box on the timeline holding it."""
    name = f"page_{position}" if page.name is None else page.name
    where = f"Page {name!r}"
    path = page.path
    if isinstance(path, PiecewisePath):
        stated: Sequence[PageAccolade | None] = (
            [None] * len(path.pieces) if page.accolades is None else page.accolades
        )
        if len(stated) != len(path.pieces):
            raise ValueError(
                f"{where} states {len(stated)} accolades; its time path has "
                f"{len(path.pieces)} pieces."
            )
        accolades = []
        for k, (piece, accolade) in enumerate(zip(path.pieces, stated), start=1):
            accolade_name = None if accolade is None else accolade.name
            accolades.append(
                _line_plan(
                    piece,
                    accolade,
                    f"accolade_{k}" if accolade_name is None else accolade_name,
                    f"{where}, accolade {k}",
                )
            )
        plan = _PagePlan(page=page, name=name, accolades=accolades, line=None)
        pieces = path
    else:
        if page.accolades:
            raise ValueError(
                f"{where} states accolades, which are the pieces of a "
                f"PiecewisePath; its time path is a {type(path).__name__}."
            )
        stated = [None]
        plan = _PagePlan(
            page=page,
            name=name,
            accolades=None,
            line=_line_plan(path, None, None, where),
        )
        pieces = PiecewisePath((path,))

    holders = plan.holders
    on_page = 0
    for k, (holder, accolade) in enumerate(zip(holders, stated), start=1):
        for j, box in enumerate(() if accolade is None else accolade.boxes, start=1):
            at = f"{where}, accolade {k}, Box {j}"
            _on_pixels(box.shape, at)
            area = holder.area
            if area is None:
                raise ValueError(
                    f"{at} stands on a piece of the time path that is no "
                    "horizontal line: Boxes are placed along horizontal lines only."
                )
            x_min, x_max = box.shape.x_min.value, box.shape.x_max.value
            if x_min < area.x_min.value or x_max > area.x_max.value:
                raise ValueError(
                    f"{at} spans x {x_min} to {x_max}, which leaves its accolade, "
                    f"spanning x {area.x_min.value} to {area.x_max.value}."
                )
            on_page += 1
            holder.boxes.append(box)
            holder.order.append(on_page)
    for j, box in enumerate(page.boxes, start=1):
        at = f"{where}, Box {j}"
        _on_pixels(box.shape, at)
        if not any(holder.area is not None for holder in holders):
            raise ValueError(
                f"{at}: Boxes are placed along horizontal lines only; the page's "
                f"time path is a {type(path).__name__}."
            )
        try:
            index = pieces.piece_of(box.shape)
        except ValueError as error:
            raise ValueError(f"{at}: {error}") from error
        on_page += 1
        holders[index].boxes.append(box)
        holders[index].order.append(on_page)
    return plan


def _given_ids(
    parameter: str, ids: Ids, kind: str, positions: list[dict[str, int]]
) -> list[str] | None:
    """The ids *ids* gives the elements at *positions*, or None when it gives none.

    Args:
        parameter: The argument's name, for messages (``"box_ids"``).
        ids: The argument.
        kind: ``"page"``, ``"accolade"`` or ``"Box"``.
        positions: Per element in document order, its template fields.

    Raises:
        ValueError: If a sequence holds another number of ids than there are
            elements, a template names a field its kind is not filled from,
            or an id is invalid.
    """
    if ids is None:
        return None
    if isinstance(ids, str):
        allowed = _KIND_FIELDS[kind]
        for _, field_name, _, _ in Formatter().parse(ids):
            if field_name is not None and field_name not in allowed:
                raise ValueError(
                    f"{parameter} holds the template field {{{field_name}}}; a "
                    f"template is filled from {', '.join(_TEMPLATE_FIELDS)}, of "
                    f"which {_KIND_ARTICLES[kind]} {kind} has {', '.join(allowed)}."
                )
        given = [ids.format(**position) for position in positions]
    else:
        given = list(ids)
        if len(given) != len(positions):
            raise ValueError(
                f"{parameter} names {len(given)} {_KIND_PLURALS[kind]}; this call "
                f"creates {len(positions)}."
            )
    for element_id in given:
        validate_id(element_id)
    return given


def _plan(
    pages: Iterable[Page],
    *,
    uid: str | None,
    volume: bool,
    page_ids: Ids,
    accolade_ids: Ids,
    box_ids: Ids,
) -> list[_PagePlan]:
    """Plan every timeline and Box of a call and give each its id.

    Raises:
        ValueError: Every refusal of a call (see the module docstring).
    """
    plans = [_plan_page(page, position) for position, page in enumerate(pages, start=1)]

    page_positions = [{"page": p, "n": p} for p in range(1, len(plans) + 1)]
    accolade_positions = []
    box_positions = []
    boxes_before = 0
    for p, plan in enumerate(plans, start=1):
        for a, _ in enumerate(plan.accolades or (), start=1):
            accolade_positions.append(
                {"page": p, "accolade": a, "n": len(accolade_positions) + 1}
            )
        for a, holder in enumerate(plan.holders, start=1):
            for b, on_page in enumerate(holder.order, start=1):
                box_positions.append(
                    {"page": p, "accolade": a, "box": b, "n": boxes_before + on_page}
                )
        boxes_before += sum(len(holder.boxes) for holder in plan.holders)

    given_pages = _given_ids("page_ids", page_ids, "page", page_positions)
    given_accolades = _given_ids(
        "accolade_ids", accolade_ids, "accolade", accolade_positions
    )
    given_boxes = _given_ids("box_ids", box_ids, "Box", box_positions)
    if given_boxes is not None and not isinstance(box_ids, str):
        # A sequence is in document order; the Boxes are held accolade by accolade.
        given_boxes = [given_boxes[position["n"] - 1] for position in box_positions]

    if uid is not None:
        validate_id(uid)
    accolade_iter = iter(given_accolades or ())
    box_iter = iter(given_boxes or ())
    for p, plan in enumerate(plans):
        plan.uid = plan.page.source_id if given_pages is None else given_pages[p]
        for accolade in plan.accolades or ():
            accolade.uid = next(accolade_iter) if given_accolades is not None else None
        for holder in plan.holders:
            holder.box_ids = [
                next(box_iter) if given_boxes is not None else box.source_id
                for box in holder.boxes
            ]
        if plan.line is not None:
            plan.line.uid = plan.uid

    holders_of: dict[str, str] = {} if uid is None else {uid: "the volume"}
    # Event IDs are unique across a timeline tree: a page's for one root per
    # page, the volume's for a volume. A Box is an event of its accolade (or
    # of its page along one line), and so is the Segment row naming each
    # child timeline of the tree.
    scope = "volume" if volume else "page"
    boxes_of: dict[str, tuple[object, str]] = {}
    children: dict[str, str] = {}
    for plan in plans:
        if scope == "page":
            boxes_of, children = {}, {}
        named = [(plan.uid, f"page {plan.name!r}")] + [
            (accolade.uid, f"accolade {accolade.name!r} of page {plan.name!r}")
            for accolade in plan.accolades or ()
        ]
        for timeline_id, holder_name in named:
            if timeline_id is None:
                continue
            if timeline_id in holders_of:
                raise ValueError(
                    f"The id {timeline_id!r} is given to {holders_of[timeline_id]} "
                    f"and to {holder_name}: every timeline of a call has its own id."
                )
            holders_of[timeline_id] = holder_name
        # The holders of Boxes are the accolades, or the page itself along
        # one line.
        box_holders = named if plan.accolades is None else named[1:]
        child_timelines = named[1:] if scope == "page" else named
        children.update(
            (timeline_id, holder_name)
            for timeline_id, holder_name in child_timelines
            if timeline_id is not None
        )
        for holder, (_, holder_name) in zip(plan.holders, box_holders):
            for box_id in holder.box_ids:
                if box_id is None:
                    continue
                first = boxes_of.get(box_id)
                if first is not None and first[0] is holder:
                    raise ValueError(
                        f"The id {box_id!r} is given to two Boxes of {holder_name}: "
                        f"every Box of a {scope} has its own id."
                    )
                if first is not None:
                    raise ValueError(
                        f"The id {box_id!r} is given to a Box of {first[1]} and to a "
                        f"Box of {holder_name}: every Box of a {scope} has its own id."
                    )
                boxes_of[box_id] = (holder, holder_name)
        for box_id, (_, holder_name) in boxes_of.items():
            if box_id in children:
                raise ValueError(
                    f"The id {box_id!r} is given to a Box of {holder_name} and to "
                    f"{children[box_id]}: the Boxes and the timelines of a {scope} "
                    "have ids of their own."
                )
    return plans


# endregion


# region Building


def _timeline(
    owner: Loader[Any],
    cls: type[_TimelineT],
    uid: str | None,
    **arguments: Any,
) -> _TimelineT:
    """Make one timeline for *owner*, its id *uid* or minted from its minter
    (:func:`~timetoalign.timelines.factory.mint_timeline_id`)."""
    from timetoalign.timelines.factory import mint_timeline_id

    timeline_id = mint_timeline_id(_UNIT, owner._id_minter) if uid is None else uid
    return cls(unit=_UNIT, number_type=NumberType.int, uid=timeline_id, **arguments)


def _build_line(
    owner: Loader[Any],
    plan: _AccoladePlan,
    image: str | None,
    source: Source | None,
) -> DiscreteGraphicalTimeline:
    """Build the timeline along one piece, with its area and its Boxes.

    The Boxes of a page read from the file *source* name its extraction and
    their locators.
    """
    meta: dict[str, Any] = {}
    if plan.area is not None:
        meta["box"] = plan.area.to_dict()
    if image is not None:
        meta["image"] = image
    timeline = _timeline(
        owner,
        DiscreteGraphicalTimeline,
        plan.uid,
        length=plan.length,
        name=plan.name,
        meta=meta,
    )
    if plan.boxes:
        assert plan.area is not None
        # A Box the source names no id of mints one by the event-type rule.
        minted = iter(
            mint_event_ids(
                ["Box"] * sum(box_id is None for box_id in plan.box_ids),
                owner._id_minter,
            ).to_pylist()
        )
        boxes = [
            Box.from_shape(
                box.shape,
                origin=plan.area.x_min,
                id=next(minted) if box_id is None else box_id,
            )
            for box, box_id in zip(plan.boxes, plan.box_ids)
        ]
        if source is None:
            timeline.add_events(boxes)
        else:
            owner._add_events(
                timeline,
                [
                    {**box.to_event_row(), **owner._stamp(source, page_box.locator)}
                    for box, page_box in zip(boxes, plan.boxes, strict=True)
                ],
            )
    return timeline


def _build_page(
    owner: Loader[Any], plan: _PagePlan
) -> DiscreteGraphicalTimeline | SegmentLine[DiscreteGraphicalTimeline]:
    """Build one page: a segment line of its accolades, or its one line."""
    image = plan.page.image
    timeline: DiscreteGraphicalTimeline | SegmentLine[DiscreteGraphicalTimeline]
    if plan.accolades is None:
        assert plan.line is not None
        plan.line.name = plan.name
        timeline = _build_line(owner, plan.line, image, plan.page.source)
    else:
        timeline = _timeline(
            owner,
            SegmentLine[DiscreteGraphicalTimeline],
            plan.uid,
            length=0,
            name=plan.name,
            meta={} if image is None else {"image": image},
        )
        for accolade in plan.accolades:
            timeline.append_segment(
                _build_line(owner, accolade, image, plan.page.source)
            )
    for statement_property, value in plan.page.statements:
        timeline.add_statement(statement_property, value)
    return timeline


def build_pages(
    owner: Loader[Any],
    pages: Iterable[Page],
    *,
    page_ids: Ids = None,
    accolade_ids: Ids = None,
    box_ids: Ids = None,
) -> list[DiscreteGraphicalTimeline | SegmentLine[DiscreteGraphicalTimeline]]:
    """Build one root timeline per page, in the order given.

    Args:
        owner: The loader whose minter the minted ids are drawn from.
        pages: The pages; each has a path.
        page_ids: The pages' ids (see "Identity" in the module docstring).
        accolade_ids: The accolades' ids, likewise.
        box_ids: The Boxes' ids, likewise.

    Returns:
        The pages' timelines, each locked: a
        ``SegmentLine[DiscreteGraphicalTimeline]`` of accolades for a
        ``PiecewisePath``, else one ``DiscreteGraphicalTimeline``.

    Raises:
        ValueError: If an ids argument does not fit the call, one id is given
            to two timelines, to two Boxes of one page or to a Box and an
            accolade of its page, a path or an
            area holds a number the pixel axis cannot hold exactly, a stated
            accolade area is not its piece's x extent, or a Box lies on no
            piece of its page's path or leaves its accolade. Nothing is
            built or minted then.

    Examples:
        See the module docstring.
    """
    plans = _plan(
        pages,
        uid=None,
        volume=False,
        page_ids=page_ids,
        accolade_ids=accolade_ids,
        box_ids=box_ids,
    )
    timelines = []
    for plan in plans:
        timeline = _build_page(owner, plan)
        timeline.lock()
        timelines.append(timeline)
    return timelines


def build_volume(
    owner: Loader[Any],
    pages: Iterable[Page],
    *,
    uid: str | None = None,
    name: str | None = None,
    image: str | None = None,
    statements: Sequence[tuple[str, Any]] = (),
    page_ids: Ids = None,
    accolade_ids: Ids = None,
    box_ids: Ids = None,
) -> SegmentLine:
    """Build the volume: a locked segment line of the pages, in the order given.

    Args:
        owner: The loader whose minter the minted ids are drawn from.
        pages: The pages, at least one; all built of accolades (each path a
            ``PiecewisePath``) or none.
        uid: The volume's id; minted ``dgt_…`` when None.
        name: The volume's name (the source's title).
        image: A reference to the volume's images as a whole, where the
            source writes one (``meta["image"]``).
        statements: ``(property, value)`` pairs the volume states.
        page_ids: The pages' ids (see "Identity" in the module docstring).
        accolade_ids: The accolades' ids, likewise.
        box_ids: The Boxes' ids, likewise.

    Returns:
        ``SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]`` where the
        pages are built of accolades, else
        ``SegmentLine[DiscreteGraphicalTimeline]``; its length is the sum of
        its pages' lengths.

    Raises:
        ValueError: As :func:`build_pages`; also if there is no page, or
            some pages are built of accolades and others are not (the
            segments of a segment line are of one class).

    Examples:
        See the module docstring.
    """
    plans = _plan(
        pages,
        uid=uid,
        volume=True,
        page_ids=page_ids,
        accolade_ids=accolade_ids,
        box_ids=box_ids,
    )
    if not plans:
        raise ValueError("A volume needs at least one page.")
    of_accolades = [plan.name for plan in plans if plan.accolades is not None]
    plain = [plan.name for plan in plans if plan.accolades is None]
    if of_accolades and plain:
        raise ValueError(
            "The pages of a volume are all built of accolades or none is: page "
            f"{of_accolades[0]!r} has a PiecewisePath, page {plain[0]!r} has none."
        )
    page_class = (
        SegmentLine[DiscreteGraphicalTimeline]
        if of_accolades
        else DiscreteGraphicalTimeline
    )
    volume = _timeline(
        owner,
        SegmentLine[page_class],
        uid,
        length=0,
        name=name,
        meta={} if image is None else {"image": image},
    )
    for statement_property, value in statements:
        volume.add_statement(statement_property, value)
    for plan in plans:
        volume.append_segment(_build_page(owner, plan))
    volume.lock()
    return volume


# endregion
