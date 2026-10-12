"""Pure ASCII timeline rendering - zero external dependencies.

This module provides ASCII/Unicode visualization for TimeToAlign! objects:
- Timelines with nested children (one child per row)
- TimelineGroups with boxed layout
- AlignmentBundles with multiple groups
- ScoreFlowControllers with flow control markers
- Flows with playthrough section sequences
- Flow comparisons (side-by-side diffs)

Design principles:
- Six distinct characters for the six timeline types
- One child per row with positioned bar showing extent on parent scale
- Truncation for many children (first N, ellipsis, last N)
- Unified rendering inside/outside groups
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from timetoalign.alignment import AlignmentBundle
    from timetoalign.timelines import Timeline, TimelineGroup
    from timetoalign.timelines.flow import Flow, ScoreFlowController


# region Diagram


class Diagram:
    """Rich-display wrapper for ASCII diagram strings.

    Returned by every ``.diagram()`` method so that Jupyter automatically
    renders the diagram via ``_repr_html_`` while ``str(diagram)`` and
    ``print(diagram)`` still produce plain text.

    ``Diagram`` is intentionally **not** a ``str`` subclass — subclassing
    ``str`` would make Jupyter use the string repr (with quotes) instead
    of calling ``_repr_html_``.
    """

    __slots__ = ("_text",)

    def __init__(self, text: str) -> None:
        self._text = text

    # Plain-text output
    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return self._text

    # Jupyter rich display
    def _repr_html_(self) -> str:
        import html

        escaped = html.escape(self._text)
        return (
            '<pre style="'
            "font-family: 'JetBrains Mono', Menlo, Consolas, "
            "'DejaVu Sans Mono', 'Liberation Mono', monospace; "
            "line-height: 1.2; white-space: pre;"
            f'">{escaped}</pre>'
        )

    # Allow concatenation / equality with plain strings
    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self._text == other
        if isinstance(other, Diagram):
            return self._text == other._text
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._text)

    def __contains__(self, item: str) -> bool:
        return item in self._text

    def __len__(self) -> int:
        return len(self._text)

    def __add__(self, other: str) -> str:
        return self._text + other

    def __radd__(self, other: str) -> str:
        return other + self._text

    # Delegate common str methods so Diagram is drop-in compatible
    def split(self, *args: Any, **kwargs: Any) -> list[str]:
        return self._text.split(*args, **kwargs)

    def splitlines(self, *args: Any, **kwargs: Any) -> list[str]:
        return self._text.splitlines(*args, **kwargs)

    def startswith(self, *args: Any, **kwargs: Any) -> bool:
        return self._text.startswith(*args, **kwargs)

    def endswith(self, *args: Any, **kwargs: Any) -> bool:
        return self._text.endswith(*args, **kwargs)

    def strip(self, *args: Any, **kwargs: Any) -> str:
        return self._text.strip(*args, **kwargs)

    def count(self, *args: Any, **kwargs: Any) -> int:
        return self._text.count(*args, **kwargs)

    def find(self, *args: Any, **kwargs: Any) -> int:
        return self._text.find(*args, **kwargs)

    def replace(self, *args: Any, **kwargs: Any) -> str:
        return self._text.replace(*args, **kwargs)


# endregion


# region Character Sets

# Timeline type characters (6 types)
# Keys are (NumberType.name, Domain.name) - note lowercase enum names
# NumberType: int (discrete), float (continuous), fraction (rational)
# Domain: graphical, physical, logical
TIMELINE_CHARS: dict[tuple[str, str], str] = {
    ("float", "graphical"): "=",
    (
        "int",
        "graphical",
    ): "\u2236",  # U+2236 RATIO (visually identical to colon, avoids Quarto fenced-div parsing)
    ("float", "physical"): "~",
    ("int", "physical"): "\u22c5",  # U+22C5 middle dot
    ("float", "logical"): "_",
    ("int", "logical"): ",",
    # fraction type uses same as float (continuous)
    ("fraction", "graphical"): "=",
    ("fraction", "physical"): "~",
    ("fraction", "logical"): "_",
}

# ASCII fallback for environments without Unicode
TIMELINE_CHARS_ASCII: dict[tuple[str, str], str] = {
    ("float", "graphical"): "=",
    (
        "int",
        "graphical",
    ): "\u2236",  # U+2236 RATIO (same as Unicode; avoids Quarto fenced-div parsing)
    ("float", "physical"): "~",
    ("int", "physical"): ".",  # fallback for middle dot
    ("float", "logical"): "_",
    ("int", "logical"): ",",
    # fraction type uses same as float (continuous)
    ("fraction", "graphical"): "=",
    ("fraction", "physical"): "~",
    ("fraction", "logical"): "_",
}

# Tree drawing characters
TREE_CHARS: dict[str, str] = {
    "branch": "\u251c",  # ├
    "last": "\u2514",  # └
    "vertical": "\u2502",  # │
    "horizontal": "\u2500",  # ─
}

TREE_CHARS_ASCII: dict[str, str] = {
    "branch": "+",
    "last": "+",
    "vertical": "|",
    "horizontal": "-",
}

# Box drawing characters
BOX_CHARS: dict[str, str] = {
    "top_left": "\u250c",  # ┌
    "top_right": "\u2510",  # ┐
    "bottom_left": "\u2514",  # └
    "bottom_right": "\u2518",  # ┘
    "horizontal": "\u2500",  # ─
    "vertical": "\u2502",  # │
}

BOX_CHARS_ASCII: dict[str, str] = {
    "top_left": "+",
    "top_right": "+",
    "bottom_left": "+",
    "bottom_right": "+",
    "horizontal": "-",
    "vertical": "|",
}

# Region display characters
REGION_CHARS: dict[str, str] = {
    "bar": "\u2550",  # ═  Region span fill (double line)
    "prefix": "\u2504",  # ┄  Row prefix (distinguishes from children's ├─)
    "left": "\u2590",  # ▐  Left boundary marker
    "right": "\u258c",  # ▌  Right boundary marker
}

REGION_CHARS_ASCII: dict[str, str] = {
    "bar": "=",
    "prefix": "~",
    "left": "[",
    "right": "]",
}

# Conversion map display characters
CMAP_CHARS: dict[str, str] = {
    "prefix": "\u21a6",  # ↦  RIGHTWARDS ARROW FROM BAR (maps-to)
    "arrow": "\u2192",  # →  RIGHTWARDS ARROW (source → target)
}

CMAP_CHARS_ASCII: dict[str, str] = {
    "prefix": ">",
    "arrow": "->",
}

# Flow control display characters (BMP-safe — avoids SMP musical symbols)
FLOW_CHARS: dict[str, str] = {
    "repeat_start": "\u2551:",  # ║:
    "repeat_end": ":\u2551",  # :║
    "segno": "\u00a7",  # §  U+00A7 SECTION SIGN (widely supported)
    "coda": "\u2295",  # ⊕  U+2295 CIRCLED PLUS (BMP, well-supported)
    "section_break": "\u2551",  # ║  U+2551 BOX DRAWINGS DOUBLE VERTICAL
    # Heavy (bold-reading) junction glyphs that mark a section break in the
    # section ruler: the right edge of the closing section, the left edge of
    # the opening section, and the opening section's volta corner.
    "break_right": "\u252b",  # ┫  U+252B BOX DRAWINGS HEAVY VERTICAL AND LEFT
    "break_left": "\u2523",  # ┣  U+2523 BOX DRAWINGS HEAVY VERTICAL AND RIGHT
    "break_volta": "\u250f",  # ┏  U+250F BOX DRAWINGS HEAVY DOWN AND RIGHT
    "arrow": "\u2192",  # →  U+2192 RIGHTWARDS ARROW
    "match": "=",
    "mismatch": "\u2260",  # ≠  U+2260 NOT EQUAL TO
    "volta_corner": "\u250c",  # ┌  U+250C BOX DRAWINGS LIGHT DOWN AND RIGHT
    "volta_top": "\u2500",  # ─  U+2500 BOX DRAWINGS LIGHT HORIZONTAL
    "volta_end": "\u2510",  # ┐  U+2510 BOX DRAWINGS LIGHT DOWN AND LEFT
}

FLOW_CHARS_ASCII: dict[str, str] = {
    "repeat_start": "|:",
    "repeat_end": ":|",
    "segno": "S",
    "coda": "@",
    "section_break": "||",
    "break_right": "#",
    "break_left": "#",
    "break_volta": "#",
    "arrow": "->",
    "match": "=",
    "mismatch": "!=",
    "volta_corner": "+",
    "volta_top": "-",
    "volta_end": "+",
}

# endregion

# region Configuration Defaults

DEFAULT_WIDTH: int = 70
DEFAULT_MAX_CHILDREN: int = 6
DEFAULT_NAME_WIDTH: int = 12

#: Width of one tree level: a branch glyph, a horizontal, and a space
#: (``"├─ "``).  Each nesting level spends this much of the name column, which
#: is what keeps coordinates and bars in the root's columns at every depth.
_TREE_GUTTER: int = 3

#: Narrowest bar a diagram will draw.  The name column may grow to fit a deep
#: tree only until the bar would fall below this.
_MIN_BAR_WIDTH: int = 20

# endregion

# region Helper Functions


def _format_coordinate(value: float) -> str:
    """Format a coordinate value for display.

    Args:
        value: The coordinate value.

    Returns:
        Formatted string (integer if whole number, else 1 decimal).
    """
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}"


def _elide_name(name: str, max_width: int) -> str:
    """Elide a name if it exceeds max width.

    Args:
        name: The name to potentially elide.
        max_width: Maximum allowed width.

    Returns:
        Original name or elided version with '...' suffix.
    """
    if len(name) <= max_width:
        return name
    if max_width <= 3:
        return name[:max_width]
    return name[: max_width - 3] + "..."


def _get_timeline_char(timeline: "Timeline", use_unicode: bool = True) -> str:
    """Get the display character for a timeline type.

    Args:
        timeline: The timeline to get character for.
        use_unicode: Whether to use Unicode characters.

    Returns:
        Single character representing the timeline type.
    """
    chars = TIMELINE_CHARS if use_unicode else TIMELINE_CHARS_ASCII
    key = (timeline.number_type.name, timeline.domain.name)
    return chars.get(key, "?")


def _get_children_to_display(
    children: list[tuple[float, Any]],
    max_children: int,
) -> tuple[list[tuple[float, Any]], int, list[tuple[float, Any]]]:
    """Determine which children to display with truncation.

    Args:
        children: List of (offset, child) tuples, sorted by offset.
        max_children: Maximum to display.

    Returns:
        Tuple of (first_children, omitted_count, last_children).
        If no truncation needed, omitted_count is 0 and last_children is empty.
    """
    if len(children) <= max_children:
        return children, 0, []

    # Show first half, last half (adjusted for odd numbers)
    first_count = (max_children + 1) // 2
    last_count = max_children - first_count

    first = children[:first_count]
    last = children[-last_count:] if last_count > 0 else []
    omitted = len(children) - first_count - last_count

    return first, omitted, last


def _build_child_row(
    child_offset: float,
    child_length: float,
    child_name: str,
    child_char: str,
    parent_length: float,
    bar_width: int,
    name_width: int,
    coord_width: int,
    is_last: bool,
    tree_chars: dict[str, str],
    n_events: int = 0,
    ancestor_is_last: tuple[bool, ...] = (),
) -> str:
    """Build a single child row with positioned bar.

    Args:
        child_offset: Where child starts on the root scale.
        child_length: Length of the child.
        child_name: Display name for the child.
        child_char: Timeline character for child's type.
        parent_length: Total length of the root timeline.
        bar_width: Width of the bar area in characters.
        name_width: Max width for child name.
        coord_width: Width for coordinate fields.
        is_last: True if this is the last child (use └ instead of ├).
        tree_chars: Tree drawing character set.
        n_events: Number of events in the child timeline.
        ancestor_is_last: Whether each ancestor is last among its siblings.

    Returns:
        Formatted row string.
    """
    # The tree gutter and the name share one fixed-width block, so a row at
    # any depth keeps its coordinates and bar in the root's columns: every
    # ancestor level spends three of the name's characters on its glyph.
    prefix = tree_chars["last"] if is_last else tree_chars["branch"]
    ancestor_prefix = "".join(
        "   " if ancestor_last else f"{tree_chars['vertical']}  "
        for ancestor_last in ancestor_is_last
    )
    label_width = _TREE_GUTTER + name_width
    display_name = _elide_name(
        child_name, max(1, label_width - len(ancestor_prefix) - _TREE_GUTTER)
    )
    label = f"{ancestor_prefix}{prefix}{tree_chars['horizontal']} {display_name}"

    # Calculate bar position and width
    if parent_length > 0:
        start_pos = int((child_offset / parent_length) * bar_width)
        end_pos = int(((child_offset + child_length) / parent_length) * bar_width)
    else:
        start_pos = 0
        end_pos = bar_width

    bar_len = max(1, end_pos - start_pos)

    # Build the bar with proper positioning
    bar_area = [" "] * bar_width
    for i in range(bar_len):
        if start_pos + i < bar_width:
            bar_area[start_pos + i] = child_char

    # Format coordinates
    entry_coord = _format_coordinate(child_offset)
    exit_coord = _format_coordinate(child_offset + child_length)

    # Append event count after exit coordinate
    events_suffix = f" ({n_events} events)" if n_events > 0 else ""

    return (
        f"  {label:<{label_width}} "
        f"{entry_coord:>{coord_width}} "
        f"{''.join(bar_area)} "
        f"{exit_coord}{events_suffix}"
    )


def _append_child_rows(
    lines: list[str],
    timeline: "Timeline",
    *,
    root_length: float,
    root_offset: float,
    remaining_depth: int | None,
    max_children: int,
    bar_width: int,
    name_width: int,
    coord_width: int,
    tree_chars: dict[str, str],
    use_unicode: bool,
    line_prefix: str,
    ancestor_is_last: tuple[bool, ...] = (),
) -> None:
    """Append child rows recursively in root timeline coordinates.

    Args:
        lines: Diagram lines receiving rendered child rows.
        timeline: Timeline whose children are rendered at this level.
        root_length: Length of the root timeline.
        root_offset: Absolute offset of ``timeline`` on the root.
        remaining_depth: Levels still allowed, or ``None`` for no limit.
        max_children: Maximum children shown at each level.
        bar_width: Width of the root bar area in characters.
        name_width: Maximum width for each child name.
        coord_width: Width for coordinate fields.
        tree_chars: Tree drawing character set.
        use_unicode: Whether to use Unicode timeline characters.
        line_prefix: Diagram indentation placed before every child row.
        ancestor_is_last: Whether each ancestor is last among its siblings.
    """
    if remaining_depth == 0 or timeline.n_children == 0:
        return

    child_info: list[tuple[float, Any]] = []
    for child_id, child in timeline._children.items():
        offset = timeline._child_offsets[child_id]
        child_info.append((float(offset.value), child))
    child_info.sort(key=lambda item: item[0])

    first, omitted, last = _get_children_to_display(child_info, max_children)
    next_depth = None if remaining_depth is None else remaining_depth - 1

    def append_child(
        local_offset: float,
        child: "Timeline",
        *,
        is_last: bool,
    ) -> None:
        absolute_offset = root_offset + local_offset
        row = _build_child_row(
            child_offset=absolute_offset,
            child_length=float(child.length.value),
            child_name=child.name,
            child_char=_get_timeline_char(child, use_unicode),
            parent_length=root_length,
            bar_width=bar_width,
            name_width=name_width,
            coord_width=coord_width,
            is_last=is_last,
            tree_chars=tree_chars,
            n_events=child.n_events,
            ancestor_is_last=ancestor_is_last,
        )
        lines.append(line_prefix + row)

        if next_depth != 0:
            _append_child_rows(
                lines,
                child,
                root_length=root_length,
                root_offset=absolute_offset,
                remaining_depth=next_depth,
                max_children=max_children,
                bar_width=bar_width,
                name_width=name_width,
                coord_width=coord_width,
                tree_chars=tree_chars,
                use_unicode=use_unicode,
                line_prefix=line_prefix,
                ancestor_is_last=(*ancestor_is_last, is_last),
            )

    for index, (offset, child) in enumerate(first):
        is_last = omitted == 0 and index == len(first) - 1
        append_child(offset, child, is_last=is_last)

    if omitted > 0:
        ancestor_prefix = "".join(
            "   " if ancestor_last else f"{tree_chars['vertical']}  "
            for ancestor_last in ancestor_is_last
        )
        ellipsis = f"  {ancestor_prefix}{tree_chars['vertical']}  ... ({omitted} more children)"
        lines.append(line_prefix + ellipsis)

    for index, (offset, child) in enumerate(last):
        append_child(offset, child, is_last=index == len(last) - 1)


def _measure_name_width(
    timeline: "Timeline",
    *,
    remaining_depth: int | None,
    max_children: int,
    minimum: int,
) -> int:
    """Return the name-column width the rendered child tree needs.

    Walks exactly the children :func:`_append_child_rows` would render — same
    depth limit, same per-level truncation — and asks each one how much of the
    name column its own name plus its ancestors' glyphs claim. A child at
    level *d* spends ``d * _TREE_GUTTER`` characters on those glyphs, so the
    column must be that much wider than the name alone for the coordinates and
    the bar to stay in the root's columns.

    Every level gets the same name budget, *minimum*: a name longer than that
    is elided as it always was, and the column grows only to absorb the tree
    glyphs of the deepest rendered branch. A one-level diagram therefore keeps
    the default width exactly.

    Args:
        timeline: Root timeline of the rendering.
        remaining_depth: Levels still allowed, or ``None`` for no limit.
        max_children: Maximum children shown at each level.
        minimum: Name budget each level gets, and the floor for the result.

    Returns:
        The required name-column width, never below *minimum*.
    """
    required = minimum

    def visit(parent: "Timeline", level: int, remaining: int | None) -> None:
        nonlocal required
        if remaining == 0 or parent.n_children == 0:
            return
        children = sorted(
            (
                (float(parent._child_offsets[child_id].value), child)
                for child_id, child in parent._children.items()
            ),
            key=lambda item: item[0],
        )
        first, _, last = _get_children_to_display(children, max_children)
        next_remaining = None if remaining is None else remaining - 1
        for _, child in (*first, *last):
            required = max(
                required, level * _TREE_GUTTER + min(len(child.name), minimum)
            )
            if next_remaining != 0:
                visit(child, level + 1, next_remaining)

    visit(timeline, 0, remaining_depth)
    return required


def _build_region_row(
    region_start: float,
    region_end: float,
    region_name: str,
    parent_length: float,
    bar_width: int,
    name_width: int,
    coord_width: int,
    region_chars: dict[str, str],
) -> str:
    """Build a single region row with positioned span.

    Similar to _build_child_row() but uses region-specific characters
    and does not need is_last (no tree structure for regions).

    Args:
        region_start: Start coordinate of the region.
        region_end: End coordinate of the region.
        region_name: Display name for the region.
        parent_length: Total length of parent timeline.
        bar_width: Width of the bar area in characters.
        name_width: Max width for region name.
        coord_width: Width for coordinate fields.
        region_chars: Region character set (REGION_CHARS or REGION_CHARS_ASCII).

    Returns:
        Formatted row string.
    """
    display_name = _elide_name(region_name, name_width)

    # Calculate bar position and width
    if parent_length > 0:
        start_pos = int((region_start / parent_length) * bar_width)
        end_pos = int((region_end / parent_length) * bar_width)
    else:
        start_pos = 0
        end_pos = bar_width

    bar_len = max(1, end_pos - start_pos)

    # Build the bar with boundary markers and fill
    bar_area = [" "] * bar_width
    for i in range(bar_len):
        pos = start_pos + i
        if pos < bar_width:
            if i == 0:
                bar_area[pos] = region_chars["left"]
            elif i == bar_len - 1:
                bar_area[pos] = region_chars["right"]
            else:
                bar_area[pos] = region_chars["bar"]

    # Format coordinates
    entry_coord = _format_coordinate(region_start)
    exit_coord = _format_coordinate(region_end)

    return (
        f"  {region_chars['prefix']} "
        f"{display_name:<{name_width}} "
        f"{entry_coord:>{coord_width}} "
        f"{''.join(bar_area)} "
        f"{exit_coord}"
    )


def _get_cmap_bar_char(
    cmap: Any,
    use_unicode: bool = True,
) -> str:
    """Determine the bar character for a conversion map's target unit.

    Uses the target unit's domain and discreteness to select the timeline
    character that would represent a derivative timeline in that unit.
    Falls back to ``'·'`` (unicode) or ``'.'`` (ASCII) when the target
    unit is unknown.

    Args:
        cmap: A ConversionMap instance.
        use_unicode: Whether to use Unicode characters.

    Returns:
        Single character for the bar fill.
    """
    chars = TIMELINE_CHARS if use_unicode else TIMELINE_CHARS_ASCII
    target_unit = cmap.target_unit
    if target_unit is not None:
        number_key = "int" if target_unit.is_discrete else "float"
        domain_key = target_unit.domain.name
        return chars.get((number_key, domain_key), "\u22c5" if use_unicode else ".")
    return "\u22c5" if use_unicode else "."


def _build_cmap_row(
    cmap: Any,
    bar_width: int,
    name_width: int,
    coord_width: int,
    use_unicode: bool,
    cmap_chars: dict[str, str],
) -> str:
    """Build a single conversion map row with a full-span bar.

    The bar uses the timeline character corresponding to the target unit's
    domain and number type, visually suggesting "this timeline could be
    converted to this type."

    Args:
        cmap: A ConversionMap instance.
        bar_width: Width of the bar area in characters.
        name_width: Max width for the map name.
        coord_width: Width for coordinate fields.
        use_unicode: Whether to use Unicode characters.
        cmap_chars: C-Map character set (CMAP_CHARS or CMAP_CHARS_ASCII).

    Returns:
        Formatted row string.
    """
    bar_char = _get_cmap_bar_char(cmap, use_unicode)
    bar = bar_char * bar_width
    display_name = _elide_name(cmap.name, name_width)

    # Build description: "ClassName(source → target)" or "ClassName(id)"
    arrow = cmap_chars["arrow"]
    if cmap.source_unit and cmap.target_unit:
        description = (
            f"{cmap.__class__.__name__}"
            f"({cmap.source_unit} {arrow} {cmap.target_unit})"
        )
    else:
        description = f"{cmap.__class__.__name__}({cmap.id})"

    return (
        f"  {cmap_chars['prefix']} "
        f"{display_name:<{name_width}} "
        f"{'':>{coord_width}} "
        f"{bar} "
        f"{description}"
    )


# endregion

# region Timeline Diagram


def timeline_diagram(
    timeline: "Timeline",
    width: int = DEFAULT_WIDTH,
    show_children: bool = True,
    max_children: int = DEFAULT_MAX_CHILDREN,
    indent: int = 0,
    unicode: bool = True,
    parent_id: str | None = None,
    show: set[str] | None = None,
    depth: bool | int = True,
) -> Diagram:
    """Generate ASCII diagram for a timeline.

    Args:
        timeline: The Timeline to render.
        width: Total width of the diagram in characters.
        show_children: Whether to show child timelines (one per row).
        max_children: Maximum children to show before truncating.
        indent: Left indentation (for nested rendering).
        unicode: Use Unicode characters (True) or ASCII fallback (False).
        parent_id: If set, indicates this is a child of parent_id (for annotation).
        show: Optional set controlling which elements appear. Supported values:
            ``"children"`` (child timelines), ``"regions"`` (named regions),
            and ``"cmaps"`` (attached conversion maps).
            When ``None``, behaviour is exactly as before (children shown if
            ``show_children=True``, no regions or cmaps).  The ``show_children``
            parameter takes precedence for backwards compatibility.
        depth: Child levels to render. ``True`` renders all levels,
            ``False`` renders direct children only, and a non-negative integer
            renders at most that many levels below the root.

    Returns:
        Multi-line string with ASCII diagram.

    Raises:
        ValueError: If ``depth`` is a negative integer.

    Examples:
        >>> print(timeline_diagram(my_timeline))
        DiscreteGraphicalTimeline[dgt_1:1] (11 events, 5 children)
        0 ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶ 4835 pixels
          ├─ system_1     0   ∶∶∶∶∶∶∶                        967
          ├─ system_2   967          ∶∶∶∶∶∶∶∶               1934
          └─ ...
    """
    # chars = TIMELINE_CHARS if unicode else TIMELINE_CHARS_ASCII
    tree = TREE_CHARS if unicode else TREE_CHARS_ASCII

    if isinstance(depth, bool):
        remaining_depth = None if depth else 1
    else:
        if depth < 0:
            raise ValueError("depth must be non-negative")
        remaining_depth = depth

    # Resolve show set: None means legacy behaviour
    _show_children = show_children  # backwards compat takes precedence
    _show_regions = False
    _show_cmaps = False
    if show is not None:
        if "children" not in show:
            _show_children = False
        if "regions" in show:
            _show_regions = True
        if "cmaps" in show:
            _show_cmaps = True
    # show_children=False overrides show={"children"}
    if not show_children:
        _show_children = False

    lines: list[str] = []
    prefix = " " * indent

    # 1. Header line
    header = f"{timeline.class_name}[{timeline.id}]"
    if parent_id:
        header += f" (child of {parent_id})"

    details: list[str] = []
    # Compute total events including children
    own_events = timeline.n_events
    child_events = sum(c.n_events for c in timeline._children.values())
    total_events = own_events + child_events
    if total_events > 0:
        if child_events > 0 and own_events > 0:
            details.append(f"{total_events} events ({own_events} own)")
        else:
            details.append(f"{total_events} events")
    if timeline.n_children > 0:
        details.append(f"{timeline.n_children} children")
    if timeline.n_regions > 0:
        details.append(f"{timeline.n_regions} regions")
    if timeline.n_conversion_maps > 0:
        details.append(f"{timeline.n_conversion_maps} cmaps")
    if details:
        header += f" ({', '.join(details)})"
    lines.append(prefix + header)

    # 2. Calculate dimensions
    length_value = float(timeline.length.value)
    end_label = _format_coordinate(length_value)
    unit_label = str(timeline.unit)

    # Coordinate width based on largest coordinate
    coord_width = max(len(end_label), 5)

    # Calculate left margin to align with child bars
    # Child structure: "  {tree glyphs + name:<gutter + name_width} "
    #                  "{coord:>coord_width} {bar}"
    # For parent, we use empty tree prefix space and show unit label after bar
    # Left margin: 2 (indent) + the label block + 1 + coord_width + 1
    # Nested children spend part of the name column on their ancestors' tree
    # glyphs, so widen it to what the rendered tree needs — up to the point
    # where the bar would fall below its minimum.
    right_label_len = len(end_label) + 1 + len(unit_label)
    fixed_width = 2 + _TREE_GUTTER + 1 + coord_width + 1 + right_label_len + 1
    name_width = DEFAULT_NAME_WIDTH
    if _show_children and remaining_depth != 0:
        name_width = min(
            _measure_name_width(
                timeline,
                remaining_depth=remaining_depth,
                max_children=max_children,
                minimum=DEFAULT_NAME_WIDTH,
            ),
            max(DEFAULT_NAME_WIDTH, width - indent - fixed_width - _MIN_BAR_WIDTH),
        )
    left_margin = 2 + _TREE_GUTTER + name_width + 1 + coord_width + 1

    # Bar width: total - indent - left_margin - " end unit" - margins
    bar_width = width - indent - left_margin - right_label_len - 1
    bar_width = max(bar_width, _MIN_BAR_WIDTH)

    # 3. Get timeline character and build parent bar
    line_char = _get_timeline_char(timeline, unicode)
    bar = line_char * bar_width

    # 4. Format parent bar line (aligned with child bars)
    # Structure matches child: padding + "0" coord + bar + end coord + unit
    padding = " " * (left_margin - coord_width - 1)  # Space for tree/name area
    bar_line = f"{prefix}{padding}{'0':>{coord_width}} {bar} {end_label} {unit_label}"
    lines.append(bar_line)

    # 5. Render children recursively
    if _show_children and timeline.n_children > 0 and remaining_depth != 0:
        _append_child_rows(
            lines,
            timeline,
            root_length=length_value,
            root_offset=0.0,
            remaining_depth=remaining_depth,
            max_children=max_children,
            bar_width=bar_width,
            name_width=name_width,
            coord_width=coord_width,
            tree_chars=tree,
            use_unicode=unicode,
            line_prefix=prefix,
        )

    # 6. Render regions (below children)
    if _show_regions and timeline.n_regions > 0:
        rgn_chars = REGION_CHARS if unicode else REGION_CHARS_ASCII
        # Collect and sort regions by start coordinate
        region_info: list[tuple[float, float, str]] = []
        for region in timeline.iter_regions():
            region_info.append(
                (
                    float(region.start.value),
                    float(region.end.value),
                    region.id if region.name is None else region.name,
                )
            )
        region_info.sort(key=lambda x: x[0])

        parent_length = length_value
        for r_start, r_end, r_name in region_info:
            row = _build_region_row(
                region_start=r_start,
                region_end=r_end,
                region_name=r_name,
                parent_length=parent_length,
                bar_width=bar_width,
                name_width=name_width,
                coord_width=coord_width,
                region_chars=rgn_chars,
            )
            lines.append(prefix + row)

    # 7. Render conversion maps (below regions)
    if _show_cmaps and timeline.n_conversion_maps > 0:
        cm_chars = CMAP_CHARS if unicode else CMAP_CHARS_ASCII
        for cmap in timeline._conversion_maps.values():
            row = _build_cmap_row(
                cmap=cmap,
                bar_width=bar_width,
                name_width=name_width,
                coord_width=coord_width,
                use_unicode=unicode,
                cmap_chars=cm_chars,
            )
            lines.append(prefix + row)

    return Diagram("\n".join(lines))


# endregion

# region Group Diagram


def group_diagram(
    group: "TimelineGroup",
    width: int = DEFAULT_WIDTH,
    show_children: bool = True,
    max_children: int = DEFAULT_MAX_CHILDREN,
    unicode: bool = True,
    depth: bool | int = True,
) -> Diagram:
    """Generate ASCII diagram for a TimelineGroup.

    Args:
        group: The TimelineGroup to render.
        width: Total width of the diagram in characters.
        show_children: Whether to expand child timelines.
        max_children: Maximum children per timeline.
        unicode: Use Unicode characters (True) or ASCII fallback (False).
        depth: Child levels to render for each member timeline.

    Returns:
        Multi-line string with ASCII diagram.

    Examples:
        >>> print(group_diagram(my_group))
        TimelineGroup[my_group] (2 timelines)
        ┌────────────────────────────────────────────────────────────┐
        │ DiscreteGraphicalTimeline[dgt_1:1] (11 events, 5 children)  │
        │ 0 ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶ 4835 pixels    │
        │   ├─ system_1     0   ∶∶∶∶∶∶∶                    967       │
        │   └─ ...                                                   │
        │                                                            │
        │ ContinuousPhysicalTimeline[audio:1] (0 events)             │
        │ 0 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 150.0 seconds  │
        └────────────────────────────────────────────────────────────┘
    """
    box = BOX_CHARS if unicode else BOX_CHARS_ASCII

    lines: list[str] = []

    # Header
    lines.append(f"TimelineGroup[{group.id}] ({group.n_timelines} timelines)")

    # Collect content lines for all timelines
    content_lines: list[str] = []
    timeline_list = list(group)

    for idx, tl in enumerate(timeline_list):
        # Render each timeline normally (with reduced width for box borders)
        tl_diagram = timeline_diagram(
            tl,
            width=width - 4,  # Account for "│ " prefix and " │" suffix
            show_children=show_children,
            max_children=max_children,
            unicode=unicode,
            depth=depth,
        )
        for line in tl_diagram.split("\n"):
            content_lines.append(line)

        # Add blank line between timelines (except after last)
        if idx < len(timeline_list) - 1:
            content_lines.append("")

    # Calculate box width based on content
    if content_lines:
        max_content_width = max(len(line) for line in content_lines)
    else:
        max_content_width = width - 4

    inner_width = max(max_content_width, 20)
    box_width = inner_width + 4  # "│ " + content + " │"

    # Build box
    lines.append(
        box["top_left"] + box["horizontal"] * (box_width - 2) + box["top_right"]
    )

    for line in content_lines:
        padded = line.ljust(inner_width)
        lines.append(f"{box['vertical']} {padded} {box['vertical']}")

    lines.append(
        box["bottom_left"] + box["horizontal"] * (box_width - 2) + box["bottom_right"]
    )

    return Diagram("\n".join(lines))


# endregion

# region Bundle Diagram


DEFAULT_MAX_STANDALONE: int = 6


def bundle_diagram(
    bundle: "AlignmentBundle",
    width: int = 80,
    show_children: bool = True,
    max_children: int = DEFAULT_MAX_CHILDREN,
    max_standalone: int = DEFAULT_MAX_STANDALONE,
    unicode: bool = True,
    depth: bool | int = True,
) -> Diagram:
    """Generate ASCII diagram for an AlignmentBundle.

    Renders grouped timelines inside their group boxes, then any
    standalone timelines (not in a group) as a proportionally-scaled
    list.  When timelines share the same unit, the longest fills the
    full bar width and shorter ones are drawn proportionally.

    Args:
        bundle: The AlignmentBundle to render.
        width: Total width of the diagram in characters.
        show_children: Whether to expand child timelines.
        max_children: Maximum children per timeline.
        max_standalone: Maximum standalone timelines to display before
            truncating with an ellipsis.
        unicode: Use Unicode characters (True) or ASCII fallback (False).
        depth: Child levels to render in nested timeline diagrams.

    Returns:
        Diagram object (displays as ASCII in terminal, rich HTML in Jupyter).

    Examples:
        >>> print(bundle_diagram(my_bundle))
        AlignmentBundle[thoresen_alignment]

          TimelineGroup[dgt1_group] (2 timelines)
          ┌──────────────────────────────────────────────────────┐
          │ DiscreteGraphicalTimeline[dgt_1:1] (11 events)        │
          │ 0 ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶ 4835 pixels    │
          └──────────────────────────────────────────────────────┘

          MatchClaims: 5 segment-to-segment
    """
    lines: list[str] = []

    # Header
    lines.append(f"AlignmentBundle[{bundle.id}]")
    lines.append("")

    # Render each group with indentation
    for group in bundle.groups.values():
        group_str = group_diagram(
            group,
            width=width - 2,  # Account for indentation
            show_children=show_children,
            max_children=max_children,
            unicode=unicode,
            depth=depth,
        )
        for line in group_str.split("\n"):
            lines.append("  " + line)
        lines.append("")

    # Collect standalone timelines (not in any group)
    standalone_ids = [
        uid for uid in bundle.timelines if bundle.timeline_to_group.get(uid) is None
    ]

    if standalone_ids:
        standalone_tls = [bundle.timelines[uid] for uid in standalone_ids]

        lines.append(f"  Standalone timelines ({len(standalone_tls)}):")

        # Build the list to render (with truncation)
        if len(standalone_tls) <= max_standalone:
            to_render = list(enumerate(standalone_tls))
            omitted = 0
        else:
            first_count = (max_standalone + 1) // 2
            last_count = max_standalone - first_count
            first = list(enumerate(standalone_tls[:first_count]))
            last = [
                (
                    len(standalone_tls) - last_count + i,
                    standalone_tls[-(last_count - i)],
                )
                for i in range(last_count)
            ]
            omitted = len(standalone_tls) - first_count - last_count
            to_render = first  # render first, then ellipsis, then last

        # Find the maximum length across all standalone timelines
        # (for proportional bar widths)
        max_length = max(float(tl.length.value) for tl in standalone_tls)
        if max_length <= 0:
            max_length = 1.0

        # Determine layout dimensions
        name_width = DEFAULT_NAME_WIDTH
        # Longest coordinate label
        max_end_label = _format_coordinate(max_length)
        coord_width = max(len(max_end_label), 5)
        unit_label = str(standalone_tls[0].unit)
        right_label_len = coord_width + 1 + len(unit_label)
        # Full bar width for the longest timeline
        full_bar_width = (
            width - 4 - name_width - 1 - coord_width - 1 - right_label_len - 1
        )
        full_bar_width = max(full_bar_width, 10)

        def _render_standalone_row(tl: "Timeline") -> str:
            length = float(tl.length.value)
            end_label = _format_coordinate(length)
            tl_unit = str(tl.unit)
            line_char = _get_timeline_char(tl, unicode)

            # Proportional bar width
            if max_length > 0:
                proportion = length / max_length
            else:
                proportion = 1.0
            bar_w = max(int(full_bar_width * proportion), 1)
            bar = line_char * bar_w

            elided = _elide_name(tl.name or tl.id, name_width)

            n_ev = tl.n_events
            ev_str = f" ({n_ev} ev)" if n_ev > 0 else ""

            return (
                f"    {elided:<{name_width}} "
                f"{'0':>{coord_width}} "
                f"{bar} "
                f"{end_label} {tl_unit}{ev_str}"
            )

        for _idx, tl in to_render:
            lines.append(_render_standalone_row(tl))

        if omitted > 0:
            lines.append(f"    ... ({omitted} more)")
            for _idx, tl in last:
                lines.append(_render_standalone_row(tl))

        lines.append("")

    # Match claims summary. The bundle's own count spans BOTH claim stores —
    # the per-claim Python list and the columnar claim fields (the latter via
    # row counts, no claim materialised) — so a dense audio-to-audio bundle
    # whose claims live only in a MatchClaimField reports its true count.
    n_matches = getattr(bundle, "n_cross_group_claims", 0)
    if n_matches > 0:
        lines.append(f"  MatchClaims: {n_matches}")
    else:
        lines.append("  MatchClaims: 0")

    return Diagram("\n".join(lines))


# endregion

# region Flow Control Diagram


def flow_control_diagram(
    controller: "ScoreFlowController",
    width: int = DEFAULT_WIDTH,
    unicode: bool = True,
    show_graph: bool = True,
    show_legend: bool = True,
    mode: str = "auto",
) -> Diagram:
    """Generate ASCII diagram for a ScoreFlowController.

    Shows the folded score map with atomic sections and flow control markers.

    Three rendering modes are available:

    - ``"full"``: One slot per MC — detailed but wide. Best for small
      scores (< ~20 MCs). Shows repeat markers and volta brackets
      aligned to individual MC positions.
    - ``"sections"``: One slot per AtomicSection — compact overview.
      Each section shows its ID centered over its MC range. Flow control
      markers attach to sections. Works well for any size score.
    - ``"table"``: Vertical table listing each section on its own row
      with ID, MC range, MC count, flow events, and transitions.
      Most compact; no spatial layout.
    - ``"auto"`` (default): Chooses ``"full"`` if all MCs fit within
      *width*; otherwise ``"sections"``.

    Args:
        controller: The ScoreFlowController to render.
        width: Total width of the diagram in characters.
        unicode: Use Unicode characters (True) or ASCII fallback (False).
        show_graph: Whether to show section transition graph.
        show_legend: Whether to show flow control event legend.
        mode: Rendering mode — ``"auto"``, ``"full"``, ``"sections"``,
            or ``"table"``.

    Returns:
        Multi-line string with ASCII diagram.
    """
    fc = FLOW_CHARS if unicode else FLOW_CHARS_ASCII

    units = list(controller.iter_units())
    sections = controller.get_sections()  # AtomicSections

    # Count flow control events
    n_flow_events = sum(
        1 for u in units if u.flow_control_types or u.start_repeat or u.end_repeat
    )

    n_mcs = len(units)

    # Resolve "auto" mode
    if mode == "auto":
        # 4 chars per MC + 3 chars prefix — if it fits in width, use full
        if n_mcs * 4 + 3 <= width:
            mode = "full"
        else:
            mode = "sections"

    # Header (shared)
    lines: list[str] = []
    lines.append(
        f"ScoreFlowController "
        f"({n_mcs} MCs, {len(sections)} atomic sections, "
        f"{n_flow_events} flow events)"
    )

    # Dispatch to rendering mode
    if mode == "full":
        _render_full_ruler(lines, units, sections, width, fc, unicode)
    elif mode == "sections":
        _render_sections_ruler(lines, units, sections, width, fc, unicode)
    elif mode == "table":
        _render_table(lines, units, sections, fc)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected auto/full/sections/table)")

    # Legend (shared across modes)
    if show_legend:
        _render_legend(lines, units, sections, fc)

    # Section transition graph (shared across modes)
    if show_graph and sections:
        _render_graph(lines, sections, fc, controller=controller)

    return Diagram("\n".join(lines))


def _render_full_ruler(
    lines: list[str],
    units: list[Any],
    sections: list[Any],
    width: int,
    fc: dict[str, str],
    unicode: bool,
) -> None:
    """Render per-MC ruler (original detailed mode for small scores)."""
    n_mcs = len(units)
    if n_mcs == 0:
        return

    mc_numbers = [u.mc for u in units]
    col_width = max(4, (width - 6) // n_mcs)
    total_ruler_width = col_width * n_mcs

    # MC ruler row
    ruler_parts: list[str] = ["MC "]
    for mc in mc_numbers:
        ruler_parts.append(f"{mc:>{col_width}}")
    lines.append("".join(ruler_parts))

    # Build MC -> slot position lookup
    mc_to_col: dict[int, int] = {}
    for idx, mc in enumerate(mc_numbers):
        mc_to_col[mc] = 3 + idx * col_width

    # Section spans row
    section_row = [" "] * (3 + total_ruler_width)
    tree = TREE_CHARS if unicode else TREE_CHARS_ASCII
    for sec in sections:
        start_idx = (
            mc_numbers.index(sec.mc_start) if sec.mc_start in mc_numbers else None
        )
        end_mc = sec.mc_end - 1
        end_idx = mc_numbers.index(end_mc) if end_mc in mc_numbers else None

        if start_idx is not None and end_idx is not None:
            col_start = 3 + start_idx * col_width
            col_end = 3 + (end_idx + 1) * col_width - 1

            if start_idx == end_idx:
                mid = col_start + col_width // 2
                if mid < len(section_row):
                    section_row[mid] = sec.id[0]
            else:
                h = tree["horizontal"]
                if col_start < len(section_row):
                    section_row[col_start] = tree["branch"]
                if col_end < len(section_row):
                    section_row[col_end] = "\u2524" if unicode else "|"
                span = col_end - col_start - 1
                if span > 0:
                    id_str = sec.id
                    if len(id_str) > span:
                        id_str = id_str[:span]
                    pad_left = (span - len(id_str)) // 2
                    pad_right = span - len(id_str) - pad_left
                    fill = h * pad_left + id_str + h * pad_right
                    for ci, ch in enumerate(fill):
                        pos = col_start + 1 + ci
                        if pos < len(section_row):
                            section_row[pos] = ch

    lines.append("".join(section_row).rstrip())

    # Flow control markers row (repeats)
    fc_row = [" "] * (3 + total_ruler_width)
    for unit in units:
        col = mc_to_col.get(unit.mc, 0)
        if unit.start_repeat:
            marker = fc["repeat_start"]
            for ci, ch in enumerate(marker):
                if col + ci < len(fc_row):
                    fc_row[col + ci] = ch
        if unit.end_repeat:
            marker = fc["repeat_end"]
            start_pos = col + col_width - len(marker)
            for ci, ch in enumerate(marker):
                if start_pos + ci < len(fc_row) and start_pos + ci >= 0:
                    fc_row[start_pos + ci] = ch

    fc_str = "".join(fc_row).rstrip()
    if fc_str.strip():
        lines.append(fc_str)

    # Volta brackets row
    volta_row = [" "] * (3 + total_ruler_width)
    has_volta = False
    volta_units = [(u, mc_to_col.get(u.mc, 0)) for u in units if u.volta is not None]
    for vi, (unit, col) in enumerate(volta_units):
        has_volta = True
        corner = fc["volta_corner"]
        top = fc["volta_top"]
        num_str = str(unit.volta)
        bracket = corner + num_str
        if vi > 0:
            _prev_unit, prev_col = volta_units[vi - 1]
            end_char = fc["volta_end"]
            close_pos = col - 1
            if close_pos > prev_col and close_pos < len(volta_row):
                volta_row[close_pos] = end_char
        for ci, ch in enumerate(bracket):
            if col + ci < len(volta_row):
                volta_row[col + ci] = ch
        for ci in range(len(bracket), col_width - 1):
            if col + ci < len(volta_row):
                volta_row[col + ci] = top

    if has_volta:
        lines.append("".join(volta_row).rstrip())

    # Marker / jump-instruction row.
    #
    # LEFT-anchored MARKERS (segno §, coda destination ⊕) mark a point on
    # the timeline. RIGHT-anchored JUMP/BREAK labels (fine, to⊕, DSaC,
    # DSaF, ...) fire at the end of their measure and get two slots of
    # trailing padding so they sit clearly inside the section and don't
    # touch the next slot's left-anchored content. We merge both onto
    # one row when they fit, and split into two rows on collision.
    right_pad = 2 if col_width >= 4 else 1
    left_entries: list[tuple[int, str]] = []
    right_entries: list[tuple[int, str]] = []
    collision = False
    for unit in units:
        col = mc_to_col.get(unit.mc, 0)
        ltext, rtext = _glyphs_for_unit(unit, fc)
        if ltext:
            left_entries.append((col, ltext))
        if rtext:
            right_entries.append((col + col_width - right_pad - len(rtext), rtext))
        if ltext and rtext and len(ltext) + len(rtext) + right_pad > col_width:
            collision = True

    def _draw_full(entries: list[tuple[int, str]]) -> str:
        if not entries:
            return ""
        row = [" "] * (3 + total_ruler_width)
        for start, text in entries:
            for ci, ch in enumerate(text):
                pos = start + ci
                if 0 <= pos < len(row):
                    row[pos] = ch
        return "".join(row).rstrip()

    if collision:
        for row_str in (_draw_full(left_entries), _draw_full(right_entries)):
            if row_str.strip():
                lines.append(row_str)
    else:
        row_str = _draw_full(left_entries + right_entries)
        if row_str.strip():
            lines.append(row_str)


def _build_section_slot(
    sec: Any,
    units_in_sec: list[Any],
    sec_volta: int | None,
    col_width: int,
    fc: dict[str, str],
    tree: dict[str, str],
    unicode: bool,
    break_after: bool = False,
    break_before: bool = False,
) -> tuple[str, str, str, str, str]:
    """Return per-section slot strings for the four ruler rows.

    Returns ``(id_line, range_line, repeat_line, ltext, rtext)``. The
    first three are exactly ``col_width`` chars wide; the last two are
    the raw left- and right-anchored marker glyph strings, which the
    caller composes into either one merged marker row or two split rows.

    A section break is a flow control element and so must read on the
    schema itself: ``break_after`` (this section closes on a Break) draws
    the slot's right edge with the heavy junction glyph, and
    ``break_before`` (the preceding section closed on a Break) draws the
    left edge — branch or volta corner — heavy. The two adjacent heavy
    edges render the break as a bold ``┫┣`` junction, distinct from an
    ordinary ``┤├`` section boundary.

    All alignment inside the slot is done with ``str.ljust``,
    ``str.rjust`` and ``str.center`` against ``col_width``. No absolute
    slot offsets exist here, by design — that is what makes vertical
    misalignment between the four rows structurally impossible.
    """
    h = tree["horizontal"]
    branch = fc["break_left"] if break_before else tree["branch"]
    right_branch = fc["break_right"] if break_after else ("┤" if unicode else "|")
    volta_corner = fc["break_volta"] if break_before else fc["volta_corner"]
    mc_count = sec.mc_end - sec.mc_start

    def _center(s: str, w: int, fill: str = " ") -> str:
        # Centre `s` in width `w` with any extra padding on the RIGHT,
        # matching the pre-refactor pad arithmetic. (Python's str.center
        # puts extras on the LEFT, which would shift slots by one for
        # odd-length labels.)
        pad_left = (w - len(s)) // 2
        pad_right = w - len(s) - pad_left
        return fill * pad_left + s + fill * pad_right

    # --- id_line ---
    # Two slot shapes, both filling col_width with `─` (never spaces):
    #   regular: ├──ID──┤   (LEFT = branch,             RIGHT = right_branch)
    #   volta:   ┌N─ID──    (LEFT = volta_corner + N,   RIGHT = "")
    # The leading `─` after the volta number is the first character of the
    # centered middle, so both shapes share `inner_w = col_width - 2`.
    inner_w = max(0, col_width - 2)
    middle = _center(sec.id, inner_w, h)
    if sec_volta is not None:
        left = f"{volta_corner}{sec_volta}"
        right = ""
    else:
        left = branch
        right = right_branch
    id_line = (left + middle + right)[:col_width].ljust(col_width, h)

    # --- range_line ---
    if mc_count == 1:
        label = str(sec.mc_start)
    else:
        label = f"{sec.mc_start}-{sec.mc_end - 1}"
    if len(label) > col_width - 1:
        label = label[: col_width - 1]
    range_line = _center(label, col_width)

    # --- repeat_line ---
    has_start = any(u.start_repeat for u in units_in_sec)
    has_end = any(u.end_repeat for u in units_in_sec)
    rs = fc["repeat_start"]
    re_ = fc["repeat_end"]
    if has_start and has_end:
        gap = max(0, col_width - len(rs) - len(re_))
        repeat_line = (rs + " " * gap + re_)[:col_width].ljust(col_width)
    elif has_start:
        repeat_line = rs[:col_width].ljust(col_width)
    elif has_end:
        repeat_line = re_[:col_width].rjust(col_width)
    else:
        repeat_line = " " * col_width

    # --- marker glyphs (left + right) ---
    ltexts: list[str] = []
    rtexts: list[str] = []
    for u in units_in_sec:
        lt, rt = _glyphs_for_unit(u, fc)
        if lt:
            ltexts.append(lt)
        if rt:
            rtexts.append(rt)
    ltext = " ".join(ltexts)
    rtext = " ".join(rtexts)

    return id_line, range_line, repeat_line, ltext, rtext


def _compose_marker_slot(ltext: str, rtext: str, col_width: int) -> str:
    """Compose one section's marker slot from its left and right glyphs."""
    if ltext and rtext:
        gap = max(0, col_width - len(ltext) - len(rtext))
        return (ltext + " " * gap + rtext)[:col_width].ljust(col_width)
    if ltext:
        return ltext[:col_width].ljust(col_width)
    if rtext:
        return rtext[:col_width].rjust(col_width)
    return " " * col_width


def _render_sections_ruler(
    lines: list[str],
    units: list[Any],
    sections: list[Any],
    width: int,
    fc: dict[str, str],
    unicode: bool,
) -> None:
    """Render per-section ruler (compact mode for large scores).

    Each AtomicSection contributes a fixed-width slot to every row; the
    rows are produced by concatenating per-section slot strings (plus a
    leading prefix). Vertical alignment is structural — every row's
    slot positions are a direct consequence of the same per-section
    slot widths — so the marker row cannot drift out of alignment with
    the section slots above it.
    """
    if not sections:
        return

    n_secs = len(sections)
    tree = TREE_CHARS if unicode else TREE_CHARS_ASCII

    prefix_w = 4  # "    " indent
    col_width = max(6, (width - prefix_w) // n_secs)
    prefix = " " * prefix_w

    mc_to_sec: dict[int, str] = {}
    for sec in sections:
        for mc in range(sec.mc_start, sec.mc_end):
            mc_to_sec[mc] = sec.id

    units_by_sec: dict[str, list[Any]] = {sec.id: [] for sec in sections}
    for u in units:
        sec_id = mc_to_sec.get(u.mc)
        if sec_id is not None:
            units_by_sec[sec_id].append(u)

    unit_by_mc: dict[int, Any] = {u.mc: u for u in units}

    # A section opens a volta bracket when its first measure carries a volta
    # number. The bracket spans the whole section, so the number is read off
    # ``mc_start`` alone — NOT by requiring every measure in the section to
    # share one volta value (a first ending whose section also contains
    # post-ending measures would otherwise be missed).
    sec_volta: dict[str, int] = {}
    for sec in sections:
        first = unit_by_mc.get(sec.mc_start)
        if first is not None and first.volta is not None:
            sec_volta[sec.id] = first.volta

    # A section closes on a Break when its last measure carries a section
    # break. The Break sits at a section junction (it forces the boundary),
    # so the closing section's right edge and the next section's left edge
    # are both drawn heavy to render the break on the schema.
    ends_with_break: list[bool] = []
    for sec in sections:
        last = unit_by_mc.get(sec.mc_end - 1)
        ends_with_break.append(bool(last is not None and last.section_break))

    slots = [
        _build_section_slot(
            sec,
            units_by_sec[sec.id],
            sec_volta.get(sec.id),
            col_width,
            fc,
            tree,
            unicode,
            break_after=ends_with_break[i],
            break_before=(i > 0 and ends_with_break[i - 1]),
        )
        for i, sec in enumerate(sections)
    ]

    id_row = prefix + "".join(s[0] for s in slots)
    range_row = prefix + "".join(s[1] for s in slots)
    repeat_row = prefix + "".join(s[2] for s in slots)

    lines.append(id_row.rstrip())
    lines.append(range_row.rstrip())
    if repeat_row.strip():
        lines.append(repeat_row.rstrip())

    # Marker row(s). If any section has both a left and a right marker
    # that cannot fit with at least one slot of gap between them,
    # split into a left-only row above and a right-only row below.
    collision = any(
        s[3] and s[4] and len(s[3]) + len(s[4]) + 1 > col_width for s in slots
    )
    if collision:
        left_row = prefix + "".join(
            (s[3].ljust(col_width) if s[3] else " " * col_width) for s in slots
        )
        right_row = prefix + "".join(
            (s[4].rjust(col_width) if s[4] else " " * col_width) for s in slots
        )
        for row in (left_row, right_row):
            if row.strip():
                lines.append(row.rstrip())
    else:
        marker_row = prefix + "".join(
            _compose_marker_slot(s[3], s[4], col_width) for s in slots
        )
        if marker_row.strip():
            lines.append(marker_row.rstrip())


def _render_table(
    lines: list[str],
    units: list[Any],
    sections: list[Any],
    fc: dict[str, str],
) -> None:
    """Render compact vertical table listing each section."""
    if not sections:
        return

    # Build mc -> unit lookup for flow control annotation
    mc_to_unit: dict[int, Any] = {u.mc: u for u in units}

    # Table header
    lines.append("")
    lines.append(f"  {'ID':<4} {'MCs':<12} {'#':>3}  {'Flow control'}")
    lines.append(f"  {'──':<4} {'───────────':<12} {'──':>3}  {'────────────'}")

    for sec in sections:
        mc_range = f"[{sec.mc_start}, {sec.mc_end})"
        mc_count = sec.mc_end - sec.mc_start

        # Collect flow events in this section
        events: list[str] = []
        for mc in range(sec.mc_start, sec.mc_end):
            u = mc_to_unit.get(mc)
            if u is None:
                continue
            if u.start_repeat:
                events.append(f"{fc['repeat_start']}")
            if u.end_repeat:
                target = u.next[0] if u.next else "?"
                events.append(f"{fc['repeat_end']}{fc['arrow']}{target}")
            if u.volta is not None:
                events.append(f"v{u.volta}")
            if u.fine:
                events.append("fine")
            jump = _get_jump_marker(u, sections, fc)
            if jump:
                events.append(jump)

        fc_str = ", ".join(events) if events else ""
        lines.append(f"  {sec.id:<4} {mc_range:<12} {mc_count:>3}  {fc_str}")

    # Transitions
    lines.append("")
    for sec in sections:
        targets = list(sec.to) if sec.to else []
        if targets:
            lines.append(f"  {sec.id} {fc['arrow']} [{', '.join(targets)}]")


def _get_jump_marker(
    unit: Any,
    sections: list[Any],
    fc: dict[str, str],
) -> str:
    """Return the short jump-instruction label drawn under a unit's MC slot.

    Uses the abbreviated DSaC / DSaF / DCaC / DCaF / DS / DC / →Coda
    labels so they fit in a single slot. Does NOT include `fine`
    markers — those are surfaced separately by the row drawing code.
    """
    if not unit.jump_from:
        return ""
    return _jump_label(unit.flow_control_types, short=True) or ""


def _marker_glyph_text(unit: Any, fc: dict[str, str]) -> str:
    """Return LEFT-anchored marker glyphs for a unit: segno (§), coda (⊕).

    These mark a *destination* on the timeline and are drawn at the section's
    leftmost slot. ``fine`` and "to coda" markers are RIGHT-anchored
    (handled by `_glyphs_for_unit`) because they fire at the end of a section.
    A coda marker that is *also* a jump source is a "to coda" instruction and
    is therefore excluded here.
    """
    parts: list[str] = []
    if unit.segno:
        parts.append(fc["segno"])
    if (
        unit.coda
        and not unit.jump_from
        and "to_coda" not in (unit.flow_control_types or ())
    ):
        parts.append(fc["coda"])
    return "".join(parts)


def _glyphs_for_unit(unit: Any, fc: dict[str, str]) -> tuple[str, str]:
    """Split a unit's flow-control glyphs into ``(left, right)`` strings.

    LEFT-anchored glyphs sit at the section's leftmost slot and mark a
    *point* on the timeline (segno destination, coda destination).

    RIGHT-anchored glyphs sit at the section's right edge (with one slot of
    trailing padding handled by the caller) and represent instructions that
    fire at the *end* of the section:

    - ``fine`` — Break that closes the section
    - ``"to "`` + coda glyph — a coda marker acting as a jump source, or an
      explicit ``to_coda`` flow-control type
    - ``DSaC`` / ``DSaF`` / ``DCaC`` / ``DCaF`` / ``DS`` / ``DC`` jump labels
    """
    ltext = _marker_glyph_text(unit, fc)
    rparts: list[str] = []

    types = unit.flow_control_types or ()
    is_to_coda = (unit.coda and unit.jump_from) or "to_coda" in types
    if is_to_coda:
        rparts.append(f"to{fc['coda']}")

    if unit.fine:
        rparts.append("fine")

    if unit.jump_from:
        label = _jump_label(types, short=True)
        # to_coda is rendered as "to ⊕" above; suppress the generic label.
        if label is not None and "to_coda" not in types:
            rparts.append(label)

    return ltext, " ".join(rparts)


_JUMP_LABELS_LONG: tuple[tuple[str, str], ...] = (
    ("dal_segno_al_coda", "D.S. al Coda"),
    ("dal_segno_al_fine", "D.S. al Fine"),
    ("da_capo_al_coda", "D.C. al Coda"),
    ("da_capo_al_fine", "D.C. al Fine"),
    ("dal_segno", "D.S."),
    ("da_capo", "D.C."),
    ("to_coda", "to Coda"),
)
_JUMP_LABELS_SHORT: tuple[tuple[str, str], ...] = (
    ("dal_segno_al_coda", "DSaC"),
    ("dal_segno_al_fine", "DSaF"),
    ("da_capo_al_coda", "DCaC"),
    ("da_capo_al_fine", "DCaF"),
    ("dal_segno", "DS"),
    ("da_capo", "DC"),
    ("to_coda", "→Coda"),
)


def _jump_label(fct: tuple[str, ...], short: bool = False) -> str | None:
    """Resolve the most specific jump-instruction label from flow_control_types."""
    table = _JUMP_LABELS_SHORT if short else _JUMP_LABELS_LONG
    for fc_type, label in table:
        if fc_type in fct:
            return label
    return None


def _section_id_for_mc(mc: int, sections: list[Any]) -> str:
    for sec in sections:
        if sec.mc_start <= mc < sec.mc_end:
            return sec.id
    return "?"


def _format_unit_events(
    unit: Any, sections: list[Any], fc: dict[str, str]
) -> list[str]:
    """Return human-readable strings for every flow event at this MC."""
    events: list[str] = []
    if unit.start_repeat:
        events.append(f"repeat_start (section {_section_id_for_mc(unit.mc, sections)})")
    if unit.end_repeat:
        target = unit.next[0] if unit.next else "?"
        events.append(f"repeat_end {fc['arrow']} MC {target}")
    if unit.volta is not None:
        events.append(
            f"volta {unit.volta} (section {_section_id_for_mc(unit.mc, sections)})"
        )
    if unit.segno:
        events.append(f"segno marker '{unit.segno}'")
    if unit.coda:
        events.append(f"coda marker '{unit.coda}'")
    if unit.fine:
        events.append("fine")
    if unit.jump_from:
        label = _jump_label(unit.flow_control_types)
        if label is not None:
            bits = [label]
            if unit.jump_bwd:
                bits.append(f"{fc['arrow']} {unit.jump_bwd}")
            if unit.play_until:
                bits.append(f"play until {unit.play_until}")
            if unit.jump_fwd and unit.jump_fwd != "fine":
                bits.append(f"then {fc['arrow']} {unit.jump_fwd}")
            events.append(" ".join(bits))
    if unit.section_break:
        events.append("section_break")
    return events


def _render_legend(
    lines: list[str],
    units: list[Any],
    sections: list[Any],
    fc: dict[str, str],
) -> None:
    """Render the flow-control legend, one line per MC."""
    lines.append("")
    lines.append("Flow control:")
    for unit in units:
        events = _format_unit_events(unit, sections, fc)
        if not events:
            continue
        lines.append(f"  MC {unit.mc:>3}: {'; '.join(events)}")


def _render_graph(
    lines: list[str],
    sections: list[Any],
    fc: dict[str, str],
    controller: Any | None = None,
) -> None:
    """Render section transition graph and (optionally) the default flow."""
    lines.append("")
    lines.append("Section transitions:")
    entries: list[str] = []
    for sec in sections:
        targets = list(sec.to) if sec.to else []
        entries.append(f"{sec.id} {fc['arrow']} [{', '.join(targets)}]")

    per_row = 4
    for i in range(0, len(entries), per_row):
        end = i + per_row
        chunk = entries[i:end]
        lines.append("  " + "    ".join(chunk))

    # Atomic-section sequence under the default flow (e.g., A → B → C → B …)
    if controller is None:
        return
    try:
        from timetoalign.core.enums import FlowMode

        flow = controller.compute_flow(FlowMode.default)
        sequence = flow.to_atomic_sequence()
    except Exception:
        return
    if not sequence:
        return
    lines.append("")
    lines.append("Atomic flow (default):")
    arrow = f" {fc['arrow']} "
    lines.append("  " + arrow.join(sequence))


# endregion

# region Flow Diagram


def flow_diagram(
    flow_obj: "Flow",
    width: int = DEFAULT_WIDTH,
    unicode: bool = True,
    show_mcs: bool = False,
    show_reasons: bool = True,
) -> Diagram:
    """Generate ASCII diagram for a Flow object.

    Shows the playthrough section sequence for a computed flow.

    Args:
        flow_obj: The Flow to render.
        width: Total width of the diagram in characters.
        unicode: Use Unicode characters (True) or ASCII fallback (False).
        show_mcs: Whether to expand MC sequences per section.
        show_reasons: Whether to annotate why each section starts.

    Returns:
        Multi-line string with ASCII diagram.
    """
    fc = FLOW_CHARS if unicode else FLOW_CHARS_ASCII

    lines: list[str] = []
    sections = flow_obj.sections
    folded = flow_obj.folded_length
    unfolded = flow_obj.unfolded_length
    ratio = unfolded / folded if folded > 0 else 0.0

    # 1. Header
    lines.append(
        f"Flow({flow_obj.mode.value}): {folded} folded "
        f"{fc['arrow']} {unfolded} unfolded "
        f"(\u00d7{ratio:.2f}), {len(sections)} sections"
    )
    lines.append("")

    # 2. Table
    if show_reasons:
        lines.append(f" {'#':>3}  {'MCs':<12} {'Sections':<12} {'Reason'}")
        lines.append(
            f" {'──':>3}  {'───────────':<12} " f"{'────────':<12} {'──────────────'}"
        )
    else:
        lines.append(f" {'#':>3}  {'MCs':<12} {'Sections':<12}")
        lines.append(f" {'──':>3}  {'───────────':<12} {'────────':<12}")

    # Try to get controller for rich reason annotations
    ctrl = flow_obj.controller

    for idx, sec in enumerate(sections):
        step = idx + 1
        mc_range = f"[{sec.mc_start}, {sec.mc_end})"
        sec_ids = ";".join(sec.atomic_section_ids)

        # Derive reason
        reason = ""
        if show_reasons:
            if idx == 0:
                reason = "start"
            else:
                prev = sections[idx - 1]
                if sec.mc_start == prev.mc_end:
                    reason = fc["arrow"]
                elif sec.mc_start < prev.mc_end:
                    # Jumped backward
                    if ctrl is not None:
                        try:
                            all_units = list(ctrl.iter_units())
                            prev_last_mc = prev.mc_end - 1
                            prev_unit = next(
                                (u for u in all_units if u.mc == prev_last_mc),
                                None,
                            )
                            if prev_unit and prev_unit.jump_from:
                                fct = prev_unit.flow_control_types
                                if "da_capo" in fct:
                                    reason = f"D.C. {fc['arrow']} {sec.mc_start}"
                                elif "dal_segno" in fct:
                                    reason = f"D.S. {fc['arrow']} {sec.mc_start}"
                                else:
                                    reason = f"repeat {fc['arrow']} {sec.mc_start}"
                            elif prev_unit and prev_unit.end_repeat:
                                reason = f"repeat {fc['arrow']} {sec.mc_start}"
                            else:
                                reason = f"jump \u2190 {sec.mc_start}"
                        except (ValueError, StopIteration):
                            reason = f"jump \u2190 {sec.mc_start}"
                    else:
                        reason = f"jump \u2190 {sec.mc_start}"
                else:
                    # Jumped forward
                    if ctrl is not None:
                        try:
                            all_units = list(ctrl.iter_units())
                            prev_last_mc = prev.mc_end - 1
                            prev_unit = next(
                                (u for u in all_units if u.mc == prev_last_mc),
                                None,
                            )
                            if prev_unit and prev_unit.jump_from:
                                fct = prev_unit.flow_control_types
                                if "to_coda" in fct:
                                    reason = "coda"
                                else:
                                    reason = f"skip {fc['arrow']} {sec.mc_start}"
                            else:
                                reason = f"skip {fc['arrow']} {sec.mc_start}"
                        except (ValueError, StopIteration):
                            reason = f"jump {fc['arrow']} {sec.mc_start}"
                    else:
                        reason = f"jump {fc['arrow']} {sec.mc_start}"

        if show_reasons:
            lines.append(f" {step:>3} {mc_range:<12} {sec_ids:<12} {reason}")
        else:
            lines.append(f" {step:>3} {mc_range:<12} {sec_ids:<12}")

        # Optionally expand MC numbers
        if show_mcs:
            mc_seq = list(range(sec.mc_start, sec.mc_end))
            lines.append(f"      MCs: {', '.join(str(m) for m in mc_seq)}")

    # 3. Footer - atomic sequence
    seq = flow_obj.to_atomic_sequence()
    lines.append("")
    lines.append(f"Sequence: {' '.join(seq)}")

    return Diagram("\n".join(lines))


# endregion

# region Flow Comparison Diagram


def flow_comparison_diagram(
    flow_a: "Flow",
    flow_b: "Flow",
    width: int = 80,
    unicode: bool = True,
) -> Diagram:
    """Generate side-by-side comparison of two Flows.

    Args:
        flow_a: First Flow to compare.
        flow_b: Second Flow to compare.
        width: Total width of the diagram in characters.
        unicode: Use Unicode characters (True) or ASCII fallback (False).

    Returns:
        Multi-line string with comparison diagram.
    """
    fc = FLOW_CHARS if unicode else FLOW_CHARS_ASCII

    lines: list[str] = []
    name_a = flow_a.mode.value
    name_b = flow_b.mode.value

    # Header
    lines.append(f"Flow comparison: {name_a} vs {name_b}")
    lines.append("")

    # Column layout
    col_w = max(16, (width - 12) // 2)
    lines.append(f" {'#':>3}  {name_a:<{col_w}} {name_b:<{col_w}} {'':>5}")
    lines.append(
        f" {'──':>3}  "
        f"{'─' * min(col_w, 15)}{' ' * max(0, col_w - 15)} "
        f"{'─' * min(col_w, 15)}{' ' * max(0, col_w - 15)} "
        f"{'─────'}"
    )

    secs_a = flow_a.sections
    secs_b = flow_b.sections
    max_rows = max(len(secs_a), len(secs_b))
    n_matching = 0

    for i in range(max_rows):
        step = i + 1

        if i < len(secs_a):
            sa = secs_a[i]
            a_str = (
                f"[{sa.mc_start}, {sa.mc_end})  " f"{';'.join(sa.atomic_section_ids)}"
            )
        else:
            a_str = "---"

        if i < len(secs_b):
            sb = secs_b[i]
            b_str = (
                f"[{sb.mc_start}, {sb.mc_end})  " f"{';'.join(sb.atomic_section_ids)}"
            )
        else:
            b_str = "---"

        # Compare
        if i < len(secs_a) and i < len(secs_b):
            sa = secs_a[i]
            sb = secs_b[i]
            if (
                sa.mc_start == sb.mc_start
                and sa.mc_end == sb.mc_end
                and sa.atomic_section_ids == sb.atomic_section_ids
            ):
                mark = fc["match"]
                n_matching += 1
            else:
                diffs: list[str] = []
                if sa.mc_start != sb.mc_start:
                    diffs.append("mc_start")
                if sa.mc_end != sb.mc_end:
                    diffs.append("mc_end")
                if sa.atomic_section_ids != sb.atomic_section_ids:
                    diffs.append("sections")
                mark = f"{fc['mismatch']} \u2190 {','.join(diffs)}"
        else:
            mark = fc["mismatch"]

        lines.append(f" {step:>3} {a_str:<{col_w}} {b_str:<{col_w}} {mark}")

    # Summary footer
    lines.append("")
    lines.append(
        f" {name_a}: {len(secs_a)} sections, " f"{flow_a.unfolded_length} unfolded"
    )
    lines.append(
        f" {name_b}: {len(secs_b)} sections, " f"{flow_b.unfolded_length} unfolded"
    )
    lines.append(f" Matching: {n_matching}/{max_rows} sections identical")

    return Diagram("\n".join(lines))


# endregion
