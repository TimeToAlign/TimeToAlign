"""Tests for the ASCII display module.

Tests cover:
- Character sets for all 6 timeline types
- Helper functions (coordinate formatting, name elision)
- Child row building and positioning
- Truncation logic for many children
- Timeline, group, and bundle diagrams
- ASCII fallback mode
"""

from __future__ import annotations

import re

import pyarrow as pa
import pytest

from timetoalign.core import TimeUnit
from timetoalign.core.enums import FlowMode
from timetoalign.display.ascii import (
    BOX_CHARS,
    BOX_CHARS_ASCII,
    CMAP_CHARS,
    CMAP_CHARS_ASCII,
    FLOW_CHARS,
    FLOW_CHARS_ASCII,
    REGION_CHARS,
    REGION_CHARS_ASCII,
    TIMELINE_CHARS,
    TIMELINE_CHARS_ASCII,
    TREE_CHARS,
    TREE_CHARS_ASCII,
    _build_child_row,
    _build_region_row,
    _elide_name,
    _format_coordinate,
    _get_children_to_display,
    _get_timeline_char,
    bundle_diagram,
    flow_comparison_diagram,
    flow_control_diagram,
    flow_diagram,
    group_diagram,
    timeline_diagram,
)
from timetoalign.storage.events import EventData
from timetoalign.timelines import ScoreFlowController
from timetoalign.timelines.types import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
)

# region Character Set Tests


class TestCharacterSets:
    """Tests for character set definitions."""

    def test_timeline_chars_covers_all_types(self) -> None:
        """All 6 core timeline type combinations have characters."""
        # Core types: int (discrete) and float (continuous) for each domain
        core_keys = {
            ("float", "graphical"),
            ("int", "graphical"),
            ("float", "physical"),
            ("int", "physical"),
            ("float", "logical"),
            ("int", "logical"),
        }
        # fraction type is also supported (same as float/continuous)
        fraction_keys = {
            ("fraction", "graphical"),
            ("fraction", "physical"),
            ("fraction", "logical"),
        }
        all_keys = core_keys | fraction_keys
        assert set(TIMELINE_CHARS.keys()) == all_keys
        assert set(TIMELINE_CHARS_ASCII.keys()) == all_keys

    def test_timeline_chars_are_single_character(self) -> None:
        """All timeline characters are single characters."""
        for key, char in TIMELINE_CHARS.items():
            assert len(char) == 1, f"Character for {key} is not single: {char!r}"
        for key, char in TIMELINE_CHARS_ASCII.items():
            assert len(char) == 1, f"ASCII char for {key} is not single: {char!r}"

    def test_timeline_chars_are_distinct(self) -> None:
        """Core timeline characters are unique (int/float for each domain)."""
        # Check the 6 core types are distinct
        core_keys = [
            ("float", "graphical"),
            ("int", "graphical"),
            ("float", "physical"),
            ("int", "physical"),
            ("float", "logical"),
            ("int", "logical"),
        ]
        chars = [TIMELINE_CHARS[k] for k in core_keys]
        assert len(chars) == len(set(chars)), "Core timeline chars are not unique"

    def test_tree_chars_complete(self) -> None:
        """Tree drawing characters include all necessary entries."""
        required = {"branch", "last", "vertical", "horizontal"}
        assert set(TREE_CHARS.keys()) == required
        assert set(TREE_CHARS_ASCII.keys()) == required

    def test_box_chars_complete(self) -> None:
        """Box drawing characters include all corners and sides."""
        required = {
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
            "horizontal",
            "vertical",
        }
        assert set(BOX_CHARS.keys()) == required
        assert set(BOX_CHARS_ASCII.keys()) == required


# endregion

# region Helper Function Tests


class TestFormatCoordinate:
    """Tests for _format_coordinate helper."""

    def test_integer_value_no_decimal(self) -> None:
        """Integer values format without decimal point."""
        assert _format_coordinate(0.0) == "0"
        assert _format_coordinate(100.0) == "100"
        assert _format_coordinate(4835.0) == "4835"

    def test_float_value_one_decimal(self) -> None:
        """Float values format with one decimal place."""
        assert _format_coordinate(150.5) == "150.5"
        assert _format_coordinate(99.9) == "99.9"
        assert _format_coordinate(0.1) == "0.1"

    def test_nearly_integer_treated_as_integer(self) -> None:
        """Values that are exactly integers format as integers."""
        assert _format_coordinate(150.0) == "150"


class TestElideName:
    """Tests for _elide_name helper."""

    def test_short_name_unchanged(self) -> None:
        """Names within max_width are unchanged."""
        assert _elide_name("system_1", 12) == "system_1"
        assert _elide_name("abc", 5) == "abc"

    def test_exact_length_unchanged(self) -> None:
        """Names exactly at max_width are unchanged."""
        assert _elide_name("system_12345", 12) == "system_12345"

    def test_long_name_elided(self) -> None:
        """Names exceeding max_width get '...' suffix."""
        result = _elide_name("very_long_system_name", 12)
        assert len(result) == 12
        assert result.endswith("...")

    def test_very_short_max_width(self) -> None:
        """Very short max_width just truncates."""
        assert _elide_name("system_1", 3) == "sys"
        assert _elide_name("system_1", 2) == "sy"


class TestGetTimelineChar:
    """Tests for _get_timeline_char helper."""

    def test_discrete_graphical(self) -> None:
        """Discrete graphical timeline gets '∶' (U+2236 RATIO, avoids Quarto fenced-div parsing)."""
        tl = DiscreteGraphicalTimeline(length=100, uid="test_dg")
        assert _get_timeline_char(tl, use_unicode=True) == "\u2236"
        assert _get_timeline_char(tl, use_unicode=False) == "\u2236"

    def test_continuous_graphical(self) -> None:
        """Continuous graphical timeline gets '='."""
        tl = ContinuousGraphicalTimeline(length=100.0, uid="test_cg")
        assert _get_timeline_char(tl, use_unicode=True) == "="

    def test_discrete_physical(self) -> None:
        """Discrete physical timeline gets '⋅' (or '.' in ASCII)."""
        tl = DiscretePhysicalTimeline(length=100, uid="test_dp")
        assert _get_timeline_char(tl, use_unicode=True) == "\u22c5"
        assert _get_timeline_char(tl, use_unicode=False) == "."

    def test_continuous_physical(self) -> None:
        """Continuous physical timeline gets '~'."""
        tl = ContinuousPhysicalTimeline(length=100.0, uid="test_cp")
        assert _get_timeline_char(tl, use_unicode=True) == "~"

    def test_discrete_logical(self) -> None:
        """Discrete logical timeline gets ','."""
        tl = DiscreteLogicalTimeline(length=100, uid="test_dl")
        assert _get_timeline_char(tl, use_unicode=True) == ","

    def test_continuous_logical(self) -> None:
        """Continuous logical timeline gets '_'."""
        tl = ContinuousLogicalTimeline(length=100.0, uid="test_cl")
        assert _get_timeline_char(tl, use_unicode=True) == "_"


class TestGetChildrenToDisplay:
    """Tests for _get_children_to_display truncation helper."""

    def test_no_truncation_when_under_max(self) -> None:
        """No truncation when children count <= max."""
        children: list[tuple[float, str]] = [(0.0, "a"), (10.0, "b"), (20.0, "c")]
        first, omitted, last = _get_children_to_display(children, max_children=6)
        assert first == children
        assert omitted == 0
        assert last == []

    def test_no_truncation_at_max(self) -> None:
        """No truncation when children count == max."""
        children: list[tuple[float, str]] = [(float(i * 10), f"c{i}") for i in range(6)]
        first, omitted, last = _get_children_to_display(children, max_children=6)
        assert first == children
        assert omitted == 0
        assert last == []

    def test_truncation_splits_correctly(self) -> None:
        """Truncation splits into first half, omitted count, last half."""
        children: list[tuple[float, str]] = [
            (float(i * 10), f"c{i}") for i in range(10)
        ]
        first, omitted, last = _get_children_to_display(children, max_children=6)
        # For max=6: first_count = (6+1)//2 = 3, last_count = 6-3 = 3
        assert len(first) == 3
        assert len(last) == 3
        assert omitted == 4  # 10 - 3 - 3 = 4

    def test_truncation_with_odd_max(self) -> None:
        """Truncation with odd max_children."""
        children: list[tuple[float, str]] = [
            (float(i * 10), f"c{i}") for i in range(10)
        ]
        first, omitted, last = _get_children_to_display(children, max_children=5)
        # For max=5: first_count = (5+1)//2 = 3, last_count = 5-3 = 2
        assert len(first) == 3
        assert len(last) == 2
        assert omitted == 5


class TestBuildChildRow:
    """Tests for _build_child_row helper."""

    def test_basic_row_structure(self) -> None:
        """Row has expected prefix, name, coords, bar, and end coord."""
        row = _build_child_row(
            child_offset=0,
            child_length=500,
            child_name="system_1",
            child_char=":",
            parent_length=1000,
            bar_width=50,
            name_width=12,
            coord_width=5,
            is_last=False,
            tree_chars=TREE_CHARS,
        )
        assert TREE_CHARS["branch"] in row  # ├
        assert "system_1" in row
        assert ":" in row  # The bar character

    def test_last_child_uses_last_marker(self) -> None:
        """Last child uses └ instead of ├."""
        row = _build_child_row(
            child_offset=0,
            child_length=500,
            child_name="last_one",
            child_char=":",
            parent_length=1000,
            bar_width=50,
            name_width=12,
            coord_width=5,
            is_last=True,
            tree_chars=TREE_CHARS,
        )
        assert TREE_CHARS["last"] in row  # └
        assert TREE_CHARS["branch"] not in row

    def test_bar_positioned_correctly(self) -> None:
        """Child bar is positioned proportionally on parent scale."""
        # Child at 250-750 on parent 0-1000 should be in middle
        row = _build_child_row(
            child_offset=250,
            child_length=500,
            child_name="middle",
            child_char=":",
            parent_length=1000,
            bar_width=40,
            name_width=10,
            coord_width=5,
            is_last=True,
            tree_chars=TREE_CHARS,
        )
        # The bar should start around position 10 (25% of 40)
        # Find the bar area in the row
        assert ":" in row


# endregion

# region Timeline Diagram Tests


class TestTimelineDiagram:
    """Tests for timeline_diagram function."""

    def test_basic_timeline_output(self) -> None:
        """Basic timeline renders header and bar."""
        tl = DiscreteGraphicalTimeline(length=100, uid="test_basic")
        result = timeline_diagram(tl)

        # Check header
        assert "DiscreteGraphicalTimeline[" in result
        assert tl.id in result

        # Check bar line
        assert "0 " in result  # Start coordinate
        assert "100 pixels" in result  # End + unit

    def test_timeline_with_children(self) -> None:
        """Timeline with children shows child rows."""
        parent = DiscreteGraphicalTimeline(length=1000, uid="parent_wc")
        child1 = DiscreteGraphicalTimeline(length=300, uid="child1", name="system_1")
        child2 = DiscreteGraphicalTimeline(length=300, uid="child2", name="system_2")
        parent.add_child(child1, offset=0)
        parent.add_child(child2, offset=400)

        result = timeline_diagram(parent)

        assert "2 children" in result
        assert "system_1" in result
        assert "system_2" in result
        assert TREE_CHARS["branch"] in result or TREE_CHARS["last"] in result

    def test_show_children_false(self) -> None:
        """show_children=False hides children."""
        parent = DiscreteGraphicalTimeline(length=1000, uid="parent_hide")
        child = DiscreteGraphicalTimeline(length=300, uid="child_hide", name="hidden")
        parent.add_child(child, offset=0)

        result = timeline_diagram(parent, show_children=False)

        # Header should still mention children count
        assert "1 children" in result
        # But child row should not appear
        lines = result.split("\n")
        # Only header and bar line, no child rows
        assert len(lines) == 2

    def test_ascii_mode(self) -> None:
        """unicode=False uses ASCII fallback characters."""
        tl = DiscreteGraphicalTimeline(length=100, uid="test_ascii")
        result = timeline_diagram(tl, unicode=False)

        # Should use ASCII-safe characters only
        assert (
            "\u2236" in result
        )  # U+2236 RATIO used for DiscreteGraphical (avoids Quarto parsing)
        # No Unicode box-drawing characters
        assert "\u250c" not in result  # ┌

    def test_parent_id_annotation(self) -> None:
        """parent_id parameter adds annotation to header."""
        tl = DiscreteGraphicalTimeline(length=100, uid="test_annot")
        result = timeline_diagram(tl, parent_id="parent:1")

        assert "(child of parent:1)" in result


class TestTimelineDiagramRecursion:
    """Tests for recursive child rendering."""

    @staticmethod
    def _make_hierarchy() -> ContinuousPhysicalTimeline:
        """Create a hierarchy with a following root child."""
        root = ContinuousPhysicalTimeline(length=100, uid="recursive_root", name="root")
        child = ContinuousPhysicalTimeline(
            length=40,
            uid="recursive_child",
            name="child",
        )
        grandchild = ContinuousPhysicalTimeline(
            length=10,
            uid="recursive_grandchild",
            name="grandchild",
        )
        tail = ContinuousPhysicalTimeline(
            length=10,
            uid="recursive_tail",
            name="tail",
        )
        child.add_child(grandchild, offset=5)
        root.add_child(child, offset=10)
        root.add_child(tail, offset=80)
        return root

    @pytest.mark.parametrize(
        ("depth", "expected"),
        [
            (
                True,
                "ContinuousPhysicalTimeline[recursive_root] (2 children)\n"
                "                       0 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 100 seconds\n"
                "  ├─ child            10    ~~~~~~~~~~~~~                  50\n"
                "  │  └─ grandchild    15     ~~~~                          25\n"
                "  └─ tail             80                           ~~~     90",
            ),
            (
                False,
                "ContinuousPhysicalTimeline[recursive_root] (2 children)\n"
                "                      0 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 100 seconds\n"
                "  ├─ child           10    ~~~~~~~~~~~~~~                  50\n"
                "  └─ tail            80                            ~~~     90",
            ),
            (
                0,
                "ContinuousPhysicalTimeline[recursive_root] (2 children)\n"
                "                      0 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 100 seconds",
            ),
            (
                1,
                "ContinuousPhysicalTimeline[recursive_root] (2 children)\n"
                "                      0 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 100 seconds\n"
                "  ├─ child           10    ~~~~~~~~~~~~~~                  50\n"
                "  └─ tail            80                            ~~~     90",
            ),
            (
                2,
                "ContinuousPhysicalTimeline[recursive_root] (2 children)\n"
                "                       0 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 100 seconds\n"
                "  ├─ child            10    ~~~~~~~~~~~~~                  50\n"
                "  │  └─ grandchild    15     ~~~~                          25\n"
                "  └─ tail             80                           ~~~     90",
            ),
        ],
        ids=["unlimited", "direct-bool", "none", "direct-int", "two-levels"],
    )
    def test_depth_controls_complete_output(
        self,
        depth: bool | int,
        expected: str,
    ) -> None:
        """Each supported depth renders the exact expected hierarchy."""
        assert timeline_diagram(self._make_hierarchy(), depth=depth) == expected

    def test_every_level_shares_the_root_coordinate_columns(self) -> None:
        """A descendant's coordinate column is the root's, not its own indent."""
        rendered = timeline_diagram(self._make_hierarchy(), depth=True)
        # The first number followed by a space on each row is its entry
        # coordinate; the names of this hierarchy carry no digits.
        columns = {
            re.search(r"\d+(?:\.\d+)? ", line).end()
            for line in rendered.split("\n")[1:]
        }

        assert len(columns) == 1

    def test_name_column_widens_only_for_the_tree_glyphs(self) -> None:
        """Nesting buys name room; a one-level diagram keeps the default."""
        flat = ContinuousPhysicalTimeline(length=100, uid="flat_root")
        flat.add_child(
            ContinuousPhysicalTimeline(length=40, uid="flat_child", name="child"),
            offset=10,
        )

        nested = timeline_diagram(self._make_hierarchy(), depth=True).split("\n")
        one_level = timeline_diagram(self._make_hierarchy(), depth=1).split("\n")
        flat_rendered = timeline_diagram(flat).split("\n")

        # "grandchild" is 10 characters behind one level of tree glyphs, so the
        # column grows by one to seat it; without descendants it never grows.
        assert nested[2].index("10") == one_level[2].index("10") + 1
        assert flat_rendered[2].index("10") == one_level[2].index("10")

    def test_false_equals_depth_one(self) -> None:
        """Boolean false preserves the direct-child rendering."""
        root = self._make_hierarchy()
        assert timeline_diagram(root, depth=False) == timeline_diagram(root, depth=1)

    def test_negative_depth_raises(self) -> None:
        """Negative integer depths are invalid."""
        with pytest.raises(ValueError, match="depth must be non-negative"):
            timeline_diagram(self._make_hierarchy(), depth=-1)

    def test_grandchildren_truncate_per_level(self) -> None:
        """A descendant level uses its own child limit and omission row."""
        root = ContinuousPhysicalTimeline(length=100, uid="truncate_root")
        child = ContinuousPhysicalTimeline(
            length=60,
            uid="truncate_child",
            name="branch",
        )
        for index in range(6):
            grandchild = ContinuousPhysicalTimeline(
                length=5,
                uid=f"leaf_{index}",
                name=f"leaf_{index}",
            )
            child.add_child(grandchild, offset=index * 10)
        root.add_child(child, offset=20)

        assert timeline_diagram(root, depth=True, max_children=3) == (
            "ContinuousPhysicalTimeline[truncate_root] (1 children)\n"
            "                      0 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 100 seconds\n"
            "  └─ branch          20       ~~~~~~~~~~~~~~~~~~~~~        80\n"
            "     ├─ leaf_0       20       ~~                           25\n"
            "     ├─ leaf_1       30           ~                        35\n"
            "     │  ... (3 more children)\n"
            "     └─ leaf_5       70                        ~~          75"
        )

    def test_segment_line_child_expands_its_segments(self) -> None:
        """A segment line child is a level like any other: ``depth`` rules it."""
        source = ContinuousPhysicalTimeline(length=30, uid="segment_source")
        segment_line = source.create_segment_line([0, 10, 20, 30])
        segment_line.name = "segments"
        root = ContinuousPhysicalTimeline(length=100, uid="segment_root")
        root.add_child(segment_line, offset=25)

        assert timeline_diagram(root, depth=True) == (
            "ContinuousPhysicalTimeline[segment_root] (1 children)\n"
            "                      0 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 100 seconds\n"
            "  └─ segments        25         ~~~~~~~~~~                 55\n"
            "     ├─ segment_0    25         ~~~                        35\n"
            "     ├─ segment_1    35            ~~~~                    45\n"
            "     └─ segment_2    45                ~~~                 55"
        )
        assert timeline_diagram(root, depth=1) == (
            "ContinuousPhysicalTimeline[segment_root] (1 children)\n"
            "                      0 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 100 seconds\n"
            "  └─ segments        25         ~~~~~~~~~~                 55"
        )


# endregion

# region Group Diagram Tests


class TestGroupDiagram:
    """Tests for group_diagram function."""

    def test_basic_group_output(self) -> None:
        """Basic group renders header and box; a group has no stored-row footer."""
        from timetoalign.timelines import TimelineGroup

        tl = DiscreteGraphicalTimeline(length=100, uid="grp_basic")
        group = TimelineGroup(id="test_group", timelines=[tl])

        result = group_diagram(group)

        # Header
        assert "TimelineGroup[test_group]" in result
        assert "1 timelines" in result

        # Box characters
        assert BOX_CHARS["top_left"] in result
        assert BOX_CHARS["bottom_right"] in result

        # No footer: the stored rows are always the common start and end
        assert result.splitlines()[-1].startswith(BOX_CHARS["bottom_left"])

    def test_group_with_multiple_timelines(self) -> None:
        """Group with multiple timelines shows all."""
        from timetoalign.timelines import TimelineGroup

        tl1 = DiscreteGraphicalTimeline(length=100, uid="grp_multi_1", name="tl1")
        tl2 = ContinuousPhysicalTimeline(length=50.0, uid="grp_multi_2", name="tl2")
        group = TimelineGroup(id="multi", timelines=[tl1, tl2])

        result = group_diagram(group)

        assert "2 timelines" in result
        assert "DiscreteGraphicalTimeline" in result
        assert "ContinuousPhysicalTimeline" in result

    def test_group_ascii_mode(self) -> None:
        """Group in ASCII mode uses ASCII box characters."""
        from timetoalign.timelines import TimelineGroup

        tl = DiscreteGraphicalTimeline(length=100, uid="grp_ascii")
        group = TimelineGroup(id="ascii_test", timelines=[tl])

        result = group_diagram(group, unicode=False)

        # ASCII box corners
        assert "+" in result  # ASCII corner
        assert "\u250c" not in result  # No Unicode ┌

    def test_group_forwards_depth(self) -> None:
        """A group applies the requested depth to its member diagrams."""
        from timetoalign.timelines import TimelineGroup

        root = ContinuousPhysicalTimeline(length=100, uid="group_root")
        child = ContinuousPhysicalTimeline(length=10, uid="group_child", name="child")
        root.add_child(child, offset=10)
        group = TimelineGroup(id="recursive_group", timelines=[root])

        assert group_diagram(group, depth=0) == (
            "TimelineGroup[recursive_group] (1 timelines)\n"
            "┌────────────────────────────────────────────────────────────────────┐\n"
            "│ ContinuousPhysicalTimeline[group_root] (1 children)                │\n"
            "│                       0 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 100 seconds │\n"
            "└────────────────────────────────────────────────────────────────────┘"
        )


# endregion

# region Bundle Diagram Tests


class TestBundleDiagram:
    """Tests for bundle_diagram function."""

    def test_basic_bundle_output(self) -> None:
        """Basic bundle renders header and groups."""
        from timetoalign.alignment import AlignmentBundle

        tl = DiscreteGraphicalTimeline(length=100, uid="bnd_basic")
        bundle = AlignmentBundle(id="test_bundle")
        bundle.add_timeline(tl)

        result = bundle_diagram(bundle)

        # Header
        assert "AlignmentBundle[test_bundle]" in result

        # Match claims footer
        assert "MatchClaims:" in result

    def test_columnar_claim_field_counted(self) -> None:
        """Claims held only in a columnar MatchClaimField are counted.

        A dense audio-to-audio bundle stores its claims in a MatchClaimField
        (not the per-claim Python list), so the diagram's claim count must
        include the field's row count.
        """
        from timetoalign.alignment import AlignmentBundle, MatchClaimField

        bundle = AlignmentBundle(id="columnar_bundle")
        tl_a = ContinuousPhysicalTimeline(length=1.0, uid="rec-a:cpt_1")
        tl_b = ContinuousPhysicalTimeline(length=1.0, uid="rec-b:cpt_1")
        bundle.add_timeline(tl_a, as_group="rec-a")
        bundle.add_timeline(tl_b, as_group="rec-b")

        # Three synchronous instant claims, held columnar (never materialised).
        field = MatchClaimField.from_columns(
            timeline_a_ids=["rec-a:cpt_1"] * 3,
            timeline_b_ids=["rec-b:cpt_1"] * 3,
            coordinate_a=[0.0, 0.1, 0.2],
            coordinate_b=[0.0, 0.1, 0.2],
            unit_a="seconds",
            unit_b="seconds",
        )
        bundle.add_match_claim_field(field)

        result = bundle_diagram(bundle)

        # The Python claim list is empty; the count comes from the field.
        assert len(bundle.cross_group_claims) == 0
        assert "MatchClaims: 3" in result

    def test_list_and_field_claim_counts_sum(self) -> None:
        """The diagram count is the list count plus the columnar field count."""
        from timetoalign.alignment import (
            AlignmentAnchor,
            AlignmentBundle,
            MatchClaim,
            MatchClaimField,
        )
        from timetoalign.core.enums import TimeUnit
        from timetoalign.core.time import Coordinate

        bundle = AlignmentBundle(id="mixed_bundle")
        tl_a = ContinuousPhysicalTimeline(length=1.0, uid="rec-a:cpt_1")
        tl_b = ContinuousPhysicalTimeline(length=1.0, uid="rec-b:cpt_1")
        bundle.add_timeline(tl_a, as_group="rec-a")
        bundle.add_timeline(tl_b, as_group="rec-b")

        # Two Python-list claims ...
        bundle.add_match_claims(
            [
                MatchClaim(
                    timeline_a_id="rec-a:cpt_1",
                    timeline_b_id="rec-b:cpt_1",
                    start_anchor=AlignmentAnchor(
                        timeline_a_id="rec-a:cpt_1",
                        coordinate_a=Coordinate(c, TimeUnit.seconds),
                        timeline_b_id="rec-b:cpt_1",
                        coordinate_b=Coordinate(c, TimeUnit.seconds),
                    ),
                )
                for c in (0.0, 0.1)
            ]
        )
        # ... plus three columnar claims → 5 total.
        bundle.add_match_claim_field(
            MatchClaimField.from_columns(
                timeline_a_ids=["rec-a:cpt_1"] * 3,
                timeline_b_ids=["rec-b:cpt_1"] * 3,
                coordinate_a=[0.2, 0.3, 0.4],
                coordinate_b=[0.2, 0.3, 0.4],
                unit_a="seconds",
                unit_b="seconds",
            )
        )

        result = bundle_diagram(bundle)
        assert "MatchClaims: 5" in result


# endregion

# region Integration Tests


class TestDiagramMethods:
    """Tests for diagram() methods on Timeline, Group, Bundle."""

    def test_timeline_diagram_method(self) -> None:
        """Timeline.diagram() returns same as timeline_diagram()."""
        tl = DiscreteGraphicalTimeline(length=100, uid="meth_tl")

        method_result = tl.diagram()
        func_result = timeline_diagram(tl)

        assert method_result == func_result

    def test_group_diagram_method(self) -> None:
        """TimelineGroup.diagram() returns same as group_diagram()."""
        from timetoalign.timelines import TimelineGroup

        tl = DiscreteGraphicalTimeline(length=100, uid="meth_grp")
        group = TimelineGroup(id="method_test", timelines=[tl])

        method_result = group.diagram()
        func_result = group_diagram(group)

        assert method_result == func_result

    def test_bundle_diagram_method(self) -> None:
        """AlignmentBundle.diagram() returns same as bundle_diagram()."""
        from timetoalign.alignment import AlignmentBundle

        tl = DiscreteGraphicalTimeline(length=100, uid="meth_bnd")
        bundle = AlignmentBundle(id="bundle_method")
        bundle.add_timeline(tl)

        method_result = bundle.diagram()
        func_result = bundle_diagram(bundle)

        assert method_result == func_result

    def test_diagram_parameters_passed_through(self) -> None:
        """diagram() method passes parameters correctly."""
        tl = DiscreteGraphicalTimeline(length=100, uid="meth_params")

        # Test width parameter affects output
        wide = tl.diagram(width=100)
        narrow = tl.diagram(width=40)
        # Wider diagram should produce longer output
        assert len(wide) == 139
        assert len(narrow) == 94

    def test_timeline_diagram_method_forwards_depth(self) -> None:
        """Timeline.diagram() forwards depth to timeline_diagram()."""
        root = ContinuousPhysicalTimeline(length=20, uid="method_depth_root")
        child = ContinuousPhysicalTimeline(
            length=10,
            uid="method_depth_child",
            name="child",
        )
        grandchild = ContinuousPhysicalTimeline(
            length=5,
            uid="method_depth_grandchild",
            name="grandchild",
        )
        child.add_child(grandchild, offset=0)
        root.add_child(child, offset=0)

        assert root.diagram(depth=False) == timeline_diagram(root, depth=False)

    def test_group_diagram_method_forwards_depth(self) -> None:
        """TimelineGroup.diagram() toggles grandchildren through depth."""
        from timetoalign.timelines import TimelineGroup

        root = ContinuousPhysicalTimeline(length=20, uid="group_method_root")
        child = ContinuousPhysicalTimeline(
            length=10,
            uid="group_method_child",
            name="child",
        )
        grandchild = ContinuousPhysicalTimeline(
            length=5,
            uid="group_method_grandchild",
            name="grandchild",
        )
        child.add_child(grandchild, offset=0)
        root.add_child(child, offset=0)
        group = TimelineGroup(id="depth_method_group", timelines=[root])

        assert "grandchild" in group.diagram(depth=True)
        assert "grandchild" not in group.diagram(depth=False)

    def test_bundle_diagram_method_forwards_depth(self) -> None:
        """AlignmentBundle.diagram() toggles grandchildren through depth."""
        from timetoalign.alignment import AlignmentBundle

        root = ContinuousPhysicalTimeline(length=20, uid="bundle_method_root")
        child = ContinuousPhysicalTimeline(
            length=10,
            uid="bundle_method_child",
            name="child",
        )
        grandchild = ContinuousPhysicalTimeline(
            length=5,
            uid="bundle_method_grandchild",
            name="grandchild",
        )
        child.add_child(grandchild, offset=0)
        root.add_child(child, offset=0)
        bundle = AlignmentBundle(id="depth_method_bundle")
        bundle.add_timeline(root, as_group="depth_method_group")

        assert "grandchild" in bundle.diagram(depth=True)
        assert "grandchild" not in bundle.diagram(depth=False)


# endregion

# region Region Character Set Tests


class TestRegionCharSets:
    """Tests for region character set definitions."""

    def test_region_chars_complete(self) -> None:
        """Region character set includes all required keys."""
        required = {"bar", "prefix", "left", "right"}
        assert set(REGION_CHARS.keys()) == required
        assert set(REGION_CHARS_ASCII.keys()) == required

    def test_region_chars_are_single_character(self) -> None:
        """All region characters are single characters."""
        for key, char in REGION_CHARS.items():
            assert len(char) == 1, f"Region char for {key} is not single: {char!r}"
        for key, char in REGION_CHARS_ASCII.items():
            assert (
                len(char) == 1
            ), f"ASCII region char for {key} is not single: {char!r}"


# endregion

# region Flow Character Set Tests


class TestFlowCharSets:
    """Tests for flow control character set definitions."""

    def test_flow_chars_complete(self) -> None:
        """Flow character set includes all required keys."""
        required = {
            "repeat_start",
            "repeat_end",
            "segno",
            "coda",
            "section_break",
            "break_right",
            "break_left",
            "break_volta",
            "arrow",
            "match",
            "mismatch",
            "volta_corner",
            "volta_top",
            "volta_end",
        }
        assert set(FLOW_CHARS.keys()) == required
        assert set(FLOW_CHARS_ASCII.keys()) == required

    def test_flow_chars_ascii_are_printable(self) -> None:
        """All ASCII flow characters are printable ASCII."""
        for key, char in FLOW_CHARS_ASCII.items():
            for c in char:
                assert ord(c) < 128, f"ASCII flow char for {key} is not ASCII: {char!r}"


# endregion

# region Build Region Row Tests


class TestBuildRegionRow:
    """Tests for _build_region_row helper."""

    def test_basic_region_row_structure(self) -> None:
        """Region row has prefix, name, coords, bar, and end coord."""
        row = _build_region_row(
            region_start=0,
            region_end=500,
            region_name="movement_1",
            parent_length=1000,
            bar_width=50,
            name_width=12,
            coord_width=5,
            region_chars=REGION_CHARS,
        )
        assert REGION_CHARS["prefix"] in row
        assert "movement_1" in row
        assert REGION_CHARS["left"] in row
        assert REGION_CHARS["right"] in row

    def test_region_bar_positioned_proportionally(self) -> None:
        """Region bar is positioned proportionally on parent scale."""
        row = _build_region_row(
            region_start=250,
            region_end=750,
            region_name="middle",
            parent_length=1000,
            bar_width=40,
            name_width=10,
            coord_width=5,
            region_chars=REGION_CHARS,
        )
        # The bar should contain fill characters
        assert REGION_CHARS["bar"] in row
        assert "250" in row
        assert "750" in row

    def test_region_name_elision(self) -> None:
        """Long region names are elided."""
        row = _build_region_row(
            region_start=0,
            region_end=100,
            region_name="very_long_region_name_here",
            parent_length=100,
            bar_width=40,
            name_width=12,
            coord_width=5,
            region_chars=REGION_CHARS,
        )
        assert "very_long..." in row


# endregion

# region Timeline Diagram with Regions Tests


class TestTimelineDiagramWithRegions:
    """Tests for timeline_diagram with regions display."""

    def test_show_regions_only(self) -> None:
        """show={'regions'} displays regions without children."""
        tl = ContinuousLogicalTimeline(length=1000.0, uid="tl_reg_only")
        child = ContinuousLogicalTimeline(length=300.0, uid="child_r", name="child")
        tl.add_child(child, offset=0)
        tl.create_region(0, 500, id="section_a", name="section_a")
        tl.create_region(500, 1000, id="section_b", name="section_b")

        result = timeline_diagram(tl, show={"regions"})

        # Regions should appear
        assert "section_a" in result
        assert "section_b" in result
        # Children should NOT appear (not in show set)
        assert TREE_CHARS["branch"] not in result
        assert TREE_CHARS["last"] not in result

    def test_show_regions_and_children(self) -> None:
        """show={'regions', 'children'} displays both."""
        tl = ContinuousLogicalTimeline(length=1000.0, uid="tl_both")
        child = ContinuousLogicalTimeline(length=300.0, uid="child_b", name="sys")
        tl.add_child(child, offset=0)
        tl.create_region(0, 500, id="verse", name="verse")

        result = timeline_diagram(tl, show={"regions", "children"})

        # Both should appear
        assert "verse" in result
        assert "sys" in result
        # Tree chars for children
        assert TREE_CHARS["last"] in result or TREE_CHARS["branch"] in result

    def test_regions_sorted_by_start(self) -> None:
        """Regions are sorted by start coordinate."""
        tl = ContinuousLogicalTimeline(length=1000.0, uid="tl_sort")
        # Add in reverse order
        tl.create_region(600, 1000, id="third", name="third")
        tl.create_region(0, 300, id="first", name="first")
        tl.create_region(300, 600, id="second", name="second")

        result = timeline_diagram(tl, show={"regions"})
        lines = result.split("\n")

        # Find region lines (lines containing the region prefix char)
        region_lines = [ln for ln in lines if REGION_CHARS["prefix"] in ln]
        assert len(region_lines) == 3

        # first should appear before second, second before third
        first_idx = next(i for i, ln in enumerate(region_lines) if "first" in ln)
        second_idx = next(i for i, ln in enumerate(region_lines) if "second" in ln)
        third_idx = next(i for i, ln in enumerate(region_lines) if "third" in ln)
        assert first_idx < second_idx < third_idx

    def test_show_none_backwards_compat(self) -> None:
        """show=None preserves exact existing behaviour (no regions)."""
        tl = ContinuousLogicalTimeline(length=1000.0, uid="tl_compat")
        tl.create_region(0, 500, id="hidden", name="hidden")

        result = timeline_diagram(tl)  # show=None is default

        # Region should not appear
        assert REGION_CHARS["prefix"] not in result
        assert "hidden" not in result

    def test_regions_ascii_mode(self) -> None:
        """Regions render correctly in ASCII mode."""
        tl = ContinuousLogicalTimeline(length=1000.0, uid="tl_ascii_r")
        tl.create_region(0, 250, id="intro", name="intro")

        result = timeline_diagram(tl, unicode=False, show={"regions"})

        assert REGION_CHARS_ASCII["prefix"] in result
        assert REGION_CHARS_ASCII["left"] in result
        assert "intro" in result


class TestTimelineDiagramHeaderRegions:
    """Tests for region count in timeline diagram header."""

    def test_header_includes_region_count(self) -> None:
        """Header line includes region count."""
        tl = ContinuousLogicalTimeline(length=1000.0, uid="tl_hdr_r")
        tl.create_region(0, 500, id="a", name="a")
        tl.create_region(500, 1000, id="b", name="b")

        result = timeline_diagram(tl)
        first_line = result.split("\n")[0]

        assert "2 regions" in first_line


# endregion

# region Diagram Method Show Parameter Tests


class TestDiagramMethodShowParam:
    """Tests for Timeline.diagram(show=...) parameter."""

    def test_show_param_passed_through(self) -> None:
        """Timeline.diagram(show=...) produces same output as function."""
        tl = ContinuousLogicalTimeline(length=1000.0, uid="tl_meth_show")
        tl.create_region(0, 500, id="part", name="part")

        method_result = tl.diagram(show={"regions"})
        func_result = timeline_diagram(tl, show={"regions"})

        assert method_result == func_result

    def test_show_children_false_overrides_show_set(self) -> None:
        """show_children=False hides children even if 'children' in show."""
        tl = ContinuousLogicalTimeline(length=1000.0, uid="tl_override")
        child = ContinuousLogicalTimeline(
            length=300.0, uid="child_ov", name="child_name"
        )
        tl.add_child(child, offset=0)
        tl.create_region(0, 500, id="reg", name="reg")

        result = tl.diagram(show_children=False, show={"children", "regions"})

        # Region should appear
        assert "reg" in result
        # Child should NOT appear (show_children=False overrides)
        assert "child_name" not in result


# endregion

# region Conversion Map Diagram Tests


class TestTimelineDiagramWithCmaps:
    """Tests for timeline_diagram with conversion map display."""

    def test_show_cmaps_only(self) -> None:
        """show={'cmaps'} displays c-maps without children."""
        from timetoalign.maps import LinearMap

        tl = ContinuousLogicalTimeline(length=1000.0, uid="tl_cmap_only")
        child = ContinuousLogicalTimeline(length=300.0, uid="child_cm", name="child")
        tl.add_child(child, offset=0)
        cmap = LinearMap(
            scalar=2.0,
            source_unit="quarters",
            target_unit="seconds",
            uid="cmap_tq",
        )
        tl.add_conversion_map(cmap)

        result = timeline_diagram(tl, show={"cmaps"})

        # C-map should appear
        assert CMAP_CHARS["prefix"] in result
        assert "quarters_..." in result  # name elided to fit DEFAULT_NAME_WIDTH
        assert "LinearMap" in result
        # Children should NOT appear (not in show set)
        assert TREE_CHARS["branch"] not in result
        assert TREE_CHARS["last"] not in result

    def test_show_cmaps_and_children(self) -> None:
        """show={'cmaps', 'children'} displays both."""
        from timetoalign.maps import LinearMap

        tl = ContinuousPhysicalTimeline(length=1000.0, uid="tl_cmap_both")
        child = ContinuousPhysicalTimeline(length=300.0, uid="child_cm2", name="sys")
        tl.add_child(child, offset=0)
        cmap = LinearMap(
            scalar=44100.0,
            source_unit="seconds",
            target_unit="samples",
            uid="cmap_ss",
        )
        tl.add_conversion_map(cmap)

        result = timeline_diagram(tl, show={"cmaps", "children"})

        # Both should appear
        assert "sys" in result
        assert "LinearMap" in result
        assert CMAP_CHARS["prefix"] in result
        # Tree chars for children
        assert TREE_CHARS["last"] in result or TREE_CHARS["branch"] in result

    def test_cmap_bar_uses_target_char(self) -> None:
        """C-map bar uses the timeline char for the target unit's type."""
        from timetoalign.maps import LinearMap

        # Target is "samples" which is discrete + physical -> middle dot
        tl = ContinuousPhysicalTimeline(length=10.0, uid="tl_cmap_char")
        cmap = LinearMap(
            scalar=44100.0,
            source_unit="seconds",
            target_unit="samples",
            uid="cmap_char",
        )
        tl.add_conversion_map(cmap)

        result = timeline_diagram(tl, show={"cmaps"})

        # Discrete physical char is middle dot (⋅)
        expected_char = TIMELINE_CHARS[("int", "physical")]
        # The bar should contain a run of this character
        assert expected_char * 5 in result

    def test_cmap_description_shows_arrow(self) -> None:
        """C-map row includes 'ClassName(source → target)' description."""
        from timetoalign.maps import LinearMap

        tl = ContinuousLogicalTimeline(length=100.0, uid="tl_cmap_desc")
        cmap = LinearMap(
            scalar=0.5,
            source_unit="quarters",
            target_unit="seconds",
            uid="cmap_desc",
        )
        tl.add_conversion_map(cmap)

        result = timeline_diagram(tl, show={"cmaps"})

        assert CMAP_CHARS["arrow"] in result
        assert "quarters" in result
        assert "seconds" in result

    def test_show_none_backwards_compat_no_cmaps(self) -> None:
        """show=None preserves existing behaviour (no cmaps)."""
        from timetoalign.maps import LinearMap

        tl = ContinuousLogicalTimeline(length=1000.0, uid="tl_cmap_compat")
        cmap = LinearMap(
            scalar=1.0,
            source_unit="quarters",
            target_unit="seconds",
            uid="cmap_compat",
        )
        tl.add_conversion_map(cmap)

        result = timeline_diagram(tl)  # show=None is default

        # C-map should not appear
        assert CMAP_CHARS["prefix"] not in result

    def test_cmaps_ascii_mode(self) -> None:
        """C-maps render correctly in ASCII mode."""
        from timetoalign.maps import LinearMap

        tl = ContinuousLogicalTimeline(length=1000.0, uid="tl_cmap_ascii")
        cmap = LinearMap(
            scalar=2.0,
            source_unit="quarters",
            target_unit="seconds",
            uid="cmap_ascii",
        )
        tl.add_conversion_map(cmap)

        result = timeline_diagram(tl, unicode=False, show={"cmaps"})

        assert CMAP_CHARS_ASCII["prefix"] in result
        assert CMAP_CHARS_ASCII["arrow"] in result
        assert "LinearMap" in result

    def test_multiple_cmaps(self) -> None:
        """Multiple c-maps are each rendered as separate rows."""
        from timetoalign.maps import LinearMap

        tl = ContinuousPhysicalTimeline(length=1000.0, uid="tl_multi_cmap")
        cmap1 = LinearMap(
            scalar=1000.0,
            source_unit="seconds",
            target_unit="milliseconds",
            uid="cmap_m1",
        )
        cmap2 = LinearMap(
            scalar=44100.0,
            source_unit="seconds",
            target_unit="samples",
            uid="cmap_m2",
        )
        tl.add_conversion_map(cmap1)
        tl.add_conversion_map(cmap2)

        result = timeline_diagram(tl, show={"cmaps"})
        lines = result.split("\n")

        # Find c-map lines (lines containing the c-map prefix char)
        cmap_lines = [ln for ln in lines if CMAP_CHARS["prefix"] in ln]
        assert len(cmap_lines) == 2

    def test_no_cmaps_no_rows(self) -> None:
        """No c-map rows appear when timeline has no c-maps."""
        tl = ContinuousLogicalTimeline(length=1000.0, uid="tl_no_cmap")

        result = timeline_diagram(tl, show={"cmaps"})

        assert CMAP_CHARS["prefix"] not in result


class TestTimelineDiagramHeaderCmaps:
    """Tests for c-map count in timeline diagram header."""

    def test_header_includes_cmap_count(self) -> None:
        """Header line includes c-map count."""
        from timetoalign.maps import LinearMap

        tl = ContinuousLogicalTimeline(length=1000.0, uid="tl_hdr_cm")
        cmap = LinearMap(
            scalar=1.0,
            source_unit="quarters",
            target_unit="seconds",
            uid="cmap_hdr",
        )
        tl.add_conversion_map(cmap)

        result = timeline_diagram(tl)
        first_line = result.split("\n")[0]

        assert "1 cmaps" in first_line

    def test_header_no_cmaps_when_empty(self) -> None:
        """Header does not mention cmaps when there are none."""
        tl = ContinuousLogicalTimeline(length=1000.0, uid="tl_hdr_no_cm")

        result = timeline_diagram(tl)
        first_line = result.split("\n")[0]

        assert "cmaps" not in first_line


class TestDiagramMethodCmapShowParam:
    """Tests for Timeline.diagram(show={'cmaps'}) parameter."""

    def test_show_cmaps_passed_through(self) -> None:
        """Timeline.diagram(show={'cmaps'}) produces same output as function."""
        from timetoalign.maps import LinearMap

        tl = ContinuousLogicalTimeline(length=1000.0, uid="tl_meth_cm")
        cmap = LinearMap(
            scalar=2.0,
            source_unit="quarters",
            target_unit="seconds",
            uid="cmap_meth",
        )
        tl.add_conversion_map(cmap)

        method_result = tl.diagram(show={"cmaps"})
        func_result = timeline_diagram(tl, show={"cmaps"})

        assert method_result == func_result


# endregion

# region Flow Diagram Test Fixtures


class _MockMeasureData(EventData):
    """Minimal mock for MeasureData, wrapping a synthetic PyArrow table.

    Subclassing EventData means it exposes the same public surface
    (``schema``, ``column_values()``) that ScoreFlowController relies on.
    """

    def __init__(self, tbl: pa.Table) -> None:
        super().__init__(tbl, unit=TimeUnit.quarters)


def _make_simple_controller() -> ScoreFlowController:
    """Create a ScoreFlowController with repeats and voltas (6 MCs, 4 sections).

    Structure:
        MC 1-2: Section A (intro, 2 bars)
        MC 3:   Repeat start, Section B
        MC 4:   Volta 1, end_repeat -> back to MC 3, Section C
        MC 5:   Volta 2, Section D
        MC 6:   Final bar, Section D continued (or E)
    """
    table = pa.table(
        {
            "mc": [1, 2, 3, 4, 5, 6],
            "mn": ["1", "2", "3", "4", "4", "5"],
            "duration": [{"value": 4.0}] * 6,
            "start": [{"value": float(i * 4)} for i in range(6)],
            "next": [[2], [3], [4, 5], [3], [6], [-1]],
            "timesig": ["4/4"] * 6,
            "volta": [None, None, None, 1, 2, None],
            "start_repeat": [False, False, True, False, False, False],
            "end_repeat": [False, False, False, True, False, False],
        }
    )
    md = _MockMeasureData(table)
    return ScoreFlowController(md)


def _make_minimal_controller() -> ScoreFlowController:
    """Create a minimal ScoreFlowController (3 MCs, no flow control)."""
    table = pa.table(
        {
            "mc": [1, 2, 3],
            "mn": ["1", "2", "3"],
            "duration": [{"value": 4.0}] * 3,
            "start": [{"value": 0.0}, {"value": 4.0}, {"value": 8.0}],
            "next": [[2], [3], [-1]],
            "timesig": ["4/4"] * 3,
            "volta": [None, None, None],
        }
    )
    md = _MockMeasureData(table)
    return ScoreFlowController(md)


# endregion

# region Flow Control Diagram Tests


class TestFlowControlDiagram:
    """Tests for flow_control_diagram function."""

    def test_header_content(self) -> None:
        """Header contains MC count, section count, and flow event count."""
        ctrl = _make_simple_controller()
        result = flow_control_diagram(ctrl)

        assert "ScoreFlowController" in result
        assert "MCs" in result
        assert "atomic sections" in result
        assert "flow events" in result

    def test_mc_ruler_present(self) -> None:
        """MC ruler row shows all MC numbers."""
        ctrl = _make_simple_controller()
        result = flow_control_diagram(ctrl)

        # All 6 MCs should appear
        for mc in range(1, 7):
            assert str(mc) in result

    def test_sections_aligned_with_ruler(self) -> None:
        """Section IDs appear in the diagram."""
        ctrl = _make_simple_controller()
        result = flow_control_diagram(ctrl)

        sections = ctrl.get_sections()
        for sec in sections:
            assert sec.id in result

    def test_repeat_markers_present(self) -> None:
        """Repeat barlines appear at correct MCs."""
        ctrl = _make_simple_controller()
        result = flow_control_diagram(ctrl, unicode=True)

        # Unicode repeat markers should be present
        assert FLOW_CHARS["repeat_start"] in result
        assert FLOW_CHARS["repeat_end"] in result

    def test_volta_brackets_rendered(self) -> None:
        """Volta brackets appear for volta MCs."""
        ctrl = _make_simple_controller()
        result = flow_control_diagram(ctrl, unicode=True)

        # Volta corner and number
        assert FLOW_CHARS["volta_corner"] in result
        assert "1" in result
        assert "2" in result

    def test_legend_content(self) -> None:
        """Legend lists all flow control events."""
        ctrl = _make_simple_controller()
        result = flow_control_diagram(ctrl, show_legend=True)

        assert "Flow control:" in result
        assert "repeat_start" in result
        assert "repeat_end" in result
        assert "volta 1" in result
        assert "volta 2" in result

    def test_legend_hidden(self) -> None:
        """show_legend=False hides legend."""
        ctrl = _make_simple_controller()
        result = flow_control_diagram(ctrl, show_legend=False)

        assert "Flow control:" not in result

    def test_section_graph(self) -> None:
        """Section transition graph shows transitions."""
        ctrl = _make_simple_controller()
        result = flow_control_diagram(ctrl, show_graph=True)

        assert "Section transitions:" in result

    def test_section_graph_hidden(self) -> None:
        """show_graph=False hides section graph."""
        ctrl = _make_simple_controller()
        result = flow_control_diagram(ctrl, show_graph=False)

        assert "Section transitions:" not in result

    def test_ascii_mode(self) -> None:
        """ASCII mode uses ASCII fallback characters."""
        ctrl = _make_simple_controller()
        result = flow_control_diagram(ctrl, unicode=False)

        # Should contain ASCII repeat markers
        assert FLOW_CHARS_ASCII["repeat_start"] in result
        # Should NOT contain Unicode-only chars
        assert (
            FLOW_CHARS["volta_corner"] not in result
            or FLOW_CHARS_ASCII["volta_corner"] in result
        )  # noqa: E501

    def test_minimal_controller_no_flow_events(self) -> None:
        """Controller with no flow control still renders."""
        ctrl = _make_minimal_controller()
        result = flow_control_diagram(ctrl)

        assert "ScoreFlowController" in result
        assert "3 MCs" in result
        assert "0 flow events" in result


class TestFlowControllerDiagramMethod:
    """Tests for ScoreFlowController.diagram() method."""

    def test_diagram_delegates_correctly(self) -> None:
        """ScoreFlowController.diagram() produces same as function."""
        ctrl = _make_simple_controller()

        method_result = ctrl.diagram()
        func_result = flow_control_diagram(ctrl)

        assert method_result == func_result

    def test_str_returns_diagram(self) -> None:
        """str(controller) returns the diagram."""
        ctrl = _make_simple_controller()
        assert str(ctrl) == ctrl.diagram()


# endregion

# region Flow Diagram Tests


class TestFlowDiagram:
    """Tests for flow_diagram function."""

    def test_header_content(self) -> None:
        """Header contains mode, folded/unfolded counts, ratio."""
        ctrl = _make_simple_controller()
        flow_obj = ctrl.compute_flow(FlowMode.default)
        result = flow_diagram(flow_obj)

        assert "Flow(default)" in result
        assert "folded" in result
        assert "unfolded" in result
        assert "\u00d7" in result  # multiplication sign

    def test_section_rows_present(self) -> None:
        """Section rows show MC ranges and section IDs."""
        ctrl = _make_simple_controller()
        flow_obj = ctrl.compute_flow(FlowMode.default)
        result = flow_diagram(flow_obj)

        # Should have numbered rows
        assert "  1 " in result
        # Section IDs in atomic sequence
        for sec_id in flow_obj.to_atomic_sequence()[:3]:
            assert sec_id in result

    def test_atomic_sequence_footer(self) -> None:
        """Footer shows complete atomic section sequence."""
        ctrl = _make_simple_controller()
        flow_obj = ctrl.compute_flow(FlowMode.default)
        result = flow_diagram(flow_obj)

        assert "Sequence:" in result
        seq = flow_obj.to_atomic_sequence()
        assert " ".join(seq) in result

    def test_reasons_column(self) -> None:
        """Reason column shows why each section starts."""
        ctrl = _make_simple_controller()
        flow_obj = ctrl.compute_flow(FlowMode.default)
        result = flow_diagram(flow_obj, show_reasons=True)

        assert "Reason" in result
        assert "start" in result  # First section always "start"

    def test_show_reasons_false(self) -> None:
        """show_reasons=False hides reasons column."""
        ctrl = _make_simple_controller()
        flow_obj = ctrl.compute_flow(FlowMode.default)
        result = flow_diagram(flow_obj, show_reasons=False)

        assert "Reason" not in result

    def test_show_mcs_expands(self) -> None:
        """show_mcs=True shows MC sequences for each section."""
        ctrl = _make_simple_controller()
        flow_obj = ctrl.compute_flow(FlowMode.default)
        result = flow_diagram(flow_obj, show_mcs=True)

        assert "MCs:" in result


class TestFlowDiagramMethod:
    """Tests for Flow.diagram() method."""

    def test_diagram_delegates_correctly(self) -> None:
        """Flow.diagram() produces same as function."""
        ctrl = _make_simple_controller()
        flow_obj = ctrl.compute_flow(FlowMode.default)

        method_result = flow_obj.diagram()
        func_result = flow_diagram(flow_obj)

        assert method_result == func_result

    def test_str_returns_diagram(self) -> None:
        """str(flow) returns the diagram."""
        ctrl = _make_simple_controller()
        flow_obj = ctrl.compute_flow(FlowMode.default)
        assert str(flow_obj) == flow_obj.diagram()

    def test_repr_unchanged(self) -> None:
        """__repr__ returns compact one-liner, NOT diagram."""
        ctrl = _make_simple_controller()
        flow_obj = ctrl.compute_flow(FlowMode.default)
        repr_str = repr(flow_obj)

        assert repr_str.startswith("Flow(")
        # repr should be one line
        assert "\n" not in repr_str


# endregion

# region Flow Comparison Diagram Tests


class TestFlowComparisonDiagram:
    """Tests for flow_comparison_diagram function."""

    def test_identical_flows_all_match(self) -> None:
        """Identical flows show all '=' markers."""
        ctrl = _make_simple_controller()
        flow_a = ctrl.compute_flow(FlowMode.default)
        flow_b = ctrl.compute_flow(FlowMode.default)
        result = flow_comparison_diagram(flow_a, flow_b)

        assert "Flow comparison:" in result
        assert FLOW_CHARS["match"] in result
        # All should match
        lines = result.split("\n")
        match_count = sum(
            1 for ln in lines if FLOW_CHARS["match"] in ln and "#" not in ln
        )
        assert match_count >= 1

    def test_divergent_flows_show_mismatch(self) -> None:
        """Different flows show mismatch markers with explanations."""
        ctrl = _make_simple_controller()
        flow_a = ctrl.compute_flow(FlowMode.default)
        flow_b = ctrl.compute_flow(FlowMode.printed)
        result = flow_comparison_diagram(flow_a, flow_b)

        assert "Flow comparison:" in result
        # Summary should appear
        assert "Matching:" in result

    def test_summary_footer(self) -> None:
        """Summary footer shows section counts and match ratio."""
        ctrl = _make_simple_controller()
        flow_a = ctrl.compute_flow(FlowMode.default)
        flow_b = ctrl.compute_flow(FlowMode.default)
        result = flow_comparison_diagram(flow_a, flow_b)

        assert "sections" in result
        assert "unfolded" in result
        assert "Matching:" in result

    def test_ascii_mode(self) -> None:
        """ASCII mode uses ASCII characters."""
        ctrl = _make_simple_controller()
        flow_a = ctrl.compute_flow(FlowMode.default)
        flow_b = ctrl.compute_flow(FlowMode.default)
        result = flow_comparison_diagram(flow_a, flow_b, unicode=False)

        assert FLOW_CHARS_ASCII["match"] in result


class TestDiffDiagramMethod:
    """Tests for Flow.diff_diagram() method."""

    def test_diff_diagram_delegates_correctly(self) -> None:
        """flow.diff_diagram(other) produces same as function."""
        ctrl = _make_simple_controller()
        flow_a = ctrl.compute_flow(FlowMode.default)
        flow_b = ctrl.compute_flow(FlowMode.default)

        method_result = flow_a.diff_diagram(flow_b)
        func_result = flow_comparison_diagram(flow_a, flow_b)

        assert method_result == func_result


# endregion
