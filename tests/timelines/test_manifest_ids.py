"""Identity of manifest elements: one id form, given ids kept, unique per manifest."""

from __future__ import annotations

import gc
import json
import pickle
from collections.abc import Callable, Iterator
from fractions import Fraction as F
from typing import Any

import pytest
from timetoshare import validate

from timetoalign.alignment import AlignmentBundle
from timetoalign.core import Coordinate, FlowControlElement, TimeUnit
from timetoalign.loader.alignment import ParangonadaLoader
from timetoalign.loader.graphical.iiif import IIIFManifestLoader
from timetoalign.loader.midi.score import ScoreMidiLoader
from timetoalign.loader.physical.audio import AudioLoader
from timetoalign.loader.score import Ms3Loader
from timetoalign.loader.tabular import LabLoader
from timetoalign.maps import (
    ChainMap,
    CombinationMap,
    ConstantMap,
    IntervalToConstantMap,
    LinearMap,
    MeasureRecordMap,
    PiecewiseMap,
    QuartersToFloatingMeasures,
    ScalarMap,
    ShiftMap,
    TableMap,
)
from timetoalign.testdata import ensure_data
from timetoalign.timelines import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
    MeasureMap,
    SegmentLine,
    Timeline,
    TimelineGroup,
)
from timetoalign.timelines.flowcontrol import Break, Jump, Marker
from timetoalign.timelines.regions import Region

SEGNO = FlowControlElement.segno
DAL_SEGNO = FlowControlElement.dal_segno

SCORE = ensure_data("score")
MIDI = ensure_data("midi")
AUDIO = ensure_data("audio")
PARANGONAR = ensure_data("parangonar")
SUPRA = ensure_data("supra")
FIXTURES = ensure_data("fixtures")

QUARTERS = TimeUnit.quarters


def _q(value: Any) -> Coordinate:
    return Coordinate(value, QUARTERS)


# region One form, one generator


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda: ContinuousLogicalTimeline(), "clt_1"),
        (lambda: DiscreteLogicalTimeline(), "dlt_1"),
        (lambda: ContinuousPhysicalTimeline(), "cpt_1"),
        (lambda: DiscretePhysicalTimeline(), "dpt_1"),
        (lambda: ContinuousGraphicalTimeline(), "cgt_1"),
        (lambda: DiscreteGraphicalTimeline(), "dgt_1"),
        (lambda: Timeline(unit="seconds"), "cpt_1"),
        (lambda: Timeline(unit="ticks"), "dlt_1"),
        (lambda: Timeline(unit="pixels"), "dgt_1"),
        (lambda: SegmentLine[ContinuousLogicalTimeline](), "clt_1"),
        (lambda: LinearMap(scalar=2), "lmap_1"),
        (lambda: TableMap(x_values=[0, 4], y_values=[0.0, 2.5]), "tmap_1"),
        (
            lambda: QuartersToFloatingMeasures(x_values=[0, 4], y_values=[1.0, 2.0]),
            "tmap_1",
        ),
        (
            lambda: PiecewiseMap(breaks=[0, 8], maps=[ShiftMap(offset=1)]),
            "pmap_1",
        ),
        (lambda: ScalarMap(2), "cmap_1"),
        (lambda: Region(_q(0), _q(4)), "region_1"),
        (lambda: Break(_q(4)), "flow_marker_1"),
        (lambda: Jump(_q(4), _q(0)), "jump_from_1"),
        (lambda: Marker(_q(0), control_type=SEGNO), "jump_to_1"),
    ],
)
def test_standalone_elements_mint_their_first_id(
    build: Callable[[], Any], expected: str
) -> None:
    assert build().id == expected


def _reference(*, reverse: bool = False) -> Timeline:
    """Build the reference manifest, its collections in either order."""
    root = ContinuousLogicalTimeline(length=32)
    steps: list[Callable[[], Any]] = [
        lambda: root.create_child(length=8, offset=0),
        lambda: root.create_child(length=8, offset=8),
        lambda: root.create_region(0, 8, name="A"),
        lambda: root.create_region(8, 16, name="B"),
        lambda: root.flow_control.add_break(Break(_q(8))),
        lambda: root.flow_control.add_break(Break(_q(16))),
        lambda: root.add_conversion_map(
            LinearMap(scalar=2, source_unit="quarters", target_unit="seconds")
        ),
    ]
    for step in reversed(steps) if reverse else steps:
        step()
    marker = root.flow_control.add_marker(
        Marker(_q(8), name="segno", control_type=SEGNO)
    )
    root.flow_control.add_jump(Jump(_q(24), _q(8), target=marker.id))
    return root


REFERENCE_IDS = {
    "timeline": "clt_1",
    "children": {"clt_2", "clt_3"},
    "regions": {"region_1", "region_2"},
    "breaks": {"flow_marker_1", "flow_marker_2"},
    "markers": ["jump_to_1"],
    "jumps": [("jump_from_1", "jump_to_1")],
    "maps": ["lmap_1"],
}


def _ids(manifest: dict[str, Any]) -> dict[str, Any]:
    (payload,) = manifest["timelines"]
    flow = payload["flow_control"]
    return {
        "timeline": payload["id"],
        "children": {child["id"] for child in payload["children"]},
        "regions": {entry["id"] for entry in payload["regions"]},
        "breaks": {entry["id"] for entry in flow if entry["type"] == "flow_marker"},
        "markers": [entry["id"] for entry in flow if entry["type"] == "jump_to"],
        "jumps": [
            (entry["id"], entry.get("target"))
            for entry in flow
            if entry["type"] == "jump_from"
        ],
        "maps": [entry["id"] for entry in payload["conversion_maps"]],
    }


class TestDeterminism:
    """Minted ids depend on the manifest only, never on the process history."""

    def test_reference_ids(self) -> None:
        assert _ids(_reference().to_dict()) == REFERENCE_IDS

    def test_unrelated_construction_first_changes_nothing(self) -> None:
        reference = _reference().to_dict()
        for _ in range(5):
            ContinuousLogicalTimeline(length=4)
        for _ in range(4):
            LinearMap(scalar=3)
        assert _reference().to_dict() == reference

    def test_reverse_insertion_order_mints_the_same_ids(self) -> None:
        assert _ids(_reference(reverse=True).to_dict()) == REFERENCE_IDS

    def test_smallest_free_number(self) -> None:
        root = ContinuousLogicalTimeline(length=32)
        root.add_child(ContinuousLogicalTimeline(length=4, uid="clt_3"), 0)
        assert root.create_child(length=4, offset=4).id == "clt_2"
        assert root.create_child(length=4, offset=8).id == "clt_4"

    def test_no_old_form(self) -> None:
        for cls in (
            Timeline,
            ContinuousLogicalTimeline,
            DiscreteLogicalTimeline,
            ContinuousPhysicalTimeline,
            DiscretePhysicalTimeline,
            ContinuousGraphicalTimeline,
            DiscreteGraphicalTimeline,
        ):
            timeline_id = cls().id
            assert not timeline_id.startswith("tl:")
            assert "_" in timeline_id


# endregion

# region Given ids


GIVEN = [
    "performance",
    "a",
    "Bass Kick",
    "score:clt_7",
    "score/1",
    "part 2",
    "https://example.org/op35#m1",
]


def _given_manifest() -> Timeline:
    root = ContinuousLogicalTimeline(length=32, uid="performance")
    root.add_child(ContinuousLogicalTimeline(length=8, uid="Bass Kick"), 0)
    root.add_child(ContinuousLogicalTimeline(length=8, uid="score:clt_7"), 8)
    root.create_region(0, 8, id="a", name="A")
    root.add_conversion_map(ScalarMap(2, uid="raw"))
    root.flow_control.add_break(Break(_q(8), id="b1"))
    root.flow_control.add_marker(
        Marker(_q(8), id="segno", name="segno", control_type=SEGNO)
    )
    root.flow_control.add_jump(Jump(_q(24), _q(8), id="ds", target="segno"))
    return root


class TestGivenIds:
    """Whatever id is given is the id, through a JSON round trip."""

    def test_given_ids_are_kept(self) -> None:
        assert _ids(_given_manifest().to_dict()) == {
            "timeline": "performance",
            "children": {"Bass Kick", "score:clt_7"},
            "regions": {"a"},
            "breaks": {"b1"},
            "markers": ["segno"],
            "jumps": [("ds", "segno")],
            "maps": ["raw"],
        }

    @pytest.mark.parametrize("build", [_given_manifest, _reference])
    def test_round_trip_is_text_identical(self, build: Callable[[], Timeline]) -> None:
        payload = build().to_dict()
        text = json.dumps(payload)
        restored = Timeline.from_dict(json.loads(text))
        assert json.dumps(restored.to_dict()) == text

    @pytest.mark.parametrize("refused", ["", "a\tb", "a\nb"])
    @pytest.mark.parametrize(
        "build",
        [
            lambda uid: ContinuousLogicalTimeline(uid=uid),
            lambda uid: ScalarMap(2, uid=uid),
            lambda uid: Region(_q(0), _q(4), id=uid),
            lambda uid: Break(_q(4), id=uid),
            lambda uid: Jump(_q(4), _q(0), id=uid),
            lambda uid: Marker(_q(0), id=uid, control_type=SEGNO),
        ],
        ids=["timeline", "map", "region", "break", "jump", "marker"],
    )
    def test_refused_ids(self, build: Callable[[str], Any], refused: str) -> None:
        with pytest.raises(ValueError, match="Invalid id"):
            build(refused)

    @pytest.mark.parametrize("given", GIVEN)
    def test_accepted_ids(self, given: str) -> None:
        assert ContinuousLogicalTimeline(uid=given).id == given
        assert Region(_q(0), _q(4), id=given).id == given


# endregion

# region Collisions


def _with_region() -> ContinuousLogicalTimeline:
    timeline = ContinuousLogicalTimeline(length=16)
    timeline.create_region(0, 4, name="intro")
    return timeline


class TestCollisions:
    """Ids are unique within one manifest across all kinds."""

    def test_minted_ids_are_minted_again_when_joining(self) -> None:
        root = _with_region()
        child = _with_region()
        root.add_child(child, 0)
        assert (root.id, root.list_regions()) == ("clt_1", ["region_1"])
        assert (child.id, child.list_regions()) == ("clt_2", ["region_2"])
        assert root.list_children() == ["clt_2"]
        segments = [
            event["id"]
            for event in root._events
            if event.get("event_type") == "Segment"
        ]
        assert segments == ["clt_2"]

    def test_a_grandchild_is_renamed_under_its_parent(self) -> None:
        root = ContinuousLogicalTimeline(length=16)
        root.create_child(length=4)  # clt_2
        child = ContinuousLogicalTimeline(length=8)
        grandchild = child.create_child(length=4)  # clt_2 inside the child
        root.add_child(child, 8)
        assert (child.id, grandchild.id) == ("clt_3", "clt_4")
        assert child.list_children() == ["clt_4"]
        assert child.get_child("clt_4") is grandchild

    def test_a_marker_minted_again_takes_its_jumps_along(self) -> None:
        root = ContinuousLogicalTimeline(length=16)
        root.flow_control.add_marker(Marker(_q(0), control_type=SEGNO))
        child = ContinuousLogicalTimeline(length=8)
        marker = child.flow_control.add_marker(
            Marker(_q(2), name="segno", control_type=SEGNO)
        )
        child.flow_control.add_jump(Jump(_q(6), _q(2), target=marker.id))
        root.add_child(child, 0)
        assert [m.id for m in child.flow_control.markers] == ["jump_to_2"]
        assert [j.target for j in child.flow_control.jumps] == ["jump_to_2"]

    def test_given_timeline_id_collision_raises(self) -> None:
        root = ContinuousLogicalTimeline(length=16, uid="score")
        with pytest.raises(ValueError) as error:
            root.add_child(ContinuousLogicalTimeline(length=4, uid="score"), 0)
        assert "timeline 'score'" in str(error.value)
        assert "already held by timeline 'score'" in str(error.value)

    def test_given_id_collides_across_kinds(self) -> None:
        root = ContinuousLogicalTimeline(length=16, uid="score")
        root.create_region(0, 4, id="intro")
        child = ContinuousLogicalTimeline(length=4, uid="intro")
        with pytest.raises(
            ValueError, match="already held by region 'intro' of timeline 'score'"
        ):
            root.add_child(child, 0)
        # Atomic: nothing changed on either side.
        assert root.list_children() == []
        assert root.list_regions() == ["intro"]
        assert root.length.value == 16
        assert child.id == "intro"

    def test_region_beside_a_child_with_its_id_raises(self) -> None:
        root = ContinuousLogicalTimeline(length=16)
        root.add_child(ContinuousLogicalTimeline(length=4, uid="intro"), 0)
        with pytest.raises(ValueError, match="already held by timeline 'intro'"):
            root.create_region(0, 4, id="intro")
        assert root.list_regions() == []

    def test_given_map_id_held_by_a_child_map_raises(self) -> None:
        root = ContinuousLogicalTimeline(length=16)
        child = root.create_child(length=4)
        child.add_conversion_map(ScalarMap(2, uid="tempo"))
        with pytest.raises(
            ValueError,
            match="already held by conversion map 'tempo' of timeline 'clt_2'",
        ):
            root.add_conversion_map(ScalarMap(3, uid="tempo"))
        assert root.n_conversion_maps == 0


def _free_bundle() -> AlignmentBundle:
    """A manifest whose every kind of element carries a free id."""
    score = ContinuousLogicalTimeline(length=16, uid="score/1")
    score.add_child(ContinuousLogicalTimeline(length=8, uid="part 2"), 0)
    score.create_region(0, 4, id="intro: A/B")
    score.add_conversion_map(ScalarMap(2, uid="urn:tempo"))
    score.add_measure_map(
        MeasureMap(_bars().measures, id="https://example.org/op35/bars")
    )
    score.flow_control.add_break(Break(_q(8), id="fine / end"))
    score.flow_control.add_marker(Marker(_q(4), id="segno 1", control_type=SEGNO))
    score.flow_control.add_jump(Jump(_q(12), _q(4), id="D.S./1", target="segno 1"))
    perf = ContinuousPhysicalTimeline(length=20.0, uid="https://example.org/take#1")
    bundle = AlignmentBundle(id="https://example.org/op35")
    bundle.add_timeline(score)
    bundle.add_timeline(perf, as_group="score and take")
    return bundle


def _bars() -> MeasureMap:
    return MeasureMap.from_rows([{"count": 1, "time_signature": "4/4"}])


class TestFreeIds:
    """Slashes, spaces, colons and URIs are ids, through a manifest round trip."""

    def test_every_kind_keeps_its_free_id(self) -> None:
        payload = _free_bundle().to_dict()
        text = json.dumps(payload)
        read = AlignmentBundle.from_dict(json.loads(text))

        assert read.id == "https://example.org/op35"
        score = read.timelines["score/1"]
        assert score.list_children() == ["part 2"]
        assert score.list_regions() == ["intro: A/B"]
        assert sorted(cmap.id for cmap in score._all_conversion_maps()) == [
            "cmap_1",
            "urn:tempo",
        ]
        assert score.measure_map.id == "https://example.org/op35/bars"
        assert [e.id for e in score.flow_control.breaks] == ["fine / end"]
        assert [m.id for m in score.flow_control.markers] == ["segno 1"]
        assert [(j.id, j.target) for j in score.flow_control.jumps] == [
            ("D.S./1", "segno 1")
        ]
        assert sorted(read.timelines) == [
            "https://example.org/take#1",
            "score/1",
        ]
        assert payload["measure_maps"][0]["id"] == "https://example.org/op35/bars"
        assert read.to_dict() == payload
        assert json.dumps(read.to_dict()) == text


def _root_pair(
    score: dict[str, Any], perf: dict[str, Any], **keys: Any
) -> dict[str, Any]:
    """A manifest of two quarters roots ``score`` and ``perf`` of length 16."""
    return {
        "timetoshare": "0.1.0",
        **keys,
        "timelines": [
            {"id": "score", "unit": "quarters", "length": [16, 1], **score},
            {"id": "perf", "unit": "quarters", "length": [16, 1], **perf},
        ],
    }


def _region(region_id: str) -> dict[str, Any]:
    return {"id": region_id, "start": [0, 1], "end": [4, 1]}


def _scalar(map_id: str) -> dict[str, Any]:
    return {
        "type": "LinearMap",
        "id": map_id,
        "source_unit": "quarters",
        "target_unit": "seconds",
        "scalar": 2,
    }


@pytest.mark.parametrize(
    ("data", "holders"),
    [
        (
            _root_pair(
                {"regions": [_region("intro")]}, {"regions": [_region("intro")]}
            ),
            (
                "intro",
                "region 'intro' of timeline 'score'",
                "region 'intro' of timeline 'perf'",
            ),
        ),
        (
            _root_pair(
                {"conversion_maps": [_scalar("m")]},
                {
                    "flow_control": [
                        {
                            "id": "m",
                            "type": "flow_marker",
                            "manifestation": "fine",
                            "coordinate": [4, 1],
                        }
                    ]
                },
            ),
            (
                "m",
                "conversion map 'm' of timeline 'score'",
                "flow_marker 'm' of timeline 'perf'",
            ),
        ),
        (
            _root_pair(
                {
                    "children": [
                        {
                            "id": "perf",
                            "unit": "quarters",
                            "offset": [0, 1],
                            "length": [4, 1],
                        }
                    ]
                },
                {},
            ),
            ("perf", "timeline 'perf'", "timeline 'perf'"),
        ),
        (
            _root_pair(
                {},
                {"regions": [_region("g")]},
                groups=[{"id": "g", "members": ["score"]}],
            ),
            ("g", "region 'g' of timeline 'perf'", "group 'g'"),
        ),
        (
            _root_pair(
                {},
                {
                    "flow_control": [
                        {
                            "id": "bars",
                            "type": "jump_to",
                            "manifestation": "segno",
                            "coordinate": [4, 1],
                        }
                    ]
                },
                measure_maps=[
                    {"id": "bars", "rows": [{"count": 1, "time_signature": "4/4"}]}
                ],
            ),
            ("bars", "measure map 'bars'", "jump_to 'bars' of timeline 'perf'"),
        ),
        (
            _root_pair(
                {
                    "conversion_maps": [
                        {
                            "type": "ChainMap",
                            "id": "chain",
                            "source_unit": "quarters",
                            "target_unit": "minutes",
                            "maps": [
                                _scalar("op35"),
                                {
                                    "type": "LinearMap",
                                    "id": "minutes",
                                    "source_unit": "seconds",
                                    "target_unit": "minutes",
                                    "scalar": [1, 60],
                                },
                            ],
                        }
                    ]
                },
                {},
                id="op35",
            ),
            (
                "op35",
                "the manifest",
                "conversion map 'op35' nested in conversion map 'chain' of timeline 'score'",
            ),
        ),
    ],
    ids=[
        "regions",
        "map_and_break",
        "child_and_root",
        "group",
        "measure_map",
        "manifest",
    ],
)
def test_an_id_held_twice_in_a_manifest_is_refused(
    data: dict[str, Any], holders: tuple[str, str, str]
) -> None:
    element_id, first, second = holders
    for root in data["timelines"]:
        alone = {"timetoshare": "0.1.0", "timelines": [root]}
        assert Timeline.from_dict(alone).id == root["id"]
    with pytest.raises(ValueError) as refused:
        AlignmentBundle.from_dict(data)
    assert str(refused.value) == (
        f"the id '{element_id}' is held twice in the manifest: by {first} and by "
        f"{second}"
    )


def _joining(uid: str) -> ContinuousLogicalTimeline:
    """A root of given id holding a minted region, map and marker, and a jump."""
    timeline = ContinuousLogicalTimeline(length=16, uid=uid)
    timeline.create_region(0, 4)
    timeline.add_conversion_map(LinearMap(scalar=2))
    marker = timeline.flow_control.add_marker(Marker(_q(2), control_type=SEGNO))
    timeline.flow_control.add_jump(Jump(_q(6), _q(2), id=f"ds-{uid}", target=marker.id))
    return timeline


def _held(timeline: Timeline) -> tuple[list[str], list[str], list[str], list[Any]]:
    return (
        timeline.list_regions(),
        [cmap.id for cmap in timeline._all_conversion_maps()],
        [marker.id for marker in timeline.flow_control.markers],
        [jump.target for jump in timeline.flow_control.jumps],
    )


def _group_of(*timelines: Timeline) -> TimelineGroup:
    return TimelineGroup(timelines=list(timelines))


def _bundle_of(*timelines: Timeline) -> AlignmentBundle:
    bundle = AlignmentBundle()
    for timeline in timelines:
        bundle.add_timeline(timeline)
    return bundle


SCOPES = pytest.mark.parametrize(
    ("join", "scope"),
    [(_group_of, "group 'group_1'"), (_bundle_of, "bundle 'bundle_1'")],
    ids=["group", "bundle"],
)


class TestJoiningScopes:
    """Trees joining a group or a bundle share one scope, every kind of element."""

    @SCOPES
    def test_minted_ids_of_every_kind_are_minted_again(
        self, join: Callable[..., Any], scope: str
    ) -> None:
        first, second = _joining("a"), _joining("b")
        join(first, second)

        assert _held(first) == (["region_1"], ["lmap_1"], ["jump_to_1"], ["jump_to_1"])
        assert _held(second) == (["region_2"], ["lmap_2"], ["jump_to_2"], ["jump_to_2"])

    @SCOPES
    def test_a_given_id_held_in_the_scope_is_refused_atomically(
        self, join: Callable[..., Any], scope: str
    ) -> None:
        first, second = _joining("a"), _joining("b")
        first.add_conversion_map(ScalarMap(3, uid="intro"))
        second.create_region(4, 8, id="intro")
        container = join(first)

        with pytest.raises(ValueError) as refused:
            (container.add_timeline)(second)
        assert str(refused.value) == (
            f"region 'intro' of timeline 'b' cannot join {scope}: its id 'intro' is "
            "already held by conversion map 'intro' of timeline 'a'"
        )
        assert second.list_regions() == ["region_1", "intro"]
        assert _held(second)[1:] == (["lmap_1"], ["jump_to_1"], ["jump_to_1"])
        members = (
            container.timeline_ids
            if isinstance(container, TimelineGroup)
            else sorted(container.timelines)
        )
        assert members == ["a"]

    def test_a_given_bundle_id_is_the_manifest_s(self) -> None:
        bundle = AlignmentBundle(id="study")
        with pytest.raises(ValueError) as refused:
            bundle.add_timeline(ContinuousLogicalTimeline(length=4, uid="study"))
        assert str(refused.value) == (
            "timeline 'study' cannot join bundle 'study': its id 'study' is already "
            "held by bundle 'study'"
        )
        assert bundle.timelines == {}


def test_a_given_group_id_held_by_a_region_is_refused() -> None:
    bundle = AlignmentBundle()
    bundle.add_timeline(_joining("a"))
    bundle.timelines["a"].create_region(4, 8, id="g")
    group = TimelineGroup(id="g", timelines=[ContinuousLogicalTimeline(length=4)])

    with pytest.raises(ValueError) as refused:
        bundle.add_group(group)
    assert str(refused.value) == (
        "group 'g' cannot join bundle 'bundle_1': its id 'g' is already held by "
        "region 'g' of timeline 'a'"
    )
    assert bundle.groups == {}


class TestLivingScopes:
    """While a group or a bundle holds trees, their elements take ids together."""

    @pytest.mark.parametrize("join", [_group_of, _bundle_of], ids=["group", "bundle"])
    @pytest.mark.parametrize(
        ("add", "element"),
        [
            (
                lambda timeline: timeline.add_region(Region(_q(0), _q(4), id="intro")),
                "region",
            ),
            (
                lambda timeline: timeline.add_conversion_map(ScalarMap(2, uid="intro")),
                "conversion map",
            ),
            (
                lambda timeline: timeline.flow_control.add_break(
                    Break(_q(4), id="intro")
                ),
                "flow_marker",
            ),
        ],
        ids=["region", "map", "break"],
    )
    def test_an_element_added_later_takes_its_id_in_the_scope(
        self,
        join: Callable[..., Any],
        add: Callable[[Timeline], Any],
        element: str,
    ) -> None:
        first, second = _joining("a"), ContinuousLogicalTimeline(length=16, uid="b")
        first.create_region(4, 8, id="intro")
        container = join(first, second)

        assert second.create_region(0, 4).id == "region_2"
        with pytest.raises(ValueError) as refused:
            add(second)
        assert str(refused.value) == (
            f"{element} 'intro' of timeline 'b' cannot join: its id 'intro' is "
            "already held by region 'intro' of timeline 'a'"
        )
        assert second.list_regions() == ["region_2"]
        assert second.n_conversion_maps == 0
        assert second.flow_control.breaks == []
        assert container is not None

    def test_a_removed_member_leaves_the_scope(self) -> None:
        first, second = _joining("a"), ContinuousLogicalTimeline(length=16, uid="b")
        group = _group_of(first, second)
        group.remove_timeline("b")

        assert second.create_region(0, 4).id == "region_1"

    def test_a_dropped_group_leaves_no_scope(self) -> None:
        first, second = _joining("a"), ContinuousLogicalTimeline(length=16, uid="b")
        group = _group_of(first, second)
        del group
        gc.collect()

        assert second.create_region(0, 4).id == "region_1"

    def test_a_pickled_group_keeps_its_scope(self) -> None:
        group = _group_of(_joining("a"), ContinuousLogicalTimeline(length=16, uid="b"))
        restored = pickle.loads(pickle.dumps(group))

        assert restored.get_timeline("b").create_region(0, 4).id == "region_2"

    def test_a_given_id_mints_a_minted_group_id_again(self) -> None:
        first, second = _joining("a"), ContinuousLogicalTimeline(length=16, uid="b")
        group = _group_of(first, second)
        bundle = AlignmentBundle()
        bundle.add_group(group)

        assert second.create_region(0, 4, id="group_1").id == "group_1"
        assert group.id == "group_2"
        assert list(bundle.groups) == ["group_2"]


def _holding_region(uid: str, region_id: str | None) -> ContinuousLogicalTimeline:
    timeline = ContinuousLogicalTimeline(length=16, uid=uid)
    timeline.create_region(0, 4, id=region_id)
    return timeline


def _group_behind_a_bundle(
    region_id: str | None,
) -> tuple[AlignmentBundle, TimelineGroup, ContinuousLogicalTimeline]:
    """A bundle holding ``d``, then a group of ``a``, ``b``, and the newcomer ``c``."""
    bundle = AlignmentBundle()
    bundle.add_timeline(_holding_region("d", region_id))
    group = TimelineGroup(
        timelines=[
            ContinuousLogicalTimeline(length=16, uid="a"),
            ContinuousLogicalTimeline(length=16, uid="b"),
        ]
    )
    bundle.add_group(group)
    return bundle, group, _holding_region("c", region_id)


def _empty_group_behind_a_bundle(
    region_id: str | None,
) -> tuple[AlignmentBundle, TimelineGroup, ContinuousLogicalTimeline]:
    """A bundle holding ``d`` and an empty group, and the newcomer ``c``."""
    bundle = AlignmentBundle()
    bundle.add_timeline(_holding_region("d", region_id))
    group = TimelineGroup()
    bundle.add_group(group)
    return bundle, group, _holding_region("c", region_id)


_REFUSED_C = (
    "region 'r' of timeline 'c' cannot join group 'group_1': its id 'r' is "
    "already held by region 'r' of timeline 'd'"
)


class TestGroupBehindABundle:
    """A timeline added to a group joins every bundle holding the group."""

    def test_a_given_id_held_in_the_bundle_is_refused_atomically(self) -> None:
        bundle, group, newcomer = _group_behind_a_bundle("r")

        with pytest.raises(ValueError) as refused:
            group.add_timeline(newcomer)
        assert str(refused.value) == _REFUSED_C
        assert group.timeline_ids == ["a", "b"]
        assert list(bundle.timelines) == ["d", "a", "b"]
        assert newcomer.list_regions() == ["r"]
        assert newcomer.create_region(4, 8, id="d").id == "d"

    def test_a_minted_id_is_minted_again_and_the_bundle_writes_the_newcomer(
        self,
    ) -> None:
        bundle, group, newcomer = _group_behind_a_bundle(None)
        group.add_timeline(newcomer)

        assert newcomer.list_regions() == ["region_2"]
        assert bundle.timelines["d"].list_regions() == ["region_1"]
        assert bundle.timelines["c"] is newcomer
        data = bundle.to_dict()
        assert [timeline["id"] for timeline in data["timelines"]] == [
            "a",
            "b",
            "c",
            "d",
        ]
        assert data["groups"] == [
            {"id": "group_1", "name": None, "meta": {}, "members": ["a", "b", "c"]}
        ]
        validate(data)

    def test_an_empty_group_refuses_a_given_id_held_in_the_bundle_atomically(
        self,
    ) -> None:
        bundle, group, newcomer = _empty_group_behind_a_bundle("r")

        with pytest.raises(ValueError) as refused:
            group.add_timeline(newcomer)
        assert str(refused.value) == _REFUSED_C
        assert group.timeline_ids == []
        assert list(bundle.timelines) == ["d"]
        assert bundle.groups == {"group_1": group}
        assert newcomer.list_regions() == ["r"]

    def test_an_empty_group_mints_again_and_the_bundle_writes_the_newcomer(
        self,
    ) -> None:
        bundle, group, newcomer = _empty_group_behind_a_bundle(None)
        group.add_timeline(newcomer)

        assert newcomer.list_regions() == ["region_2"]
        assert bundle.timelines["d"].list_regions() == ["region_1"]
        assert bundle.timelines["c"] is newcomer
        data = bundle.to_dict()
        assert [timeline["id"] for timeline in data["timelines"]] == ["c", "d"]
        assert data["groups"] == [
            {"id": "group_1", "name": None, "meta": {}, "members": ["c"]}
        ]
        validate(data)

    def test_an_unpickled_bundle_holds_its_empty_group_again(self) -> None:
        bundle, _group, newcomer = _empty_group_behind_a_bundle("r")
        restored = pickle.loads(pickle.dumps(bundle))

        with pytest.raises(ValueError) as refused:
            restored.groups["group_1"].add_timeline(newcomer)
        assert str(refused.value) == _REFUSED_C
        assert restored.groups["group_1"].timeline_ids == []
        assert list(restored.timelines) == ["d"]

    def test_a_group_the_bundle_no_longer_holds_does_not_consult_it(self) -> None:
        bundle, group, newcomer = _empty_group_behind_a_bundle("r")
        bundle.groups.pop("group_1")
        group.add_timeline(newcomer)

        assert group.timeline_ids == ["c"]
        assert newcomer.list_regions() == ["r"]
        assert list(bundle.timelines) == ["d"]

    def test_a_collected_bundle_is_not_consulted(self) -> None:
        bundle, group, newcomer = _empty_group_behind_a_bundle("r")
        del bundle
        gc.collect()
        group.add_timeline(newcomer)

        assert group.timeline_ids == ["c"]
        assert newcomer.list_regions() == ["r"]


def test_an_id_is_named_in_single_quotes() -> None:
    timeline = _holding_region("t", "Bach's")

    with pytest.raises(ValueError) as refused:
        timeline.create_region(4, 8, id="Bach's")
    assert str(refused.value) == (
        "region 'Bach's' of timeline 't' cannot join: its id 'Bach's' is already "
        "held by region 'Bach's' of timeline 't'"
    )


def test_a_written_jump_to_takes_an_id_of_the_whole_manifest() -> None:
    bundle = AlignmentBundle()
    for uid in ("score", "perf"):
        timeline = ContinuousLogicalTimeline(length=16, uid=uid)
        timeline.flow_control.add_jump(
            Jump(_q(12), _q(4), id=f"ds-{uid}", control_type=DAL_SEGNO)
        )
        bundle.add_timeline(timeline)

    written = {
        timeline["id"]: [
            (element["id"], element.get("target"))
            for element in timeline["flow_control"]
        ]
        for timeline in bundle.to_dict()["timelines"]
    }
    assert written == {
        "perf": [("jump_to_1", None), ("ds-perf", "jump_to_1")],
        "score": [("jump_to_2", None), ("ds-score", "jump_to_2")],
    }
    assert [jump.target for jump in bundle.timelines["score"].flow_control.jumps] == [
        None
    ]


QUARTERS_TO_SECONDS = {"source_unit": "quarters", "target_unit": "seconds", "scalar": 2}
SECONDS_TO_MINUTES = {
    "source_unit": "seconds",
    "target_unit": "minutes",
    "scalar": [1, 60],
}


def _seconds_to_minutes() -> ChainMap:
    return ChainMap(
        [
            LinearMap(scalar=2, source_unit="quarters", target_unit="seconds"),
            LinearMap(scalar=F(1, 60), source_unit="seconds", target_unit="minutes"),
        ]
    )


def _measure_and_section() -> CombinationMap:
    record = ChainMap(
        [
            LinearMap(scalar=4, source_unit="whole_notes", target_unit="quarters"),
            MeasureRecordMap(
                MeasureMap.from_rows([{"count": 1, "time_signature": "4/4"}])
            ),
        ]
    )
    section = ConstantMap(value="verse", source_unit="whole_notes")
    return CombinationMap({"measure": record, "section": section})


def _map_ids(cmap: dict[str, Any]) -> Any:
    """The ids of a written map, each nested map's under its own."""
    nested = cmap.get("maps")
    if nested is None:
        return cmap["id"]
    items = nested.values() if isinstance(nested, dict) else nested
    return (cmap["id"], [_map_ids(item) for item in items])


class TestNestedMaps:
    """Maps nested in maps hold ids of their own, unique in the manifest."""

    def test_a_chain_of_two_unnamed_maps_writes_two_ids(self) -> None:
        assert _map_ids(_seconds_to_minutes().to_dict()) == (
            "cmap_1",
            ["lmap_1", "lmap_2"],
        )

    def test_a_nested_minted_id_taken_in_the_manifest_is_minted_again(self) -> None:
        timeline = ContinuousLogicalTimeline(length=16)
        timeline.add_conversion_map(
            LinearMap(scalar=3, source_unit="quarters", target_unit="seconds")
        )
        timeline.add_conversion_map(_seconds_to_minutes())
        written = timeline.to_dict()["timelines"][0]["conversion_maps"]
        assert [_map_ids(cmap) for cmap in written] == [
            ("cmap_1", ["lmap_3", "lmap_2"]),
            "lmap_1",
        ]

    def test_a_record_map_in_a_chain_in_a_combination(self) -> None:
        combination = _measure_and_section()
        assert _map_ids(combination.to_dict()) == (
            "cmap_1",
            [("cmap_3", ["lmap_1", "cmap_2"]), "cmap_4"],
        )
        timeline = ContinuousLogicalTimeline(length=4, unit="whole_notes")
        timeline.add_conversion_map(ConstantMap(value="a", source_unit="whole_notes"))
        timeline.add_conversion_map(combination)
        payload = timeline.to_dict()
        assert [
            _map_ids(cmap) for cmap in payload["timelines"][0]["conversion_maps"]
        ] == [
            "cmap_1",
            ("cmap_5", [("cmap_3", ["lmap_1", "cmap_2"]), "cmap_4"]),
        ]
        text = json.dumps(payload)
        assert json.dumps(Timeline.from_dict(json.loads(text)).to_dict()) == text

    def test_the_constant_maps_of_an_interval_map_with_an_end(self) -> None:
        cmap = IntervalToConstantMap(
            boundaries=[0, 4], values=["a", "b"], end=8, source_unit="quarters"
        )
        assert _map_ids(cmap.to_dict()) == ("cmap_1", ["cmap_2", "cmap_3"])

    def test_a_joining_child_s_nested_map_is_minted_again(self) -> None:
        root = ContinuousLogicalTimeline(length=16)
        root.add_conversion_map(
            LinearMap(scalar=3, source_unit="quarters", target_unit="seconds")
        )
        child = ContinuousLogicalTimeline(length=8)
        chain = _seconds_to_minutes()
        child.add_conversion_map(chain)
        root.add_child(child, 0)
        assert [cmap.id for cmap in chain.maps] == ["lmap_3", "lmap_2"]
        assert list(child._conversion_maps) == ["cmap_1"]

    def test_two_nested_maps_given_one_id_raise(self) -> None:
        with pytest.raises(
            ValueError,
            match=(
                "conversion map 'a' nested in conversion map 'cmap_1' cannot join: "
                "its id 'a' is already held by conversion map 'a' nested in "
                "conversion map 'cmap_1'"
            ),
        ):
            ChainMap([LinearMap(scalar=2, uid="a"), LinearMap(scalar=3, uid="a")])

    def test_a_nested_given_id_held_in_the_manifest_raises(self) -> None:
        timeline = ContinuousLogicalTimeline(length=16)
        timeline.add_conversion_map(ScalarMap(2, uid="tempo"))
        chain = ChainMap([LinearMap(scalar=2), LinearMap(scalar=3, uid="tempo")])
        with pytest.raises(
            ValueError,
            match="already held by conversion map 'tempo' of timeline 'clt_1'",
        ):
            timeline.add_conversion_map(chain)
        assert list(timeline._conversion_maps) == ["tempo"]
        assert (chain.id, [cmap.id for cmap in chain.maps]) == (
            "cmap_1",
            ["lmap_1", "tempo"],
        )

    def test_one_map_nested_in_two_places_raises(self) -> None:
        twice = LinearMap(scalar=2)
        with pytest.raises(ValueError, match="nests one map in two places"):
            ChainMap([twice, twice])
        assert twice.id == "lmap_1"

    def test_reading_refuses_a_nested_id_repeated(self) -> None:
        payload = ContinuousLogicalTimeline(length=16, uid="piece").to_dict()
        payload["timelines"][0]["conversion_maps"] = [
            {
                "type": "ChainMap",
                "id": "cmap_1",
                "maps": [
                    {"type": "LinearMap", "id": "lmap_1", **QUARTERS_TO_SECONDS},
                    {"type": "LinearMap", "id": "lmap_1", **SECONDS_TO_MINUTES},
                ],
            }
        ]
        with pytest.raises(
            ValueError,
            match="its id 'lmap_1' is already held by conversion map 'lmap_1'",
        ):
            Timeline.from_dict(payload)


# endregion

# region Regions and markers by id


class TestKeyedById:
    """The region API takes ids; jumps target markers by id."""

    def test_regions_with_one_name_coexist(self) -> None:
        timeline = ContinuousLogicalTimeline(length=32)
        timeline.create_region(0, 8, name="Chorus")
        timeline.create_region(16, 24, name="Chorus")
        assert timeline.list_regions() == ["region_1", "region_2"]
        assert timeline.has_region("region_2")
        assert not timeline.has_region("Chorus")
        assert timeline.get_region("region_2").name == "Chorus"
        with pytest.raises(KeyError, match=r"\['region_1', 'region_2'\]"):
            timeline.get_region("Chorus")

    def test_children_from_regions_are_named_after_them(self) -> None:
        timeline = ContinuousLogicalTimeline(length=32)
        timeline.create_region(0, 8, name="Chorus")
        timeline.create_region(16, 24, name="Verse")
        child = timeline.create_child_from_region("region_1")
        assert (child.id, child.name) == ("clt_2", "Chorus")
        children = timeline.create_children_from_regions(["region_2"])
        assert [(c.id, c.name) for c in children] == [("clt_3", "Verse")]

    def test_jump_target_must_name_a_marker(self) -> None:
        timeline = ContinuousLogicalTimeline(length=32)
        with pytest.raises(ValueError, match="names no marker"):
            timeline.flow_control.add_jump(Jump(_q(8), _q(0), target="segno"))
        assert timeline.flow_control.jumps == []

    def test_dal_segno_targets_the_segno_marker_id(self) -> None:
        timeline = (
            Ms3Loader()
            .load(SCORE / "couperin_concerts/c05n05_musete.measures.tsv")
            .create_timeline()
        )
        flow = timeline.flow_control
        assert [(m.id, m.control_type, m.name, m.coordinate) for m in flow.markers] == [
            ("jump_to_1", SEGNO, "segno", _q(12)),
        ]
        assert [(b.control_type, b.coordinate) for b in flow.breaks] == [
            (FlowControlElement.fine, _q(42))
        ]
        dal_segno = [j for j in flow.jumps if j.target is not None]
        assert [(j.control_type.value, j.target) for j in dal_segno] == [
            ("dal_segno_al_fine", "jump_to_1"),
            ("dal_segno_al_fine", "jump_to_1"),
        ]

    def test_jumps_without_a_marker_target_hold_none(self) -> None:
        timeline = (
            Ms3Loader()
            .load(SCORE / "couperin_concerts/c11n08_Rondeau.measures.tsv")
            .create_timeline()
        )
        assert {j.target for j in timeline.flow_control.jumps} == {None}


# endregion

# region Loader parity


def _elements(payload: dict[str, Any]) -> Iterator[str]:
    """Yield every element id of one serialized timeline and its descendants."""
    yield payload["id"]
    for entry in payload["conversion_maps"]:
        yield entry["id"]
    for entry in payload["regions"]:
        yield entry["id"]
    for entry in payload["flow_control"]:
        yield entry["id"]
    for child in payload["children"]:
        yield from _elements(child)


LOADERS: list[tuple[str, Callable[[], Timeline]]] = [
    (
        "ms3",
        lambda: Ms3Loader(auto_discover=True)
        .load(
            SCORE
            / "flow_control/flow_only/out_of_the_flow_experience-flow_only.notes.tsv"
        )
        .create_timeline(),
    ),
    (
        "ms3_dal_segno",
        lambda: Ms3Loader()
        .load(SCORE / "couperin_concerts/c05n05_musete.measures.tsv")
        .create_timeline(),
    ),
    (
        "midi",
        lambda: ScoreMidiLoader()
        .load(MIDI / "score/beethoven_mtd.mid")
        .create_timeline(),
    ),
    (
        "lab",
        lambda: LabLoader().load(FIXTURES / "lab/regions.lab").create_timeline(),
    ),
    (
        "audio",
        lambda: AudioLoader()
        .load(AUDIO / "hard_techno/Bass Kick.mp3")
        .create_timeline(),
    ),
    (
        "parangonada",
        lambda: ParangonadaLoader()
        .load(PARANGONAR / "Beethoven_Eroica_op35-cpjku")
        .create_timelines()[0],
    ),
    (
        "iiif",
        lambda: IIIFManifestLoader()
        .load(SUPRA / "image/ifff_manifest.json")
        .create_timeline(),
    ),
]


class TestLoaderParity:
    """Every loader's manifest carries a valid, unique id on every element."""

    @pytest.mark.parametrize(
        "load", [load for _, load in LOADERS], ids=[name for name, _ in LOADERS]
    )
    def test_every_element_has_a_unique_valid_id(
        self, load: Callable[[], Timeline]
    ) -> None:
        ids = list(_elements(load().to_dict()["timelines"][0]))
        assert ids
        assert len(ids) == len(set(ids))
        for element_id in ids:
            assert isinstance(element_id, str) and element_id
            assert not any(ord(character) < 0x20 for character in element_id)
            assert not element_id.startswith("tl:")


# endregion
