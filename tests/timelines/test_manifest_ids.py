"""Identity of manifest elements: minted when built, given ids kept, unique per manifest."""

from __future__ import annotations

import gc
import json
import pickle
import re
from collections.abc import Callable, Iterator
from fractions import Fraction as F
from typing import Any

import pytest
from timetoshare import validate

from tests.helpers import bare, filed, refusal
from timetoalign.alignment import AlignmentBundle
from timetoalign.core import Coordinate, FlowManifestation, TimeUnit
from timetoalign.core.ids import IdMinter, reset_content_minters
from timetoalign.loader.alignment import ParangonadaLoader
from timetoalign.loader.graphical.iiif import IIIFManifestLoader
from timetoalign.loader.graphical.paths import VerticalLinePath
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
    MetricMap,
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
from timetoalign.timelines import groups as groups_module
from timetoalign.timelines.flow import FlowMap
from timetoalign.timelines.flowcontrol import (
    Barline,
    Break,
    Jump,
    JumpDestination,
    JumpTarget,
    SectionBreak,
)
from timetoalign.timelines.manifest import _manifest_ids_of
from timetoalign.timelines.regions import Region

SEGNO = FlowManifestation.segno

SCORE = ensure_data("score")
MIDI = ensure_data("midi")
AUDIO = ensure_data("audio")
PARANGONAR = ensure_data("parangonar")
SUPRA = ensure_data("supra")
FIXTURES = ensure_data("fixtures")

QUARTERS = TimeUnit.quarters


def _q(value: Any) -> Coordinate:
    return Coordinate(value, QUARTERS)


def _jump(at: Any, *, id: str | None = None, **place: Any) -> Jump:
    """A jump at *at* with one destination, ``to`` an id or ``by`` a distance."""
    return Jump(_q(at), [JumpDestination(**place)], id=id)


# region Minted when built


def _bars(**keys: Any) -> MeasureMap:
    return MeasureMap.from_rows([{"count": 1, "time_signature": "4/4"}], **keys)


# Each element built without an id, by its prefix; every builder takes the seed.
BUILDS: list[tuple[str, str, Callable[..., Any]]] = [
    ("clt", "clt", lambda **k: ContinuousLogicalTimeline(**k)),
    ("dlt", "dlt", lambda **k: DiscreteLogicalTimeline(**k)),
    ("cpt", "cpt", lambda **k: ContinuousPhysicalTimeline(**k)),
    ("dpt", "dpt", lambda **k: DiscretePhysicalTimeline(**k)),
    ("cgt", "cgt", lambda **k: ContinuousGraphicalTimeline(**k)),
    ("dgt", "dgt", lambda **k: DiscreteGraphicalTimeline(**k)),
    ("timeline_seconds", "cpt", lambda **k: Timeline(unit="seconds", **k)),
    ("timeline_ticks", "dlt", lambda **k: Timeline(unit="ticks", **k)),
    ("timeline_pixels", "dgt", lambda **k: Timeline(unit="pixels", **k)),
    ("segment_line", "clt", lambda **k: SegmentLine[ContinuousLogicalTimeline](**k)),
    ("linear", "cmap", lambda **k: LinearMap(scalar=2, **k)),
    ("table", "cmap", lambda **k: TableMap(x_values=[0, 4], y_values=[0.0, 2.5], **k)),
    (
        "floating",
        "cmap",
        lambda **k: QuartersToFloatingMeasures(
            x_values=[0, 4], y_values=[1.0, 2.0], **k
        ),
    ),
    (
        "piecewise",
        "cmap",
        lambda **k: PiecewiseMap(breaks=[0, 8], maps=[ShiftMap(offset=1)], **k),
    ),
    ("scalar", "cmap", lambda **k: ScalarMap(2, **k)),
    ("measure_map", "measure_map", lambda **k: _bars(**k)),
    ("region", "region", lambda **k: Region(_q(0), _q(4), **k)),
    ("break", "break", lambda **k: Break(_q(4), **k)),
    ("section_break", "section_break", lambda **k: SectionBreak(_q(4), **k)),
    ("barline", "barline", lambda **k: Barline(_q(4), **k)),
    ("jump", "jump", lambda **k: Jump(_q(4), [JumpDestination(by=-4)], **k)),
    (
        "jump_target",
        "jump_target",
        lambda **k: JumpTarget(_q(0), manifestation=SEGNO, **k),
    ),
    ("flow_map", "flow", lambda **k: FlowMap("score", ("a",), **k)),
    ("group", "group", lambda **k: TimelineGroup(**k)),
    ("bundle", "bundle", lambda **k: AlignmentBundle(**k)),
]
BUILD_IDS = [name for name, _, _ in BUILDS]
EACH = pytest.mark.parametrize(
    ("prefix", "build"), [(prefix, build) for _, prefix, build in BUILDS], ids=BUILD_IDS
)


@EACH
def test_an_element_built_without_an_id_mints_one_with_its_prefix(
    prefix: str, build: Callable[..., Any]
) -> None:
    assert re.fullmatch(rf"{prefix}_[0-9a-z]{{13}}", build().id)


@EACH
def test_a_seed_decides_the_id(prefix: str, build: Callable[..., Any]) -> None:
    ContinuousLogicalTimeline(length=99)
    assert build(seed=7).id == IdMinter(7).mint(prefix)
    assert build(seed=7).id == IdMinter(7).mint(prefix)


@EACH
def test_content_seeded_ids_are_reproducible(
    prefix: str, build: Callable[..., Any]
) -> None:
    reset_content_minters()
    first = build().id
    reset_content_minters()
    ContinuousLogicalTimeline(length=99)
    LinearMap(scalar=99)
    assert build().id == first


@EACH
def test_equal_content_built_twice_holds_two_ids(
    prefix: str, build: Callable[..., Any]
) -> None:
    assert build().id != build().id


def test_a_seed_mints_several_ids_in_order() -> None:
    timeline = ContinuousLogicalTimeline(length=12)
    regions = timeline.create_regions_from_boundaries([0, 4, 8, 12], seed=7)
    minter = IdMinter(7)
    assert [region.id for region in regions] == [
        minter.mint("region") for _ in range(3)
    ]


def _reference(*, reverse: bool = False) -> Timeline:
    """Build the reference manifest, its collections in either order."""
    root = ContinuousLogicalTimeline(length=32)
    steps: list[Callable[[], Any]] = [
        lambda: root.create_child(length=8, offset=0),
        lambda: root.create_child(length=8, offset=8),
        lambda: root.create_region(0, 8, name="A"),
        lambda: root.create_region(8, 16, name="B"),
        lambda: root.flow_control.add(Break(_q(8))),
        lambda: root.flow_control.add(Break(_q(16))),
        lambda: root.add_conversion_map(
            LinearMap(scalar=2, source_unit="quarters", target_unit="seconds")
        ),
    ]
    for step in reversed(steps) if reverse else steps:
        step()
    marker = root.flow_control.add(JumpTarget(_q(8), name="segno", manifestation=SEGNO))
    root.flow_control.add(_jump(24, to=marker.id))
    return root


def _ids(manifest: dict[str, Any]) -> dict[str, Any]:
    (payload,) = manifest["timelines"]
    flow = payload["flow_control"]
    return {
        "timeline": payload["id"],
        "children": {child["id"] for child in payload["children"]},
        "regions": {entry["id"] for entry in payload["regions"]},
        "breaks": {entry["id"] for entry in flow if entry["role"] == "break"},
        "jump_targets": [
            entry["id"] for entry in flow if entry["role"] == "jump_target"
        ],
        "jumps": [
            (entry["id"], [destination.get("to") for destination in entry["targets"]])
            for entry in flow
            if entry["role"] == "jump"
        ],
        "maps": [entry["id"] for entry in payload["conversion_maps"]],
    }


class TestDeterminism:
    """Minted ids depend on what is built, never on unrelated process history."""

    def test_one_build_writes_one_manifest(self) -> None:
        reset_content_minters()
        reference = bare(_reference().to_dict())
        reset_content_minters()
        for _ in range(5):
            ContinuousLogicalTimeline(length=4)
        for _ in range(4):
            LinearMap(scalar=3)
        assert bare(_reference().to_dict()) == reference

    def test_reverse_insertion_order_holds_the_same_ids(self) -> None:
        reset_content_minters()
        reference = _ids(bare(_reference().to_dict()))
        reset_content_minters()
        assert _ids(bare(_reference(reverse=True).to_dict())) == reference

    def test_a_given_id_is_never_parsed(self) -> None:
        root = ContinuousLogicalTimeline(length=32)
        root.add_child(ContinuousLogicalTimeline(length=4, uid="clt_3"), 0)
        child = root.create_child(length=4, offset=4)
        assert child.id != "clt_4"
        assert re.fullmatch(r"clt_[0-9a-z]{13}", child.id)

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


class TestNothingMintedWhileWriting:
    """An object written twice gives one manifest twice."""

    def test_the_reference_manifest(self) -> None:
        root = _reference()
        assert root.to_dict() == root.to_dict()

    def test_a_measure_view_names_the_measure_map_it_holds(self) -> None:
        meter = MetricMap.from_uniform(4, 4, seed=7)
        timeline = ContinuousLogicalTimeline(length=16, uid="score")
        timeline.add_conversion_map(meter)
        first = bare(timeline.to_dict())
        minter = IdMinter(7)
        assert meter.id == minter.mint("cmap")
        measure_map_id = minter.mint("measure_map")
        assert [entry["id"] for entry in first["measure_maps"]] == [measure_map_id]
        (written,) = first["timelines"][0]["conversion_maps"]
        assert written["measure_map"] == measure_map_id
        assert bare(timeline.to_dict()) == first


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
    root.flow_control.add(Break(_q(8), id="b1"))
    root.flow_control.add(
        JumpTarget(_q(8), id="segno", name="segno", manifestation=SEGNO)
    )
    root.flow_control.add(_jump(24, id="ds", to="segno"))
    return root


class TestGivenIds:
    """Whatever id is given is the id, through a JSON round trip."""

    def test_given_ids_are_kept(self) -> None:
        assert _ids(bare(_given_manifest().to_dict())) == {
            "timeline": "performance",
            "children": {"Bass Kick", "score:clt_7"},
            "regions": {"a"},
            "breaks": {"b1"},
            "jump_targets": ["segno"],
            "jumps": [("ds", ["segno"])],
            "maps": ["raw"],
        }

    @pytest.mark.parametrize("build", [_given_manifest, _reference])
    def test_round_trip_is_text_identical(self, build: Callable[[], Timeline]) -> None:
        payload = bare(build().to_dict())
        text = json.dumps(payload)
        restored = Timeline.from_dict(filed(json.loads(text)))
        assert json.dumps(bare(restored.to_dict())) == text

    @pytest.mark.parametrize("refused", ["", "a\tb", "a\nb"])
    @pytest.mark.parametrize(
        "build",
        [
            lambda uid: ContinuousLogicalTimeline(uid=uid),
            lambda uid: ScalarMap(2, uid=uid),
            lambda uid: Region(_q(0), _q(4), id=uid),
            lambda uid: Break(_q(4), id=uid),
            lambda uid: SectionBreak(_q(4), id=uid),
            lambda uid: Barline(_q(4), id=uid),
            lambda uid: _jump(4, id=uid, by=-4),
            lambda uid: JumpTarget(_q(0), id=uid, manifestation=SEGNO),
        ],
        ids=[
            "timeline",
            "map",
            "region",
            "break",
            "section_break",
            "barline",
            "jump",
            "jump_target",
        ],
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


def _seeded(uid: str | None = None, *, timeline: int = 1) -> ContinuousLogicalTimeline:
    """A timeline built with *timeline* as seed, holding a region of seed 2."""
    built = ContinuousLogicalTimeline(length=16, uid=uid, seed=timeline)
    built.create_region(0, 4, name="intro", seed=2)
    return built


class TestCollisions:
    """Ids are unique within one manifest across all kinds; nothing is renamed."""

    def test_a_minted_timeline_id_held_twice_is_refused(self) -> None:
        root, child = _seeded(), _seeded()
        held = child.id
        assert root.id == held
        with pytest.raises(ValueError) as refused:
            root.add_child(child, 0)
        assert str(refused.value) == (
            f"timeline '{held}' cannot join: its id '{held}' is already held by "
            f"timeline '{held}'"
        )
        assert root.list_children() == []
        assert (root.id, child.id) == (held, held)
        assert root.list_regions() == child.list_regions()

    def test_a_minted_region_id_held_twice_is_refused(self) -> None:
        root, child = _seeded(), _seeded(timeline=3)
        (region,) = root.list_regions()
        with pytest.raises(ValueError) as refused:
            root.add_child(child, 0)
        assert str(refused.value) == (
            f"region '{region}' of timeline '{child.id}' cannot join: its id "
            f"'{region}' is already held by region '{region}' of timeline '{root.id}'"
        )
        assert root.list_children() == []
        assert child.list_regions() == [region]

    def test_a_jump_target_held_twice_refuses_the_child(self) -> None:
        root = ContinuousLogicalTimeline(length=16)
        held = root.flow_control.add(JumpTarget(_q(0), manifestation=SEGNO, seed=5))
        child = ContinuousLogicalTimeline(length=8)
        target = child.flow_control.add(JumpTarget(_q(2), name="segno", seed=5))
        child.flow_control.add(_jump(6, to=target.id))
        assert target.id == held.id
        with pytest.raises(ValueError, match="is already held by jump_target"):
            root.add_child(child, 0)
        assert [t.id for t in child.flow_control.jump_targets] == [held.id]
        assert [j.targets[0].to for j in child.flow_control.jumps] == [held.id]
        assert root.list_children() == []

    def test_given_timeline_id_collision_raises(self) -> None:
        root = ContinuousLogicalTimeline(length=16, uid="score")
        with pytest.raises(ValueError) as error:
            root.add_child(ContinuousLogicalTimeline(length=4, uid="score"), 0)
        assert str(error.value) == (
            "timeline 'score' cannot join: its id 'score' is already held by "
            "timeline 'score'"
        )

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
            match=f"already held by conversion map 'tempo' of timeline '{child.id}'",
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
    score.flow_control.add(Break(_q(8), id="fine / end"))
    score.flow_control.add(JumpTarget(_q(4), id="segno 1", manifestation=SEGNO))
    score.flow_control.add(_jump(12, id="D.S./1", to="segno 1"))
    perf = ContinuousPhysicalTimeline(length=20.0, uid="https://example.org/take#1")
    bundle = AlignmentBundle(id="https://example.org/op35")
    bundle.add_timeline(score)
    bundle.add_timeline(perf, as_group="score and take")
    return bundle


class TestFreeIds:
    """Slashes, spaces, colons and URIs are ids, through a manifest round trip."""

    def test_every_kind_keeps_its_free_id(self) -> None:
        bundle = _free_bundle()
        map_ids = sorted(
            cmap.id for cmap in bundle.timelines["score/1"]._all_conversion_maps()
        )
        payload = bare(bundle.to_dict())
        text = json.dumps(payload)
        read = AlignmentBundle.from_dict(filed(json.loads(text)))

        assert read.id == "https://example.org/op35"
        score = read.timelines["score/1"]
        assert score.list_children() == ["part 2"]
        assert score.list_regions() == ["intro: A/B"]
        assert sorted(cmap.id for cmap in score._all_conversion_maps()) == map_ids
        assert "urn:tempo" in map_ids
        assert score.measure_map.id == "https://example.org/op35/bars"
        assert [e.id for e in score.flow_control.breaks] == ["fine / end"]
        assert [t.id for t in score.flow_control.jump_targets] == ["segno 1"]
        assert [(j.id, j.targets[0].to) for j in score.flow_control.jumps] == [
            ("D.S./1", "segno 1")
        ]
        assert sorted(read.timelines) == [
            "https://example.org/take#1",
            "score/1",
        ]
        assert payload["measure_maps"][0]["id"] == "https://example.org/op35/bars"
        assert bare(read.to_dict()) == payload
        assert json.dumps(bare(read.to_dict())) == text


def _root_pair(
    score: dict[str, Any], perf: dict[str, Any], **keys: Any
) -> dict[str, Any]:
    """A manifest of two quarters roots ``score`` and ``perf`` of length 16."""
    return {
        "timetoshare": "0.2.0",
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
    ("data", "refused_with"),
    [
        (
            _root_pair(
                {"regions": [_region("intro")]}, {"regions": [_region("intro")]}
            ),
            (
                "duplicate_id",
                "#/timelines/1/regions/0/id",
                "'intro' is not unique in the manifest, first at "
                "#/timelines/0/regions/0/id",
            ),
        ),
        (
            _root_pair(
                {"conversion_maps": [_scalar("m")]},
                {
                    "flow_control": [
                        {
                            "id": "m",
                            "role": "break",
                            "manifestation": "fine",
                            "coordinate": [4, 1],
                        }
                    ]
                },
            ),
            (
                "duplicate_id",
                "#/timelines/1/flow_control/0/id",
                "'m' is not unique in the manifest, first at "
                "#/timelines/0/conversion_maps/0/id",
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
            (
                "duplicate_id",
                "#/timelines/1/id",
                "'perf' is not unique in the manifest, first at "
                "#/timelines/0/children/0/id",
            ),
        ),
        (
            _root_pair(
                {},
                {"regions": [_region("g")]},
                groups=[{"id": "g", "members": ["score"]}],
            ),
            (
                "duplicate_id",
                "#/groups/0/id",
                "'g' is not unique in the manifest, first at #/timelines/1/regions/0/id",
            ),
        ),
        (
            _root_pair(
                {},
                {
                    "flow_control": [
                        {
                            "id": "bars",
                            "role": "jump_target",
                            "manifestation": "segno",
                            "coordinate": [4, 1],
                        }
                    ]
                },
                measure_maps=[
                    {"id": "bars", "rows": [{"count": 1, "time_signature": "4/4"}]}
                ],
            ),
            (
                "duplicate_id",
                "#/measure_maps/0/id",
                "'bars' is not unique in the manifest, first at "
                "#/timelines/1/flow_control/0/id",
            ),
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
                "manifest_id",
                "#/id",
                "'op35' is not unique in the manifest, first at "
                "#/timelines/0/conversion_maps/0/maps/0/id",
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
    data: dict[str, Any], refused_with: tuple[str, str, str]
) -> None:
    for root in data["timelines"]:
        alone = {"timetoshare": "0.2.0", "timelines": [root]}
        assert Timeline.from_dict(filed(alone)).id == root["id"]
    with pytest.raises(ValueError) as refused:
        AlignmentBundle.from_dict(filed(data))
    assert refusal(refused.value) == refused_with


def _joining(uid: str, *, seeded: bool = True) -> ContinuousLogicalTimeline:
    """A root of given id holding a region, a map and a jump target (each of
    one seed when *seeded*, so that two such trees collide), and a jump."""
    seeds = (11, 12, 13) if seeded else (None, None, None)
    timeline = ContinuousLogicalTimeline(length=16, uid=uid)
    timeline.create_region(0, 4, seed=seeds[0])
    timeline.add_conversion_map(LinearMap(scalar=2, seed=seeds[1]))
    target = timeline.flow_control.add(
        JumpTarget(_q(2), manifestation=SEGNO, seed=seeds[2])
    )
    timeline.flow_control.add(_jump(6, id=f"ds-{uid}", to=target.id))
    return timeline


def _held(timeline: Timeline) -> tuple[list[str], list[str], list[str], list[Any]]:
    return (
        timeline.list_regions(),
        [cmap.id for cmap in timeline._all_conversion_maps()],
        [target.id for target in timeline.flow_control.jump_targets],
        [jump.targets[0].to for jump in timeline.flow_control.jumps],
    )


def _group_of(*timelines: Timeline) -> TimelineGroup:
    return TimelineGroup(timelines=list(timelines))


def _bundle_of(*timelines: Timeline) -> AlignmentBundle:
    bundle = AlignmentBundle()
    for timeline in timelines:
        bundle.add_timeline(timeline)
    return bundle


def _scope(container: TimelineGroup | AlignmentBundle) -> str:
    kind = "group" if isinstance(container, TimelineGroup) else "bundle"
    return f"{kind} '{container.id}'"


def _members(container: TimelineGroup | AlignmentBundle) -> list[str]:
    if isinstance(container, TimelineGroup):
        return container.timeline_ids
    return list(container.timelines)


SCOPES = pytest.mark.parametrize(
    "join", [_group_of, _bundle_of], ids=["group", "bundle"]
)


class TestJoiningScopes:
    """Trees joining a group or a bundle share one scope, every kind of element."""

    @SCOPES
    def test_ids_of_one_seed_collide_and_are_refused_atomically(
        self, join: Callable[..., Any]
    ) -> None:
        first, second = _joining("a"), _joining("b")
        before = _held(second)
        container = join(first)
        (map_id,) = before[1]

        with pytest.raises(ValueError) as refused:
            container.add_timeline(second)
        assert str(refused.value) == (
            f"conversion map '{map_id}' of timeline 'b' cannot join "
            f"{_scope(container)}: its id '{map_id}' is already held by "
            f"conversion map '{map_id}' of timeline 'a'"
        )
        assert _held(second) == before
        assert _members(container) == ["a"]

    @SCOPES
    def test_trees_built_from_content_join(self, join: Callable[..., Any]) -> None:
        first, second = _joining("a", seeded=False), _joining("b", seeded=False)
        before = (_held(first), _held(second))
        container = join(first, second)

        assert (_held(first), _held(second)) == before
        assert _members(container) == ["a", "b"]

    @SCOPES
    def test_a_given_id_held_in_the_scope_is_refused_atomically(
        self, join: Callable[..., Any]
    ) -> None:
        first, second = _joining("a", seeded=False), _joining("b", seeded=False)
        first.add_conversion_map(ScalarMap(3, uid="intro"))
        second.create_region(4, 8, id="intro")
        before = _held(second)
        container = join(first)

        with pytest.raises(ValueError) as refused:
            container.add_timeline(second)
        assert str(refused.value) == (
            f"region 'intro' of timeline 'b' cannot join {_scope(container)}: its id "
            "'intro' is already held by conversion map 'intro' of timeline 'a'"
        )
        assert _held(second) == before
        assert _members(container) == ["a"]

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
        f"group 'g' cannot join bundle '{bundle.id}': its id 'g' is already held by "
        "region 'g' of timeline 'a'"
    )
    assert bundle.groups == {}


def test_a_minted_group_id_is_never_renamed() -> None:
    second = ContinuousLogicalTimeline(length=16, uid="b")
    group = _group_of(_joining("a"), second)
    bundle = AlignmentBundle()
    bundle.add_group(group)
    group_id = group.id

    with pytest.raises(ValueError) as refused:
        second.create_region(0, 4, id=group_id)
    assert str(refused.value) == (
        f"region '{group_id}' of timeline 'b' cannot join: its id '{group_id}' is "
        f"already held by group '{group_id}'"
    )
    assert group.id == group_id
    assert list(bundle.groups) == [group_id]
    assert second.list_regions() == []


class TestLivingScopes:
    """While a group or a bundle holds trees, their elements are checked together."""

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
                lambda timeline: timeline.flow_control.add(Break(_q(4), id="intro")),
                "break",
            ),
        ],
        ids=["region", "map", "break"],
    )
    def test_an_element_added_later_is_checked_in_the_scope(
        self,
        join: Callable[..., Any],
        add: Callable[[Timeline], Any],
        element: str,
    ) -> None:
        first, second = _joining("a"), ContinuousLogicalTimeline(length=16, uid="b")
        first.create_region(4, 8, id="intro")
        container = join(first, second)

        with pytest.raises(ValueError) as refused:
            add(second)
        assert str(refused.value) == (
            f"{element} 'intro' of timeline 'b' cannot join: its id 'intro' is "
            "already held by region 'intro' of timeline 'a'"
        )
        assert second.list_regions() == []
        assert second.n_conversion_maps == 0
        assert second.flow_control.breaks == ()
        assert container is not None

    @staticmethod
    def _pair() -> tuple[ContinuousLogicalTimeline, ContinuousLogicalTimeline]:
        first, second = _joining("a"), ContinuousLogicalTimeline(length=16, uid="b")
        first.create_region(4, 8, id="intro")
        return first, second

    def test_a_removed_member_leaves_the_scope(self) -> None:
        first, second = self._pair()
        group = _group_of(first, second)
        group.remove_timeline("b")

        assert second.create_region(0, 4, id="intro").id == "intro"

    def test_a_dropped_group_leaves_no_scope(self) -> None:
        first, second = self._pair()
        group = _group_of(first, second)
        del group
        gc.collect()

        assert second.create_region(0, 4, id="intro").id == "intro"

    def test_a_pickled_group_keeps_its_scope(self) -> None:
        restored = pickle.loads(pickle.dumps(_group_of(*self._pair())))

        with pytest.raises(ValueError, match="already held by region 'intro'"):
            restored.get_timeline("b").create_region(0, 4, id="intro")


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


def _refused_c(group: TimelineGroup) -> str:
    return (
        f"region 'r' of timeline 'c' cannot join group '{group.id}': its id 'r' is "
        "already held by region 'r' of timeline 'd'"
    )


class TestGroupBehindABundle:
    """A timeline added to a group joins every bundle holding the group."""

    def test_a_given_id_held_in_the_bundle_is_refused_atomically(self) -> None:
        bundle, group, newcomer = _group_behind_a_bundle("r")

        with pytest.raises(ValueError) as refused:
            group.add_timeline(newcomer)
        assert str(refused.value) == _refused_c(group)
        assert group.timeline_ids == ["a", "b"]
        assert list(bundle.timelines) == ["d", "a", "b"]
        assert newcomer.list_regions() == ["r"]
        assert newcomer.create_region(4, 8, id="d").id == "d"

    def test_minted_ids_join_and_the_bundle_writes_the_newcomer(self) -> None:
        bundle, group, newcomer = _group_behind_a_bundle(None)
        regions = (newcomer.list_regions(), bundle.timelines["d"].list_regions())
        group.add_timeline(newcomer)

        assert (
            newcomer.list_regions(),
            bundle.timelines["d"].list_regions(),
        ) == regions
        assert bundle.timelines["c"] is newcomer
        data = bare(bundle.to_dict(minimal=False))
        assert [timeline["id"] for timeline in data["timelines"]] == [
            "a",
            "b",
            "c",
            "d",
        ]
        assert data["groups"] == [
            {"id": group.id, "name": None, "meta": {}, "members": ["a", "b", "c"]}
        ]
        validate(filed(data))

    def test_an_empty_group_refuses_a_given_id_held_in_the_bundle_atomically(
        self,
    ) -> None:
        bundle, group, newcomer = _empty_group_behind_a_bundle("r")

        with pytest.raises(ValueError) as refused:
            group.add_timeline(newcomer)
        assert str(refused.value) == _refused_c(group)
        assert group.timeline_ids == []
        assert list(bundle.timelines) == ["d"]
        assert bundle.groups == {group.id: group}
        assert newcomer.list_regions() == ["r"]

    def test_an_empty_group_takes_minted_ids_and_the_bundle_writes_the_newcomer(
        self,
    ) -> None:
        bundle, group, newcomer = _empty_group_behind_a_bundle(None)
        group.add_timeline(newcomer)

        assert bundle.timelines["c"] is newcomer
        data = bare(bundle.to_dict(minimal=False))
        assert [timeline["id"] for timeline in data["timelines"]] == ["c", "d"]
        assert data["groups"] == [
            {"id": group.id, "name": None, "meta": {}, "members": ["c"]}
        ]
        validate(filed(data))

    def test_an_unpickled_bundle_holds_its_empty_group_again(self) -> None:
        bundle, group, newcomer = _empty_group_behind_a_bundle("r")
        restored = pickle.loads(pickle.dumps(bundle))
        restored_group = restored.groups[group.id]

        with pytest.raises(ValueError) as refused:
            restored_group.add_timeline(newcomer)
        assert str(refused.value) == _refused_c(restored_group)
        assert restored_group.timeline_ids == []
        assert list(restored.timelines) == ["d"]

    def test_a_group_the_bundle_no_longer_holds_does_not_consult_it(self) -> None:
        bundle, group, newcomer = _empty_group_behind_a_bundle("r")
        bundle.groups.pop(group.id)
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


def _holding_two_maps() -> ContinuousLogicalTimeline:
    """A quarters timeline holding two minted maps."""
    timeline = ContinuousLogicalTimeline(length=16)
    timeline.add_conversion_map(
        LinearMap(scalar=3, source_unit="quarters", target_unit="seconds")
    )
    timeline.add_conversion_map(
        LinearMap(scalar=F(1, 2), source_unit="quarters", target_unit="minutes")
    )
    return timeline


def _map_ids(cmap: dict[str, Any]) -> Any:
    """The ids of a written map, each nested map's under its own."""
    nested = cmap.get("maps")
    if nested is None:
        return cmap["id"]
    items = nested.values() if isinstance(nested, dict) else nested
    return (cmap["id"], [_map_ids(item) for item in items])


class TestNestedMaps:
    """Maps nested in maps hold ids of their own, unique in the manifest."""

    def test_a_chain_of_two_unnamed_maps_writes_their_ids(self) -> None:
        chain = _seconds_to_minutes()
        assert _map_ids(chain.to_dict()) == (chain.id, [m.id for m in chain.maps])
        assert len({chain.id, *(m.id for m in chain.maps)}) == 3

    def test_a_chain_joining_a_timeline_keeps_every_id(self) -> None:
        timeline = _holding_two_maps()
        held = [cmap.id for cmap in timeline._all_conversion_maps()]
        chain = _seconds_to_minutes()
        timeline.add_conversion_map(chain)
        written = bare(timeline.to_dict())["timelines"][0]["conversion_maps"]
        assert [_map_ids(cmap) for cmap in written] == [
            *held,
            (chain.id, [m.id for m in chain.maps]),
        ]

    def test_a_record_map_in_a_chain_in_a_combination(self) -> None:
        combination = _measure_and_section()
        record, section = combination._nested_maps()
        assert isinstance(record, ChainMap)
        expected = (
            combination.id,
            [(record.id, [m.id for m in record.maps]), section.id],
        )
        assert _map_ids(combination.to_dict()) == expected
        timeline = ContinuousLogicalTimeline(length=4, unit="whole_notes")
        constant = ConstantMap(value="a", source_unit="whole_notes")
        timeline.add_conversion_map(constant)
        timeline.add_conversion_map(combination)
        payload = bare(timeline.to_dict())
        assert [
            _map_ids(cmap) for cmap in payload["timelines"][0]["conversion_maps"]
        ] == [constant.id, expected]
        text = json.dumps(payload)
        assert (
            json.dumps(bare(Timeline.from_dict(filed(json.loads(text))).to_dict()))
            == text
        )

    def test_the_constant_maps_of_an_interval_map_with_an_end(self) -> None:
        cmap = IntervalToConstantMap(
            boundaries=[0, 4], values=["a", "b"], end=8, source_unit="quarters"
        )
        assert _map_ids(cmap.to_dict()) == (
            cmap.id,
            [piece.id for piece in cmap._nested_maps()],
        )

    def test_a_joining_child_s_nested_map_keeps_its_id(self) -> None:
        root = _holding_two_maps()
        child = ContinuousLogicalTimeline(length=8)
        chain = _seconds_to_minutes()
        ids = (chain.id, [cmap.id for cmap in chain.maps])
        child.add_conversion_map(chain)
        root.add_child(child, 0)
        assert (chain.id, [cmap.id for cmap in chain.maps]) == ids
        assert list(child._conversion_maps) == [chain.id]

    def test_two_nested_maps_given_one_id_raise(self) -> None:
        with pytest.raises(
            ValueError,
            match=(
                "conversion map 'a' nested in conversion map 'chain' cannot join: "
                "its id 'a' is already held by conversion map 'a' nested in "
                "conversion map 'chain'"
            ),
        ):
            ChainMap(
                [LinearMap(scalar=2, uid="a"), LinearMap(scalar=3, uid="a")],
                uid="chain",
            )

    def test_two_nested_maps_of_one_seed_raise(self) -> None:
        held = IdMinter(3).mint("cmap")
        with pytest.raises(ValueError, match=f"its id '{held}' is already held"):
            ChainMap(
                [LinearMap(scalar=2, seed=3), LinearMap(scalar=3, seed=3)],
                uid="chain",
            )

    def test_a_nested_given_id_held_in_the_manifest_raises(self) -> None:
        timeline = ContinuousLogicalTimeline(length=16, uid="score")
        timeline.add_conversion_map(ScalarMap(2, uid="tempo"))
        chain = ChainMap([LinearMap(scalar=2), LinearMap(scalar=3, uid="tempo")])
        ids = (chain.id, [cmap.id for cmap in chain.maps])
        with pytest.raises(
            ValueError,
            match="already held by conversion map 'tempo' of timeline 'score'",
        ):
            timeline.add_conversion_map(chain)
        assert list(timeline._conversion_maps) == ["tempo"]
        assert (chain.id, [cmap.id for cmap in chain.maps]) == ids

    def test_one_map_nested_in_two_places_raises(self) -> None:
        twice = LinearMap(scalar=2)
        held = twice.id
        with pytest.raises(ValueError, match="nests one map in two places"):
            ChainMap([twice, twice])
        assert twice.id == held

    def test_reading_refuses_a_nested_id_repeated(self) -> None:
        payload = bare(ContinuousLogicalTimeline(length=16, uid="piece").to_dict())
        payload["timelines"][0]["conversion_maps"] = [
            {
                "type": "ChainMap",
                "id": "cmap_1",
                "source_unit": "quarters",
                "target_unit": "minutes",
                "maps": [
                    {"type": "LinearMap", "id": "cmap_2", **QUARTERS_TO_SECONDS},
                    {"type": "LinearMap", "id": "cmap_2", **SECONDS_TO_MINUTES},
                ],
            }
        ]
        with pytest.raises(ValueError) as refused:
            Timeline.from_dict(filed(payload))
        assert refusal(refused.value) == (
            "duplicate_id",
            "#/timelines/0/conversion_maps/0/maps/1/id",
            "'cmap_2' is not unique in the manifest, first at "
            "#/timelines/0/conversion_maps/0/maps/0/id",
        )


# endregion

# region Regions and jump targets by id


def _target(flow: Any, target_id: str | None) -> Any:
    """The jump target *target_id* names, as manifestation, name, coordinate."""
    if target_id is None:
        return None
    (target,) = [t for t in flow.jump_targets if t.id == target_id]
    return (target.manifestation, target.name, target.coordinate.value)


class TestKeyedById:
    """The region API takes ids; a destination names its jump target by id."""

    def test_regions_with_one_name_coexist(self) -> None:
        timeline = ContinuousLogicalTimeline(length=32)
        first = timeline.create_region(0, 8, name="Chorus")
        second = timeline.create_region(16, 24, name="Chorus")
        assert first.id != second.id
        assert timeline.list_regions() == [first.id, second.id]
        assert timeline.has_region(second.id)
        assert not timeline.has_region("Chorus")
        assert timeline.get_region(second.id).name == "Chorus"
        with pytest.raises(KeyError, match=re.escape(str([first.id, second.id]))):
            timeline.get_region("Chorus")

    def test_children_from_regions_are_named_after_them(self) -> None:
        timeline = ContinuousLogicalTimeline(length=32)
        chorus = timeline.create_region(0, 8, name="Chorus")
        verse = timeline.create_region(16, 24, name="Verse")
        child = timeline.create_child_from_region(chorus.id)
        assert child.name == "Chorus"
        assert re.fullmatch(r"clt_[0-9a-z]{13}", child.id)
        (other,) = timeline.create_children_from_regions([verse.id])
        assert other.name == "Verse"
        assert len({child.id, other.id, chorus.id, verse.id}) == 4

    def test_a_destination_must_name_a_jump_target(self) -> None:
        timeline = ContinuousLogicalTimeline(length=32, uid="score")
        with pytest.raises(ValueError) as refused:
            timeline.flow_control.add(_jump(8, id="ds", to="segno"))
        assert str(refused.value) == (
            "jump 'ds' goes on at 'segno', which names no jump target of this "
            "registry and no place of its timeline. Available jump targets: []. "
            "Available places: ['score']"
        )
        assert timeline.flow_control.elements == ()

    def test_dal_segno_names_the_segno_s_id(self) -> None:
        timeline = (
            Ms3Loader()
            .load(SCORE / "couperin_concerts/c05n05_musete.measures.tsv")
            .create_timeline()
        )
        flow = timeline.flow_control
        assert sorted(
            (target.coordinate.value, target.name or "", str(target.manifestation))
            for target in flow.jump_targets
        ) == [
            (12, "segno", str(SEGNO)),
            (42, "", str(FlowManifestation.repeat_start)),
            (42, "1e", "None"),
            (84, "", str(FlowManifestation.repeat_start)),
            (84, "2e", "None"),
        ]
        (segno,) = [t for t in flow.jump_targets if t.name == "segno"]
        assert [(b.manifestation, b.coordinate) for b in flow.breaks] == [
            (FlowManifestation.fine, _q(42))
        ]
        assert [
            destination.to
            for jump in flow.jumps
            for destination in jump.targets
            if destination.manifestation is FlowManifestation.dal_segno_al_fine
        ] == [segno.id, segno.id]

    def test_a_destination_without_a_printed_place_names_none(self) -> None:
        timeline = (
            Ms3Loader()
            .load(SCORE / "couperin_concerts/c11n08_Rondeau.measures.tsv")
            .create_timeline()
        )
        flow = timeline.flow_control
        repeat_end = FlowManifestation.repeat_end
        repeat_start = FlowManifestation.repeat_start
        da_capo = FlowManifestation.da_capo
        starts = sorted(
            target.coordinate.value
            for target in flow.jump_targets
            if target.manifestation is repeat_start
        )
        # The first repeat end has no printed start; a da capo has no place.
        # The two destinations without a manifestation go on to a couplet
        # (the markers 2e and 3e) after a da capo.
        assert sorted(
            (
                (
                    jump.coordinate.value,
                    str(destination.manifestation),
                    _target(flow, destination.to),
                )
                for jump in flow.jumps
                for destination in jump.targets
            ),
            key=_by_place,
        ) == sorted(
            [
                (12, str(repeat_end), None),
                (12, "None", _target_named(flow, "2e")),
                (12, "None", _target_named(flow, "3e")),
                (24, str(da_capo), None),
                (24, str(repeat_end), (repeat_start, None, starts[0])),
                (36, str(da_capo), None),
                (36, str(repeat_end), (repeat_start, None, starts[1])),
                (F(171, 2), str(repeat_end), (repeat_start, None, starts[2])),
            ],
            key=_by_place,
        )


def _by_place(row: tuple[Any, str, Any]) -> tuple[Any, str, str]:
    """Sort key of a destination row: its jump's coordinate, then the rest."""
    return (row[0], row[1], repr(row[2]))


def _target_named(flow: Any, name: str) -> Any:
    (target,) = [t for t in flow.jump_targets if t.name == name]
    return (target.manifestation, target.name, target.coordinate.value)


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
        "audio_continuous",
        lambda: AudioLoader()
        .load(AUDIO / "hard_techno/Bass Kick.mp3")
        .create_continuous_timeline(),
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
        .create_timelines(VerticalLinePath(x=2048, y0=0, y1=299400, tolerance=2048))[0],
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
        ids = list(_elements(bare(load().to_dict(minimal=False))["timelines"][0]))
        assert ids
        assert len(ids) == len(set(ids))
        for element_id in ids:
            assert isinstance(element_id, str) and element_id
            assert not any(ord(character) < 0x20 for character in element_id)
            assert not element_id.startswith("tl:")


# endregion


# region Ids read once per manifest


def _flat_bundle(roots: int) -> AlignmentBundle:
    bundle = AlignmentBundle()
    for index in range(roots):
        timeline = ContinuousLogicalTimeline(length=F(32), uid=f"t{index}")
        timeline.add_conversion_map(
            LinearMap(scalar=2, source_unit="quarters", target_unit="seconds")
        )
        timeline.create_region(0, 4, name="r")
        bundle.add_timeline(timeline)
    return bundle


@pytest.fixture
def id_passes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record each pass over the trees a group or bundle holds."""
    passes: list[int] = []
    held_ids = groups_module._held_ids

    def counting(timelines: Any) -> dict[str, str]:
        passes.append(1)
        return held_ids(timelines)

    monkeypatch.setattr(groups_module, "_held_ids", counting)
    return passes


class TestIdsReadOncePerManifest:
    """A manifest's ids are read in one pass, however many roots it has."""

    @pytest.mark.parametrize("roots", [3, 12])
    def test_writing_reads_the_ids_once(self, id_passes: list[int], roots: int) -> None:
        bundle = _flat_bundle(roots)
        id_passes.clear()
        bundle.to_dict()
        assert len(id_passes) == 1

    @pytest.mark.parametrize("roots", [3, 12])
    def test_reading_admits_no_root_again(
        self, id_passes: list[int], roots: int
    ) -> None:
        data = _flat_bundle(roots).to_dict()
        id_passes.clear()
        bundle = AlignmentBundle.from_dict(data)
        assert len(id_passes) == 0
        assert set(bundle.timelines) == {f"t{index}" for index in range(roots)}
        assert bundle.to_dict() == data

    def test_ids_read_together_are_the_ids_read_root_by_root(self) -> None:
        a = ContinuousLogicalTimeline(length=F(8), uid="a")
        a.create_region(0, 1, id="ra")
        b = ContinuousLogicalTimeline(length=F(8), uid="b")
        b_map = LinearMap(scalar=2)
        b.add_conversion_map(b_map)
        c = ContinuousLogicalTimeline(length=F(8), uid="c")
        d = ContinuousLogicalTimeline(length=F(8), uid="d")
        first = AlignmentBundle(id="first")
        first.add_group(TimelineGroup(id="g", timelines=[a, b]))
        first.add_timeline(c)
        second = AlignmentBundle(id="second")
        second.add_timeline(c)
        second.add_timeline(d)
        roots = [a, b, c]

        one_by_one: dict[str, str] = {}
        for root in roots:
            for element_id, holder in root._manifest_ids().items():
                one_by_one.setdefault(element_id, holder)
        together = _manifest_ids_of(roots)

        assert together == one_by_one
        assert together == {
            "a": "timeline 'a'",
            "ra": "region 'ra' of timeline 'a'",
            "b": "timeline 'b'",
            b_map.id: f"conversion map '{b_map.id}' of timeline 'b'",
            "c": "timeline 'c'",
            "d": "timeline 'd'",
            "g": "group 'g'",
            "first": "bundle 'first'",
            "second": "bundle 'second'",
        }


# endregion
