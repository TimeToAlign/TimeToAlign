"""Identity of manifest elements: one id form, given ids kept, unique per manifest."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from fractions import Fraction
from typing import Any

import pytest

from timetoalign.core import Coordinate, TimeUnit
from timetoalign.loader.alignment import ParangonadaLoader
from timetoalign.loader.graphical.iiif import IIIFManifestLoader
from timetoalign.loader.midi.score import ScoreMidiLoader
from timetoalign.loader.physical.audio import AudioLoader
from timetoalign.loader.score import Ms3Loader
from timetoalign.loader.tabular import LabLoader
from timetoalign.maps import (
    LinearMap,
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
    SegmentLine,
    Timeline,
)
from timetoalign.timelines.flowcontrol import Break, Jump, Marker
from timetoalign.timelines.regions import Region

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
        (lambda: Break(_q(4)), "break_1"),
        (lambda: Jump(_q(4), _q(0)), "jump_1"),
        (lambda: Marker(_q(0)), "marker_1"),
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
    marker = root.flow_control.add_marker(Marker(_q(8), name="segno"))
    root.flow_control.add_jump(Jump(_q(24), _q(8), target=marker.id))
    return root


REFERENCE_IDS = {
    "timeline": "clt_1",
    "children": {"clt_2", "clt_3"},
    "regions": {"region_1", "region_2"},
    "breaks": {"break_1", "break_2"},
    "markers": ["marker_1"],
    "jumps": [("jump_1", "marker_1")],
    "maps": ["lmap_1"],
}


def _ids(payload: dict[str, Any]) -> dict[str, Any]:
    flow = payload["flow_control"]
    return {
        "timeline": payload["id"],
        "children": set(payload["children"]),
        "regions": {entry["id"] for entry in payload["regions"]},
        "breaks": {entry["id"] for entry in flow["breaks"]},
        "markers": [entry["id"] for entry in flow["markers"]],
        "jumps": [(entry["id"], entry["target"]) for entry in flow["jumps"]],
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


GIVEN = ["performance", "a", "Bass Kick", "score:clt_7"]


def _given_manifest() -> Timeline:
    root = ContinuousLogicalTimeline(length=32, uid="performance")
    root.add_child(ContinuousLogicalTimeline(length=8, uid="Bass Kick"), 0)
    root.add_child(ContinuousLogicalTimeline(length=8, uid="score:clt_7"), 8)
    root.create_region(0, 8, id="a", name="A")
    root.add_conversion_map(ScalarMap(2, uid="raw"))
    root.flow_control.add_break(Break(_q(8), id="b1"))
    root.flow_control.add_marker(Marker(_q(8), id="segno", name="segno"))
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

    @pytest.mark.parametrize("refused", ["", "a/b", "a\tb", "a\nb"])
    @pytest.mark.parametrize(
        "build",
        [
            lambda uid: ContinuousLogicalTimeline(uid=uid),
            lambda uid: ScalarMap(2, uid=uid),
            lambda uid: Region(_q(0), _q(4), id=uid),
            lambda uid: Break(_q(4), id=uid),
            lambda uid: Jump(_q(4), _q(0), id=uid),
            lambda uid: Marker(_q(0), id=uid),
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
        root.flow_control.add_marker(Marker(_q(0)))
        child = ContinuousLogicalTimeline(length=8)
        marker = child.flow_control.add_marker(Marker(_q(2), name="segno"))
        child.flow_control.add_jump(Jump(_q(6), _q(2), target=marker.id))
        root.add_child(child, 0)
        assert [m.id for m in child.flow_control.markers] == ["marker_2"]
        assert [j.target for j in child.flow_control.jumps] == ["marker_2"]

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
        assert [(m.id, m.name, m.coordinate) for m in flow.markers] == [
            ("marker_1", "segno", _q(12)),
            ("marker_2", "fine", _q(Fraction(81, 2))),
        ]
        dal_segno = [j for j in flow.jumps if j.target is not None]
        assert [(j.control_type.value, j.target) for j in dal_segno] == [
            ("dal_segno_al_fine", "marker_1"),
            ("dal_segno_al_fine", "marker_1"),
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
    for kind in ("breaks", "markers", "jumps"):
        for entry in payload["flow_control"][kind]:
            yield entry["id"]
    for child_id, child in payload["children"].items():
        assert child["timeline"]["id"] == child_id
        yield from _elements(child["timeline"])


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
        ids = list(_elements(load().to_dict()))
        assert ids
        assert len(ids) == len(set(ids))
        for element_id in ids:
            assert isinstance(element_id, str) and element_id
            assert "/" not in element_id
            assert not any(ord(character) < 0x20 for character in element_id)
            assert not element_id.startswith("tl:")


# endregion
