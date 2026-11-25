"""FlowMaps on timelines without flow control (see README, *FlowMaps*).

Every expected interval and position is worked out by hand from the offsets
and bounds the fixtures state, and written as a literal.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from timetoalign.alignment import AlignmentBundle
from timetoalign.core import (
    Coordinate,
    Duration,
    FlowConvention,
    Gap,
    IdCoordinate,
    Interval,
    NumberType,
    TimeUnit,
)
from timetoalign.core.provenance import Provenance
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscretePhysicalTimeline,
    FlowMap,
    PlacedChunk,
    PlacedGap,
    ResolvedFlowMap,
    Timeline,
    UnknownLengthError,
)

Q = TimeUnit.quarters
S = TimeUnit.seconds


def q(value: Any) -> Coordinate:
    return Coordinate(Fraction(value), Q)


def s(value: float) -> Coordinate:
    return Coordinate(value, S)


def _exact(coordinates: list[Coordinate]) -> list[tuple[Any, type, TimeUnit]]:
    """Value, the value's type and unit of each coordinate."""
    return [(c.value, type(c.value), c.unit) for c in coordinates]


@pytest.fixture
def concert() -> ContinuousLogicalTimeline:
    """The quarters case of the README: on the root's axis ``first`` is
    [0, 8), ``second`` [8, 20), ``trio`` [12, 18), ``coda`` [18, 20)."""
    root = ContinuousLogicalTimeline(length=40, uid="set")
    root.create_child(8, 0, uid="first")
    second = root.create_child(12, 8, uid="second")
    second.create_child(6, 4, uid="trio")
    second.create_region(10, 12, id="coda")
    root.create_region(0, 3, id="intro")
    root.create_region(30, 40, id="tail")
    return root


@pytest.fixture
def audio() -> ContinuousPhysicalTimeline:
    """The seconds case of the README."""
    root = ContinuousPhysicalTimeline(length=30.0, uid="audio")
    root.create_region(0.5, 10.25, id="verse")
    root.create_region(10.25, 20.0, id="chorus")
    return root


ORDER = ["second", Gap(4), "first", "trio", "coda", "intro"]


class TestQuarters:
    """Chunks from children at two depths, a child's Region, the root's
    Region, and a gap, on a fraction axis."""

    def test_the_chunks_are_placed_one_after_the_other(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        resolved = concert.create_flow_map(ORDER).resolve(concert)
        assert resolved == ResolvedFlowMap(
            "set",
            Q,
            NumberType.fraction,
            (
                PlacedChunk(Interval(start=q(8), end=q(20)), q(0)),
                PlacedGap(Duration(Fraction(4), Q), q(12)),
                PlacedChunk(Interval(start=q(0), end=q(8)), q(16)),
                PlacedChunk(Interval(start=q(12), end=q(18)), q(24)),
                PlacedChunk(Interval(start=q(18), end=q(20)), q(30)),
                PlacedChunk(Interval(start=q(0), end=q(3)), q(32)),
            ),
        )
        assert [(chunk.start.value, chunk.end.value) for chunk in resolved.chunks] == [
            (0, 12),
            (12, 16),
            (16, 24),
            (24, 30),
            (30, 32),
            (32, 35),
        ]
        assert resolved.length == q(35)
        assert type(resolved.length.value) is Fraction

    @pytest.mark.parametrize(
        ("position", "played_at"),
        [
            (2, [18, 34]),
            (13, [5, 25]),
            (19, [11, 31]),
            (Fraction(25, 2), [Fraction(9, 2), Fraction(49, 2)]),
            (20, []),
            (25, []),
        ],
    )
    def test_unfold(
        self, concert: ContinuousLogicalTimeline, position: Any, played_at: list[Any]
    ) -> None:
        resolved = concert.create_flow_map(ORDER).resolve(concert)
        assert _exact(resolved.unfold(position)) == [
            (Fraction(value), Fraction, Q) for value in played_at
        ]

    @pytest.mark.parametrize(
        ("position", "root_position"),
        [
            (0, 8),
            (11, 19),
            (16, 0),
            (24, 12),
            (31, 19),
            (34, 2),
            (12, None),
            (15, None),
        ],
    )
    def test_fold(
        self,
        concert: ContinuousLogicalTimeline,
        position: int,
        root_position: int | None,
    ) -> None:
        resolved = concert.create_flow_map(ORDER).resolve(concert)
        folded = resolved.fold(position)
        if root_position is None:
            assert folded is None
        else:
            assert folded is not None
            assert _exact([folded]) == [(Fraction(root_position), Fraction, Q)]

    @pytest.mark.parametrize("position", [35, -1, 100])
    def test_fold_refuses_a_position_outside_the_virtual_axis(
        self, concert: ContinuousLogicalTimeline, position: int
    ) -> None:
        resolved = concert.create_flow_map(ORDER).resolve(concert)
        with pytest.raises(ValueError, match="lies outside the virtual axis"):
            resolved.fold(position)

    def test_a_chunk_may_occur_any_number_of_times(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        resolved = concert.create_flow_map(["intro", "intro", "intro"]).resolve(concert)
        assert resolved.length == q(9)
        assert [c.value for c in resolved.unfold(1)] == [1, 4, 7]

    def test_a_flow_map_without_chunks_is_an_axis_of_length_zero(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        resolved = concert.create_flow_map([]).resolve(concert)
        assert resolved.length == q(0)
        assert resolved.unfold(2) == []
        with pytest.raises(ValueError, match="lies outside the virtual axis"):
            resolved.fold(0)


class TestSeconds:
    """Chunks from Regions and a gap on a float axis."""

    ORDER = ["chorus", "verse", Gap(1.5), "chorus"]

    def test_the_chunks_are_placed_one_after_the_other(
        self, audio: ContinuousPhysicalTimeline
    ) -> None:
        resolved = audio.create_flow_map(self.ORDER).resolve(audio)
        assert resolved.chunks == (
            PlacedChunk(Interval(start=s(10.25), end=s(20.0)), s(0.0)),
            PlacedChunk(Interval(start=s(0.5), end=s(10.25)), s(9.75)),
            PlacedGap(Duration(1.5, S), s(19.5)),
            PlacedChunk(Interval(start=s(10.25), end=s(20.0)), s(21.0)),
        )
        assert _exact([resolved.length]) == [(30.75, float, S)]
        assert resolved.number_type is NumberType.float

    def test_unfold_and_fold(self, audio: ContinuousPhysicalTimeline) -> None:
        resolved = audio.create_flow_map(self.ORDER).resolve(audio)
        assert _exact(resolved.unfold(12.0)) == [(1.75, float, S), (22.75, float, S)]
        assert resolved.unfold(0.25) == []
        assert _exact([resolved.fold(10.0)]) == [(0.75, float, S)]  # type: ignore[list-item]
        assert resolved.fold(20.0) is None
        assert _exact([resolved.fold(30.0)]) == [(19.25, float, S)]  # type: ignore[list-item]

    def test_an_integer_given_on_a_float_axis_answers_in_floats(
        self, audio: ContinuousPhysicalTimeline
    ) -> None:
        resolved = audio.create_flow_map(self.ORDER).resolve(audio)
        assert _exact(resolved.unfold(12)) == [(1.75, float, S), (22.75, float, S)]


class TestPositionForms:
    """Raw values, Coordinates and IdCoordinates in; Coordinates out."""

    def test_every_form_of_one_position_gives_one_answer(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        resolved = concert.create_flow_map(ORDER).resolve(concert)
        expected = [q(18), q(34)]
        for position in (
            2,
            2.0,
            Fraction(2),
            Coordinate(2, Q),
            IdCoordinate(2, Q, timeline_id="set"),
        ):
            assert resolved.unfold(position) == expected
            assert all(type(c) is Coordinate for c in resolved.unfold(position))
        for position in (16, 16.0, Fraction(16), Coordinate(16, Q)):
            assert resolved.fold(position) == q(0)
        assert resolved.fold(IdCoordinate(16, Q, timeline_id="clt_9")) == q(0)

    def test_a_coordinate_in_another_unit_is_refused(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        resolved = concert.create_flow_map(ORDER).resolve(concert)
        with pytest.raises(
            ValueError, match="is in seconds; the axis of timeline 'set'"
        ):
            resolved.unfold(Coordinate(2.0, S))
        with pytest.raises(
            ValueError, match="is in seconds; the axis of timeline 'set'"
        ):
            resolved.fold(Coordinate(2.0, S))

    def test_an_id_coordinate_of_another_timeline_is_no_root_position(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        resolved = concert.create_flow_map(ORDER).resolve(concert)
        with pytest.raises(ValueError, match="is no position of timeline 'set'"):
            resolved.unfold(IdCoordinate(2, Q, timeline_id="first"))

    def test_a_root_position_is_no_virtual_position(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        resolved = concert.create_flow_map(ORDER).resolve(concert)
        with pytest.raises(ValueError, match="not of the virtual axis: unfold it"):
            resolved.fold(IdCoordinate(16, Q, timeline_id="set"))


class TestGaps:
    """A gap keeps its length, in the number type of the axis."""

    def test_a_gap_is_held_in_the_number_type_of_the_axis(
        self, concert: ContinuousLogicalTimeline, audio: ContinuousPhysicalTimeline
    ) -> None:
        on_fractions = concert.create_flow_map(["first", Gap(4)])
        (gap,) = (chunk for chunk in on_fractions.chunks if isinstance(chunk, Gap))
        assert (gap.length, type(gap.length)) == (4, Fraction)
        assert on_fractions.to_dict()["chunks"] == ["first", {"gap": [4, 1]}]
        on_floats = audio.create_flow_map(["verse", Gap(4)])
        (gap,) = (chunk for chunk in on_floats.chunks if isinstance(chunk, Gap))
        assert (gap.length, type(gap.length)) == (4.0, float)
        assert on_floats.to_dict()["chunks"] == ["verse", {"gap": 4.0}]

    def test_a_gap_that_is_no_value_of_the_axis_is_refused(
        self, audio: ContinuousPhysicalTimeline
    ) -> None:
        samples = DiscretePhysicalTimeline(length=100, uid="wav")
        samples.create_region(0, 10, id="start")
        with pytest.raises(ValueError, match="the gap of 4.5 in FlowMap 'f' is no int"):
            samples.create_flow_map(["start", Gap(4.5)], id="f")
        assert samples.list_flow_maps() == []
        whole = samples.create_flow_map(["start", Gap(4.0)], id="f")
        assert [type(c.length) for c in whole.chunks if isinstance(c, Gap)] == [int]
        assert whole.resolve(samples).length == Coordinate(14, TimeUnit.samples)
        with pytest.raises(
            ValueError, match="the gap of 1/3 in FlowMap 'g' is no float"
        ):
            audio.create_flow_map(["verse", Gap(Fraction(1, 3))], id="g")
        assert audio.list_flow_maps() == []

    @pytest.mark.parametrize("length", [0, -1, Fraction(-1, 2), 0.0])
    def test_a_gap_is_positive(self, length: Any) -> None:
        with pytest.raises(ValueError, match="the length of a gap is positive"):
            Gap(length)

    @pytest.mark.parametrize("length", ["4", None, True, float("nan"), float("inf")])
    def test_a_gap_has_a_finite_number_as_its_length(self, length: Any) -> None:
        with pytest.raises(ValueError, match="the length of a gap is"):
            Gap(length)


class TestRefusals:
    """What a FlowMap may not name; nothing joins the timeline."""

    def test_a_chunk_naming_nothing_in_the_tree(
        self, concert: ContinuousLogicalTimeline, audio: ContinuousPhysicalTimeline
    ) -> None:
        for unknown in ("nowhere", "verse", "audio"):
            with pytest.raises(ValueError) as raised:
                concert.create_flow_map(["first", unknown], id="f")
            assert str(raised.value) == (
                f"chunk '{unknown}' of FlowMap 'f' names no region, child timeline "
                "or virtual timeline in the tree of 'set'"
            )
        assert concert.list_flow_maps() == []

    def test_the_root_itself_is_no_chunk(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        with pytest.raises(ValueError, match="names its own timeline 'set' as a chunk"):
            concert.create_flow_map(["first", "set"])
        with pytest.raises(ValueError, match="names its own timeline 'set' as a chunk"):
            FlowMap("set", ["set"])
        assert concert.list_flow_maps() == []

    def test_a_chunk_is_an_id_or_a_gap(self) -> None:
        with pytest.raises(
            ValueError, match="a chunk of FlowMap 'f' is an ID or a Gap"
        ):
            FlowMap("set", [(0, 4)], id="f")  # type: ignore[list-item]

    def test_a_flow_map_of_another_timeline(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        other = FlowMap("elsewhere", ["first"], id="f")
        with pytest.raises(ValueError) as raised:
            concert.add_flow_map(other)
        assert str(raised.value) == (
            "FlowMap 'f' is a FlowMap of timeline 'elsewhere', not of 'set'"
        )
        with pytest.raises(ValueError, match="not of 'set'"):
            other.resolve(concert)
        assert concert.list_flow_maps() == []

    def test_a_flow_map_names_a_root_timeline(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        second = concert.get_child("second")
        with pytest.raises(ValueError) as raised:
            second.create_flow_map(["trio"], id="f")
        assert str(raised.value) == (
            "FlowMap 'f' cannot be read on timeline 'second', a child of 'set': "
            "a FlowMap names a root timeline"
        )
        assert second.list_flow_maps() == []

    def test_a_root_holding_flow_maps_that_becomes_a_child_is_not_written(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        concert.create_flow_map(["first"], id="f")
        evening = ContinuousLogicalTimeline(length=100, uid="evening")
        evening.add_child(concert, offset=10)
        with pytest.raises(ValueError) as raised:
            evening.to_dict()
        assert str(raised.value) == (
            "#/flow_maps/0/timeline: 'set' is a child timeline; a FlowMap names a "
            "root timeline"
        )

    def test_a_child_of_unknown_length_is_no_chunk(self) -> None:
        root = ContinuousLogicalTimeline(length=8, uid="root")
        root.create_child(None, 0, uid="open")
        with pytest.raises(UnknownLengthError) as raised:
            root.create_flow_map(["open"])
        assert str(raised.value) == (
            "Cannot read it as a chunk of a FlowMap: the length of timeline 'open' "
            "is unknown"
        )
        assert root.list_flow_maps() == []

    def test_one_canonical_flow_map_per_timeline(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        concert.create_flow_map(["first"], id="one", canonical=True)
        concert.create_flow_map(["second"], id="plain")
        with pytest.raises(ValueError) as raised:
            concert.create_flow_map(["second"], id="two", canonical=True)
        assert str(raised.value) == (
            "FlowMap 'two' is a second canonical FlowMap of timeline 'set', after 'one'"
        )
        assert concert.list_flow_maps() == ["one", "plain"]

    def test_a_given_id_held_in_the_manifest(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        with pytest.raises(ValueError, match="'coda'"):
            concert.create_flow_map(["first"], id="coda")
        assert concert.list_flow_maps() == []

    def test_an_unknown_flow_map_is_a_key_error_naming_the_held_ones(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        concert.create_flow_map(["first"], id="one")
        with pytest.raises(KeyError, match=r"Available FlowMaps: \['one'\]"):
            concert.get_flow_map("two")


class TestChunkObjects:
    """``create_flow_map`` takes the elements themselves or their IDs."""

    def test_regions_children_and_virtual_timelines_are_named_by_their_ids(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        twice = concert.create_flow_map(["first", "first"], id="twice")
        z = concert.create_virtual_timeline(twice, id="z")
        flow_map = concert.create_flow_map(
            [concert.get_region("intro"), concert.get_child("second"), z, Gap(2)]
        )
        assert flow_map.chunks == ("intro", "second", "z", Gap(Fraction(2)))
        assert flow_map.id == "flow_map_1"
        assert flow_map.timeline == "set"
        assert concert.get_flow_map("flow_map_1") is flow_map


class TestVirtualTimelineAsChunk:
    """A virtual timeline stands for the stretches of the FlowMap it applies."""

    def test_its_stretches_are_played_in_its_place(
        self, concert: ContinuousLogicalTimeline
    ) -> None:
        twice = concert.create_flow_map(["first", "first"], id="twice")
        concert.create_virtual_timeline(twice, id="z")
        resolved = concert.create_flow_map(["z", Gap(1), "second"]).resolve(concert)
        assert [
            (
                (chunk.source.start.value, chunk.source.end.value)
                if isinstance(chunk, PlacedChunk)
                else None
            )
            for chunk in resolved.chunks
        ] == [(0, 8), (0, 8), None, (8, 20)]
        assert resolved.length == q(29)
        assert resolved.unfold(2) == [q(2), q(10)]
        assert resolved.fold(16) is None
        assert resolved.fold(17) == q(8)

    @staticmethod
    def _manifest(**virtual: Any) -> dict[str, Any]:
        return {
            "timetoshare": "0.2.0",
            "timelines": [
                {
                    "id": "#root",
                    "unit": "quarters",
                    "length": [16, 1],
                    "regions": [{"id": "#a", "start": [0, 1], "end": [4, 1]}],
                }
            ],
            "flow_maps": [{"id": "#f", "timeline": "#root", "chunks": ["#a", "#v"]}],
            "virtual_timelines": [
                {
                    "id": "#v",
                    "map": "#f",
                    "timeline": "#root",
                    "unit": "quarters",
                    "number_type": "fraction",
                    "length": [4, 1],
                    **virtual,
                }
            ],
        }

    def test_a_flow_map_naming_itself_is_refused_when_read_against_its_root(
        self,
    ) -> None:
        manifest = self._manifest()
        root = Timeline.from_dict(manifest)
        assert root.to_dict() == Timeline.from_dict(root.to_dict()).to_dict()
        with pytest.raises(ValueError) as raised:
            root.get_flow_map("f").resolve(root)
        assert str(raised.value) == (
            "FlowMap 'f' names itself as a chunk, through the virtual timelines "
            "its chunks name"
        )

    def test_a_virtual_timeline_in_another_unit_is_no_chunk(self) -> None:
        root = Timeline.from_dict(
            self._manifest(unit="seconds", number_type="float", length=4.0)
        )
        with pytest.raises(ValueError) as raised:
            root.get_flow_map("f").resolve(root)
        assert str(raised.value) == (
            "chunk 'v' of FlowMap 'f' is a virtual timeline in seconds; the axis "
            "of its root 'root' is in quarters"
        )


EDITOR = "https://example.org/people/editor"
ENTRY = {
    "stated_by": [EDITOR],
    "date": "2026-11-25",
    "method": "manual",
    "beliefs": [{"actor": EDITOR, "belief": "true", "date": "2026-11-25"}],
}


class TestDictionaryForm:
    """The manifest's object and the FlowMap are one statement."""

    FULL = {
        "id": "encore",
        "timeline": "set",
        "chunks": ["second", {"gap": [4, 1]}, "first"],
        "convention": "single",
        "canonical": True,
        "name": "Encore order",
        "provenance": [ENTRY],
    }

    def test_every_key_is_read(self) -> None:
        flow_map = FlowMap.from_dict(self.FULL)
        assert flow_map == FlowMap(
            "set",
            ("second", Gap(Fraction(4)), "first"),
            convention=FlowConvention.single,
            canonical=True,
            name="Encore order",
            provenance=(Provenance.from_dict(ENTRY),),
            id="encore",
        )
        assert flow_map.convention is FlowConvention.single
        assert type(flow_map.chunks[1].length) is Fraction  # type: ignore[union-attr]
        assert flow_map.to_dict() == self.FULL
        assert flow_map.to_dict(minimal=False) == self.FULL

    def test_the_minimal_form_leaves_the_defaults_out(self) -> None:
        minimal = {"id": "f", "timeline": "set", "chunks": ["first"]}
        flow_map = FlowMap.from_dict(minimal)
        assert flow_map == FlowMap("set", ["first"], id="f")
        assert (flow_map.convention, flow_map.canonical, flow_map.name) == (
            None,
            False,
            None,
        )
        assert flow_map.provenance == ()
        assert flow_map.to_dict() == minimal
        assert flow_map.to_dict(minimal=False) == {
            **minimal,
            "canonical": False,
            "name": None,
        }

    @pytest.mark.parametrize(
        ("written", "length"), [([4, 1], Fraction(4)), (4.5, 4.5), (4, 4)]
    )
    def test_a_gap_is_read_as_its_json_form_says(
        self, written: Any, length: Any
    ) -> None:
        data = {"id": "f", "timeline": "set", "chunks": [{"gap": written}]}
        flow_map = FlowMap.from_dict(data)
        (gap,) = flow_map.chunks
        assert isinstance(gap, Gap)
        assert (gap.length, type(gap.length)) == (length, type(length))
        assert flow_map.to_dict() == data

    def test_a_convention_of_another_word_is_kept(self) -> None:
        data = {
            "id": "f",
            "timeline": "set",
            "chunks": ["first"],
            "convention": "as recorded in 1953",
        }
        flow_map = FlowMap.from_dict(data)
        assert flow_map.convention == "as recorded in 1953"
        assert not isinstance(flow_map.convention, FlowConvention)
        assert flow_map.to_dict() == data
        assert FlowMap("set", ["first"], convention="printed").convention is (
            FlowConvention.printed
        )

    def test_unknown_keys_are_kept_on_the_flow_map_and_on_a_gap(self) -> None:
        data = {
            "id": "f",
            "timeline": "set",
            "chunks": ["first", {"gap": [2, 1], "x-why": "applause"}],
            "x-edition": 3,
        }
        flow_map = FlowMap.from_dict(data)
        assert flow_map.to_dict() == data
        assert flow_map.to_dict(minimal=False) == {
            **data,
            "canonical": False,
            "name": None,
        }

    @pytest.mark.parametrize(
        ("broken", "message"),
        [
            ({"timeline": "set", "chunks": []}, "a FlowMap lacks the key"),
            (
                {"id": "f", "timeline": "set", "chunks": "first"},
                "the chunks of FlowMap 'f' must be an array",
            ),
            (
                {"id": "f", "timeline": "set", "chunks": [{"hole": 4}]},
                "a chunk of FlowMap 'f' is an ID or {gap: length}",
            ),
            (
                {"id": "f", "timeline": "set", "chunks": [{"gap": [0, 1]}]},
                "a gap of FlowMap 'f': the length of a gap is positive",
            ),
            (
                {"id": "f", "timeline": "set", "chunks": [], "provenance": [{"x": 1}]},
                "the provenance of FlowMap 'f'",
            ),
        ],
    )
    def test_a_malformed_object_is_refused(
        self, broken: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError) as raised:
            FlowMap.from_dict(broken)
        assert message in str(raised.value)

    def test_is_computed_needs_a_convention_and_the_method_inferred(self) -> None:
        inferred = Provenance(method="inferred")
        assert FlowMap(
            "set", [], convention="default", provenance=[inferred]
        ).is_computed
        assert not FlowMap("set", [], convention="default").is_computed
        assert not FlowMap("set", [], provenance=[inferred]).is_computed
        assert not FlowMap(
            "set", [], convention="default", provenance=[ENTRY]  # type: ignore[list-item]
        ).is_computed


class TestIdentity:
    """IDs are minted in the manifest, and references follow a new ID."""

    @staticmethod
    def _score() -> ContinuousLogicalTimeline:
        score = ContinuousLogicalTimeline(length=8)
        section = score.create_region(0, 4)
        flow_map = score.create_flow_map([section, section])
        score.create_virtual_timeline(flow_map)
        return score

    def test_ids_are_minted_in_the_timeline_s_manifest(self) -> None:
        score = self._score()
        assert score.id == "clt_1"
        assert score.list_regions() == ["region_1"]
        assert score.list_flow_maps() == ["flow_map_1"]
        # The root holds clt_1, so the virtual timeline is minted above it.
        assert score.list_virtual_timelines() == ["clt_2"]
        second = score.create_flow_map(["region_1"])
        assert second.id == "flow_map_2"

    def test_a_second_tree_joining_a_bundle_is_minted_again_and_keeps_its_references(
        self,
    ) -> None:
        first, second = self._score(), self._score()
        bundle = AlignmentBundle()
        bundle.add_timeline(first)
        bundle.add_timeline(second)
        assert first.id == "clt_1"
        assert first.get_flow_map("flow_map_1").chunks == ("region_1", "region_1")
        assert first.list_virtual_timelines() == ["clt_2"]
        # clt_1, clt_2, region_1 and flow_map_1 are held: each is minted above.
        assert second.id == "clt_3"
        assert second.list_regions() == ["region_2"]
        assert second.list_flow_maps() == ["flow_map_2"]
        assert second.list_virtual_timelines() == ["clt_4"]
        flow_map = second.get_flow_map("flow_map_2")
        assert (flow_map.timeline, flow_map.chunks) == (
            "clt_3",
            ("region_2", "region_2"),
        )
        virtual = second.get_virtual_timeline("clt_4")
        assert (virtual.map, virtual.timeline) == ("flow_map_2", "clt_3")
        assert virtual.resolve(second).length == q(8)
        assert AlignmentBundle.from_dict(bundle.to_dict()).to_dict() == bundle.to_dict()
