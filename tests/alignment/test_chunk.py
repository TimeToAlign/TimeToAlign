"""The sentinels NOMATCH and CHUNK on the sides of a MatchClaim.

Validation strategy: ``tests/alignment/README.md``, section "The Sentinels
NOMATCH and CHUNK".
"""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import timetoshare
from timetoshare.tables import TableError, fixture_path

from timetoalign import CHUNK, NOMATCH
from timetoalign.alignment import (
    AlignmentBundle,
    MatchClaim,
    MatchClaimField,
    MatchGraph,
)
from timetoalign.core import (
    Coordinate,
    EventReference,
    Projection,
    TimeUnit,
    validate_id,
)
from timetoalign.timelines.manifest import read_projection_table
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
)

# region Fixtures


def _bundle() -> AlignmentBundle:
    """A score (quarters) holding n4, n8 and an audio (seconds) holding o20, o40."""
    score = ContinuousLogicalTimeline(length=16, uid="score")
    score.add_events(
        [
            {"id": "n4", "event_type": "Note", "instant": 4},
            {"id": "n8", "event_type": "Note", "instant": 8},
        ]
    )
    audio = ContinuousPhysicalTimeline(length=60.0, uid="audio")
    audio.add_events(
        [
            {"id": "o20", "event_type": "Note", "instant": 20.0},
            {"id": "o40", "event_type": "Note", "instant": 40.0},
        ]
    )
    bundle = AlignmentBundle()
    bundle.add_timeline(score, as_group="score")
    bundle.add_timeline(audio, as_group="audio")
    return bundle


#: Two claims between held events, and the same positions in three spellings.
EVENTS = [("n4", "score", "o20", "audio"), ("n8", "score", "o40", "audio")]
CHUNKS = [
    ({"start": 4}, "score", {"start": 20.0}, "audio"),
    ({"start": 8}, "score", {"start": 40.0}, "audio"),
]
EVENT_CHUNK = [
    ({"id": "n4", "start": 4}, "score", {"start": 20.0}, "audio"),
    ({"id": "n8", "start": 8}, "score", {"start": 40.0}, "audio"),
]


def _n_events(bundle: AlignmentBundle) -> tuple[int, int]:
    return bundle.timelines["score"].n_events, bundle.timelines["audio"].n_events


def _nodes(bundle: AlignmentBundle) -> list[list[tuple[str, object]]]:
    """The potential MatchStamps: each component's nodes, sorted."""
    graph = MatchGraph(bundle.get_match_claims(), source=bundle)
    return sorted(
        sorted(
            (node.timeline_id, node.value)
            for timeline_id in component.timeline_ids
            for node in component.get_nodes_for_timeline(timeline_id)
        )
        for component in graph.split_components()
    )


def _stamps(bundle: AlignmentBundle) -> list[tuple[dict, bool]]:
    """The MatchStamps at three score positions, two on an anchor, one between."""
    return [
        (stamp.to_dict(), stamp.is_interpolated)
        for stamp in (bundle.get_matchstamp_at(q, "score") for q in (4, 6, 8))
    ]


# endregion


class TestTheSameAnchorsAsEvents:
    """A CHUNK side answers every query as an event at its position does."""

    def test_from_events(self) -> None:
        bundle = _bundle()
        score, audio = bundle.timelines["score"], bundle.timelines["audio"]
        events = MatchClaim.from_events(
            {"id": "n4", "start": 4}, score, {"id": "o20", "start": 20.0}, audio
        )
        chunk = MatchClaim.from_events({"start": 4}, score, {"start": 20.0}, audio)
        event_chunk = MatchClaim.from_events(
            {"id": "n4", "start": 4}, score, {"start": 20.0}, audio
        )
        assert chunk.start_anchor == event_chunk.start_anchor == events.start_anchor
        assert (chunk.event_a_id, chunk.event_b_id) == (CHUNK, CHUNK)
        assert (event_chunk.event_a_id, event_chunk.event_b_id) == ("n4", CHUNK)
        assert chunk.chunk_sides == ("score", "audio")
        assert event_chunk.chunk_sides == ("audio",)
        assert events.chunk_sides == ()

    def test_an_interval(self) -> None:
        bundle = _bundle()
        score, audio = bundle.timelines["score"], bundle.timelines["audio"]
        chunk = MatchClaim.from_events(
            {"start": 4, "end": 8},
            score,
            {"start": 20.0, "end": 40.0},
            audio,
            end_coord_key="end",
        )
        assert chunk.get_interval_for("score").start == Coordinate(4, TimeUnit.quarters)
        assert chunk.get_interval_for("audio").end == Coordinate(40.0, TimeUnit.seconds)
        assert chunk.chunk_sides == ("score", "audio")

    @pytest.mark.parametrize("pairs", [CHUNKS, EVENT_CHUNK], ids=["chunk", "event"])
    def test_create_match_claims(self, pairs: list) -> None:
        events = _bundle()
        stated = events.create_match_claims(EVENTS)
        bundle = _bundle()
        claims = bundle.create_match_claims(pairs)
        assert [c.start_anchor for c in claims] == [c.start_anchor for c in stated]
        assert (
            _nodes(bundle)
            == _nodes(events)
            == [
                [("audio", 20.0), ("score", Fraction(4))],
                [("audio", 40.0), ("score", Fraction(8))],
            ]
        )
        assert _stamps(bundle) == _stamps(events)
        assert [c.get_matchstamp().to_dict() for c in claims] == [
            c.get_matchstamp().to_dict() for c in stated
        ]
        assert [c.get_matchstamp(from_graph=False).to_dict() for c in claims] == [
            c.get_matchstamp(from_graph=False).to_dict() for c in stated
        ]

    def test_from_columns(self) -> None:
        events = _bundle()
        stated = events.create_match_claims(EVENTS)
        field = MatchClaimField.from_columns(
            ["score", "score"],
            ["audio", "audio"],
            [CHUNK, "n8"],
            [CHUNK, CHUNK],
            [4, 8],
            [20.0, 40.0],
            timelines={
                "score": ("quarters", "fraction"),
                "audio": ("seconds", "float"),
            },
        )
        assert [c.start_anchor for c in field] == [c.start_anchor for c in stated]
        assert [c.chunk_sides for c in field] == [("score", "audio"), ("audio",)]
        bundle = _bundle()
        bundle.add_match_claim_field(field)
        assert _nodes(bundle) == _nodes(events)
        assert _stamps(bundle) == _stamps(events)


class TestNoEventStandsForAChunkSide:
    def test_two_positions_add_no_projection(self) -> None:
        bundle = _bundle()
        bundle.create_match_claims(CHUNKS)
        assert _n_events(bundle) == (2, 2)

    def test_a_position_beside_an_event_or_nomatch_is_a_projection(self) -> None:
        bundle = _bundle()
        of_event, of_none = bundle.create_match_claims(
            [
                ("n4", "score", {"start": 30.0}, "audio"),
                ({"start": 12}, "score", NOMATCH, "audio"),
            ]
        )
        assert _n_events(bundle) == (3, 3)
        assert bundle.timelines["audio"].get_event(of_event.event_b_id)[
            "projection_of"
        ] == {"timeline_id": "score", "event_id": "n4"}
        projection = bundle.timelines["score"].get_event(of_none.event_a_id)
        assert (projection["event_type"], projection["projection_of"]) == (
            "Projection",
            None,
        )
        assert of_none.event_b_id == NOMATCH
        assert (of_event.chunk_sides, of_none.chunk_sides) == ((), ())

    def test_a_conceptual_pair_of_positions_gives_two_projections(self) -> None:
        bundle = _bundle()
        (claim,) = bundle.create_match_claims(
            [({"start": 2}, "score", {"start": 10.0}, "audio")], synchronous=False
        )
        assert _n_events(bundle) == (3, 3)
        assert claim.is_conceptual
        assert claim.chunk_sides == ()

    def test_a_chunk_side_grows_an_unlocked_timeline(self) -> None:
        bundle = _bundle()
        bundle.create_match_claims([({"start": 4}, "score", {"start": 75.0}, "audio")])
        assert bundle.timelines["audio"].length == Coordinate(75.0, TimeUnit.seconds)
        assert _n_events(bundle) == (2, 2)


def _every_kind() -> list[MatchClaim]:
    """A CHUNK–CHUNK, an event–CHUNK, an event–event, a NOMATCH, a conceptual claim."""
    bundle = _bundle()
    score, audio = bundle.timelines["score"], bundle.timelines["audio"]
    return [
        MatchClaim.from_events({"start": 2}, score, {"start": 10.0}, audio, id="k1"),
        MatchClaim.from_events(
            {"id": "n4", "start": 4}, score, {"start": 20.0}, audio, id="k2"
        ),
        MatchClaim.from_events(
            {"id": "n8", "start": 8},
            score,
            {"id": "o40", "start": 40.0},
            audio,
            id="k3",
        ),
        MatchClaim.nomatch({"id": "n4"}, score, audio, id="k4"),
        MatchClaim.from_events(
            {"id": "n8", "start": 8},
            score,
            {"id": "o20", "start": 20.0},
            audio,
            is_synchronous=False,
            id="k5",
        ),
    ]


class TestTheTableHoldsTheSentinels:
    def test_columns(self) -> None:
        table = MatchClaimField.from_claims(_every_kind()).table
        assert table.column("event_a_id").to_pylist() == [CHUNK, "n4", "n8", "n4", "n8"]
        assert table.column("event_b_id").to_pylist() == [
            CHUNK,
            CHUNK,
            "o40",
            NOMATCH,
            "o20",
        ]
        assert (
            table.column("event_a_id").null_count,
            table.column("event_b_id").null_count,
        ) == (0, 0)
        assert table.column("event_b_name").to_pylist() == [None] * 5

    def test_parquet_and_dict(self, tmp_path: Path) -> None:
        claims = _every_kind()
        field = MatchClaimField.from_claims(claims)
        field.to_parquet(tmp_path)
        assert MatchClaimField.from_parquet(tmp_path).to_claims() == claims
        assert pq.read_table(tmp_path / "match_claims.parquet").column(
            "event_b_id"
        ).to_pylist() == [CHUNK, CHUNK, "o40", NOMATCH, "o20"]
        assert MatchClaimField.from_dict(field.to_dict()).to_claims() == claims
        assert [row["event_a_id"] for row in field.to_dict()["rows"]] == [
            CHUNK,
            "n4",
            "n8",
            "n4",
            "n8",
        ]

    def test_a_package(self, tmp_path: Path) -> None:
        bundle = _bundle()
        bundle.add_match_claims(_every_kind())
        bundle.to_file(tmp_path / "package.yaml")
        read = AlignmentBundle.from_file(tmp_path / "package.yaml")
        assert read.match_claim_table == bundle.match_claim_table
        assert [c.chunk_sides for c in read.get_match_claims()] == [
            ("score", "audio"),
            ("audio",),
            (),
            (),
            (),
        ]
        assert not (tmp_path / "projections.parquet").exists()


class TestTheReaderRefuses:
    @staticmethod
    def _table(column: str, values: list) -> pa.Table:
        table = MatchClaimField.from_claims(_every_kind()).table
        index = table.column_names.index(column)
        return table.set_column(index, column, pa.array(values, type=pa.string()))

    @pytest.mark.parametrize(
        ("column", "values", "message"),
        [
            (
                "event_a_id",
                [CHUNK, None, "n8", "n4", "n8"],
                "MatchClaim table row 1 names no event_a_id or no event_b_id (a side "
                "without an event holds NOMATCH_SENTINEL or CHUNK_SENTINEL).",
            ),
            (
                "event_a_id",
                [CHUNK, "n4", "n8", "n4", CHUNK],
                "MatchClaim table row 4 is not synchronous and names CHUNK_SENTINEL, "
                "which stands only on a synchronous MatchClaim.",
            ),
            (
                "event_b_id",
                [NOMATCH, CHUNK, "o40", NOMATCH, "o20"],
                "MatchClaim table row 0 is synchronous and names NOMATCH_SENTINEL, "
                "which stands only on a MatchClaim that is not synchronous.",
            ),
            (
                "event_a_id",
                [CHUNK, "n4", "n8", NOMATCH, "n8"],
                "MatchClaim table row 3 names NOMATCH_SENTINEL on both sides; a "
                "NOMATCH MatchClaim names an event on its other side.",
            ),
            (
                "event_a_name",
                ["bar 1", None, None, None, None],
                "MatchClaim table row 0 has a name on a sentinel side; a sentinel "
                "side has no name.",
            ),
        ],
    )
    def test_a_row_no_claim_could_be(
        self, column: str, values: list, message: str
    ) -> None:
        with pytest.raises(ValueError, match=re.escape(message)):
            MatchClaimField.from_table(self._table(column, values))

    def test_from_columns_a_null(self) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Row 0 names nothing on side a: a side of a MatchClaim names an "
                "event, or CHUNK_SENTINEL for a position on a timeline holding no "
                "event for it."
            ),
        ):
            MatchClaimField.from_columns(
                ["score"],
                ["audio"],
                [None],
                [CHUNK],
                [4],
                [20.0],
                timelines={
                    "score": ("quarters", "fraction"),
                    "audio": ("seconds", "float"),
                },
            )

    def test_a_conceptual_from_events_without_an_id(self) -> None:
        bundle = _bundle()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "event_b names no event id: a conceptual MatchClaim names two "
                "events. Give a position without an event of its own a Projection "
                "event first."
            ),
        ):
            MatchClaim.from_events(
                {"id": "n4", "start": 4},
                bundle.timelines["score"],
                {"start": 20.0},
                bundle.timelines["audio"],
                is_synchronous=False,
            )


def test_the_filters_read_the_sentinels() -> None:
    field = MatchClaimField.from_claims(_every_kind())
    assert field.filter(nomatch_only=True).table.column("id").to_pylist() == ["k4"]
    assert field.filter(synchronous_only=True).table.column("id").to_pylist() == [
        "k1",
        "k2",
        "k3",
    ]
    bundle = _bundle()
    bundle.add_match_claims(_every_kind())
    assert [c.id for c in bundle.get_match_claims(nomatch_only=True)] == ["k4"]


class TestNoElementTakesASentinelId:
    @pytest.mark.parametrize("sentinel", [NOMATCH, CHUNK])
    def test_refused(self, sentinel: str) -> None:
        message = f"Invalid id {sentinel!r}: it is reserved for a sentinel"
        with pytest.raises(ValueError, match=re.escape(message)):
            validate_id(sentinel)
        with pytest.raises(ValueError, match=re.escape(message)):
            Projection(id=sentinel, start=Coordinate(1, TimeUnit.quarters))
        timeline = ContinuousLogicalTimeline(length=4, uid="t")
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Event id '{sentinel}' is reserved for a sentinel of a "
                f"MatchClaim's side; give the event another id"
            ),
        ):
            timeline.add_events([{"id": sentinel, "event_type": "Note", "instant": 1}])
        assert timeline.n_events == 0


def test_text_renderings_name_the_sentinel() -> None:
    chunk, event_chunk, _, nomatch, _ = _every_kind()
    assert repr(chunk) == (
        "MatchClaim(instant: score@2 quarters [CHUNK] <-> audio@10 seconds [CHUNK])"
    )
    assert repr(event_chunk) == (
        "MatchClaim(instant: score@4 quarters <-> audio@20 seconds [CHUNK])"
    )
    assert str(event_chunk) == (
        "MatchClaim (synchronous, instant)\n"
        "  Timeline A:  score  @4 quarters\n"
        "  Event A:     n4\n"
        "  Timeline B:  audio  @20 seconds\n"
        "  Event B:     CHUNK"
    )
    html = chunk._repr_html_()
    assert html.count("<td colspan='2'>CHUNK</td>") == 2
    assert "SENTINEL" not in html
    assert "<td colspan='2'>NOMATCH</td>" in nomatch._repr_html_()


def _sentinel_id_message(where: str, sentinel: str) -> str:
    """The one refusal of a sentinel given as the event a side names."""
    return (
        f"{where} names '{sentinel}' as an event, a string reserved for a "
        f"sentinel: a CHUNK side is an event dict stating its position without "
        f"'id', a NOMATCH side the constant NOMATCH."
    )


class TestASentinelNamesNoEvent:
    """A reserved string as an event dict's id, or as the event a Projection stands for."""

    @pytest.mark.parametrize("sentinel", [NOMATCH, CHUNK])
    @pytest.mark.parametrize("synchronous", [True, False])
    @pytest.mark.parametrize("side", ["event_a", "event_b"])
    def test_from_events(self, sentinel: str, synchronous: bool, side: str) -> None:
        score = ContinuousLogicalTimeline(length=16, uid="score")
        audio = ContinuousPhysicalTimeline(length=60.0, uid="audio")
        sides = {
            "event_a": {"id": "n4", "start": 4},
            "event_b": {"id": "o20", "start": 20.0},
        }
        sides[side] = {**sides[side], "id": sentinel}
        with pytest.raises(ValueError) as caught:
            MatchClaim.from_events(
                sides["event_a"],
                score,
                sides["event_b"],
                audio,
                is_synchronous=synchronous,
            )
        assert str(caught.value) == _sentinel_id_message(side, sentinel)

    @pytest.mark.parametrize("sentinel", [NOMATCH, CHUNK])
    def test_nomatch(self, sentinel: str) -> None:
        score = ContinuousLogicalTimeline(length=16, uid="score")
        with pytest.raises(ValueError) as caught:
            MatchClaim.nomatch({"id": sentinel}, score, "audio")
        assert str(caught.value) == _sentinel_id_message("event", sentinel)

    @pytest.mark.parametrize("sentinel", [NOMATCH, CHUNK])
    def test_a_projection_of_a_sentinel(self, sentinel: str) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(f"Invalid id '{sentinel}': it is reserved for a sentinel"),
        ):
            Projection(
                start=Coordinate(1, TimeUnit.quarters),
                projection_of=EventReference(timeline_id="audio", event_id=sentinel),
            )
        message = (
            f"A Projection stands for an event, never for a sentinel: its "
            f"projection_of names '{sentinel}', a string reserved for a "
            f"MatchClaim's side without an event"
        )
        reference = {"timeline_id": "audio", "event_id": sentinel}
        score = ContinuousLogicalTimeline(length=16, uid="score")
        with pytest.raises(ValueError) as caught:
            score.add_events(
                [{"event_type": "Projection", "instant": 1, "projection_of": reference}]
            )
        assert str(caught.value) == message
        with pytest.raises(ValueError) as caught:
            score.add_projections(
                [{"event_type": "Projection", "start": 1, "projection_of": reference}]
            )
        assert str(caught.value) == message
        assert score.n_events == 0

    @pytest.mark.parametrize("sentinel", [NOMATCH, CHUNK])
    def test_a_projections_table_naming_a_sentinel(
        self, sentinel: str, tmp_path: Path
    ) -> None:
        """The fixture's first row, ``projection_10`` of ``onset_1``, now of a sentinel."""
        table = pq.read_table(fixture_path("projections.parquet"))
        manifest = timetoshare.loads(
            fixture_path("manifest.yaml").read_text(encoding="utf-8")
        )
        reference = table.column("projection_of").combine_chunks()
        event_ids = reference.field("event_id").to_pylist()
        assert event_ids[0] == "onset_1"
        changed = pa.StructArray.from_arrays(
            [reference.field("timeline_id"), pa.array([sentinel] + event_ids[1:])],
            fields=list(reference.type),
            mask=reference.is_null(),
        )
        path = tmp_path / "projections.parquet"
        pq.write_table(
            table.set_column(
                table.schema.get_field_index("projection_of"), "projection_of", changed
            ),
            path,
        )
        with pytest.raises(TableError) as caught:
            read_projection_table(path, manifest)
        problem = caught.value.problem
        assert (problem.code, problem.row, problem.column, problem.message) == (
            "invalid_projection_of",
            0,
            ("projection_of", "event_id"),
            f"'{sentinel}' is reserved for a sentinel",
        )


class TestRefusedBeforeAnythingIsAdded:
    """Each refusal comes before the call's first Projection is added."""

    @pytest.mark.parametrize(
        ("pair", "message"),
        [
            (
                (CHUNK, "score", {"start": 20.0}, "audio"),
                _sentinel_id_message("Pair 1", CHUNK),
            ),
            (
                ({"id": CHUNK, "start": 2}, "score", {"start": 10.0}, "audio"),
                _sentinel_id_message("Pair 1", CHUNK),
            ),
            (
                ("n8", "score", {"id": NOMATCH, "start": 10.0}, "audio"),
                _sentinel_id_message("Pair 1", NOMATCH),
            ),
            (
                ({"start": 2, "name": "bar 1"}, "score", {"start": 10.0}, "audio"),
                "Pair 1: the side on timeline 'score' states the name 'bar 1', but it "
                "is CHUNK_SENTINEL: the pair states two positions where neither "
                "timeline holds an event, and a sentinel side has no name.",
            ),
            (
                ({"start": 2}, "score", {"start": 10.0, "end": 12.0}, "audio"),
                "Pair 1 states two positions where neither timeline holds an event "
                "(CHUNK_SENTINEL on both sides) and an end on side B only: the "
                "claim's anchors hold its positions, an interval an end on both "
                "sides. State the end on both sides, or on neither.",
            ),
            (
                ({"start": 2}, "score", {"start": float("nan")}, "audio"),
                "cannot store non-finite value nan as a number cell",
            ),
            (
                (NOMATCH, "score", NOMATCH, "audio"),
                "Pair 1: event_a and event_b cannot both be NOMATCH_SENTINEL",
            ),
        ],
    )
    def test_refused(self, pair: tuple, message: str) -> None:
        bundle = _bundle()
        with pytest.raises(ValueError) as caught:
            bundle.create_match_claims(
                [("n4", "score", {"start": 30.0}, "audio"), pair]
            )
        assert str(caught.value) == message
        assert _n_events(bundle) == (2, 2)
        assert len(bundle.match_claim_table) == 0

    def test_none_is_no_side(self) -> None:
        bundle = _bundle()
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Pair 0: a side of a claimed pair is an event id, an event dict or "
                "NOMATCH_SENTINEL, not NoneType"
            ),
        ):
            bundle.create_match_claims([("n4", "score", None, "audio")])
