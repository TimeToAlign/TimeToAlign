"""Virtual timelines as members of groups, and virtual timelines going
through a group (see README, *Virtual timelines in groups*).
"""

from __future__ import annotations

import copy
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import timetoshare

from timetoalign.alignment import AlignmentBundle, MatchClaim
from timetoalign.core import Coordinate, IdCoordinate, Interval, TimeUnit
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    FlowMap,
    Timeline,
    TimelineGroup,
    VirtualTimeline,
)
from timetoalign.timelines.flow import VirtualMember

Q = TimeUnit.quarters
S = TimeUnit.seconds
FIXTURES = Path(os.path.commonpath(timetoshare.fixture_paths())) / "flow"


def q(value: Any) -> Coordinate:
    return Coordinate(Fraction(value), Q)


def s(value: float) -> Coordinate:
    return Coordinate(value, S)


# region The exposition grouped with a recording


def _exposition() -> tuple[Timeline, VirtualTimeline, TimelineGroup]:
    """The fixture's score, its virtual timeline and a group of it with a
    recording of 90.0 seconds."""
    score = Timeline.from_file(FIXTURES / "sonata_exposition.minimal.yaml")
    virtual = score.get_virtual_timeline("virtual_1")
    recording = ContinuousPhysicalTimeline(length=90.0, uid="recording")
    group = TimelineGroup(
        id="performance", timelines=[virtual.as_member(score), recording]
    )
    return score, virtual, group


# Each position on virtual_1, the recording's, and the written position.
BOUNDARIES = [(0, 0.0, 0), (12, 30.0, 12), (16, 40.0, 0), (28, 70.0, 16)]


class TestVirtualMember:
    """The member answers for the virtual timeline's axis and nothing else."""

    @pytest.mark.parametrize(("played", "seconds", "_written"), BOUNDARIES)
    def test_the_group_carries_a_virtual_position_to_the_recording(
        self, played: int, seconds: float, _written: int
    ) -> None:
        _score, _virtual, group = _exposition()
        stamp = group.get_timestamp_at(played, "virtual_1")
        assert stamp.coordinates == {"virtual_1": q(played), "recording": s(seconds)}
        assert type(stamp.coordinates["virtual_1"].value) is Fraction
        assert type(stamp.coordinates["recording"].value) is float
        assert stamp.is_interpolated is (played != 0)

    @pytest.mark.parametrize(("played", "seconds", "written"), BOUNDARIES)
    def test_the_group_carries_the_recording_to_the_written_score(
        self, played: int, seconds: float, written: int
    ) -> None:
        score, virtual, group = _exposition()
        stamp = group.get_timestamp_at(seconds, "recording")
        assert stamp.coordinates == {"recording": s(seconds), "virtual_1": q(played)}
        on_virtual = stamp.get_coordinate_for("virtual_1", format="coordinate")
        assert type(on_virtual.value) is Fraction
        assert virtual.fold(on_virtual, score) == q(written)

    def test_the_common_end(self) -> None:
        score, virtual, group = _exposition()
        stamp = group.get_timestamp_at(36, "virtual_1")
        assert stamp.coordinates == {"virtual_1": q(36), "recording": s(90.0)}
        assert stamp.is_interpolated is False
        with pytest.raises(ValueError, match="outside the virtual axis"):
            virtual.fold(36, score)
        with pytest.raises(ValueError, match="outside range for 'virtual_1'"):
            group.get_timestamp_at(37, "virtual_1")

    def test_the_table_the_interval_and_the_display_order(self) -> None:
        _score, _virtual, group = _exposition()
        assert group.timeline_ids == ["virtual_1", "recording"]
        frame = group.get_timestamp_table(format="dataframe")
        assert frame.columns.tolist() == ["virtual_1 (quarters)", "recording (seconds)"]
        assert frame["virtual_1 (quarters)"].tolist() == [Fraction(0), Fraction(36)]
        assert frame["recording (seconds)"].tolist() == [0.0, 90.0]
        assert group.get_interval_for("virtual_1") == Interval(start=q(0), end=q(36))

    def test_one_virtual_timeline_gives_one_member(self) -> None:
        score, virtual, group = _exposition()
        member = virtual.as_member(score)
        assert member is group["virtual_1"]
        assert isinstance(member, VirtualMember)
        with pytest.raises(ValueError, match="already in group"):
            group.add_timeline(member)
        other = ContinuousLogicalTimeline(length=24, uid="other")
        with pytest.raises(ValueError, match="is not held by the tree of 'other'"):
            virtual.as_member(other)

    def test_its_id_is_held_in_the_group(self) -> None:
        _score, _virtual, group = _exposition()
        with pytest.raises(ValueError, match="virtual_1"):
            group.add_timeline(ContinuousLogicalTimeline(length=4, uid="virtual_1"))
        assert group.timeline_ids == ["virtual_1", "recording"]

    def test_the_member_holds_nothing(self) -> None:
        _score, _virtual, group = _exposition()
        member = group["virtual_1"]
        assert member.length == q(36)
        assert member.name == "Score, as played"
        assert member.get_events().table.num_rows == 0
        with pytest.raises(AttributeError, match="as an axis only"):
            member.add_events([{"id": "e", "event_type": "Note", "instant": 1}])
        with pytest.raises(AttributeError, match="its map gives"):
            member.length = q(40)  # type: ignore[misc]
        with pytest.raises(AttributeError, match="not on its member"):
            member.name = "renamed"

    def test_a_copy_of_the_group_answers_alike(self) -> None:
        _score, _virtual, group = _exposition()
        copied = copy.deepcopy(group)
        stamp = copied.get_timestamp_at(70.0, "recording")
        assert stamp.coordinates == {"recording": s(70.0), "virtual_1": q(28)}


# endregion


# region The manifest


class TestManifest:
    """A group naming a virtual timeline is read and written back unchanged."""

    def test_the_literal_is_read_and_written_back(self) -> None:
        data = timetoshare.loads(
            (FIXTURES / "sonata_exposition.minimal.yaml").read_text(encoding="utf-8")
        )
        data["timelines"].insert(
            0, {"id": "#recording", "unit": "seconds", "length": 90.0}
        )
        data["groups"] = [
            {"id": "#performance", "members": ["#recording", "#virtual_1"]}
        ]
        data["graphical"] = {"order": {"#performance": ["#virtual_1", "#recording"]}}
        bundle = AlignmentBundle.from_dict(data)
        assert bundle.to_dict() == data
        group = bundle.get_group("performance")
        assert group.timeline_ids == ["virtual_1", "recording"]
        assert group.get_timestamp_at(28, "virtual_1").coordinates == {
            "virtual_1": q(28),
            "recording": s(70.0),
        }

    @pytest.mark.parametrize("minimal", [True, False])
    def test_the_group_is_written_and_read_back(self, minimal: bool) -> None:
        _score, _virtual, group = _exposition()
        data = group.to_dict(minimal=minimal)
        assert [timeline["id"] for timeline in data["timelines"]] == [
            "#recording",
            "#score",
        ]
        assert [virtual["id"] for virtual in data["virtual_timelines"]] == [
            "#virtual_1"
        ]
        assert [group["members"] for group in data["groups"]] == [
            ["#recording", "#virtual_1"]
        ]
        assert data["graphical"]["order"] == {
            "#performance": ["#virtual_1", "#recording"]
        }
        read = TimelineGroup.from_dict(data)
        assert read.to_dict(minimal=minimal) == data
        assert read.timeline_ids == ["virtual_1", "recording"]
        for played, seconds, _written in BOUNDARIES:
            assert read.get_timestamp_at(played, "virtual_1").coordinates == {
                "virtual_1": q(played),
                "recording": s(seconds),
            }

    def test_a_bundle_writes_the_group_back(self) -> None:
        _score, _virtual, group = _exposition()
        bundle = AlignmentBundle().add_group(group)
        assert sorted(bundle.timelines) == ["recording", "score", "virtual_1"]
        for minimal in (True, False):
            data = bundle.to_dict(minimal=minimal)
            assert [timeline["id"] for timeline in data["timelines"]] == [
                "#recording",
                "#score",
            ]
            assert AlignmentBundle.from_dict(data).to_dict(minimal=minimal) == data


# endregion


# region Through a group


def _reading(length: Any = 24) -> tuple[ContinuousLogicalTimeline, FlowMap]:
    """The score of length 24 with a [0, 16), b [16, 24) and the FlowMap a a b."""
    score = ContinuousLogicalTimeline(length=length, uid="score")
    a = score.create_region(0, 16, id="a")
    b = score.create_region(16, 24, id="b")
    return score, score.create_flow_map([a, a, b], id="reading")


def _through(timeline: Timeline, applied_to: str) -> VirtualTimeline:
    """State that the reading is applied to *applied_to* through ``scores``."""
    return timeline._tree_root().add_virtual_timeline(
        VirtualTimeline("reading", applied_to, q(40), through="scores", id="played")
    )


class TestThrough:
    """A FlowMap of one member applied to another member of its group."""

    @staticmethod
    def _edition() -> tuple[ContinuousLogicalTimeline, VirtualTimeline, TimelineGroup]:
        """The edition, the reading applied to it, and the group, which only
        lives while it is referenced."""
        score, _ = _reading()
        edition = ContinuousLogicalTimeline(length=30, uid="edition")
        group = TimelineGroup(id="scores", timelines=[score, edition])
        return edition, _through(edition, "edition"), group

    @pytest.mark.parametrize(
        ("position", "played"),
        [(0, [0, 16]), (5, [4, 20]), (15, [12, 28]), (20, [32]), (30, [])],
    )
    def test_unfold(self, position: int, played: list[int]) -> None:
        edition, virtual, _group = self._edition()
        unfolded = virtual.unfold(position, edition)
        assert unfolded == [q(value) for value in played]
        assert all(type(value.value) is Fraction for value in unfolded)

    @pytest.mark.parametrize(
        ("played", "position"),
        [
            (0, Fraction(0)),
            (20, Fraction(5)),
            (32, Fraction(20)),
            (37, Fraction(105, 4)),
            (39, Fraction(115, 4)),
        ],
    )
    def test_fold(self, played: int, position: Fraction) -> None:
        edition, virtual, _group = self._edition()
        folded = virtual.fold(played, edition)
        assert folded == q(position)
        assert folded is not None and type(folded.value) is Fraction

    def test_the_virtual_axis_is_the_reading(self) -> None:
        edition, virtual, _group = self._edition()
        resolved = virtual.resolve(edition)
        assert resolved.timeline == "score"
        assert resolved.length == q(40)
        with pytest.raises(ValueError, match="outside the virtual axis"):
            virtual.fold(40, edition)
        with pytest.raises(ValueError, match="no position of timeline 'edition'"):
            virtual.unfold(IdCoordinate(Fraction(5), Q, "score"), edition)

    def test_a_timeline_inside_a_member(self) -> None:
        score, _ = _reading()
        volume = ContinuousLogicalTimeline(length=40, uid="volume")
        edition = ContinuousLogicalTimeline(length=30, uid="edition")
        volume.add_child(edition, offset=5)
        group = TimelineGroup(id="scores", timelines=[score, volume])
        virtual = _through(volume, "edition")
        assert virtual.unfold(5, volume) == [q(6), q(22)]
        assert virtual.fold(22, volume) == q(5)
        assert virtual.fold(36, volume) == q(Fraction(85, 3))
        # The group stands between them: the volume, not the edition, is a member.
        assert group.timeline_ids == ["score", "volume"]

    def test_the_number_type_of_the_member_governs(self) -> None:
        score, _ = _reading()
        recording = ContinuousPhysicalTimeline(length=60.0, uid="recording")
        group = TimelineGroup(id="scores", timelines=[score, recording])
        virtual = _through(recording, "recording")
        assert virtual.unfold(15.0, recording) == [q(6), q(22)]
        folded = virtual.fold(33, recording)
        assert folded == s(42.5)
        assert folded is not None and type(folded.value) is float
        assert group.timeline_ids == ["score", "recording"]

    @pytest.mark.parametrize("minimal", [True, False])
    def test_written_and_read_back(self, minimal: bool) -> None:
        score, _ = _reading()
        edition = ContinuousLogicalTimeline(length=30, uid="edition")
        group = TimelineGroup(id="scores", timelines=[score, edition])
        _through(edition, "edition")
        data = group.to_dict(minimal=minimal)
        assert [virtual["id"] for virtual in data["virtual_timelines"]] == ["#played"]
        assert data["virtual_timelines"][0]["through"] == "#scores"
        read = TimelineGroup.from_dict(data)
        assert read.to_dict(minimal=minimal) == data
        read_edition = read["edition"]
        virtual = read_edition.get_virtual_timeline("played")
        assert virtual.unfold(5, read_edition) == [q(4), q(20)]
        assert virtual.fold(37, read_edition) == q(Fraction(105, 4))

    def test_a_group_holding_nothing_of_the_tree_is_refused(self) -> None:
        score, _ = _reading()
        edition = ContinuousLogicalTimeline(length=30, uid="edition")
        group = TimelineGroup(id="other", timelines=[score, edition])
        with pytest.raises(
            ValueError,
            match="which is no group holding a timeline of the tree of 'edition'",
        ):
            _through(edition, "edition")
        assert edition.list_virtual_timelines() == []
        assert group.timeline_ids == ["score", "edition"]

    def test_a_chunk_going_through_a_group_is_refused(self) -> None:
        edition, virtual, _group = self._edition()
        with pytest.raises(ValueError, match="going through 'scores'"):
            edition.create_flow_map([virtual.id])


# endregion


# region MatchClaims


class TestMatchClaims:
    """A bundle admits a MatchClaim naming a virtual timeline of its group."""

    def test_a_claim_between_the_virtual_timeline_and_the_recording(self) -> None:
        _score, _virtual, group = _exposition()
        bundle = AlignmentBundle().add_group(group)
        member, recording = group["virtual_1"], group["recording"]
        claim = MatchClaim.from_events(
            {"id": "played_28", "start": 28},
            member,
            {"id": "onset_72", "start": 72.0},
            recording,
        )
        bundle.add_match_claims([claim])
        (held,) = bundle.get_match_claims(timeline_id="virtual_1")
        assert (held.timeline_a_id, held.timeline_b_id) == ("virtual_1", "recording")
        stamp = bundle.get_matchstamp_at(28, "virtual_1")
        assert stamp.coordinates == {"virtual_1": q(28), "recording": s(72.0)}


# endregion
