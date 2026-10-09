"""Tests for Agent, AlignmentAnchor, MatchClaim, and MatchMetadata scalars.

The whole claim family is now frozen pydantic v2 models. ``MatchMetadata`` is
``{agent: Agent, certainty: float}`` — the old free-form ``decision_criteria``
/ ``created_at`` / ``notes`` / ``algorithm_params`` fields are gone. Anchor
coordinates are unit-bearing ``Coordinate`` values.
"""

from __future__ import annotations

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from timetoalign.alignment import Agent, AlignmentAnchor, MatchClaim, MatchMetadata
from timetoalign.core import AgentType, Coordinate, IdCoordinate, TimeUnit

# region Fixtures


@pytest.fixture
def basic_anchor() -> AlignmentAnchor:
    """Create a basic alignment anchor."""
    return AlignmentAnchor(
        timeline_a_id="score:1",
        coordinate_a=Coordinate(100.0, TimeUnit.quarters),
        timeline_b_id="recording:1",
        coordinate_b=Coordinate(45.5, TimeUnit.seconds),
    )


@pytest.fixture
def basic_agent() -> Agent:
    """Create a basic software agent."""
    return Agent(
        name="test_user",
        type=AgentType.software,
        identifier="manual_alignment",
    )


@pytest.fixture
def basic_metadata(basic_agent: Agent) -> MatchMetadata:
    """Create basic match metadata."""
    return MatchMetadata(agent=basic_agent)


# endregion


# region Agent & MatchMetadata Tests


class TestMatchMetadata:
    """Tests for the Agent and MatchMetadata pydantic scalars."""

    def test_agent_type_members(self) -> None:
        """AgentType has exactly the two members human and software."""
        assert {member.name for member in AgentType} == {"human", "software"}
        assert str(AgentType.human) == "human"
        assert str(AgentType.software) == "software"

    def test_basic_creation(self, basic_agent: Agent) -> None:
        """Test creating metadata with required fields."""
        meta = MatchMetadata(agent=basic_agent)

        assert meta.agent is basic_agent
        assert meta.agent.name == "test_user"
        assert meta.certainty == 1.0  # default

    def test_agent_roundtrip(self, basic_agent: Agent) -> None:
        """Agent.to_dict / from_dict reconstructs identically."""
        d = basic_agent.to_dict()
        assert d == {
            "name": "test_user",
            "type": "software",
            "identifier": "manual_alignment",
        }
        restored = Agent.from_dict(d)
        assert restored == basic_agent
        assert restored.type is AgentType.software

    def test_metadata_roundtrip(self, basic_metadata: MatchMetadata) -> None:
        """MatchMetadata.to_dict / from_dict round-trips the nested Agent."""
        d = basic_metadata.to_dict()
        assert d == {
            "agent": {
                "name": "test_user",
                "type": "software",
                "identifier": "manual_alignment",
            },
            "certainty": 1.0,
        }
        restored = MatchMetadata.from_dict(d)
        assert restored == basic_metadata

    def test_certainty_validation(self, basic_agent: Agent) -> None:
        """Test certainty must be in [0, 1]."""
        with pytest.raises(ValidationError, match="Certainty must be in"):
            MatchMetadata(agent=basic_agent, certainty=-0.1)

        with pytest.raises(ValidationError, match="Certainty must be in"):
            MatchMetadata(agent=basic_agent, certainty=1.1)

    def test_certainty_boundaries(self, basic_agent: Agent) -> None:
        """Test certainty boundary values are valid."""
        meta_0 = MatchMetadata(agent=basic_agent, certainty=0.0)
        meta_1 = MatchMetadata(agent=basic_agent, certainty=1.0)
        assert meta_0.certainty == 0.0
        assert meta_1.certainty == 1.0

    def test_to_dict(self, basic_metadata: MatchMetadata) -> None:
        """Test serialization to dictionary."""
        d = basic_metadata.to_dict()

        assert d["agent"]["name"] == "test_user"
        assert d["agent"]["type"] == "software"
        assert d["certainty"] == 1.0
        assert set(d) == {"agent", "certainty"}

    def test_from_dict_roundtrip(self, basic_metadata: MatchMetadata) -> None:
        """Test serialization round-trip."""
        d = basic_metadata.to_dict()
        restored = MatchMetadata.from_dict(d)

        assert restored.agent == basic_metadata.agent
        assert restored.certainty == basic_metadata.certainty

    def test_frozen_model(self, basic_metadata: MatchMetadata) -> None:
        """Test that MatchMetadata is immutable."""
        with pytest.raises(ValidationError):
            basic_metadata.certainty = 0.5  # type: ignore


# endregion


# region AlignmentAnchor Tests


class TestAlignmentAnchor:
    """Tests for AlignmentAnchor dataclass (pure coordinate pair)."""

    def test_basic_creation(self) -> None:
        """Test creating an anchor with required fields."""
        anchor = AlignmentAnchor(
            timeline_a_id="tl1",
            coordinate_a=Coordinate(10.0, TimeUnit.number),
            timeline_b_id="tl2",
            coordinate_b=Coordinate(20.0, TimeUnit.number),
        )

        assert anchor.timeline_a_id == "tl1"
        assert anchor.coordinate_a.value == 10.0
        assert anchor.timeline_b_id == "tl2"
        assert anchor.coordinate_b.value == 20.0

    def test_raw_float_coordinates_are_rejected(self) -> None:
        """Anchors require explicit unit-bearing Coordinate values."""
        with pytest.raises(ValidationError):
            AlignmentAnchor(
                timeline_a_id="tl1",
                coordinate_a=10.0,
                timeline_b_id="tl2",
                coordinate_b=20.0,
            )

    def test_no_id_field(self) -> None:
        """Test that AlignmentAnchor has no id field."""
        anchor = AlignmentAnchor(
            timeline_a_id="tl1",
            coordinate_a=Coordinate(0.0, TimeUnit.number),
            timeline_b_id="tl2",
            coordinate_b=Coordinate(0.0, TimeUnit.number),
        )
        assert not hasattr(anchor, "id")

    def test_no_claim_fields(self) -> None:
        """Test that AlignmentAnchor has no is_explicit or is_synchronous."""
        anchor = AlignmentAnchor(
            timeline_a_id="tl1",
            coordinate_a=Coordinate(0.0, TimeUnit.number),
            timeline_b_id="tl2",
            coordinate_b=Coordinate(0.0, TimeUnit.number),
        )
        assert not hasattr(anchor, "is_explicit")
        assert not hasattr(anchor, "is_synchronous")

    def test_timelines_property(self, basic_anchor: AlignmentAnchor) -> None:
        """Test timelines property."""
        assert basic_anchor.timelines == ("score:1", "recording:1")

    def test_coordinates_property(self, basic_anchor: AlignmentAnchor) -> None:
        """Test coordinates property."""
        assert basic_anchor.coordinates == (
            Coordinate(100.0, TimeUnit.quarters),
            Coordinate(45.5, TimeUnit.seconds),
        )

    def test_get_coordinate_for(self, basic_anchor: AlignmentAnchor) -> None:
        """Test getting coordinate for specific timeline."""
        score_coordinate = basic_anchor.get_coordinate_for(
            "score:1", format="coordinate"
        )
        recording_coordinate = basic_anchor.get_coordinate_for(
            "recording:1", format="coordinate"
        )
        assert score_coordinate == Coordinate(100.0, TimeUnit.quarters)
        assert recording_coordinate == Coordinate(45.5, TimeUnit.seconds)
        with pytest.raises(KeyError):
            basic_anchor.get_coordinate_for("other")

    def test_get_coordinate_for_preserves_fraction_and_unit(self) -> None:
        anchor = AlignmentAnchor(
            timeline_a_id="score",
            coordinate_a=Coordinate(Fraction(7, 3), TimeUnit.quarters),
            timeline_b_id="audio",
            coordinate_b=Coordinate(1.25, TimeUnit.seconds),
        )
        coordinate = anchor.get_coordinate_for("score")
        assert coordinate is not None
        assert coordinate.value == Fraction(7, 3)
        assert coordinate.unit is TimeUnit.quarters

    def test_connects(self, basic_anchor: AlignmentAnchor) -> None:
        """Test connects method."""
        assert basic_anchor.connects("score:1") is True
        assert basic_anchor.connects("recording:1") is True
        assert basic_anchor.connects("other") is False

    def test_connects_both(self, basic_anchor: AlignmentAnchor) -> None:
        """Test connects_both method."""
        assert basic_anchor.connects_both("score:1", "recording:1") is True
        assert basic_anchor.connects_both("recording:1", "score:1") is True
        assert basic_anchor.connects_both("score:1", "other") is False

    def test_to_dict(self, basic_anchor: AlignmentAnchor) -> None:
        """Test serialization to dictionary."""
        d = basic_anchor.to_dict()

        assert d["timeline_a_id"] == "score:1"
        # quarters count in exact ratios, so the anchor holds one and the
        # dictionary records it. seconds are a measurement and stay float.
        assert d["coordinate_a"] == {
            "unit": "quarters",
            "number_type": "fraction",
            "value": [100, 1],
        }
        assert d["timeline_b_id"] == "recording:1"
        assert d["coordinate_b"] == {
            "unit": "seconds",
            "number_type": "float",
            "value": 45.5,
        }
        # No is_explicit, is_synchronous, or id in dict
        assert "is_explicit" not in d
        assert "is_synchronous" not in d
        assert "id" not in d

    def test_from_dict_roundtrip(self, basic_anchor: AlignmentAnchor) -> None:
        """Test serialization round-trip."""
        d = basic_anchor.to_dict()
        restored = AlignmentAnchor.from_dict(d)

        assert restored.timeline_a_id == basic_anchor.timeline_a_id
        assert restored.coordinate_a == basic_anchor.coordinate_a
        assert restored.timeline_b_id == basic_anchor.timeline_b_id
        assert restored.coordinate_b == basic_anchor.coordinate_b

    def test_frozen_model(self, basic_anchor: AlignmentAnchor) -> None:
        """Test that AlignmentAnchor is immutable (frozen pydantic model)."""
        with pytest.raises(ValidationError):
            basic_anchor.coordinate_a = 200.0  # type: ignore

    def test_repr(self, basic_anchor: AlignmentAnchor) -> None:
        """Test string representation."""
        r = repr(basic_anchor)
        assert "AlignmentAnchor" in r
        assert "score:1" in r
        assert "recording:1" in r

    def test_value_equality(self) -> None:
        """Test that anchors with same values are equal (value objects)."""
        a1 = AlignmentAnchor(
            timeline_a_id="tl1",
            coordinate_a=Coordinate(10.0, TimeUnit.number),
            timeline_b_id="tl2",
            coordinate_b=Coordinate(20.0, TimeUnit.number),
        )
        a2 = AlignmentAnchor(
            timeline_a_id="tl1",
            coordinate_a=Coordinate(10.0, TimeUnit.number),
            timeline_b_id="tl2",
            coordinate_b=Coordinate(20.0, TimeUnit.number),
        )
        assert a1 == a2


# endregion


# region MatchClaim Tests


class TestMatchClaim:
    """Tests for MatchClaim dataclass (top-level timeline IDs)."""

    def test_instant_creation(self, basic_anchor: AlignmentAnchor) -> None:
        """Test creating instant match (single anchor)."""
        claim = MatchClaim(
            timeline_a_id="score:1",
            timeline_b_id="recording:1",
            start_anchor=basic_anchor,
        )

        assert claim.is_interval is False
        assert claim.start_anchor is basic_anchor
        assert claim.end_anchor is None
        assert claim.timeline_a_id == "score:1"
        assert claim.timeline_b_id == "recording:1"

    def test_interval_creation(self) -> None:
        """Test creating interval match (two anchors)."""
        start = AlignmentAnchor(
            timeline_a_id="dgt1",
            coordinate_a=Coordinate(0.0, TimeUnit.number),
            timeline_b_id="dgt2",
            coordinate_b=Coordinate(0.0, TimeUnit.number),
        )
        end = AlignmentAnchor(
            timeline_a_id="dgt1",
            coordinate_a=Coordinate(975.0, TimeUnit.number),
            timeline_b_id="dgt2",
            coordinate_b=Coordinate(866.0, TimeUnit.number),
        )
        claim = MatchClaim(
            timeline_a_id="dgt1",
            timeline_b_id="dgt2",
            start_anchor=start,
            end_anchor=end,
        )

        assert claim.is_interval is True
        assert claim.start_anchor is start
        assert claim.end_anchor is end

    def test_auto_generated_id(self, basic_anchor: AlignmentAnchor) -> None:
        """Test ID is auto-generated if not provided."""
        claim = MatchClaim(
            timeline_a_id="score:1",
            timeline_b_id="recording:1",
            start_anchor=basic_anchor,
        )
        assert claim.id.startswith("claim:MatchClaim")

    def test_explicit_id(self, basic_anchor: AlignmentAnchor) -> None:
        """Test explicit ID is preserved."""
        claim = MatchClaim(
            timeline_a_id="score:1",
            timeline_b_id="recording:1",
            start_anchor=basic_anchor,
            id="my_claim",
        )
        assert claim.id == "my_claim"

    def test_mismatched_anchors_raises(self) -> None:
        """Test error when anchors connect different timelines."""
        start = AlignmentAnchor(
            timeline_a_id="tl1",
            coordinate_a=Coordinate(0.0, TimeUnit.number),
            timeline_b_id="tl2",
            coordinate_b=Coordinate(0.0, TimeUnit.number),
        )
        end = AlignmentAnchor(
            timeline_a_id="tl1",
            coordinate_a=Coordinate(100.0, TimeUnit.number),
            timeline_b_id="tl3",  # Different!
            coordinate_b=Coordinate(100.0, TimeUnit.number),
        )

        with pytest.raises(ValueError, match="must connect same timelines"):
            MatchClaim(
                timeline_a_id="tl1",
                timeline_b_id="tl2",
                start_anchor=start,
                end_anchor=end,
            )

    def test_synchronous_requires_anchor(self) -> None:
        """Test that synchronous claims require a start_anchor."""
        with pytest.raises(ValueError, match="require a start_anchor"):
            MatchClaim(
                timeline_a_id="tl1",
                timeline_b_id="tl2",
                start_anchor=None,
                is_synchronous=True,
            )

    def test_non_synchronous_rejects_anchor(self) -> None:
        """Test that non-synchronous claims must not have anchors."""
        anchor = AlignmentAnchor(
            timeline_a_id="tl1",
            coordinate_a=Coordinate(0.0, TimeUnit.number),
            timeline_b_id="tl2",
            coordinate_b=Coordinate(0.0, TimeUnit.number),
        )
        with pytest.raises(ValueError, match="must not have anchors"):
            MatchClaim(
                timeline_a_id="tl1",
                timeline_b_id="tl2",
                start_anchor=anchor,
                is_synchronous=False,
            )

    def test_non_synchronous_claim(self) -> None:
        """Test creating a non-synchronous claim (no anchors)."""
        claim = MatchClaim(
            timeline_a_id="tl1",
            timeline_b_id="tl2",
            is_synchronous=False,
        )
        assert claim.start_anchor is None
        assert claim.end_anchor is None
        assert claim.anchors == []
        assert claim.timeline_a_id == "tl1"
        assert claim.timeline_b_id == "tl2"

    def test_anchor_timeline_mismatch_raises(self) -> None:
        """Test error when anchor timelines don't match claim timelines."""
        anchor = AlignmentAnchor(
            timeline_a_id="tl1",
            coordinate_a=Coordinate(0.0, TimeUnit.number),
            timeline_b_id="tl2",
            coordinate_b=Coordinate(0.0, TimeUnit.number),
        )
        with pytest.raises(ValueError, match="must match claim timelines"):
            MatchClaim(
                timeline_a_id="tl1",
                timeline_b_id="tl3",  # Doesn't match anchor
                start_anchor=anchor,
            )

    def test_timelines_property(self, basic_anchor: AlignmentAnchor) -> None:
        """Test timelines property."""
        claim = MatchClaim(
            timeline_a_id="score:1",
            timeline_b_id="recording:1",
            start_anchor=basic_anchor,
        )
        assert claim.timelines == ("score:1", "recording:1")

    def test_get_coordinates_for_instant(self, basic_anchor: AlignmentAnchor) -> None:
        """Test get_coordinates_for with instant match."""
        claim = MatchClaim(
            timeline_a_id="score:1",
            timeline_b_id="recording:1",
            start_anchor=basic_anchor,
        )

        start = claim.get_coordinate_for("score:1")
        assert start.value == 100.0

    def test_get_coordinates_for_interval(self) -> None:
        """Test get_coordinates_for with interval match."""
        start = AlignmentAnchor(
            timeline_a_id="dgt1",
            coordinate_a=Coordinate(0.0, TimeUnit.number),
            timeline_b_id="dgt2",
            coordinate_b=Coordinate(0.0, TimeUnit.number),
        )
        end = AlignmentAnchor(
            timeline_a_id="dgt1",
            coordinate_a=Coordinate(975.0, TimeUnit.number),
            timeline_b_id="dgt2",
            coordinate_b=Coordinate(866.0, TimeUnit.number),
        )
        claim = MatchClaim(
            timeline_a_id="dgt1",
            timeline_b_id="dgt2",
            start_anchor=start,
            end_anchor=end,
        )

        interval = claim.get_interval_for("dgt1")
        assert interval.start.value == 0.0
        assert interval.end.value == 975.0

        interval = claim.get_interval_for("dgt2")
        assert interval.start.value == 0.0
        assert interval.end.value == 866.0

    def test_get_coordinates_invalid_timeline(
        self, basic_anchor: AlignmentAnchor
    ) -> None:
        """Test error when getting coordinates for non-connected timeline."""
        claim = MatchClaim(
            timeline_a_id="score:1",
            timeline_b_id="recording:1",
            start_anchor=basic_anchor,
        )

        with pytest.raises(KeyError, match="other"):
            claim.get_coordinate_for("other")

    def test_get_coordinates_non_synchronous_raises(self) -> None:
        """Test error when getting coordinates from non-synchronous claim."""
        claim = MatchClaim(
            timeline_a_id="tl1",
            timeline_b_id="tl2",
            is_synchronous=False,
        )
        with pytest.raises(ValueError, match="no anchors"):
            claim.get_coordinate_for("tl1")

    def test_connects(self, basic_anchor: AlignmentAnchor) -> None:
        """Test connects method."""
        claim = MatchClaim(
            timeline_a_id="score:1",
            timeline_b_id="recording:1",
            start_anchor=basic_anchor,
        )

        assert claim.connects("score:1") is True
        assert claim.connects("recording:1") is True
        assert claim.connects("other") is False

    def test_connects_both(self, basic_anchor: AlignmentAnchor) -> None:
        """Test connects_both method."""
        claim = MatchClaim(
            timeline_a_id="score:1",
            timeline_b_id="recording:1",
            start_anchor=basic_anchor,
        )

        assert claim.connects_both("score:1", "recording:1") is True
        assert claim.connects_both("score:1", "other") is False

    def test_anchors_property_instant(self, basic_anchor: AlignmentAnchor) -> None:
        """Test anchors property for instant match."""
        claim = MatchClaim(
            timeline_a_id="score:1",
            timeline_b_id="recording:1",
            start_anchor=basic_anchor,
        )
        assert claim.anchors == [basic_anchor]

    def test_anchors_property_interval(self) -> None:
        """Test anchors property for interval match."""
        start = AlignmentAnchor(
            timeline_a_id="tl1",
            coordinate_a=Coordinate(0.0, TimeUnit.number),
            timeline_b_id="tl2",
            coordinate_b=Coordinate(0.0, TimeUnit.number),
        )
        end = AlignmentAnchor(
            timeline_a_id="tl1",
            coordinate_a=Coordinate(100.0, TimeUnit.number),
            timeline_b_id="tl2",
            coordinate_b=Coordinate(100.0, TimeUnit.number),
        )
        claim = MatchClaim(
            timeline_a_id="tl1",
            timeline_b_id="tl2",
            start_anchor=start,
            end_anchor=end,
        )

        assert claim.anchors == [start, end]

    def test_anchors_property_non_synchronous(self) -> None:
        """Test anchors property for non-synchronous claim returns empty list."""
        claim = MatchClaim(
            timeline_a_id="tl1",
            timeline_b_id="tl2",
            is_synchronous=False,
        )
        assert claim.anchors == []

    def test_with_metadata(
        self, basic_anchor: AlignmentAnchor, basic_metadata: MatchMetadata
    ) -> None:
        """Test creating claim with metadata."""
        claim = MatchClaim(
            timeline_a_id="score:1",
            timeline_b_id="recording:1",
            start_anchor=basic_anchor,
            metadata=basic_metadata,
        )

        assert claim.metadata is basic_metadata
        assert claim.metadata.agent.name == "test_user"

    def test_instant_factory(self) -> None:
        """Test instant match via direct construction."""
        claim = MatchClaim(
            timeline_a_id="score",
            timeline_b_id="recording",
            start_anchor=AlignmentAnchor(
                timeline_a_id="score",
                coordinate_a=Coordinate(0.0, TimeUnit.number),
                timeline_b_id="recording",
                coordinate_b=Coordinate(0.0, TimeUnit.number),
            ),
        )

        assert claim.is_interval is False
        assert claim.timeline_a_id == "score"
        assert claim.timeline_b_id == "recording"
        assert claim.start_anchor.coordinate_a.value == 0.0

    def test_interval_factory(self) -> None:
        """Test interval match via direct construction."""
        claim = MatchClaim(
            timeline_a_id="dgt1",
            timeline_b_id="dgt2",
            start_anchor=AlignmentAnchor(
                timeline_a_id="dgt1",
                coordinate_a=Coordinate(0.0, TimeUnit.number),
                timeline_b_id="dgt2",
                coordinate_b=Coordinate(0.0, TimeUnit.number),
            ),
            end_anchor=AlignmentAnchor(
                timeline_a_id="dgt1",
                coordinate_a=Coordinate(975.0, TimeUnit.number),
                timeline_b_id="dgt2",
                coordinate_b=Coordinate(866.0, TimeUnit.number),
            ),
        )

        assert claim.is_interval is True
        assert claim.timeline_a_id == "dgt1"
        assert claim.timeline_b_id == "dgt2"
        interval = claim.get_interval_for("dgt1")
        assert interval.start.value == 0.0
        assert interval.end.value == 975.0

    def test_from_events_factory(self) -> None:
        """Test from_events() factory method (case a: event-based)."""
        claim = MatchClaim.from_events(
            event_a={"start": 100.0},
            tl_a_id="score",
            event_b={"start": 45.5},
            tl_b_id="recording",
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
        )

        assert claim.is_synchronous is True
        assert claim.timeline_a_id == "score"
        assert claim.timeline_b_id == "recording"
        assert claim.start_anchor is not None
        assert claim.start_anchor.coordinate_a.value == 100.0
        assert claim.start_anchor.coordinate_b.value == 45.5

    def test_from_events_with_interval(self) -> None:
        """Test from_events() with end coordinate (case a: interval)."""
        claim = MatchClaim.from_events(
            event_a={"start": 0.0, "end": 100.0},
            tl_a_id="tl1",
            event_b={"start": 0.0, "end": 200.0},
            tl_b_id="tl2",
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
            end_coord_key="end",
        )

        assert claim.is_interval is True
        assert claim.end_anchor is not None
        assert claim.end_anchor.coordinate_a.value == 100.0
        assert claim.end_anchor.coordinate_b.value == 200.0

    def test_from_projection_factory(self) -> None:
        """Test from_projection() factory method (case b: projection)."""
        claim = MatchClaim.from_projection(
            event={"start": 100.0},
            source_tl_id="score",
            target_tl_id="recording",
            target_coord=45.5,
            source_unit=TimeUnit.number,
            target_unit=TimeUnit.number,
        )

        assert claim.is_synchronous is True
        assert claim.timeline_a_id == "score"
        assert claim.timeline_b_id == "recording"
        assert claim.start_anchor is not None
        assert claim.start_anchor.coordinate_a.value == 100.0
        assert claim.start_anchor.coordinate_b.value == 45.5

    def test_coordinate_inputs_supply_units(self) -> None:
        """Coordinate and IdCoordinate inputs make explicit units optional."""
        projected = MatchClaim.from_projection(
            event={"start": 100.0},
            source_tl_id="score",
            target_tl_id="recording",
            target_coord=IdCoordinate(45.5, TimeUnit.seconds, "recording"),
            source_unit=TimeUnit.number,
        )
        implicit = MatchClaim.implicit(
            tl_a_id="score",
            coord_a=Coordinate(100.0, TimeUnit.quarters),
            tl_b_id="recording",
            coord_b=IdCoordinate(45.5, TimeUnit.seconds, "recording"),
        )

        assert projected.start_anchor.coordinate_b == IdCoordinate(
            45.5, TimeUnit.seconds, "recording"
        )
        assert implicit.start_anchor.coordinate_a == Coordinate(
            100.0, TimeUnit.quarters
        )
        assert implicit.start_anchor.coordinate_b == IdCoordinate(
            45.5, TimeUnit.seconds, "recording"
        )

    def test_coordinate_input_mismatches_raise(self) -> None:
        """Conflicting units and timeline IDs are rejected."""
        with pytest.raises(ValueError, match="unit"):
            MatchClaim.implicit(
                tl_a_id="score",
                coord_a=Coordinate(100.0, TimeUnit.quarters),
                tl_b_id="recording",
                coord_b=45.5,
                unit_a=TimeUnit.seconds,
                unit_b=TimeUnit.seconds,
            )
        with pytest.raises(ValueError, match="timeline_id"):
            MatchClaim.from_projection(
                event={"start": 100.0},
                source_tl_id="score",
                target_tl_id="recording",
                target_coord=IdCoordinate(45.5, TimeUnit.seconds, "other"),
                source_unit=TimeUnit.number,
            )
        with pytest.raises(ValueError, match="explicit unit"):
            MatchClaim.from_projection(
                event={"start": 100.0},
                source_tl_id="score",
                target_tl_id="recording",
                target_coord=Coordinate(45.5, TimeUnit.seconds),
                source_unit=TimeUnit.number,
                target_unit=TimeUnit.frames,
            )

    def test_fraction_claim_dict_is_json_safe_and_exact(self) -> None:
        """Fraction coordinates serialize to JSON and restore exactly."""
        claim = MatchClaim(
            timeline_a_id="score",
            timeline_b_id="recording",
            start_anchor=AlignmentAnchor(
                timeline_a_id="score",
                coordinate_a=Coordinate(Fraction(7, 3), TimeUnit.quarters),
                timeline_b_id="recording",
                coordinate_b=Coordinate(2.0, TimeUnit.seconds),
            ),
        )

        serialized = claim.to_dict()
        json.dumps(serialized)
        restored = MatchClaim.from_dict(serialized)

        assert restored.start_anchor.coordinate_a.value == Fraction(7, 3)

    def test_nomatch_factory(self) -> None:
        """Test nomatch() factory method (case c: no match)."""
        claim = MatchClaim.nomatch(
            event={"start": 100.0},
            source_tl_id="score",
            target_tl_id="recording",
            unit=TimeUnit.number,
        )

        assert claim.is_synchronous is False
        assert claim.start_anchor is None
        assert claim.end_anchor is None
        assert claim.timeline_a_id == "score"
        assert claim.timeline_b_id == "recording"
        assert claim.anchors == []

    def test_implicit_factory(self) -> None:
        """Test implicit() factory method (case d: inferred)."""
        claim = MatchClaim.implicit(
            tl_a_id="tl1",
            coord_a=100.0,
            tl_b_id="tl2",
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
            coord_b=200.0,
        )

        assert claim.is_synchronous is True
        assert claim.is_explicit is False
        assert claim.start_anchor is not None
        assert claim.start_anchor.coordinate_a.value == 100.0
        assert claim.timeline_a_id == "tl1"
        assert claim.timeline_b_id == "tl2"

    def test_implicit_with_source_claim(self) -> None:
        """Test implicit() tracks the source claim."""
        source = MatchClaim(
            timeline_a_id="tl1",
            timeline_b_id="tl2",
            start_anchor=AlignmentAnchor(
                timeline_a_id="tl1",
                coordinate_a=Coordinate(100.0, TimeUnit.number),
                timeline_b_id="tl2",
                coordinate_b=Coordinate(200.0, TimeUnit.number),
            ),
        )
        implicit = MatchClaim.implicit(
            tl_a_id="tl1",
            coord_a=100.0,
            tl_b_id="tl3",
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
            coord_b=300.0,
            source_claim=source,
        )

        assert implicit.source_claim_id == source.id

    def test_to_dict_instant(self, basic_anchor: AlignmentAnchor) -> None:
        """Test serialization of instant match."""
        claim = MatchClaim(
            timeline_a_id="score:1",
            timeline_b_id="recording:1",
            start_anchor=basic_anchor,
        )
        d = claim.to_dict()

        assert "id" in d
        assert "timeline_a_id" in d
        assert "timeline_b_id" in d
        assert "start_anchor" in d
        assert d["end_anchor"] is None
        assert d["is_explicit"] is True
        assert d["is_synchronous"] is True

    def test_to_dict_interval(self) -> None:
        """Test serialization of interval match."""
        claim = MatchClaim(
            timeline_a_id="tl1",
            timeline_b_id="tl2",
            start_anchor=AlignmentAnchor(
                timeline_a_id="tl1",
                coordinate_a=Coordinate(0.0, TimeUnit.number),
                timeline_b_id="tl2",
                coordinate_b=Coordinate(0.0, TimeUnit.number),
            ),
            end_anchor=AlignmentAnchor(
                timeline_a_id="tl1",
                coordinate_a=Coordinate(100.0, TimeUnit.number),
                timeline_b_id="tl2",
                coordinate_b=Coordinate(100.0, TimeUnit.number),
            ),
        )
        d = claim.to_dict()

        assert d["start_anchor"] is not None
        assert d["end_anchor"] is not None

    def test_from_dict_roundtrip_instant(self, basic_anchor: AlignmentAnchor) -> None:
        """Test serialization round-trip for instant match."""
        claim = MatchClaim(
            timeline_a_id="score:1",
            timeline_b_id="recording:1",
            start_anchor=basic_anchor,
        )
        d = claim.to_dict()
        restored = MatchClaim.from_dict(d)

        assert restored.id == claim.id
        assert restored.is_interval is False
        assert restored.timeline_a_id == claim.timeline_a_id

    def test_from_dict_roundtrip_interval(self) -> None:
        """Test serialization round-trip for interval match."""
        claim = MatchClaim(
            timeline_a_id="tl1",
            timeline_b_id="tl2",
            start_anchor=AlignmentAnchor(
                timeline_a_id="tl1",
                coordinate_a=Coordinate(0.0, TimeUnit.number),
                timeline_b_id="tl2",
                coordinate_b=Coordinate(0.0, TimeUnit.number),
            ),
            end_anchor=AlignmentAnchor(
                timeline_a_id="tl1",
                coordinate_a=Coordinate(100.0, TimeUnit.number),
                timeline_b_id="tl2",
                coordinate_b=Coordinate(200.0, TimeUnit.number),
            ),
            metadata=MatchMetadata(
                agent=Agent(name="test", type=AgentType.software, identifier="test")
            ),
        )
        d = claim.to_dict()
        restored = MatchClaim.from_dict(d)

        assert restored.id == claim.id
        assert restored.is_interval is True
        assert restored.metadata is not None
        assert restored.metadata.agent.name == "test"

    def test_from_dict_roundtrip_non_synchronous(self) -> None:
        """Test serialization round-trip for non-synchronous claim."""
        claim = MatchClaim(
            timeline_a_id="tl1",
            timeline_b_id="tl2",
            is_synchronous=False,
        )
        d = claim.to_dict()
        restored = MatchClaim.from_dict(d)

        assert restored.is_synchronous is False
        assert restored.start_anchor is None
        assert restored.timeline_a_id == "tl1"
        assert restored.timeline_b_id == "tl2"

    def test_frozen_model(self, basic_anchor: AlignmentAnchor) -> None:
        """Test that MatchClaim is immutable (frozen pydantic model)."""
        claim = MatchClaim(
            timeline_a_id="score:1",
            timeline_b_id="recording:1",
            start_anchor=basic_anchor,
        )
        with pytest.raises(ValidationError):
            claim.is_explicit = False  # type: ignore

    def test_repr_instant(self, basic_anchor: AlignmentAnchor) -> None:
        """Test string representation of instant match."""
        claim = MatchClaim(
            timeline_a_id="score:1",
            timeline_b_id="recording:1",
            start_anchor=basic_anchor,
        )
        r = repr(claim)

        assert "instant" in r
        assert "score:1" in r
        assert "recording:1" in r

    def test_repr_interval(self) -> None:
        """Test string representation of interval match."""
        claim = MatchClaim(
            timeline_a_id="dgt1",
            timeline_b_id="dgt2",
            start_anchor=AlignmentAnchor(
                timeline_a_id="dgt1",
                coordinate_a=Coordinate(0.0, TimeUnit.number),
                timeline_b_id="dgt2",
                coordinate_b=Coordinate(0.0, TimeUnit.number),
            ),
            end_anchor=AlignmentAnchor(
                timeline_a_id="dgt1",
                coordinate_a=Coordinate(975.0, TimeUnit.number),
                timeline_b_id="dgt2",
                coordinate_b=Coordinate(866.0, TimeUnit.number),
            ),
        )
        assert repr(claim) == (
            "MatchClaim(interval: dgt1[0 number-975 number] <-> "
            "dgt2[0 number-866 number] [ANCHOR])"
        )

    def test_repr_non_synchronous(self) -> None:
        """A NOMATCH claim (orphaned event) renders the NOMATCH flag.

        The orphaned event is named on exactly one side, so the claim is a
        genuine NOMATCH; the flag word must appear and never 'non-synchronous'.
        """
        claim = MatchClaim(
            timeline_a_id="tl1",
            timeline_b_id="tl2",
            is_synchronous=False,
            event_a_id="e_orphan",
        )
        r = repr(claim)
        assert "NOMATCH" in r
        assert "non-synchronous" not in r
        assert "tl1" in r
        assert "tl2" in r

    def test_repr_nomatch_with_coordinate_exact(self) -> None:
        """NOMATCH claim from nomatch() renders the source coordinate exactly.

        The event names the orphaned note (``id``), so the claim classifies as
        a genuine NOMATCH rather than a conceptual correspondence.
        """
        claim = MatchClaim.nomatch(
            event={"id": "orphan", "start": 188.8},
            source_tl_id="score:clt1",
            target_tl_id="perf:Chopin_Ashkenazy",
            unit=TimeUnit.number,
        )
        assert claim.source_coordinate.value == 188.8
        assert repr(claim) == (
            "MatchClaim(score:clt1@188.8 number <-> " "perf:Chopin_Ashkenazy [NOMATCH])"
        )

    def test_repr_nomatch_without_coordinate(self) -> None:
        """NOMATCH claim with no 'start' falls back to bare timeline_a."""
        claim = MatchClaim.nomatch(
            event={"id": "orphan"},
            source_tl_id="score:clt1",
            target_tl_id="perf:Chopin_Ashkenazy",
            unit=TimeUnit.number,
        )
        assert claim.source_coordinate is None
        assert repr(claim) == (
            "MatchClaim(score:clt1 <-> perf:Chopin_Ashkenazy [NOMATCH])"
        )

    def test_nomatch_source_coordinate_roundtrip(self) -> None:
        """to_dict()/from_dict() preserves source_coordinate for a NOMATCH claim."""
        claim = MatchClaim.nomatch(
            event={"start": 188.8},
            source_tl_id="score:clt1",
            target_tl_id="perf:Chopin_Ashkenazy",
            unit=TimeUnit.number,
        )
        restored = MatchClaim.from_dict(claim.to_dict())
        assert restored == claim
        assert restored.source_coordinate.value == 188.8

    def test_repr_synchronous_instant_with_units(self) -> None:
        """An event-match instant repr shows unit-bearing coordinates.

        Two identified events on either side make this an event_match, which
        carries no badge — the common case must stay free of NOMATCH or any
        other kind tag.
        """
        claim = MatchClaim(
            timeline_a_id="score:1",
            timeline_b_id="recording:1",
            start_anchor=AlignmentAnchor(
                timeline_a_id="score:1",
                coordinate_a=Coordinate(100.0, TimeUnit.number),
                timeline_b_id="recording:1",
                coordinate_b=Coordinate(45.5, TimeUnit.number),
            ),
            event_a_id="a1",
            event_b_id="b1",
        )
        assert claim.source_coordinate is None
        assert repr(claim) == (
            "MatchClaim(instant: score:1@100 number <-> recording:1@45.5 number)"
        )

    def test_exact_rational_rendering(self) -> None:
        """Claims render each coordinate the way its own axis writes numbers.

        Both halves are handed an exact ratio, and the two axes answer
        differently because they declare different representations: quarters
        keep ``415/24``, seconds -- which are float-canonical -- report
        ``12.5``. An anchor coordinate is a position on a named timeline, so
        the axis decides, not the caller; this is the same rule that makes a
        claim and its matchstamp agree about one position.
        """
        anchor = AlignmentAnchor(
            timeline_a_id="audio",
            coordinate_a=Coordinate(Fraction(25, 2), TimeUnit.seconds),
            timeline_b_id="score",
            coordinate_b=Coordinate(Fraction(415, 24), TimeUnit.quarters),
        )
        claim = MatchClaim(
            timeline_a_id="audio",
            timeline_b_id="score",
            start_anchor=anchor,
        )

        assert repr(claim) == (
            "MatchClaim(instant: audio@12.5 seconds <-> "
            "score@415/24 quarters [ANCHOR])"
        )
        assert str(claim) == (
            "MatchClaim (synchronous, instant)\n"
            "  Timeline A:  audio  @12.5 seconds\n"
            "  Timeline B:  score  @415/24 quarters"
        )
        assert claim._repr_html_() == (
            "<div style='font-family: monospace;'><strong>MatchClaim</strong> "
            "<span style='background: #fff3e0; padding: 0 4px; border-radius: 3px; "
            "font-size: 0.8em;'>ANCHOR</span><table style='border-collapse: collapse; "
            "margin-top: 4px;'><tbody><tr><td>Timeline A</td><td><strong>audio"
            "</strong></td><td>@12.5 seconds</td></tr><tr><td>Timeline B</td><td>"
            "<strong>score</strong></td><td>@415/24 quarters</td></tr></tbody></table>"
            "<div style='margin-top: 4px; color: #666; font-size: 0.85em;'>Try: "
            "<code>claim.get_matchstamp()</code></div></div>"
        )
        assert repr(anchor) == (
            "AlignmentAnchor(audio@12.5 seconds <-> score@415/24 quarters)"
        )
        assert str(anchor) == repr(anchor)
        assert anchor._repr_html_() == (
            "<div style='font-family: monospace;'><strong>AlignmentAnchor</strong>("
            "audio@12.5 seconds &lt;-&gt; score@415/24 quarters)</div>"
        )

    @pytest.mark.parametrize(
        ("coordinate", "expected"),
        [
            (Coordinate(480, TimeUnit.ticks), "480 ticks"),
            (Coordinate(12.3456789, TimeUnit.seconds), "12.3456789 seconds"),
        ],
    )
    def test_integer_and_float_coordinate_rendering(
        self, coordinate: Coordinate, expected: str
    ) -> None:
        """Integer discrete and float coordinates retain their display type and unit."""
        anchor = AlignmentAnchor(
            timeline_a_id="a",
            coordinate_a=coordinate,
            timeline_b_id="b",
            coordinate_b=coordinate,
        )

        assert repr(anchor) == f"AlignmentAnchor(a@{expected} <-> b@{expected})"


# endregion


# region Integration Tests


class TestClaimIntegration:
    """Integration tests for alignment claims."""

    def test_thoresen_segment_claims(self) -> None:
        """Test creating segment claims similar to Thoresen PoC.

        Creates 5 interval matches for segment correspondence:
        - DGT1 segments: 5 equal (975 px each)
        - DGT2 segments: 5 varying (866, 867, 867, 864, 864)
        """
        segment_lengths_dgt1 = [975, 975, 975, 975, 975]
        segment_lengths_dgt2 = [866, 867, 867, 864, 864]

        claims = []
        offset_dgt1 = 0
        offset_dgt2 = 0

        for i in range(5):
            claim = MatchClaim(
                timeline_a_id="dgt1",
                timeline_b_id="dgt2",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="dgt1",
                    coordinate_a=Coordinate(float(offset_dgt1), TimeUnit.number),
                    timeline_b_id="dgt2",
                    coordinate_b=Coordinate(float(offset_dgt2), TimeUnit.number),
                ),
                end_anchor=AlignmentAnchor(
                    timeline_a_id="dgt1",
                    coordinate_a=Coordinate(
                        float(offset_dgt1 + segment_lengths_dgt1[i]), TimeUnit.number
                    ),
                    timeline_b_id="dgt2",
                    coordinate_b=Coordinate(
                        float(offset_dgt2 + segment_lengths_dgt2[i]), TimeUnit.number
                    ),
                ),
                metadata=MatchMetadata(
                    agent=Agent(
                        name="analyst",
                        type=AgentType.human,
                        identifier="segment_correspondence",
                    ),
                ),
            )
            claims.append(claim)

            offset_dgt1 += segment_lengths_dgt1[i]
            offset_dgt2 += segment_lengths_dgt2[i]

        # Verify
        assert len(claims) == 5
        assert all(c.is_interval for c in claims)
        assert all(c.connects_both("dgt1", "dgt2") for c in claims)

        # Check first segment
        interval = claims[0].get_interval_for("dgt1")
        assert interval.start.value == 0.0
        assert interval.end.value == 975.0

        interval = claims[0].get_interval_for("dgt2")
        assert interval.start.value == 0.0
        assert interval.end.value == 866.0

        # Check last segment ends at total length
        end_dgt1 = claims[4].get_interval_for("dgt1").end
        end_dgt2 = claims[4].get_interval_for("dgt2").end
        assert end_dgt1.value == 4875.0
        assert end_dgt2.value == 4328.0


# endregion
