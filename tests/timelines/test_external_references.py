"""Tests for incoming external references and the ``to_dict()`` flags.

Two contracts are exercised here:

1. the per-timeline table of external annotations pointing at this
   timeline's events -- its canonical Arrow schema, the normalization
   ``add_external_references()`` applies, and the ``KeyError`` it raises for
   unknown event ids;
2. the slimmed ``Timeline.to_dict()``, whose ``events`` and
   ``external_references`` keys are absent unless requested, forwarded to
   children, and tolerated as absent by ``from_dict()``.

Every timeline here is built inline with an explicit ``uid``, so the tests
share no state and run in any order.
"""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

import pyarrow as pa
import pytest

from timetoalign.core import NumberType, TimeUnit
from timetoalign.timelines import (
    ContinuousPhysicalTimeline,
    Timeline,
)
from timetoalign.timelines.engines.external_references import (
    EXTERNAL_REFERENCE_SCHEMA,
)

# region Helpers

#: Keys every timeline object of ``to_dict()`` carries, whatever the flags.
STRUCTURAL_KEYS = {
    "id",
    "name",
    "unit",
    "number_type",
    "length",
    "is_segment_line",
    "locked",
    "meta",
    "children",
    "conversion_maps",
    "regions",
    "flow_control",
    "measure_map",
    "metric_hierarchy",
}


def _object(timeline: Timeline, **flags: bool) -> dict[str, Any]:
    """The timeline object ``to_dict`` writes for *timeline*, its one root."""
    return timeline.to_dict(**flags)["timelines"][0]


def _timeline(uid: str = "spine", length: float = 10.0) -> Timeline:
    """A seconds timeline carrying two instant events, ``e1`` and ``e2``."""
    timeline = Timeline(length=length, unit=TimeUnit.seconds, uid=uid)
    timeline.add_events(
        [
            {
                "id": "e1",
                "temporal_type": "instant",
                "event_type": "Beat",
                "instant": 0.0,
            },
            {
                "id": "e2",
                "temporal_type": "instant",
                "event_type": "Beat",
                "instant": 5.0,
            },
        ]
    )
    return timeline


def _reference_row(
    event_id: str = "e1",
    external_id: str = "p2",
) -> dict[str, Any]:
    """One fully populated external-reference row."""
    return {
        "event_id": event_id,
        "external_id": external_id,
        "access_points": [{"uri": "Analisi_1/L1.pnml", "kind": "relative_path"}],
        "comment": "Analisi_1_L1_A",
    }


def _hierarchy() -> Timeline:
    """A parent/child/grandchild hierarchy, each level with one event."""
    parent = _timeline("parent", length=20.0)
    child = _timeline("child", length=8.0)
    grandchild = _timeline("grandchild", length=6.0)
    child.add_child(grandchild, offset=1.0)
    parent.add_child(child, offset=10.0)
    return parent


# endregion


# region Schema


class TestExternalReferenceSchema:
    """The table always carries the canonical schema, empty or not."""

    def test_fresh_timeline_returns_an_empty_table(self) -> None:
        assert _timeline().external_references.num_rows == 0

    def test_fresh_timeline_returns_a_table_not_none(self) -> None:
        assert isinstance(_timeline().external_references, pa.Table)

    def test_empty_table_carries_the_canonical_schema(self) -> None:
        assert _timeline().external_references.schema.equals(EXTERNAL_REFERENCE_SCHEMA)

    def test_populated_table_carries_the_canonical_schema(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        assert timeline.external_references.schema.equals(EXTERNAL_REFERENCE_SCHEMA)

    def test_column_names_are_exact(self) -> None:
        assert EXTERNAL_REFERENCE_SCHEMA.names == [
            "event_id",
            "external_id",
            "access_points",
            "comment",
        ]

    def test_event_id_is_a_non_null_string(self) -> None:
        field = EXTERNAL_REFERENCE_SCHEMA.field("event_id")
        assert field.type == pa.string()
        assert field.nullable is False

    def test_external_id_is_a_non_null_string(self) -> None:
        field = EXTERNAL_REFERENCE_SCHEMA.field("external_id")
        assert field.type == pa.string()
        assert field.nullable is False

    def test_access_points_is_a_non_null_list_of_structs(self) -> None:
        field = EXTERNAL_REFERENCE_SCHEMA.field("access_points")
        assert field.nullable is False
        assert field.type == pa.list_(
            pa.struct(
                [
                    pa.field("uri", pa.string(), nullable=False),
                    pa.field("kind", pa.string(), nullable=False),
                ]
            )
        )

    def test_comment_is_a_nullable_string(self) -> None:
        field = EXTERNAL_REFERENCE_SCHEMA.field("comment")
        assert field.type == pa.string()
        assert field.nullable is True

    @pytest.mark.parametrize(
        ("unit", "number_type", "length"),
        [
            (TimeUnit.seconds, NumberType.float, 10.0),
            (TimeUnit.quarters, NumberType.fraction, Fraction(10, 3)),
            (TimeUnit.pixels, NumberType.int, 1920),
        ],
        ids=["seconds", "quarters", "pixels"],
    )
    def test_schema_is_independent_of_the_timeline_type(
        self, unit: TimeUnit, number_type: NumberType, length: Any
    ) -> None:
        timeline = Timeline(
            length=length, unit=unit, number_type=number_type, uid="typed"
        )
        assert timeline.external_references.schema.equals(EXTERNAL_REFERENCE_SCHEMA)


# endregion


# region Adding


class TestAddExternalReferences:
    """Rows are normalized, appended, and never deduplicated."""

    def test_a_full_row_is_stored_verbatim(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        assert timeline.external_references.to_pylist() == [
            {
                "event_id": "e1",
                "external_id": "p2",
                "access_points": [
                    {"uri": "Analisi_1/L1.pnml", "kind": "relative_path"}
                ],
                "comment": "Analisi_1_L1_A",
            }
        ]

    def test_missing_access_points_become_an_empty_list(self) -> None:
        timeline = _timeline().add_external_references(
            [{"event_id": "e1", "external_id": "p2", "comment": "no locator"}]
        )
        assert timeline.external_references.column("access_points").to_pylist() == [[]]

    def test_explicit_none_access_points_become_an_empty_list(self) -> None:
        timeline = _timeline().add_external_references(
            [{"event_id": "e1", "external_id": "p2", "access_points": None}]
        )
        assert timeline.external_references.column("access_points").to_pylist() == [[]]

    def test_missing_comment_becomes_none(self) -> None:
        timeline = _timeline().add_external_references(
            [{"event_id": "e1", "external_id": "p2"}]
        )
        assert timeline.external_references.column("comment").to_pylist() == [None]

    def test_several_access_points_keep_their_order(self) -> None:
        timeline = _timeline().add_external_references(
            [
                {
                    "event_id": "e1",
                    "external_id": "p2",
                    "access_points": [
                        {"uri": "Analisi_1/L1.pnml", "kind": "relative_path"},
                        {"uri": "https://example.org/L1.pnml", "kind": "url"},
                    ],
                }
            ]
        )
        assert timeline.external_references.column("access_points").to_pylist() == [
            [
                {"uri": "Analisi_1/L1.pnml", "kind": "relative_path"},
                {"uri": "https://example.org/L1.pnml", "kind": "url"},
            ]
        ]

    def test_calls_append_rather_than_replace(self) -> None:
        timeline = _timeline()
        timeline.add_external_references([_reference_row("e1", "p1")])
        timeline.add_external_references([_reference_row("e2", "p2")])
        assert timeline.external_references.column("external_id").to_pylist() == [
            "p1",
            "p2",
        ]

    def test_one_call_of_two_rows_matches_two_calls_of_one(self) -> None:
        batched = _timeline().add_external_references(
            [_reference_row("e1", "p1"), _reference_row("e2", "p2")]
        )
        stepwise = _timeline()
        stepwise.add_external_references([_reference_row("e1", "p1")])
        stepwise.add_external_references([_reference_row("e2", "p2")])
        assert batched.external_references.equals(stepwise.external_references)

    def test_duplicate_rows_are_kept(self) -> None:
        timeline = _timeline().add_external_references(
            [_reference_row(), _reference_row()]
        )
        assert timeline.external_references.num_rows == 2

    def test_empty_row_list_leaves_the_table_empty(self) -> None:
        timeline = _timeline().add_external_references([])
        assert timeline.external_references.num_rows == 0

    def test_the_call_returns_self(self) -> None:
        timeline = _timeline()
        assert timeline.add_external_references([_reference_row()]) is timeline

    def test_a_pyarrow_table_is_accepted(self) -> None:
        source = pa.Table.from_pylist(
            [_reference_row()], schema=EXTERNAL_REFERENCE_SCHEMA
        )
        timeline = _timeline().add_external_references(source)
        assert timeline.external_references.equals(source)

    def test_a_pyarrow_table_appends_like_dict_rows(self) -> None:
        source = pa.Table.from_pylist(
            [_reference_row("e2", "p9")], schema=EXTERNAL_REFERENCE_SCHEMA
        )
        timeline = _timeline().add_external_references([_reference_row("e1", "p1")])
        timeline.add_external_references(source)
        assert timeline.external_references.column("event_id").to_pylist() == [
            "e1",
            "e2",
        ]


class TestAddExternalReferencesRejections:
    """Malformed rows are rejected with a message naming the offender."""

    def test_an_unknown_column_is_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _timeline().add_external_references(
                [{"event_id": "e1", "external_id": "p2", "kind": "url"}]
            )
        assert (
            str(excinfo.value) == "External reference row 0: unknown column(s) 'kind'"
        )

    def test_every_unknown_column_is_named(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _timeline().add_external_references(
                [{"event_id": "e1", "external_id": "p2", "uri": "x", "zzz": 1}]
            )
        assert (
            str(excinfo.value)
            == "External reference row 0: unknown column(s) 'uri', 'zzz'"
        )

    def test_the_offending_row_index_is_reported(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _timeline().add_external_references(
                [_reference_row(), {"event_id": "e2", "external_id": "p3", "x": 1}]
            )
        assert str(excinfo.value) == "External reference row 1: unknown column(s) 'x'"

    def test_a_missing_event_id_is_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _timeline().add_external_references([{"external_id": "p2"}])
        assert (
            str(excinfo.value)
            == "External reference row 0: 'event_id' must be a string, got None"
        )

    def test_a_non_string_external_id_is_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _timeline().add_external_references([{"event_id": "e1", "external_id": 2}])
        assert (
            str(excinfo.value)
            == "External reference row 0: 'external_id' must be a string, got 2"
        )

    def test_a_non_string_comment_is_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _timeline().add_external_references(
                [{"event_id": "e1", "external_id": "p2", "comment": 7}]
            )
        assert (
            str(excinfo.value)
            == "External reference row 0: 'comment' must be a string or None, got 7"
        )

    def test_a_non_mapping_row_is_rejected(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            _timeline().add_external_references(["e1"])
        assert (
            str(excinfo.value) == "External reference row 0 must be a mapping, got 'e1'"
        )

    def test_a_scalar_access_points_cell_is_rejected(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            _timeline().add_external_references(
                [{"event_id": "e1", "external_id": "p2", "access_points": "L1.pnml"}]
            )
        assert str(excinfo.value) == (
            "External reference row 0: 'access_points' must be a list of "
            "mappings, got 'L1.pnml'"
        )

    def test_a_non_mapping_access_point_is_rejected(self) -> None:
        with pytest.raises(TypeError) as excinfo:
            _timeline().add_external_references(
                [{"event_id": "e1", "external_id": "p2", "access_points": ["x"]}]
            )
        assert str(excinfo.value) == (
            "External reference row 0: each access point must be a mapping, got 'x'"
        )

    def test_an_unknown_access_point_key_is_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _timeline().add_external_references(
                [
                    {
                        "event_id": "e1",
                        "external_id": "p2",
                        "access_points": [
                            {"uri": "L1.pnml", "kind": "url", "note": "x"}
                        ],
                    }
                ]
            )
        assert str(excinfo.value) == (
            "External reference row 0: unknown access-point key(s) 'note'"
        )

    def test_an_access_point_without_a_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _timeline().add_external_references(
                [
                    {
                        "event_id": "e1",
                        "external_id": "p2",
                        "access_points": [{"uri": "L1.pnml"}],
                    }
                ]
            )
        assert str(excinfo.value) == (
            "External reference row 0: each access point needs string 'uri' "
            "and 'kind', got {'uri': 'L1.pnml'}"
        )

    def test_a_rejected_call_leaves_the_table_unchanged(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        before = timeline.external_references
        with pytest.raises(ValueError):
            timeline.add_external_references(
                [{"event_id": "e2", "external_id": "p3", "bogus": 1}]
            )
        assert timeline.external_references.equals(before)


# endregion


# region Event-id validation


class TestExternalReferenceValidation:
    """``validate=True`` demands that every event id exists locally."""

    def test_a_known_event_id_is_accepted(self) -> None:
        timeline = _timeline().add_external_references([_reference_row("e2", "p2")])
        assert timeline.external_references.column("event_id").to_pylist() == ["e2"]

    def test_an_unknown_event_id_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            _timeline().add_external_references(
                [{"event_id": "ghost", "external_id": "p2"}]
            )

    def test_the_message_names_the_missing_id(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            _timeline().add_external_references(
                [{"event_id": "ghost", "external_id": "p2"}]
            )
        assert excinfo.value.args[0] == (
            "Timeline 'spine' has no event(s) with id: 'ghost'"
        )

    def test_the_message_names_every_missing_id_sorted(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            _timeline().add_external_references(
                [
                    {"event_id": "ghost_b", "external_id": "p1"},
                    {"event_id": "e1", "external_id": "p2"},
                    {"event_id": "ghost_a", "external_id": "p3"},
                ]
            )
        assert excinfo.value.args[0] == (
            "Timeline 'spine' has no event(s) with id: 'ghost_a', 'ghost_b'"
        )

    def test_a_repeated_missing_id_is_named_once(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            _timeline().add_external_references(
                [
                    {"event_id": "ghost", "external_id": "p1"},
                    {"event_id": "ghost", "external_id": "p2"},
                ]
            )
        assert excinfo.value.args[0] == (
            "Timeline 'spine' has no event(s) with id: 'ghost'"
        )

    def test_nothing_is_appended_when_validation_fails(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        with pytest.raises(KeyError):
            timeline.add_external_references(
                [_reference_row("e2", "p3"), {"event_id": "ghost", "external_id": "p4"}]
            )
        assert timeline.external_references.column("external_id").to_pylist() == ["p2"]

    def test_a_child_event_id_does_not_satisfy_validation(self) -> None:
        parent = Timeline(length=20.0, unit=TimeUnit.seconds, uid="parent")
        parent.add_child(_timeline("child", length=8.0), offset=10.0)
        with pytest.raises(KeyError) as excinfo:
            parent.add_external_references([{"event_id": "e1", "external_id": "p2"}])
        assert excinfo.value.args[0] == (
            "Timeline 'parent' has no event(s) with id: 'e1'"
        )

    def test_validate_false_accepts_an_unknown_event_id(self) -> None:
        timeline = _timeline().add_external_references(
            [{"event_id": "ghost", "external_id": "p2"}], validate=False
        )
        assert timeline.external_references.column("event_id").to_pylist() == ["ghost"]

    def test_validate_false_still_rejects_a_malformed_row(self) -> None:
        with pytest.raises(ValueError):
            _timeline().add_external_references(
                [{"event_id": "e1", "external_id": "p2", "bogus": 1}], validate=False
            )

    def test_an_eventless_timeline_rejects_every_id(self) -> None:
        timeline = Timeline(length=5.0, unit=TimeUnit.seconds, uid="bare")
        with pytest.raises(KeyError) as excinfo:
            timeline.add_external_references([{"event_id": "e1", "external_id": "p2"}])
        assert excinfo.value.args[0] == (
            "Timeline 'bare' has no event(s) with id: 'e1'"
        )


# endregion


# region to_dict flag matrix


class TestToDictFlags:
    """Both content keys are opt-in and otherwise absent."""

    def test_default_output_has_exactly_the_structural_keys(self) -> None:
        assert set(_object(_timeline())) == STRUCTURAL_KEYS

    def test_default_output_omits_events(self) -> None:
        assert "events" not in _object(_timeline())

    def test_default_output_omits_external_references(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        assert "external_references" not in _object(timeline)

    def test_events_flag_adds_only_the_events_key(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        assert set(_object(timeline, events=True)) == STRUCTURAL_KEYS | {"events"}

    def test_events_flag_emits_every_event(self) -> None:
        data = _object(_timeline(), events=True)
        assert [event["id"] for event in data["events"]] == ["e1", "e2"]

    def test_external_references_flag_adds_only_that_key(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        assert set(_object(timeline, external_references=True)) == STRUCTURAL_KEYS | {
            "external_references"
        }

    def test_both_flags_add_both_keys(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        assert set(
            _object(timeline, events=True, external_references=True)
        ) == STRUCTURAL_KEYS | {
            "events",
            "external_references",
        }

    def test_an_empty_reference_table_is_emitted_as_an_empty_list(self) -> None:
        data = _object(_timeline(), external_references=True)
        assert data["external_references"] == []

    def test_reference_rows_serialize_as_plain_dicts(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        data = _object(timeline, external_references=True)
        assert data["external_references"] == [
            {
                "event_id": "e1",
                "external_id": "p2",
                "access_points": [
                    {"uri": "Analisi_1/L1.pnml", "kind": "relative_path"}
                ],
                "comment": "Analisi_1_L1_A",
            }
        ]

    def test_the_reference_payload_is_json_safe(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        data = _object(timeline, events=True, external_references=True)
        assert (
            json.loads(json.dumps(data))["external_references"]
            == data["external_references"]
        )

    def test_the_flags_do_not_disturb_the_structural_values(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        plain = _object(timeline)
        rich = _object(timeline, events=True, external_references=True)
        assert {key: rich[key] for key in STRUCTURAL_KEYS} == plain

    def test_the_fraction_json_format_is_untouched(self) -> None:
        timeline = Timeline(
            length=Fraction(10, 3),
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
            uid="rational",
        )
        data = _object(timeline, events=True, external_references=True)
        assert data["length"] == [10, 3]


class TestToDictChildForwarding:
    """Both flags travel down every level of the hierarchy."""

    @staticmethod
    def _levels(data: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the payloads of ``parent``, ``child``, and ``grandchild``."""
        (child,) = data["children"]
        (grandchild,) = child["children"]
        return [data, child, grandchild]

    @pytest.mark.parametrize(
        ("events", "external_references", "expected_extra"),
        [
            (False, False, set()),
            (True, False, {"events"}),
            (False, True, {"external_references"}),
            (True, True, {"events", "external_references"}),
        ],
        ids=["neither", "events", "references", "both"],
    )
    def test_every_level_carries_the_requested_keys(
        self,
        events: bool,
        external_references: bool,
        expected_extra: set[str],
    ) -> None:
        data = _object(
            _hierarchy(), events=events, external_references=external_references
        )
        assert [set(level) for level in self._levels(data)] == [
            STRUCTURAL_KEYS | expected_extra,
            STRUCTURAL_KEYS | expected_extra | {"offset"},
            STRUCTURAL_KEYS | expected_extra | {"offset"},
        ]

    def test_child_events_are_the_child_s_own(self) -> None:
        # The trailing ``grandchild`` row is the segment event that
        # ``add_child`` records; it is regenerated by ``from_dict``.
        data = _object(_hierarchy(), events=True)
        (child,) = data["children"]
        assert [event["id"] for event in child["events"]] == [
            "e1",
            "e2",
            "grandchild",
        ]

    def test_each_level_emits_its_own_references(self) -> None:
        parent = _hierarchy()
        parent.add_external_references([_reference_row("e1", "parent-ref")])
        parent.get_child("child").add_external_references(
            [_reference_row("e2", "child-ref")]
        )

        data = _object(parent, external_references=True)
        emitted = [
            [row["external_id"] for row in level["external_references"]]
            for level in self._levels(data)
        ]
        assert emitted == [["parent-ref"], ["child-ref"], []]


# endregion


# region Round-trip


class TestFromDictTolerance:
    """``from_dict`` accepts payloads without either content key."""

    def test_a_default_payload_reconstructs_zero_events(self) -> None:
        restored = Timeline.from_dict(_timeline().to_dict())
        assert restored.n_events == 0

    def test_a_default_payload_reconstructs_an_empty_reference_table(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        restored = Timeline.from_dict(timeline.to_dict())
        assert restored.external_references.num_rows == 0

    def test_a_default_payload_preserves_the_structure(self) -> None:
        restored = Timeline.from_dict(_hierarchy().to_dict())
        assert restored.get_child("child").get_child("grandchild").length.value == 6.0

    def test_a_default_payload_preserves_child_offsets(self) -> None:
        restored = Timeline.from_dict(_hierarchy().to_dict())
        assert restored.get_child_offset("child").value == 10.0

    def test_an_explicitly_empty_reference_list_is_accepted(self) -> None:
        data = _timeline().to_dict(external_references=True)
        assert Timeline.from_dict(data).external_references.num_rows == 0


class TestExternalReferenceRoundTrip:
    """References survive serialization exactly, with or without events."""

    def test_references_round_trip_with_events(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        restored = Timeline.from_dict(
            timeline.to_dict(events=True, external_references=True)
        )
        assert restored.external_references.equals(timeline.external_references)

    def test_references_round_trip_without_events(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        restored = Timeline.from_dict(timeline.to_dict(external_references=True))
        assert restored.external_references.equals(timeline.external_references)

    def test_a_reference_to_an_absent_event_still_round_trips(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        restored = Timeline.from_dict(timeline.to_dict(external_references=True))
        assert restored.n_events == 0
        assert restored.external_references.column("event_id").to_pylist() == ["e1"]

    def test_the_restored_table_keeps_the_canonical_schema(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        restored = Timeline.from_dict(timeline.to_dict(external_references=True))
        assert restored.external_references.schema.equals(EXTERNAL_REFERENCE_SCHEMA)

    def test_child_references_round_trip(self) -> None:
        parent = _hierarchy()
        parent.get_child("child").add_external_references(
            [_reference_row("e2", "child-ref")]
        )
        restored = Timeline.from_dict(parent.to_dict(external_references=True))
        assert restored.get_child("child").external_references.column(
            "external_id"
        ).to_pylist() == ["child-ref"]

    def test_to_dict_is_a_fixpoint_under_both_flags(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        data = timeline.to_dict(events=True, external_references=True)
        restored = Timeline.from_dict(json.loads(json.dumps(data)))
        assert restored.to_dict(events=True, external_references=True) == data

    def test_to_typed_preserves_the_reference_table(self) -> None:
        timeline = _timeline().add_external_references([_reference_row()])
        typed = timeline.to_typed()
        assert isinstance(typed, ContinuousPhysicalTimeline)
        assert typed.external_references.equals(timeline.external_references)


# endregion
