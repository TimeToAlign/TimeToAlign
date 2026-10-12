"""Exact coordinate and interval retrieval through interval claims."""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign import MatchIntervalStamp
from timetoalign.alignment import AlignmentAnchor, MatchClaim, MatchClaimField
from timetoalign.alignment.bundle import AlignmentBundle
from timetoalign.alignment.graph import MatchStamp
from timetoalign.core import (
    Coordinate,
    IdCoordinate,
    IdCoordinateField,
    Interval,
    NumberType,
    TimeUnit,
)
from timetoalign.maps import ScalarMap
from timetoalign.timelines import Timeline

FM = TimeUnit.floating_measures


def _timeline(
    timeline_id: str,
    *,
    unit: TimeUnit = FM,
    number_type: NumberType = NumberType.float,
) -> Timeline:
    """Create one standalone synthetic timeline."""
    return Timeline(
        length=200,
        uid=timeline_id,
        unit=unit,
        number_type=number_type,
    )


def _bundle(*timeline_ids: str) -> AlignmentBundle:
    """Create a bundle whose timelines have no group membership."""
    bundle = AlignmentBundle(id="interval-retrieval")
    for timeline_id in timeline_ids:
        bundle.add_timeline(_timeline(timeline_id))
    return bundle


def _coordinate(value: int | float | Fraction) -> Coordinate:
    """Create one float-canonical floating-measure coordinate."""
    return Coordinate(value, FM, number_type=NumberType.float)


def _interval(start: int | float | Fraction, end: int | float | Fraction) -> Interval:
    """Create one float-canonical floating-measure interval."""
    return Interval(_coordinate(start), _coordinate(end))


def _interval_claim(
    timeline_a: str,
    start_a: int | float | Fraction,
    end_a: int | float | Fraction,
    timeline_b: str,
    start_b: int | float | Fraction,
    end_b: int | float | Fraction,
    *,
    unit: TimeUnit = FM,
    number_type: NumberType = NumberType.float,
) -> MatchClaim:
    """Create one synchronous interval claim."""

    def coordinate(value: int | float | Fraction) -> Coordinate:
        return Coordinate(value, unit, number_type=number_type)

    return MatchClaim(
        timeline_a_id=timeline_a,
        timeline_b_id=timeline_b,
        start_anchor=AlignmentAnchor(
            timeline_a_id=timeline_a,
            coordinate_a=coordinate(start_a),
            timeline_b_id=timeline_b,
            coordinate_b=coordinate(start_b),
        ),
        end_anchor=AlignmentAnchor(
            timeline_a_id=timeline_a,
            coordinate_a=coordinate(end_a),
            timeline_b_id=timeline_b,
            coordinate_b=coordinate(end_b),
        ),
    )


def _instant_claim(
    timeline_a: str,
    coordinate_a: int | float | Fraction,
    timeline_b: str,
    coordinate_b: int | float | Fraction,
) -> MatchClaim:
    """Create one synchronous instant claim."""
    return MatchClaim(
        timeline_a_id=timeline_a,
        timeline_b_id=timeline_b,
        start_anchor=AlignmentAnchor(
            timeline_a_id=timeline_a,
            coordinate_a=_coordinate(coordinate_a),
            timeline_b_id=timeline_b,
            coordinate_b=_coordinate(coordinate_b),
        ),
    )


def _coordinate_data(value: float) -> dict[str, object]:
    """Return one expected float-canonical dictionary."""
    return {
        "unit": "floating_measures",
        "number_type": "float",
        "value": value,
    }


def test_coordinate_interior_maps_to_matchstamp() -> None:
    """An interior coordinate maps affinely and remains an instant stamp."""
    bundle = _bundle("mix", "track")
    bundle.add_match_claims([_interval_claim("mix", 125, 135, "track", 33, 43)])

    stamp = bundle.get_matchstamp_at(_coordinate(130), "mix")

    assert type(stamp) is MatchStamp
    assert not isinstance(stamp, MatchIntervalStamp)
    assert stamp.coordinates == {"mix": _coordinate(130), "track": _coordinate(38)}
    assert stamp.get_coordinate_for("track", format="float") == 38.0
    assert stamp.is_interpolated is True
    assert [claim.timeline_b_id for claim in stamp.interval_claims] == ["track"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [(125.0, 33.0), (135.0, 43.0)],
)
def test_coordinate_claim_endpoints_are_exact_anchors(
    query: float, expected: float
) -> None:
    """Closed endpoint queries return stored anchors without interpolation."""
    bundle = _bundle("mix", "track")
    bundle.add_match_claims([_interval_claim("mix", 125, 135, "track", 33, 43)])

    stamp = bundle.get_matchstamp_at(query, "mix")

    assert type(stamp) is MatchStamp
    assert stamp.coordinates == {
        "mix": _coordinate(query),
        "track": _coordinate(expected),
    }
    assert stamp.is_interpolated is False
    assert stamp.anchor_edges == [("mix", "track")]


def test_crossfade_maps_each_claim_to_its_counterpart() -> None:
    """Overlapping interval claims contribute independent correspondences."""
    bundle = _bundle("mix", "track-a", "track-b")
    first = _interval_claim("mix", 125, 135, "track-a", 33, 43)
    second = _interval_claim("mix", 128, 138, "track-b", 1, 11)
    bundle.add_match_claims([first, second])

    stamp = bundle.get_matchstamp_at(130, "mix")

    assert type(stamp) is MatchStamp
    assert stamp.coordinates == {
        "mix": _coordinate(130),
        "track-a": _coordinate(38),
        "track-b": _coordinate(3),
    }
    assert [claim.timeline_b_id for claim in stamp.interval_claims] == [
        "track-a",
        "track-b",
    ]
    assert "track-a=38" in repr(stamp)
    assert "track-b=3" in repr(stamp)
    html = stamp._repr_html_()
    assert "mix" in html
    assert "track-a" in html
    assert "track-b" in html


def test_loop_boundary_ambiguity_is_per_claim_and_atomic() -> None:
    """Different mappings to one timeline never collapse into one coordinate."""
    bundle = _bundle("mix", "loop")
    first = _interval_claim("mix", 10, 20, "loop", 0, 10)
    second = _interval_claim("mix", 20, 30, "loop", 100, 110)
    bundle.add_match_claims([first, second])

    stamp = bundle.get_matchstamp_at(20, "mix")

    assert type(stamp) is MatchStamp
    assert stamp.coordinates == {"mix": _coordinate(20)}
    assert [claim.get_interval_for("loop") for claim in stamp.interval_claims] == [
        _interval(0, 10),
        _interval(100, 110),
    ]
    with pytest.raises(ValueError) as exc_info:
        stamp.get_coordinate_for("loop")
    message = str(exc_info.value)
    assert "loop" in message
    assert "10" in message
    assert "100" in message
    plain = repr(stamp)
    assert "loop=10" in plain
    assert "loop=100" in plain
    html = stamp._repr_html_()
    assert "loop" in html
    assert "10" in html
    assert "100" in html


def test_plural_coordinate_retrieval_rejects_interval_claim_ambiguity_atomically() -> (
    None
):
    """A conflicting requested timeline prevents every coordinate projection."""
    bundle = _bundle("mix", "loop")
    bundle.add_match_claims(
        [
            _interval_claim("mix", 10, 20, "loop", 0, 10),
            _interval_claim("mix", 20, 30, "loop", 100, 110),
        ]
    )

    stamp = bundle.get_matchstamp_at(20, "mix")

    assert stamp.get_coordinates_for(["mix"], format="float") == [20.0]
    with pytest.raises(ValueError) as exc_info:
        stamp.get_coordinates_for(["mix", "loop"], format="float")
    assert str(exc_info.value) == (
        "Timeline 'loop' has multiple interval-claim candidates: "
        "claim 1 mix<->loop: loop=10 floating_measures; "
        "claim 2 mix<->loop: loop=100 floating_measures"
    )


def test_repeated_equal_candidates_are_unambiguous() -> None:
    """Several claims agreeing on one mapped value publish that coordinate."""
    bundle = _bundle("mix", "loop")
    bundle.add_match_claims(
        [
            _interval_claim("mix", 10, 20, "loop", 0, 10),
            _interval_claim("mix", 20, 30, "loop", 10, 20),
        ]
    )

    stamp = bundle.get_matchstamp_at(20, "mix")

    assert stamp.coordinates == {"mix": _coordinate(20), "loop": _coordinate(10)}
    assert len(stamp.interval_claims) == 2


def test_degenerate_expanding_claim_has_no_single_counterpart() -> None:
    """A point mapped to a nonzero interval remains a per-claim alternative."""
    bundle = _bundle("mix", "track")
    bundle.add_match_claims([_interval_claim("mix", 10, 10, "track", 30, 40)])

    stamp = bundle.get_matchstamp_at(10, "mix")

    assert stamp.coordinates == {"mix": _coordinate(10)}
    with pytest.raises(ValueError, match="no single coordinate"):
        stamp.get_coordinate_for("track")
    assert "MISSING" in repr(stamp)


def test_unequal_extents_map_proportionally() -> None:
    """Affine mapping scales the offset when claim extents differ."""
    bundle = _bundle("source", "target")
    bundle.add_match_claims([_interval_claim("source", 10, 20, "target", 30, 50)])

    stamp = bundle.get_matchstamp_at(12.5, "source")

    assert stamp.coordinates == {
        "source": _coordinate(12.5),
        "target": _coordinate(35),
    }


def test_fraction_mapping_is_exact() -> None:
    """Exact operands produce the exact rational affine result."""
    source = _timeline(
        "source", unit=TimeUnit.quarters, number_type=NumberType.fraction
    )
    target = _timeline(
        "target", unit=TimeUnit.quarters, number_type=NumberType.fraction
    )
    bundle = AlignmentBundle(id="fraction-affine")
    bundle.add_timeline(source)
    bundle.add_timeline(target)
    bundle.add_match_claims(
        [
            _interval_claim(
                "source",
                Fraction(1, 3),
                Fraction(4, 3),
                "target",
                Fraction(2, 7),
                Fraction(8, 7),
                unit=TimeUnit.quarters,
                number_type=NumberType.fraction,
            )
        ]
    )

    stamp = bundle.get_matchstamp_at(
        Coordinate(Fraction(5, 6), TimeUnit.quarters), "source"
    )

    assert stamp.coordinates == {
        "source": Coordinate(Fraction(5, 6), TimeUnit.quarters),
        "target": Coordinate(Fraction(5, 7), TimeUnit.quarters),
    }
    assert isinstance(stamp.coordinates["target"].value, Fraction)


def test_interval_coordinate_feeds_standalone_closure() -> None:
    """An unambiguous mapped coordinate expands through another claim edge."""
    bundle = _bundle("A", "B", "C")
    bundle.add_match_claims(
        [
            _interval_claim("A", 10, 20, "B", 30, 40),
            _instant_claim("B", 35, "C", 99),
        ]
    )

    stamp = bundle.get_matchstamp_at(15, "A")

    assert stamp.coordinates == {
        "A": _coordinate(15),
        "B": _coordinate(35),
        "C": _coordinate(99),
    }


def test_interval_query_combines_two_matchstamps() -> None:
    """Both resolved endpoints form one exact interval per reached timeline."""
    bundle = _bundle("mix", "track")
    bundle.add_match_claims([_interval_claim("mix", 10, 20, "track", 30, 40)])

    stamp = bundle.get_matchstamp_at(_interval(12, 18), "mix")

    assert type(stamp) is MatchIntervalStamp
    assert type(stamp.start) is MatchStamp
    assert type(stamp.end) is MatchStamp
    assert stamp.axis == _interval(12, 18)
    assert stamp.present_timelines == ["mix", "track"]
    assert stamp.get_interval_for("mix") == _interval(12, 18)
    assert stamp.get_interval_for("track") == _interval(32, 38)


def test_complete_interval_stamp_data_pairs_match_endpoint_stamps() -> None:
    """Complete endpoint coverage serializes every exact coordinate without nulls."""
    bundle = _bundle("mix", "track")
    bundle.add_match_claims([_interval_claim("mix", 10, 20, "track", 30, 40)])

    stamp = bundle.get_matchstamp_at(_interval(12, 18), "mix")
    data = stamp.to_dict()

    assert data == {
        "mix": {"start": _coordinate_data(12.0), "end": _coordinate_data(18.0)},
        "track": {"start": _coordinate_data(32.0), "end": _coordinate_data(38.0)},
    }
    assert all(
        endpoint is not None for pair in data.values() for endpoint in pair.values()
    )
    assert isinstance(stamp.start, MatchStamp)
    assert isinstance(stamp.end, MatchStamp)
    assert (
        stamp.start.get_coordinate_for("mix", format="float")
        == data["mix"]["start"]["value"]
    )
    assert (
        stamp.end.get_coordinate_for("mix", format="float")
        == data["mix"]["end"]["value"]
    )
    assert (
        stamp.start.get_coordinate_for("track", format="float")
        == data["track"]["start"]["value"]
    )
    assert (
        stamp.end.get_coordinate_for("track", format="float")
        == data["track"]["end"]["value"]
    )


def test_interval_query_preserves_a_missing_start_side() -> None:
    """Endpoint-only reach is serialized and displayed without invention."""
    bundle = _bundle("mix", "track-a", "track-b")
    bundle.add_match_claims(
        [
            _interval_claim("mix", 125, 135, "track-a", 33, 43),
            _interval_claim("mix", 128, 138, "track-b", 1, 11),
        ]
    )

    stamp = bundle.get_matchstamp_at(_interval(126, 131), "mix")

    assert type(stamp) is MatchIntervalStamp
    assert stamp.present_timelines == ["mix", "track-a", "track-b"]
    assert stamp.to_dict()["track-b"] == {"start": None, "end": _coordinate_data(4.0)}
    with pytest.raises(ValueError, match="missing start"):
        stamp.get_interval_for("track-b")
    html = stamp._repr_html_()
    assert "track-b" in html
    assert "MISSING" in html


def test_interval_query_rejects_reversed_and_absent_timelines() -> None:
    """A combined interval must have both sides in non-decreasing order."""
    start = MatchStamp(
        coordinates={"mix": _coordinate(1), "track": _coordinate(10)},
        source_id="mix",
    )
    end = MatchStamp(
        coordinates={"mix": _coordinate(2), "track": _coordinate(5)},
        source_id="mix",
    )
    stamp = MatchIntervalStamp(
        source_id="mix",
        interval=_interval(1, 2),
        start_stamp=start,
        end_stamp=end,
    )

    with pytest.raises(ValueError, match="reversed endpoints"):
        stamp.get_interval_for("track")
    with pytest.raises(KeyError, match="Available timelines"):
        stamp.get_interval_for("absent")


def test_motivating_dj_set_queries() -> None:
    """The crossfade geometry resolves both a point and a partial interval."""
    bundle = _bundle("mix", "track-a", "track-b")
    bundle.add_match_claims(
        [
            (
                "mix",
                _interval(125, 135),
                "track-a",
                _interval(33, 43),
            ),
            (
                "mix",
                _interval(128, 138),
                "track-b",
                _interval(1, 11),
            ),
        ]
    )

    point = bundle.get_matchstamp_at(Coordinate(130, FM), "mix")
    interval = bundle.get_matchstamp_at(Interval(126.0, 131.0, "fm"), "mix")

    assert type(point) is MatchStamp
    assert point.coordinates == {
        "mix": _coordinate(130),
        "track-a": _coordinate(38),
        "track-b": _coordinate(3),
    }
    assert point.is_interpolated is True
    assert interval.get_interval_for("track-a") == _interval(34, 39)
    assert interval.to_dict()["track-b"] == {
        "start": None,
        "end": _coordinate_data(4.0),
    }
    assert "track-b=[MISSING, 4 floating_measures]" in repr(interval)


def test_query_form_dispatches_scalar_and_mixed_collections() -> None:
    """Coordinate and interval forms select their result forms independently."""
    bundle = _bundle("mix", "track")
    bundle.add_match_claims([_interval_claim("mix", 10, 20, "track", 30, 40)])
    interval = _interval(12, 18)

    coordinate_forms = [
        bundle.get_matchstamp_at(15, "mix"),
        bundle.get_matchstamp_at(_coordinate(15), "mix"),
        bundle.get_matchstamp_at(IdCoordinate(15, FM, "mix")),
    ]
    plural = bundle.get_matchstamps_at([15, interval, _coordinate(16)], "mix")
    dispatched = bundle.get_matchstamp([15, interval], "mix")

    assert [type(stamp) for stamp in coordinate_forms] == [
        MatchStamp,
        MatchStamp,
        MatchStamp,
    ]
    assert type(bundle.get_matchstamp_at(interval, "mix")) is MatchIntervalStamp
    assert [type(stamp) for stamp in plural] == [
        MatchStamp,
        MatchIntervalStamp,
        MatchStamp,
    ]
    assert isinstance(dispatched, list)
    assert [type(stamp) for stamp in dispatched] == [MatchStamp, MatchIntervalStamp]
    with pytest.raises(ValueError, match="timeline_id is required"):
        bundle.get_matchstamp_at(interval)
    with pytest.raises(ValueError, match="timeline_id is required"):
        bundle.get_matchstamp_at(15)


def test_matchstamp_collection_rejects_unsupported_element_before_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported collection element prevents resolution of valid predecessors."""
    bundle = _bundle("mix", "track")
    bundle.add_match_claims([_interval_claim("mix", 10, 20, "track", 30, 40)])

    def no_matchstamp_resolution(*args: object, **kwargs: object) -> object:
        raise AssertionError("get_matchstamp_at must not be called")

    monkeypatch.setattr(bundle, "get_matchstamp_at", no_matchstamp_resolution)

    with pytest.raises(TypeError) as exc_info:
        bundle.get_matchstamps_at([15, object()], "mix")
    assert (
        str(exc_info.value)
        == "Matchstamp collection element 1 has unsupported type object"
    )


def test_matchstamp_html_contains_involved_timelines() -> None:
    """Both stamp forms expose the resolved timeline identities in HTML."""
    bundle = _bundle("mix", "track")
    bundle.add_match_claims([_interval_claim("mix", 10, 20, "track", 30, 40)])

    point_html = bundle.get_matchstamp_at(15, "mix")._repr_html_()
    interval_html = bundle.get_matchstamp_at(_interval(12, 18), "mix")._repr_html_()

    assert "mix" in point_html and "track" in point_html
    assert "mix" in interval_html and "track" in interval_html


def test_coordinate_table_fills_interval_mapped_cell() -> None:
    """A coordinate-query row stores an unambiguous interval-derived value."""
    bundle = _bundle("mix", "track")
    bundle.add_match_claims([_interval_claim("mix", 10, 20, "track", 30, 40)])

    table = bundle.get_matchstamp_table([15], "mix")

    assert table.column_names == ["mix", "track"]
    assert IdCoordinateField.from_table(table, "mix")[0] == IdCoordinate(
        15, FM, "mix", number_type=NumberType.float
    )
    assert IdCoordinateField.from_table(table, "track")[0] == IdCoordinate(
        35, FM, "track", number_type=NumberType.float
    )


def test_interval_table_query_is_not_implemented() -> None:
    """The table lane rejects interval query forms before materialization."""
    bundle = _bundle("mix", "track")

    with pytest.raises(NotImplementedError, match="Interval queries"):
        bundle.get_matchstamp_table(_interval(10, 20), "mix")


def test_ambiguous_coordinate_table_query_raises() -> None:
    """An ambiguous mapped cell cannot masquerade as an unreached null cell."""
    bundle = _bundle("mix", "loop")
    bundle.add_match_claims(
        [
            _interval_claim("mix", 10, 20, "loop", 0, 10),
            _interval_claim("mix", 20, 30, "loop", 100, 110),
        ]
    )

    with pytest.raises(ValueError, match="loop"):
        bundle.get_matchstamp_table([20], "mix")


def test_match_interval_stamp_is_exported_at_both_public_levels() -> None:
    """The interval result type is available from both public namespaces."""
    import timetoalign
    import timetoalign.alignment

    assert timetoalign.MatchIntervalStamp is MatchIntervalStamp
    assert timetoalign.alignment.MatchIntervalStamp is MatchIntervalStamp
    assert "MatchIntervalStamp" in timetoalign.__all__
    assert "MatchIntervalStamp" in timetoalign.alignment.__all__


def test_standalone_instant_claim_chain_reaches_third_timeline() -> None:
    """Direct claims provide transitive closure and exact transfer fallback."""
    bundle = _bundle("A", "B", "C")
    bundle.add_match_claims(
        [_instant_claim("A", 10, "B", 20), _instant_claim("B", 20, "C", 30)]
    )

    stamp = bundle.get_matchstamp_at(10, "A")

    assert type(stamp) is MatchStamp
    assert stamp.coordinates == {
        "A": _coordinate(10),
        "B": _coordinate(20),
        "C": _coordinate(30),
    }
    assert bundle.transfer(10, "A", "C") == 30.0


def test_fraction_axis_unit_conversion_compares_once_in_exact_domain() -> None:
    """A foreign-unit fraction query lands exactly on a fraction boundary."""
    source = _timeline(
        "source", unit=TimeUnit.quarters, number_type=NumberType.fraction
    )
    source.add_conversion_map(
        ScalarMap(
            scalar=Fraction(3, 10),
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.seconds,
        )
    )
    target = _timeline(
        "target", unit=TimeUnit.quarters, number_type=NumberType.fraction
    )
    bundle = AlignmentBundle(id="fraction-conversion")
    bundle.add_timeline(source)
    bundle.add_timeline(target)
    claim = _interval_claim(
        "source",
        Fraction(1, 3),
        Fraction(2, 3),
        "target",
        Fraction(4, 3),
        Fraction(5, 3),
        unit=TimeUnit.quarters,
        number_type=NumberType.fraction,
    )
    bundle.add_match_claims([claim])

    stamp = bundle.get_matchstamp_at(
        Coordinate(
            Fraction(1, 10),
            TimeUnit.seconds,
            number_type=NumberType.fraction,
        ),
        "source",
    )

    assert type(stamp) is MatchStamp
    assert stamp.coordinates == {
        "source": Coordinate(Fraction(1, 3), TimeUnit.quarters),
        "target": Coordinate(Fraction(4, 3), TimeUnit.quarters),
    }


def test_columnar_interval_lookup_is_closed() -> None:
    """The columnar lane selects an interval row at interior and endpoints."""
    claim = _interval_claim("mix", 10, 20, "track", 30, 40)
    field = MatchClaimField.from_claims([claim])

    assert len(field.at("mix", _coordinate(10))) == 1
    assert len(field.at("mix", _coordinate(15))) == 1
    assert len(field.at("mix", _coordinate(20))) == 1
    assert len(field.at("mix", _coordinate(21))) == 0

    bundle = _bundle("mix", "track")
    bundle.add_match_claim_field(field)
    stamp = bundle.get_matchstamp_at(15, "mix")
    assert type(stamp) is MatchStamp
    assert stamp.coordinates["track"] == _coordinate(35)


def test_columnar_fraction_interval_distinguishes_same_float_value() -> None:
    """Columnar closed containment distinguishes exact fractions at one point."""
    third = Fraction(1, 3)
    same_float = Fraction(6004799503160661, 18014398509481984)
    claim = MatchClaim(
        timeline_a_id="source",
        timeline_b_id="target",
        start_anchor=AlignmentAnchor(
            timeline_a_id="source",
            coordinate_a=Coordinate(third, TimeUnit.quarters),
            timeline_b_id="target",
            coordinate_b=Coordinate(Fraction(4, 3), TimeUnit.quarters),
        ),
        end_anchor=AlignmentAnchor(
            timeline_a_id="source",
            coordinate_a=Coordinate(third, TimeUnit.quarters),
            timeline_b_id="target",
            coordinate_b=Coordinate(Fraction(4, 3), TimeUnit.quarters),
        ),
    )
    field = MatchClaimField.from_claims([claim])

    exact_matches = field.at(
        "source",
        Coordinate(third, TimeUnit.quarters, number_type=NumberType.fraction),
    )
    rounded_matches = field.at(
        "source",
        Coordinate(same_float, TimeUnit.quarters, number_type=NumberType.fraction),
    )

    assert len(exact_matches) == 1
    assert len(rounded_matches) == 0
    interval = exact_matches[0].get_interval_for("source")
    assert interval == Interval(
        Coordinate(third, TimeUnit.quarters),
        Coordinate(third, TimeUnit.quarters),
    )
    assert isinstance(interval.start.value, Fraction)
    assert isinstance(interval.end.value, Fraction)


def test_no_hit_retains_source_only_matchstamp() -> None:
    """A position outside every claim keeps a source-only instant stamp."""
    bundle = _bundle("mix", "track")
    bundle.add_match_claims([_interval_claim("mix", 10, 20, "track", 30, 40)])

    stamp = bundle.get_matchstamp_at(25, "mix")

    assert type(stamp) is MatchStamp
    assert stamp.present_timelines == ["mix"]
    assert stamp.coordinates == {"mix": _coordinate(25)}
