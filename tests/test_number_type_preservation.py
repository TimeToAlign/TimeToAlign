"""The declared number_type is preserved at every coordinate boundary.

Validation logic is documented in ``tests/core/README.md`` under "Number
type is preserved everywhere". Every value that lands on a timeline axis is
written in that axis's declared representation, however it was obtained —
typed by a caller, converted, offset-computed, interpolated, or read back
off a query. Estimation is recorded by ``is_interpolated``, never by a
number's type.
"""

from __future__ import annotations

import math
from fractions import Fraction

from timetoalign.alignment import AlignmentBundle, MatchClaim, MatchStamp
from timetoalign.alignment.warpmap import WarpMap
from timetoalign.core import Coordinate, IdCoordinate, NumberType, TimeUnit
from timetoalign.maps import LinearMap, ScalarMap, TicksToQuarters
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    TimelineGroup,
)


def test_ticks_to_quarters_preserves_exact_scalar_value() -> None:
    """An exact tick coordinate remains rational through scalar conversion."""
    ticks = DiscreteLogicalTimeline(length=1920, uid="ticks")
    ticks.add_conversion_map(TicksToQuarters(ppq=480))

    converted = ticks.convert_to(160, "quarters")

    assert converted.value == Fraction(1, 3)
    assert isinstance(converted.value, Fraction)


def test_float_input_is_expressed_in_the_target_axis_type() -> None:
    """A float query lands on an exact axis as an exact value.

    quarters are fraction-canonical, so the converted coordinate is a
    ``Fraction`` whatever the caller passed. Re-expressing the double as its
    exact dyadic is not fabrication -- the number is identical, digit for
    digit; fabrication would be inventing a tidier ratio than it really is.
    """
    ticks = DiscreteLogicalTimeline(length=1920, uid="ticks")
    ticks.add_conversion_map(TicksToQuarters(ppq=480))

    converted = ticks.convert_to(160.0, "quarters")

    assert isinstance(converted.value, Fraction)
    assert converted.value == Fraction(160.0 / 480)
    assert float(converted.value) == 160.0 / 480


def test_timestamp_coordinate_preserves_exact_child_offset() -> None:
    """Typed timestamp access retains exact parent-to-child subtraction."""
    piece = ContinuousLogicalTimeline(length=Fraction(12), uid="piece")
    movement = ContinuousLogicalTimeline(length=Fraction(4), uid="movement")
    piece.add_child(movement, offset=Fraction(11, 2))

    stamp = piece.get_timestamp(Fraction(11, 2))
    coordinate = stamp.get_coordinate_for("movement", format="coordinate")

    assert coordinate == Coordinate(Fraction(0), TimeUnit.quarters)
    assert isinstance(coordinate.value, Fraction)
    assert stamp.get_coordinate_for("movement", format="float") == 0.0


def test_float_query_is_expressed_in_the_axis_type() -> None:
    """A float query and the constructor give the same thing on one axis.

    This is the property the rule buys: a caller reasoning about a
    fraction-canonical timeline never has to inspect a value's kind to find
    out what they got, because there is one answer per axis rather than one
    per entry point.
    """
    piece = ContinuousLogicalTimeline(length=Fraction(12), uid="piece")
    movement = ContinuousLogicalTimeline(length=Fraction(4), uid="movement")
    piece.add_child(movement, offset=Fraction(11, 2))

    stamp = piece.get_timestamp(6.5)

    assert stamp.axis.value == Fraction(13, 2)
    assert isinstance(stamp.axis.value, Fraction)

    coordinate = stamp.get_coordinate_for("movement", format="coordinate")
    assert coordinate == Coordinate(Fraction(1), TimeUnit.quarters)
    assert isinstance(coordinate.value, Fraction)

    # Same literal through the constructor: one answer per axis.
    assert (
        piece.get_timestamp(9.5).axis.value == Coordinate(9.5, TimeUnit.quarters).value
    )


def test_interval_coordinate_access_preserves_exact_values() -> None:
    """Typed interval endpoints retain exact child coordinates."""
    piece = ContinuousLogicalTimeline(length=Fraction(12), uid="piece")
    movement = ContinuousLogicalTimeline(length=Fraction(4), uid="movement")
    piece.add_child(movement, offset=Fraction(11, 2))

    stamp = piece.get_interval_stamp(Fraction(11, 2), Fraction(13, 2))
    interval = stamp.get_interval("movement")

    assert interval.start == Coordinate(Fraction(0), TimeUnit.quarters)
    assert interval.end == Coordinate(Fraction(1), TimeUnit.quarters)
    assert isinstance(interval.start.value, Fraction)
    assert isinstance(interval.end.value, Fraction)


def test_group_retrieval_returns_exact_target_coordinate_or_raises() -> None:
    """Structural re-expression between group members stays exact.

    The relation between these two timelines is proven from their own
    declared origins and lengths, so converting across it is definitional
    rather than estimated and the exact value survives. Empirical
    interpolation -- reading a position off matched anchor pairs -- carries
    estimate provenance separately and is re-expressed in the target axis's
    canonical number type.
    """
    source = ContinuousLogicalTimeline(length=Fraction(12), uid="a")
    target = ContinuousLogicalTimeline(length=Fraction(24), uid="b")
    group = TimelineGroup(id="g", timelines=[source, target])

    converted = group.get_coordinate_at(
        IdCoordinate(Fraction(1, 3), TimeUnit.quarters, "a"),
        timeline_id="b",
        format="coordinate",
    )

    assert converted == Coordinate(Fraction(2, 3), TimeUnit.quarters)
    assert isinstance(converted.value, Fraction)

    partial = ContinuousLogicalTimeline(length=Fraction(2), uid="partial")
    group.add_timeline(
        partial,
        start=IdCoordinate(Fraction(2), TimeUnit.quarters, "a"),
        end=IdCoordinate(Fraction(4), TimeUnit.quarters, "a"),
    )
    import pytest

    with pytest.raises(KeyError):
        group.get_coordinate_at(
            IdCoordinate(Fraction(1), TimeUnit.quarters, "a"),
            timeline_id="partial",
        )


def test_irrational_map_output_is_expressed_in_the_axis_type() -> None:
    """Even an irrational-scaled result is written the way its axis writes.

    The computation is as inexact as its parameter; the *type* still follows
    the axis, because a type says how a number is written and not where it
    came from.
    """
    timeline = DiscreteGraphicalTimeline(length=12, uid="source")
    timeline.add_conversion_map(
        ScalarMap(
            scalar=math.sqrt(2),
            source_unit=TimeUnit.pixels,
            target_unit=TimeUnit.quarters,
        )
    )

    converted = timeline.convert_to(3, TimeUnit.quarters)

    # quarters are fraction-canonical, so the result is a Fraction -- the
    # exact dyadic of the double the computation produced, numerically
    # identical to it. That the scaling was irrational is a fact about the
    # map, not something the coordinate's type is asked to carry.
    assert isinstance(converted.value, Fraction)
    assert converted.value == Fraction(math.sqrt(2) * 3)
    assert float(converted.value) == math.sqrt(2) * 3


def test_matchstamp_typed_access_preserves_stored_fraction() -> None:
    """MatchStamp keeps raw access numeric and typed access exact."""
    stamp = MatchStamp(
        coordinates={"score": Coordinate(Fraction(1, 3), TimeUnit.quarters)},
        source_id="score",
    )

    assert stamp.get_coordinate_for("score", format="float") == 1 / 3
    coordinate = stamp.get_coordinate_for("score", format="coordinate")
    assert coordinate == Coordinate(Fraction(1, 3), TimeUnit.quarters)
    assert isinstance(coordinate.value, Fraction)


def _mixed_group() -> TimelineGroup:
    """A group whose two members declare different representations."""
    return TimelineGroup(
        id="g",
        timelines=[
            ContinuousLogicalTimeline(length=100, unit=TimeUnit.quarters, uid="clt_1"),
            ContinuousPhysicalTimeline(length=60, unit=TimeUnit.seconds, uid="cpt_1"),
        ],
    )


def test_group_axis_follows_the_addressed_timeline_not_the_argument() -> None:
    """A group stamp's axis is written the way its timeline writes numbers.

    Asserted in both directions on one group, because the failure this
    catches is reading the *argument's* Python type: that mistake spells a
    float query as a float on an exact axis and an int query as a Fraction on
    a float axis, and a one-directional check would clear half of it.
    """
    group = _mixed_group()

    exact = group.get_timestamp_at(9.5, "clt_1")
    assert exact.axis.value == Fraction(19, 2)
    assert isinstance(exact.axis.value, Fraction)

    # Same position typed exactly: one answer per axis, not one per literal.
    assert group.get_timestamp_at(Fraction(19, 2), "clt_1").axis == exact.axis
    assert isinstance(group.get_timestamp_at(10, "clt_1").axis.value, Fraction)

    # seconds are float-canonical, so an exact argument is written as a float.
    for query in (10, 10.0, Fraction(10, 1)):
        axis = group.get_timestamp_at(query, "cpt_1").axis
        assert axis.value == 10.0
        assert isinstance(axis.value, float)


def test_group_and_timeline_agree_on_the_same_position() -> None:
    """The group path and the timeline path give one answer, not two.

    A stamp from any source has identical structure and behaviour, so a
    caller who reaches a position through a group rather than through its
    member must not get a differently-typed axis.
    """
    group = _mixed_group()
    timeline = group.get_timeline("clt_1")

    direct = timeline.get_timestamp(9.5).axis
    via_group = group.get_timestamp_at(9.5, "clt_1").axis

    assert direct == via_group
    assert type(direct) is type(via_group)


def test_bundle_matchstamp_axis_follows_the_declared_type() -> None:
    """A bundle's matchstamp axis is re-expressed like every other axis.

    The bundle resolves its query against float graph nodes and WarpMap
    feeds, which is the internal lane the rule deliberately leaves alone.
    What it may not do is report that internal float as the axis: an exact
    query on a fraction-canonical timeline came back as ``79.0`` before, so
    the same position answered differently depending on the entry point.
    """
    score = ContinuousLogicalTimeline(length=100, unit=TimeUnit.quarters, uid="clt_1")
    bundle = AlignmentBundle(id="b")
    bundle.add_timeline(score, uid="clt_1", as_group="g")

    for query in (Fraction(79, 1), 79.0, 79):
        axis = bundle.get_matchstamp_at(query, "clt_1").axis
        assert axis.value == Fraction(79, 1)
        assert isinstance(axis.value, Fraction)

    fractional = bundle.get_matchstamp_at(32.5, "clt_1")
    assert fractional.axis.value == Fraction(65, 2)
    assert isinstance(fractional.axis.value, Fraction)
    assert fractional.get_coordinate_for("clt_1", format="coordinate") == Coordinate(
        Fraction(65, 2), TimeUnit.quarters
    )


def test_bundle_matchstamp_axis_stays_float_on_a_float_axis() -> None:
    """The other direction: an exact query on a float axis is written float."""
    clock = ContinuousPhysicalTimeline(length=100, unit=TimeUnit.seconds, uid="cpt_1")
    bundle = AlignmentBundle(id="b-float")
    bundle.add_timeline(clock, uid="cpt_1", as_group="g")

    for query in (25, 25.0, Fraction(25, 1)):
        axis = bundle.get_matchstamp_at(query, "cpt_1").axis
        assert axis.value == 25.0
        assert isinstance(axis.value, float)


def _claimed_bundle() -> tuple[AlignmentBundle, list[MatchClaim]]:
    """A quarters/seconds bundle whose claims are built from storage cells.

    The specimen matters: a plain Python ``2.5`` never reproduced the defect,
    because the scalar never-degrade rule has nothing exact to keep. A real
    event cell carries both sides (``{value: 2.5, numerator: 5,
    denominator: 2}``), and reading one hands the claim an exact ratio for a
    position on a float-canonical axis.
    """
    score = ContinuousLogicalTimeline(length=Fraction(16), uid="score")
    audio = ContinuousPhysicalTimeline(length=10.0, uid="audio")
    score.add_events([{"start": q, "id": f"a{q}"} for q in (0, 4, 8, 12)])
    audio.add_events(
        [{"start": s, "id": f"b{i}"} for i, s in enumerate((0.0, 2.5, 5.75, 8.0))]
    )
    score_cells = score.get_events().table.column("start").to_pylist()
    audio_cells = audio.get_events().table.column("start").to_pylist()

    bundle = AlignmentBundle(id="b-claims")
    bundle.add_timeline(score, uid="score", as_group="g-score")
    bundle.add_timeline(audio, uid="audio", as_group="g-audio")
    claims = [
        MatchClaim.from_events(
            event_a={"id": f"a{index}", "start": score_cells[index]},
            tl_a_id="score",
            event_b={"id": f"b{index}", "start": audio_cells[index]},
            tl_b_id="audio",
            unit_a=TimeUnit.quarters,
            unit_b=TimeUnit.seconds,
        )
        for index in range(4)
    ]
    bundle.add_match_claims(claims)
    return bundle, claims


def test_claim_getters_follow_the_addressed_axis_in_both_directions() -> None:
    """A claim answers per axis, exactly as the matchstamp built from it does.

    Both directions on one claim: the seconds half proves an exact input does
    not survive on a float-canonical axis, and the quarters half proves an
    integral input is still written exactly. A check of the seconds half alone
    would pass on an implementation that simply floated everything.
    """
    _, claims = _claimed_bundle()
    claim = claims[1]

    seconds = claim.get_coordinate_for("audio")
    quarters = claim.get_coordinate_for("score")

    assert seconds == IdCoordinate(2.5, TimeUnit.seconds, "audio")
    assert isinstance(seconds.value, float)
    assert quarters == IdCoordinate(Fraction(4, 1), TimeUnit.quarters, "score")
    assert isinstance(quarters.value, Fraction)

    plural = claim.get_coordinates_for(["score", "audio"])
    assert [entry.value for entry in plural] == [Fraction(4, 1), 2.5]
    assert claim.get_coordinate("audio") == seconds


def test_claim_and_its_matchstamp_report_one_position() -> None:
    """The two lanes off one claim agree, from the graph and without it."""
    _, claims = _claimed_bundle()
    claim = claims[1]

    from_claim = claim.get_coordinate_for("audio", format="coordinate")
    full = claim.get_matchstamp().get_coordinate_for("audio", format="coordinate")
    reduced = claim.get_matchstamp(from_graph=False).get_coordinate_for(
        "audio", format="coordinate"
    )

    assert from_claim == full == reduced
    for value in (from_claim.value, full.value, reduced.value):
        assert isinstance(value, float)


def test_anchor_coordinates_are_positions_on_their_timelines() -> None:
    """Anchor storage, access and rendering all carry the declared type."""
    _, claims = _claimed_bundle()
    anchor = claims[1].start_anchor

    assert anchor.coordinate_b == Coordinate(2.5, TimeUnit.seconds)
    assert isinstance(anchor.coordinate_b.value, float)
    assert anchor.get_coordinate_for("audio") == IdCoordinate(
        2.5, TimeUnit.seconds, "audio"
    )
    assert repr(anchor) == ("AlignmentAnchor(score@4 quarters <-> audio@2.5 seconds)")


def test_matchline_and_warpmap_read_the_declared_axis() -> None:
    """Anchors, source coordinates and inferred axis types all agree.

    ``WarpMap.from_match_line`` infers its axis representations from the
    anchors it is given, so an anchor carrying the wrong kind made the map
    declare a seconds axis exact. Number type is a type system, not a record
    of which Python object the value happened to arrive as.
    """
    bundle, _ = _claimed_bundle()
    line = bundle._get_or_build_match_line("score")

    assert [coordinate.value for coordinate in line.source_coordinates] == [
        Fraction(0, 1),
        Fraction(4, 1),
        Fraction(8, 1),
        Fraction(12, 1),
    ]
    assert all(
        isinstance(coordinate.value, Fraction) for coordinate in line.source_coordinates
    )

    anchors = line.get_alignment_anchors("audio")
    assert [anchor.coordinate_b.value for anchor in anchors] == [
        0.0,
        2.5,
        5.75,
        8.0,
    ]
    assert all(isinstance(anchor.coordinate_b.value, float) for anchor in anchors)

    warp = WarpMap.from_match_line(line, "audio")
    assert warp.source_number_type is NumberType.fraction
    assert warp.target_number_type is NumberType.float


def _typed_group() -> TimelineGroup:
    """A group spanning all three representations plus a C-Map column."""
    pixels = DiscreteGraphicalTimeline(length=12473, uid="dgt_1")
    pixels.add_conversion_map(
        LinearMap(
            scalar=Fraction(1, 4),
            offset=0,
            source_unit=TimeUnit.pixels,
            target_unit=TimeUnit.quarters,
        )
    )
    score = ContinuousLogicalTimeline(length=Fraction(19, 2), uid="clt_1")
    # A fraction axis converting into a float-canonical unit: the target
    # admits either, so the source axis's representation is kept -- the case
    # where a column keyed on the map's own default disagreed with the row.
    score.add_conversion_map(
        LinearMap(
            scalar=Fraction(4, 1),
            offset=0,
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.seconds,
        )
    )
    group = TimelineGroup(id="g-typed")
    group.add_timeline(pixels)
    group.add_timeline(ContinuousPhysicalTimeline(length=37.5, uid="cpt_1"))
    group.add_timeline(score)
    return group


def test_stamp_lane_and_frame_lane_write_the_same_positions() -> None:
    """The stamp getter and the frame present one set of positions.

    Asserted side by side rather than as two pinned tables, because the
    property is agreement: the group's stored timestamps are a float64
    interpolation lane, and the frame used to hand that lane's doubles
    straight to the reader while the stamp getter reported ``12473`` where
    the frame said ``12473.0``.
    """
    group = _typed_group()

    stamps = group.get_timestamps_at([0, 12473], "dgt_1")
    frame = group.get_timestamp_table(format="dataframe")
    queried = group.get_timestamp_table([0, 12473], "dgt_1", format="dataframe")

    assert list(frame.columns) == [
        "dgt_1 (pixels)",
        "cpt_1 (seconds)",
        "clt_1 (quarters)",
        "pixels_to_quarters (quarters)",
        "quarters_to_seconds (seconds)",
    ]
    assert list(queried.columns) == list(frame.columns)
    for name in frame.columns:
        assert list(queried[name]) == list(frame[name])

    for position, stamp in enumerate(stamps):
        assert stamp.get_coordinate_for("dgt_1", format="int") == [0, 12473][position]
        assert (
            stamp.get_coordinate_for("clt_1").value
            == [Fraction(0, 1), Fraction(19, 2)][position]
        )

    assert list(frame["dgt_1 (pixels)"]) == [0, 12473]
    assert list(frame["cpt_1 (seconds)"]) == [0.0, 37.5]
    assert list(frame["clt_1 (quarters)"]) == [Fraction(0, 1), Fraction(19, 2)]
    assert list(frame["pixels_to_quarters (quarters)"]) == [
        Fraction(0, 1),
        Fraction(12473, 4),
    ]
    assert list(frame["quarters_to_seconds (seconds)"]) == [0.0, 38.0]

    assert str(frame["dgt_1 (pixels)"].dtype) == "int64"
    assert str(frame["cpt_1 (seconds)"].dtype) == "float64"
    assert str(frame["clt_1 (quarters)"].dtype) == "object"
    assert str(frame["pixels_to_quarters (quarters)"].dtype) == "object"
    # float64 even though the source axis is exact: a converted reading is
    # written by its target, and seconds are float-canonical.
    assert str(frame["quarters_to_seconds (seconds)"].dtype) == "float64"


def test_a_gapped_integer_column_uses_the_nullable_dtype() -> None:
    """Whole numbers where the axis reaches, a null where it does not."""
    group = TimelineGroup(id="g-gapped")
    group.add_timeline(DiscreteGraphicalTimeline(length=1000, uid="dgt_1"))
    group.add_timeline(
        DiscreteGraphicalTimeline(length=400, uid="dgt_2"),
        start=0,
        end=500,
    )

    frame = group.get_timestamp_table(format="dataframe")
    column = frame["dgt_2 (pixels)"]

    assert str(column.dtype) == "Int64"
    assert column.isna().any()
    assert list(column.dropna()) == [0, 400]


def test_the_stored_timestamps_stay_the_float_lane() -> None:
    """The store keeps its doubles; the published table carries structs.

    Interpolation runs on doubles, so the group's own store must stay
    float64. Re-expression happens where the table is published, which is
    also where the exact side is written alongside it.
    """
    group = _typed_group()

    assert [str(field.type) for field in group._timestamp_table.schema] == [
        "double",
        "double",
        "double",
    ]
    assert group._timestamp_table.column("dgt_1").to_pylist() == [0.0, 12473.0]

    published = group.get_timestamp_table()
    assert [str(field.type) for field in published.schema] == [
        "struct<value: double, numerator: int64, denominator: int64>",
    ] * 5
    assert published.column("dgt_1").combine_chunks().field(
        "numerator"
    ).to_pylist() == [
        0,
        12473,
    ]


def _float_target_timeline() -> ContinuousLogicalTimeline:
    """An exact quarters axis mapped into two float-canonical units."""
    score = ContinuousLogicalTimeline(length=Fraction(64), uid="clt_1")
    score.add_conversion_map(
        LinearMap(
            scalar=Fraction(1, 3),
            offset=0,
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.seconds,
        )
    )
    score.add_conversion_map(
        LinearMap(
            scalar=Fraction(1, 4),
            offset=1,
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.floating_measures,
        )
    )
    return score


def test_a_float_target_reads_float_from_an_exact_source() -> None:
    """Every reader of one C-Map reports the target's representation.

    The source axis is fraction-canonical and both targets are
    float-canonical, which is the combination that decides the question: if
    the source axis had any say, an exact ``1/3`` quarters would arrive as
    an exact ratio on a seconds axis, and a ``floating_measures`` reading
    would render as a forty-digit dyadic instead of ``1.25``. The five
    readers are asserted together because the property is that they cannot
    disagree, and each is a separate code path into the same map.
    """
    score = _float_target_timeline()
    stamp = score.get_timestamp(Fraction(1))
    seconds_map = score.get_conversion_map(TimeUnit.seconds)
    measures_map = score.get_conversion_map(TimeUnit.floating_measures)

    assert stamp._conversion_rows() == [
        ("seconds", 0.3333333333333333, "seconds"),
        ("floating_measures", 1.25, "floating_measures"),
    ]
    for unit, expected in (
        (TimeUnit.seconds, 0.3333333333333333),
        (TimeUnit.floating_measures, 1.25),
    ):
        converted = stamp.get_unit(unit, format="coordinate")
        assert converted.value == expected
        assert isinstance(converted.value, float)
        assert stamp.get_conversion_for(unit.value) == expected

    assert seconds_map(Fraction(1)) == 0.3333333333333333
    assert measures_map(Fraction(1)) == 1.25
    assert isinstance(seconds_map(Fraction(1)), float)
    assert isinstance(measures_map(Fraction(1)), float)


def test_one_map_reads_the_same_from_an_exact_and_an_integral_axis() -> None:
    """Two axes, one conversion, one number.

    The same seconds-valued scaling is attached to a fraction-canonical
    quarters axis and to an integer-locked pixels axis. A reading that
    carried its source axis's representation would answer these two
    differently -- which is what a caller comparing a score position against
    a scan position would then be doing.
    """
    quarters = ContinuousLogicalTimeline(length=Fraction(100), uid="clt_1")
    pixels = DiscreteGraphicalTimeline(length=20000, uid="dgt_1")
    for timeline, source_unit in (
        (quarters, TimeUnit.quarters),
        (pixels, TimeUnit.pixels),
    ):
        timeline.add_conversion_map(
            LinearMap(
                scalar=Fraction(3278347, 7350),
                offset=0,
                source_unit=source_unit,
                target_unit=TimeUnit.seconds,
            )
        )

    from_exact = quarters.get_timestamp(Fraction(1))
    from_integral = pixels.get_timestamp(1)

    assert from_exact.get_conversion_for("seconds") == 446.03360544217685
    assert from_integral.get_conversion_for("seconds") == 446.03360544217685
    assert from_exact._conversion_rows() == from_integral._conversion_rows()


def test_a_fraction_target_stays_exact_from_any_source() -> None:
    """The other direction: a fraction-canonical target is not floated.

    Expressing per target is not a preference for floats -- it is a
    preference for the target. A quarters-valued conversion off an
    integer-locked pixels axis comes back exact, because quarters are
    fraction-canonical.
    """
    pixels = DiscreteGraphicalTimeline(length=12473, uid="dgt_1")
    pixels.add_conversion_map(
        LinearMap(
            scalar=Fraction(1, 4),
            offset=0,
            source_unit=TimeUnit.pixels,
            target_unit=TimeUnit.quarters,
        )
    )
    stamp = pixels.get_timestamp(3)

    converted = stamp.get_unit(TimeUnit.quarters, format="coordinate")
    assert converted.value == Fraction(3, 4)
    assert isinstance(converted.value, Fraction)
    assert stamp._conversion_rows() == [("quarters", Fraction(3, 4), "quarters")]


def test_an_unmatched_claim_writes_its_axis_in_both_directions() -> None:
    """A NOMATCH claim carries its position outside an anchor, and it counts.

    The anchored and interval lanes were brought onto their axes first, and
    this one was missed because its coordinate lives in a different field --
    which is exactly how the same defect reappears in a fixed area. The
    storage and every rendering of it are asserted, since the coordinate is
    read straight off the model rather than through a getter (an anchorless
    claim raises on ``get_coordinate_for``, and rightly so). Both directions:
    the exact side of the seconds cell is a sixteen-digit ratio, and a
    quarters axis must still keep its exact third.
    """
    audio = ContinuousPhysicalTimeline(length=300.0, uid="Demo2")
    audio.add_events([{"id": "sect", "event_type": "Section", "instant": 164.3}])
    score = ContinuousLogicalTimeline(length=Fraction(64), uid="score")
    score.add_events([{"id": "n1", "event_type": "Note", "instant": Fraction(1, 3)}])

    on_float_axis = MatchClaim.nomatch(
        event=audio.get_event("sect"),
        source_tl_id="Demo2",
        target_tl_id="Studio",
        unit=TimeUnit.seconds,
    )
    on_exact_axis = MatchClaim.nomatch(
        event=score.get_event("n1"),
        source_tl_id="score",
        target_tl_id="perf",
        unit=TimeUnit.quarters,
    )

    assert on_float_axis.source_coordinate == Coordinate(164.3, TimeUnit.seconds)
    assert isinstance(on_float_axis.source_coordinate.value, float)
    assert repr(on_float_axis) == (
        "MatchClaim(Demo2@164.3 seconds <-> Studio [NOMATCH])"
    )
    assert "@164.3 seconds" in str(on_float_axis)
    assert "164.3 seconds" in on_float_axis._repr_html_()

    assert on_exact_axis.source_coordinate == Coordinate(
        Fraction(1, 3), TimeUnit.quarters
    )
    assert isinstance(on_exact_axis.source_coordinate.value, Fraction)
    assert repr(on_exact_axis) == ("MatchClaim(score@1/3 quarters <-> perf [NOMATCH])")
