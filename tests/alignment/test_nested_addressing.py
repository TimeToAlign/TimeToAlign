"""A bundle answers for every timeline of its trees, nested ones included.

The scenario and every expected value are documented in ``README.md`` under
"Every Timeline of a Bundle's Trees".
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.alignment.bundle import AlignmentBundle
from timetoalign.alignment.claims import AlignmentAnchor, MatchClaim
from timetoalign.core import Coordinate, IdCoordinate, TimeUnit
from timetoalign.core.fields import field_metadata
from timetoalign.timelines import Timeline

# region Fixture

# (child id, offset on clt_1, length)
CHILDREN = (
    ("clt_2", 0, 356),
    ("clt_3", 356, 312),
    ("clt_4", 668, 324),
    ("clt_5", 992, 624),
)

# clt_6 quarters -> cpt_1 seconds
CLAIMS = ((0, 0.0), (27, 9.0))

REGISTERED = ["clt_7", "clt_1", "clt_6", "cpt_1"]

# One instant seen from every timeline that reaches it (README table, row 1).
AT_27_HALVES = {
    "clt_4": Fraction(27, 2),
    "clt_7": Fraction(27, 2),
    "clt_1": Fraction(1363, 2),
    "clt_6": Fraction(27, 2),
    "cpt_1": 4.5,
}


def _bundle() -> tuple[AlignmentBundle, dict[str, Timeline]]:
    """Build the nested score, its grouped partner and the claimed recording.

    Returns:
        The bundle and every timeline of the fixture, by id.
    """
    root = Timeline(length=1616, unit=TimeUnit.quarters, uid="clt_1")
    timelines = {"clt_1": root}
    for child_id, offset, length in CHILDREN:
        child = Timeline(length=length, unit=TimeUnit.quarters, uid=child_id)
        root.add_child(child, offset=offset)
        timelines[child_id] = child
    grandchild = Timeline(length=27, unit=TimeUnit.quarters, uid="clt_7")
    timelines["clt_4"].add_child(grandchild, offset=0)
    timelines["clt_7"] = grandchild
    partner = Timeline(length=27, unit=TimeUnit.quarters, uid="clt_6")
    timelines["clt_6"] = partner
    recording = Timeline(length=10.0, unit=TimeUnit.seconds, uid="cpt_1")
    timelines["cpt_1"] = recording

    bundle = AlignmentBundle(id="nested")
    bundle.add_timeline(grandchild, as_group="score")
    bundle.add_timeline(partner, grouped_with="clt_7")
    bundle.add_timeline(recording)
    bundle.add_match_claims(
        [
            MatchClaim(
                timeline_a_id="clt_6",
                timeline_b_id="cpt_1",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="clt_6",
                    coordinate_a=Coordinate(quarters, TimeUnit.quarters),
                    timeline_b_id="cpt_1",
                    coordinate_b=Coordinate(seconds, TimeUnit.seconds),
                ),
                is_synchronous=True,
            )
            for quarters, seconds in CLAIMS
        ]
    )
    return bundle, timelines


def _values(stamp_coordinates: dict[str, Coordinate]) -> list[tuple[str, object]]:
    """List a stamp's ``(timeline id, value)`` pairs in its order."""
    return [
        (timeline_id, coordinate.value)
        for timeline_id, coordinate in stamp_coordinates.items()
    ]


def _assert_exact_types(values: list[tuple[str, object]]) -> None:
    """Quarters values are Fractions, seconds values floats."""
    for timeline_id, value in values:
        expected = float if timeline_id == "cpt_1" else Fraction
        assert type(value) is expected, (timeline_id, value)


# endregion


# region Lookup


def test_registry_holds_only_what_was_registered() -> None:
    """Addressing a nested timeline enrols nothing."""
    bundle, _ = _bundle()

    bundle.get_matchstamp_at(Fraction(27, 2), "clt_4")

    assert bundle.timeline_ids == REGISTERED
    assert bundle.timeline_to_groups == {"clt_6": ("group_1",), "clt_7": ("group_1",)}


@pytest.mark.parametrize("timeline_id", ["clt_2", "clt_3", "clt_4", "clt_5", "clt_7"])
def test_get_timeline_finds_every_timeline_of_the_tree(timeline_id: str) -> None:
    """Depth one and two, registered or not: the timeline itself is returned."""
    bundle, timelines = _bundle()

    assert bundle.get_timeline(timeline_id) is timelines[timeline_id]


def test_unknown_timeline_names_every_known_one() -> None:
    """Registered timelines first, then each root's other descendants depth-first."""
    bundle, _ = _bundle()

    with pytest.raises(KeyError) as raised:
        bundle.get_matchstamp_at(0, "clt_9")

    assert raised.value.args[0] == (
        "Timeline 'clt_9' not in bundle. Available timelines: "
        "['clt_7', 'clt_1', 'clt_6', 'cpt_1', 'clt_2', 'clt_3', 'clt_4', 'clt_5']"
    )


# endregion


# region The middle ancestor


def test_query_on_the_middle_ancestor() -> None:
    """A query on clt_4 reaches its child, the root, the partner and the recording."""
    bundle, _ = _bundle()

    stamp = bundle.get_matchstamp_at(Fraction(27, 2), "clt_4")

    values = _values(stamp.coordinates)
    assert values == list(AT_27_HALVES.items())
    _assert_exact_types(values)
    assert stamp.source_id == "clt_4"
    assert stamp.is_interpolated is True


@pytest.mark.parametrize(
    ("source", "order"),
    [
        ("clt_1", ["clt_1", "clt_7", "clt_6", "cpt_1", "clt_4"]),
        ("clt_7", ["clt_7", "clt_1", "clt_6", "cpt_1", "clt_4"]),
        ("clt_6", ["clt_6", "clt_7", "clt_1", "cpt_1", "clt_4"]),
    ],
)
def test_middle_ancestor_appears_from_every_source(
    source: str, order: list[str]
) -> None:
    """The root, the grouped descendant and its partner all include clt_4."""
    bundle, _ = _bundle()

    stamp = bundle.get_matchstamp_at(AT_27_HALVES[source], source)

    values = _values(stamp.coordinates)
    assert values == [(timeline_id, AT_27_HALVES[timeline_id]) for timeline_id in order]
    _assert_exact_types(values)


# endregion


# region Spans


@pytest.mark.parametrize(
    ("source", "position", "expected"),
    [
        (
            "clt_1",
            Fraction(668),
            [
                ("clt_1", Fraction(668)),
                ("clt_7", Fraction(0)),
                ("clt_6", Fraction(0)),
                ("cpt_1", 0.0),
                ("clt_3", Fraction(312)),
                ("clt_4", Fraction(0)),
            ],
        ),
        ("clt_1", Fraction(100), [("clt_1", Fraction(100)), ("clt_2", Fraction(100))]),
        ("clt_5", Fraction(300), [("clt_5", Fraction(300)), ("clt_1", Fraction(1292))]),
        (
            "clt_4",
            Fraction(27),
            [
                ("clt_4", Fraction(27)),
                ("clt_7", Fraction(27)),
                ("clt_1", Fraction(695)),
                ("clt_6", Fraction(27)),
                ("cpt_1", 9.0),
            ],
        ),
    ],
)
def test_closed_spans_decide_who_appears(
    source: str, position: Fraction, expected: list[tuple[str, object]]
) -> None:
    """Siblings meet at a shared boundary; a covered position shows its covers only."""
    bundle, _ = _bundle()

    values = _values(bundle.get_matchstamp_at(position, source).coordinates)

    assert values == expected
    _assert_exact_types(values)


# endregion


# region Lanes


def test_plural_getter_follows() -> None:
    """get_matchstamps_at answers each position as the singular getter does."""
    bundle, _ = _bundle()

    stamps = bundle.get_matchstamps_at([Fraction(27, 2), Fraction(27)], "clt_4")

    assert [_values(stamp.coordinates) for stamp in stamps] == [
        list(AT_27_HALVES.items()),
        [
            ("clt_4", Fraction(27)),
            ("clt_7", Fraction(27)),
            ("clt_1", Fraction(695)),
            ("clt_6", Fraction(27)),
            ("cpt_1", 9.0),
        ],
    ]


def test_key_getter_follows() -> None:
    """An event stored on clt_4 is owned by the root and its stamp includes clt_4."""
    bundle, timelines = _bundle()
    timelines["clt_4"].add_events(
        [{"id": "note:1", "event_type": "Note", "instant": Fraction(27, 2)}]
    )

    stamp = bundle.get_matchstamp_for("note:1")

    assert stamp.source_id == "clt_1"
    assert _values(stamp.coordinates) == [
        ("clt_1", Fraction(1363, 2)),
        ("clt_7", Fraction(27, 2)),
        ("clt_6", Fraction(27, 2)),
        ("cpt_1", 4.5),
        ("clt_4", Fraction(27, 2)),
    ]


def test_table_follows() -> None:
    """The position table carries clt_4 as a timeline column, struct and scalar."""
    bundle, _ = _bundle()

    table = bundle.get_matchstamp_table([Fraction(27, 2)], "clt_4")
    frame = bundle.get_matchstamp_table([Fraction(27, 2)], "clt_4", format="dataframe")

    assert table.column_names == ["clt_1", "clt_4", "clt_6", "clt_7", "cpt_1"]
    assert table.column("clt_4").to_pylist() == [
        {"value": 13.5, "numerator": 27, "denominator": 2}
    ]
    metadata = field_metadata(table.schema.field("clt_4"))
    assert (metadata["timeline_id"], metadata["unit"]) == ("clt_4", "quarters")
    assert frame.to_dict("records") == [
        {
            "clt_1 (quarters)": Fraction(1363, 2),
            "clt_4 (quarters)": Fraction(27, 2),
            "clt_6 (quarters)": Fraction(27, 2),
            "clt_7 (quarters)": Fraction(27, 2),
            "cpt_1 (seconds)": 4.5,
        }
    ]


def test_coordinate_getter_follows() -> None:
    """get_coordinate_at transfers from and onto clt_4 like the stamp does."""
    bundle, _ = _bundle()

    from_middle = bundle.get_coordinate_at(
        IdCoordinate(Fraction(27, 2), TimeUnit.quarters, "clt_4"), "clt_1"
    )
    onto_middle = bundle.get_coordinate_at(
        IdCoordinate(Fraction(27, 2), TimeUnit.quarters, "clt_6"), "clt_4"
    )

    assert from_middle == IdCoordinate(Fraction(1363, 2), TimeUnit.quarters, "clt_1")
    assert type(from_middle.value) is Fraction
    assert onto_middle == IdCoordinate(Fraction(27, 2), TimeUnit.quarters, "clt_4")
    assert type(onto_middle.value) is Fraction


# endregion


# region Filters


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"timeline_ids": {"clt_4", "cpt_1"}}, ["cpt_1", "clt_4"]),
        ({"id_pattern": r"^clt_[47]$"}, ["clt_7", "clt_4"]),
        ({"include_units": {TimeUnit.seconds}}, ["cpt_1"]),
    ],
)
def test_filters_select_nested_timelines(filters: dict, expected: list[str]) -> None:
    """Every filter selects among all known timelines, nested ones included."""
    bundle, _ = _bundle()

    stamp = bundle.get_matchstamp_at(Fraction(1363, 2), "clt_1", **filters)

    assert _values(stamp.coordinates) == [
        (timeline_id, AT_27_HALVES[timeline_id]) for timeline_id in expected
    ]


# endregion
