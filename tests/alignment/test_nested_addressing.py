"""A bundle answers for every timeline of its trees, nested ones included.

The scenario and every expected value are documented in ``README.md`` under
"Every Timeline of a Bundle's Trees".
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.alignment.bundle import AlignmentBundle
from timetoalign.core import Coordinate, IdCoordinate, TimeUnit
from timetoalign.core.fields import field_metadata
from timetoalign.timelines import Timeline, TimelineGroup

from .helpers import match_claim

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
            match_claim(
                "clt_6",
                Coordinate(quarters, TimeUnit.quarters),
                "cpt_1",
                Coordinate(seconds, TimeUnit.seconds),
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
                ("clt_4", Fraction(0)),
            ],
        ),
        (
            "clt_3",
            Fraction(312),
            [
                ("clt_3", Fraction(312)),
                ("clt_7", Fraction(0)),
                ("clt_1", Fraction(668)),
                ("clt_6", Fraction(0)),
                ("cpt_1", 0.0),
                ("clt_4", Fraction(0)),
            ],
        ),
        (
            "clt_4",
            Fraction(0),
            [
                ("clt_4", Fraction(0)),
                ("clt_7", Fraction(0)),
                ("clt_1", Fraction(668)),
                ("clt_6", Fraction(0)),
                ("cpt_1", 0.0),
            ],
        ),
        (
            "clt_1",
            Fraction(1616),
            [("clt_1", Fraction(1616)), ("clt_5", Fraction(624))],
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
def test_holding_timelines_appear(
    source: str, position: Fraction, expected: list[tuple[str, object]]
) -> None:
    """A shared boundary is the later sibling's; a lone child's end is the child's."""
    bundle, _ = _bundle()

    values = _values(bundle.get_matchstamp_at(position, source).coordinates)

    assert values == expected
    _assert_exact_types(values)


def _siblings_bundle() -> AlignmentBundle:
    """Build ``P`` with the adjoining children ``A`` and ``B``, each with a child.

    Returns:
        A bundle holding ``P`` alone.
    """
    root = Timeline(length=8, unit=TimeUnit.quarters, uid="P")
    for child_id, offset in (("A", 0), ("B", 4)):
        child = Timeline(length=4, unit=TimeUnit.quarters, uid=child_id)
        root.add_child(child, offset=offset)
        child.add_child(
            Timeline(length=4, unit=TimeUnit.quarters, uid=f"{child_id}1"), offset=0
        )
    return AlignmentBundle(id="siblings").add_timeline(root)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (Fraction(6), [("A", Fraction(6))]),
        (Fraction(-1), [("A", Fraction(-1))]),
        (
            Fraction(4),
            [
                ("A", Fraction(4)),
                ("P", Fraction(4)),
                ("A1", Fraction(4)),
                ("B", Fraction(0)),
                ("B1", Fraction(0)),
            ],
        ),
        (Fraction(0), [("A", Fraction(0)), ("P", Fraction(0)), ("A1", Fraction(0))]),
    ],
)
def test_walk_up_starts_on_the_queried_timeline(
    position: Fraction, expected: list[tuple[str, object]]
) -> None:
    """A position off the queried child reaches no ancestor; its own end does."""
    bundle = _siblings_bundle()

    values = _values(bundle.get_matchstamp_at(position, "A").coordinates)

    assert values == expected
    assert all(type(value) is Fraction for _, value in values)


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
        ({"within": {"clt_4", "cpt_1"}}, ["cpt_1", "clt_4"]),
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


# region Claims on an unregistered nested timeline


def _claimed_child_bundle() -> AlignmentBundle:
    """Build a root whose unregistered child holds the claims to a recording.

    Returns:
        The bundle, with the two claims made through ``create_match_claims``.
    """
    root = Timeline(length=100, unit=TimeUnit.quarters, uid="clt_1")
    child = Timeline(length=40, unit=TimeUnit.quarters, uid="clt_2")
    root.add_child(child, offset=20)
    child.add_events(
        [
            {"id": "n1", "event_type": "Note", "instant": 0},
            {"id": "n2", "event_type": "Note", "instant": 40},
        ]
    )
    recording = Timeline(length=30.0, unit=TimeUnit.seconds, uid="cpt_1")
    recording.add_events(
        [
            {"id": "a1", "event_type": "Note", "instant": 5.0},
            {"id": "a2", "event_type": "Note", "instant": 25.0},
        ]
    )
    bundle = AlignmentBundle(id="claimed_child")
    bundle.add_timeline(root)
    bundle.add_timeline(recording)
    bundle.create_match_claims(
        [("n1", "clt_2", "a1", "cpt_1"), ("n2", "clt_2", "a2", "cpt_1")]
    )
    return bundle


def test_claims_are_made_on_an_unregistered_child() -> None:
    """``create_match_claims`` resolves events on a child no one registered."""
    bundle = _claimed_child_bundle()
    claims = bundle.get_match_claims()

    anchors = [
        (
            claim.start_anchor.timeline_a_id,
            claim.start_anchor.coordinate_a.value,
            claim.start_anchor.timeline_b_id,
            claim.start_anchor.coordinate_b.value,
        )
        for claim in claims
    ]

    assert bundle.timeline_ids == ["clt_1", "cpt_1"]
    assert anchors == [
        ("clt_2", Fraction(0), "cpt_1", 5.0),
        ("clt_2", Fraction(40), "cpt_1", 25.0),
    ]
    assert [(claim.event_a_id, claim.event_b_id) for claim in claims] == [
        ("n1", "a1"),
        ("n2", "a2"),
    ]
    assert [claim.is_synchronous for claim in claims] == [True, True]


def test_claims_on_an_unknown_timeline_name_every_known_one() -> None:
    """An id naming no timeline of the trees raises, naming every known one."""
    bundle = _claimed_child_bundle()

    with pytest.raises(KeyError) as raised:
        bundle.create_match_claims([("n1", "clt_9", "a1", "cpt_1")])

    assert raised.value.args[0] == (
        "Timeline 'clt_9' not in bundle. Available timelines: ['clt_1', 'cpt_1', 'clt_2']"
    )


@pytest.mark.parametrize(
    ("timeline_id", "position", "expected"),
    [
        (
            "clt_2",
            Fraction(20),
            [("clt_2", Fraction(20)), ("clt_1", Fraction(40)), ("cpt_1", 15.0)],
        ),
        (
            "clt_1",
            Fraction(40),
            [("clt_1", Fraction(40)), ("cpt_1", 15.0), ("clt_2", Fraction(20))],
        ),
        (
            "cpt_1",
            15.0,
            [("cpt_1", 15.0), ("clt_1", Fraction(40)), ("clt_2", Fraction(20))],
        ),
        ("clt_1", Fraction(10), [("clt_1", Fraction(10))]),
    ],
)
def test_warp_runs_through_the_claimed_child(
    timeline_id: str, position: object, expected: list[tuple[str, object]]
) -> None:
    """Between the anchors, every source reaches the same three coordinates."""
    bundle = _claimed_child_bundle()

    values = _values(bundle.get_matchstamp_at(position, timeline_id).coordinates)

    assert values == expected
    _assert_exact_types(values)


def test_only_registered_and_claimed_timelines_are_units() -> None:
    """The claimed child is a unit of its own beside the two registered ones."""
    bundle = _claimed_child_bundle()

    assert bundle._warp_units() == [
        (None, ("clt_1",)),
        (None, ("cpt_1",)),
        (None, ("clt_2",)),
    ]


def test_unclaimed_descendants_are_no_units(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hundred unclaimed children add no unit and no transfer."""
    root = Timeline(length=100, unit=TimeUnit.quarters, uid="score")
    for index in range(100):
        root.add_child(
            Timeline(length=1, unit=TimeUnit.quarters, uid=f"bar_{index}"), index
        )
    recording = Timeline(length=100.0, unit=TimeUnit.seconds, uid="perf")
    bundle = AlignmentBundle()
    bundle.add_timeline(root, as_group="gs")
    bundle.add_timeline(recording, as_group="gp")
    bundle.create_match_claims(
        [
            ({"start": position}, "score", {"start": float(position)}, "perf")
            for position in (0, 100)
        ]
    )
    transfers: list[str] = []
    transfer = AlignmentBundle._transfer_to_group

    def counting(self: AlignmentBundle, *args: object, **kwargs: object) -> object:
        transfers.append(str(args[1]))
        return transfer(self, *args, **kwargs)

    monkeypatch.setattr(AlignmentBundle, "_transfer_to_group", counting)

    units = bundle._warp_units()
    values = _values(bundle.get_matchstamp_at(Fraction(101, 2), "score").coordinates)

    assert [members for _, members in units] == [("score",), ("perf",)]
    assert sorted(values, key=lambda item: item[0]) == [
        ("bar_50", Fraction(1, 2)),
        ("perf", 50.5),
        ("score", Fraction(101, 2)),
    ]
    assert transfers == ["score"]


def test_transfer_runs_through_the_claimed_child() -> None:
    """The private transfer accepts the unregistered child on either side."""
    bundle = _claimed_child_bundle()

    forward = bundle.transfer(20, "clt_2", "cpt_1")
    backward = bundle.transfer(15.0, "cpt_1", "clt_2")

    assert (type(forward), forward) == (float, 15.0)
    assert (type(backward), backward) == (float, 20.0)


# endregion


# region One step per edge

DRAWING_X = 101.68572501334543
DRAWING_ROOT_X = 178.38943961314868


def _drawing_bundle() -> AlignmentBundle:
    """Build a page drawing whose nested child sits at offset 0 in its parent.

    Returns:
        The bundle holding the drawing's root.
    """
    root = Timeline(length=932.2326510101616, unit=TimeUnit.css_pixels, uid="cgt_1")
    child = Timeline(length=855.5289364103584, unit=TimeUnit.css_pixels, uid="cgt_2")
    grandchild = Timeline(
        length=713.3617548979767, unit=TimeUnit.css_pixels, uid="cgt_3"
    )
    root.add_child(child, offset=76.70371459980326)
    child.add_child(grandchild, offset=0)
    bundle = AlignmentBundle(id="drawing")
    bundle.add_timeline(root)
    return bundle


@pytest.mark.parametrize(
    ("source", "position", "expected"),
    [
        (
            "cgt_3",
            DRAWING_X,
            [("cgt_3", DRAWING_X), ("cgt_1", DRAWING_ROOT_X), ("cgt_2", DRAWING_X)],
        ),
        (
            "cgt_2",
            DRAWING_X,
            [("cgt_2", DRAWING_X), ("cgt_1", DRAWING_ROOT_X), ("cgt_3", DRAWING_X)],
        ),
        (
            "cgt_1",
            DRAWING_ROOT_X,
            [
                ("cgt_1", DRAWING_ROOT_X),
                ("cgt_2", 101.68572501334542),
                ("cgt_3", 101.68572501334542),
            ],
        ),
    ],
)
def test_zero_offset_is_the_identity_on_a_float_axis(
    source: str, position: float, expected: list[tuple[str, object]]
) -> None:
    """A child at offset 0 and its parent name one float, whoever is queried."""
    bundle = _drawing_bundle()

    values = _values(bundle.get_matchstamp_at(position, source).coordinates)

    assert values == expected
    assert all(type(value) is float for _, value in values)
    named = dict(values)
    assert named["cgt_2"] == named["cgt_3"]


def test_exact_tree_stays_exact_up_and_down() -> None:
    """Grandchild to parent to root to uncle: every step exact."""
    root = Timeline(length=10, unit=TimeUnit.quarters, uid="clt_1")
    uncle = Timeline(length=4, unit=TimeUnit.quarters, uid="clt_2")
    parent = Timeline(length=2, unit=TimeUnit.quarters, uid="clt_3")
    grandchild = Timeline(length=1, unit=TimeUnit.quarters, uid="clt_4")
    root.add_child(uncle, offset=Fraction(1, 3))
    root.add_child(parent, offset=Fraction(10, 3))
    parent.add_child(grandchild, offset=Fraction(1, 2))
    bundle = AlignmentBundle(id="exact")
    bundle.add_timeline(root)

    values = _values(bundle.get_matchstamp_at(Fraction(1, 4), "clt_4").coordinates)

    assert values == [
        ("clt_4", Fraction(1, 4)),
        ("clt_1", Fraction(49, 12)),
        ("clt_2", Fraction(15, 4)),
        ("clt_3", Fraction(3, 4)),
    ]
    assert all(type(value) is Fraction for _, value in values)


# endregion


# region Grouping a nested timeline

GROUPED_AT_27_HALVES = {
    "clt_6": Fraction(27, 2),
    "clt_7": Fraction(27, 2),
    "clt_1": Fraction(1363, 2),
    "clt_4": Fraction(27, 2),
}


def _tree() -> dict[str, Timeline]:
    """Build the main fixture's tree and the partner, nothing in a bundle.

    Returns:
        Every timeline of the tree and the partner ``clt_6``, by id.
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
    timelines["clt_6"] = Timeline(length=27, unit=TimeUnit.quarters, uid="clt_6")
    return timelines


def _stamp_from_partner(bundle: AlignmentBundle) -> dict[str, object]:
    """Return the values of the stamp at ``clt_6`` 27/2, by id."""
    return dict(_values(bundle.get_matchstamp_at(Fraction(27, 2), "clt_6").coordinates))


def test_grouped_with_names_a_nested_timeline() -> None:
    """``grouped_with`` finds an unregistered child and registers it as a member."""
    timelines = _tree()
    bundle = AlignmentBundle(id="grouped").add_timeline(timelines["clt_1"])

    bundle.add_timeline(timelines["clt_6"], grouped_with="clt_7")

    assert bundle.timeline_ids == ["clt_1", "clt_7", "clt_6"]
    assert bundle.groups["group_1"].timeline_ids == ["clt_7", "clt_6"]
    assert bundle.timeline_to_groups == {"clt_6": ("group_1",), "clt_7": ("group_1",)}
    values = _values(bundle.get_matchstamp_at(Fraction(27, 2), "clt_6").coordinates)
    assert values == [
        ("clt_6", Fraction(27, 2)),
        ("clt_1", Fraction(1363, 2)),
        ("clt_7", Fraction(27, 2)),
        ("clt_4", Fraction(27, 2)),
    ]
    _assert_exact_types(values)


@pytest.mark.parametrize(
    ("tree_first", "registered"),
    [(True, ["clt_1", "clt_7", "clt_6"]), (False, ["clt_7", "clt_1", "clt_6"])],
)
def test_add_group_with_a_nested_member(
    tree_first: bool, registered: list[str]
) -> None:
    """A group holding a nested member joins whether or not its tree is held."""
    timelines = _tree()
    bundle = AlignmentBundle(id="grouped")
    if tree_first:
        bundle.add_timeline(timelines["clt_1"])

    bundle.add_group(
        TimelineGroup(name="score", timelines=[timelines["clt_7"], timelines["clt_6"]])
    )

    assert bundle.timeline_ids == registered
    assert bundle.timeline_to_groups == {"clt_6": ("group_1",), "clt_7": ("group_1",)}
    assert _stamp_from_partner(bundle) == GROUPED_AT_27_HALVES


def test_grouped_with_an_unknown_timeline_names_every_known_one() -> None:
    """A name of no timeline of the trees raises, nested timelines listed."""
    timelines = _tree()
    bundle = AlignmentBundle(id="grouped").add_timeline(timelines["clt_1"])

    with pytest.raises(KeyError) as raised:
        bundle.add_timeline(timelines["clt_6"], grouped_with="clt_9")

    assert raised.value.args[0] == (
        "Timeline to group with 'clt_9' not in bundle. Available timelines: "
        "['clt_1', 'clt_2', 'clt_3', 'clt_4', 'clt_7', 'clt_5']"
    )
    assert bundle.timeline_ids == ["clt_1"]
    assert bundle.groups == {}


# endregion
