"""Manifest files: read in any serialization, written back as they came.

The strategy is documented in ``tests/timelines/README.md`` ("Manifest
files"). Every reference text is ``timetoshare.dumps`` of a manifest written
by hand or of a package fixture, never the library's own output.
"""

from __future__ import annotations

import copy
import difflib
import io
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from timetoshare import (
    FORMAT_CONTEXT,
    complete,
    dumps,
    fixture_paths,
    loads,
    minimal,
    validate,
    with_context,
)
from timetoshare.tables import fixture_path

from tests.helpers import refusal
from timetoalign.alignment import AlignmentBundle
from timetoalign.timelines import Timeline, TimelineGroup
from timetoalign.timelines.manifest import read_projection_table
from timetoalign.timelines.types import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
)

SERIALIZATIONS = ("yaml", "json", "jcs")

FIXTURE_ROOT = Path(os.path.commonpath(fixture_paths()))
"""The package's ``fixtures/<version>/`` directory."""

READERS: dict[str, type] = {
    ".": Timeline,
    "groups": TimelineGroup,
    "bundles": AlignmentBundle,
    "statements": AlignmentBundle,
    "flow": Timeline,
}
"""The class reading a fixture directory's manifests as a whole."""

FIXTURES = sorted(
    path.relative_to(FIXTURE_ROOT).as_posix()
    for path in fixture_paths()
    if not path.name.endswith(".minimal.yaml")
)


def _write(path: Path, text: str) -> Path:
    """Write *text* to *path* exactly, line feeds included."""
    path.write_bytes(text.encode("utf-8"))
    return path


def _read(path: Path) -> str:
    """Return the exact text of *path*."""
    return path.read_bytes().decode("utf-8")


def _round_trip(reader: type, text: str, tmp_path: Path) -> str:
    """Read *text* from a file with *reader* and return what ``to_file`` writes."""
    source = _write(tmp_path / "source.txt", text)
    target = tmp_path / "target.txt"
    assert reader.from_file(source).to_file(target) == target
    return _read(target)


SCORE = {"id": "#score", "length": [32, 1], "unit": "quarters"}
TAKE = {"id": "#take_1", "length": 64.5, "unit": "seconds"}

GROUPED = {
    "@context": dict(FORMAT_CONTEXT),
    "timetoshare": "0.2.0",
    "timelines": [SCORE, TAKE],
    "groups": [{"id": "#g", "members": ["#score", "#take_1"]}],
}
"""A minimal anonymous bundle of one group, without ``graphical``."""

STYLED = {
    **GROUPED,
    "graphical": {
        "order": {"#g": ["#take_1", "#score"]},
        "styles": {"frame-highlight": {"stroke": "#7F00FF"}},
    },
}
"""The same manifest with a ``graphical`` object."""

SIMPLE = {
    "@context": dict(FORMAT_CONTEXT),
    "timetoshare": "0.2.0",
    "timelines": [SCORE, TAKE],
    "name": "Study",
}
"""A minimal named bundle of two ungrouped timelines."""

EVENTS = [
    {
        "duration": None,
        "end": None,
        "event_type": "Beat",
        "id": "e1",
        "name": None,
        "start": 0.0,
        "temporal_type": "instant",
    },
    {
        "duration": 1.5,
        "end": 2.5,
        "event_type": "Note",
        "id": "e2",
        "name": "C4",
        "start": 1.0,
        "temporal_type": "interval",
    },
]
"""The event rows of ``take_1``: an instant and an interval."""

REFERENCES = [
    {
        "access_points": [{"kind": "relative_path", "uri": "analysis/L1.pnml"}],
        "comment": None,
        "event_id": "e1",
        "external_id": "p2",
    }
]
"""The external-reference row of ``take_1``, naming its first event."""

WITH_TABLES = {
    "@context": dict(FORMAT_CONTEXT),
    "timetoshare": "0.2.0",
    "timelines": [
        {**SCORE, "events": [], "external_references": []},
        {**TAKE, "events": EVENTS, "external_references": REFERENCES},
    ],
}
"""A bundle whose timeline objects hold ``events`` and ``external_references``."""


def _without(manifest: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Return *manifest* with *keys* left out of every timeline object."""
    timelines = [
        {key: value for key, value in timeline.items() if key not in keys}
        for timeline in manifest["timelines"]
    ]
    return {**manifest, "timelines": timelines}


# region Round trips


@pytest.mark.parametrize("manifest", [GROUPED, STYLED, SIMPLE])
def test_the_literals_are_valid_minimal_manifests(manifest: dict[str, Any]) -> None:
    """The hand-written manifests are valid, and minimal."""
    validate(manifest)
    assert dumps(minimal(manifest)) == dumps(manifest)
    assert loads(dumps(manifest)) == manifest


@pytest.mark.parametrize("fixture", FIXTURES)
def test_every_fixture_file_is_written_back_byte_for_byte(
    fixture: str, tmp_path: Path
) -> None:
    """Each form, serialization and styling of each fixture keeps its bytes.

    A fixture without ``@context`` gains the one the writer emits.
    """
    reader = READERS[Path(fixture).parent.as_posix()]
    twin = fixture.removesuffix(".yaml") + ".minimal.yaml"
    for name in (fixture, twin):
        value = loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))
        variants = [value]
        if "graphical" in value:
            variants.append({k: v for k, v in value.items() if k != "graphical"})
        for manifest in variants:
            for serialization in SERIALIZATIONS:
                text = dumps(manifest, serialization=serialization)
                written = dumps(with_context(manifest, {}), serialization=serialization)
                assert _round_trip(reader, text, tmp_path) == written, (
                    name,
                    serialization,
                    "graphical" in manifest,
                )


@pytest.mark.parametrize(
    "manifest",
    [
        WITH_TABLES,
        _without(WITH_TABLES, "external_references"),
        _without(WITH_TABLES, "events"),
    ],
    ids=["both", "events", "external_references"],
)
@pytest.mark.parametrize("serialization", SERIALIZATIONS)
def test_event_rows_and_external_references_are_written_back(
    manifest: dict[str, Any], serialization: str, tmp_path: Path
) -> None:
    """The keys a file's timeline objects hold are written back, each on its own."""
    text = dumps(manifest, serialization=serialization)
    assert _round_trip(AlignmentBundle, text, tmp_path) == text


# endregion


# region Inference


def test_a_key_at_its_default_makes_the_file_complete(tmp_path: Path) -> None:
    """One key stated at its default is enough for the complete form."""
    stated = copy.deepcopy(SIMPLE)
    stated["timelines"][0]["meta"] = {}
    text = dumps(stated)
    assert _round_trip(AlignmentBundle, text, tmp_path) == dumps(complete(stated))
    text = dumps(SIMPLE)
    assert _round_trip(AlignmentBundle, text, tmp_path) == text


def test_a_float_at_an_integer_default_makes_the_file_complete(tmp_path: Path) -> None:
    """``offset: 0.0`` states the default ``0``: complete, the float kept."""
    stated = copy.deepcopy(SIMPLE)
    stated["timelines"][1]["conversion_maps"] = [
        {
            "id": "#cmap_1",
            "offset": 0.0,
            "scalar": 48000,
            "source_unit": "seconds",
            "target_unit": "samples",
            "type": "LinearMap",
        }
    ]
    validate(stated)
    written = _round_trip(AlignmentBundle, dumps(stated), tmp_path)
    assert written == dumps(complete(stated))
    assert "        offset: 0.0\n" in written


def test_styling_is_kept_as_read(tmp_path: Path) -> None:
    """A group file without ``graphical`` stays without it; with one it keeps it."""
    group = _write(tmp_path / "group.yaml", dumps(GROUPED))
    assert "graphical" in TimelineGroup.from_file(group).to_dict()
    for manifest in (GROUPED, STYLED):
        text = dumps(manifest)
        assert _round_trip(TimelineGroup, text, tmp_path) == text


def test_a_text_that_is_not_canonical_is_written_canonical(tmp_path: Path) -> None:
    """JSON on one line is read as pretty JSON and written as pretty JSON."""
    text = (
        '{"timetoshare": "0.2.0", "name": "Study", "timelines": ['
        '{"id": "#score", "unit": "quarters", "length": [32, 1]}, '
        '{"id": "#take_1", "unit": "seconds", "length": 64.5}]}'
    )
    assert loads(text) == {k: v for k, v in SIMPLE.items() if k != "@context"}
    written = _round_trip(AlignmentBundle, text, tmp_path)
    assert written == dumps(SIMPLE, serialization="json")


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ({"minimal": False}, dumps(complete(STYLED))),
        (
            {"styling": False},
            dumps({k: v for k, v in STYLED.items() if k != "graphical"}),
        ),
        ({"serialization": "jcs"}, dumps(STYLED, serialization="jcs")),
    ],
    ids=["minimal", "styling", "serialization"],
)
def test_each_argument_overrides_one_inference(
    argument: dict[str, Any], expected: str, tmp_path: Path
) -> None:
    """Each argument changes what it names and keeps the other two as read."""
    source = _write(tmp_path / "group.yaml", dumps(STYLED))
    target = TimelineGroup.from_file(source).to_file(tmp_path / "out", **argument)
    assert _read(target) == expected


def test_a_file_is_validated_as_it_is_written(tmp_path: Path) -> None:
    """Styling the format refuses blocks a styled file only.

    The fixture names timestamps at 0 and 480 ticks; shortened to 400, the
    timeline's styling names a coordinate beyond its end.
    """
    timeline = Timeline.from_file(FIXTURE_ROOT / "discrete_logical.minimal.yaml")
    timeline.length = 400

    with pytest.raises(ValueError) as refused:
        timeline.to_file(tmp_path / "styled.yaml")
    code, path, message = refusal(refused.value)
    assert (code, path) == (
        "timestamp_coordinate",
        "#/graphical/timestamps/1/coordinate",
    )
    assert "'timestamp_2'" in message
    assert not (tmp_path / "styled.yaml").exists()

    written = timeline.to_file(tmp_path / "plain.yaml", styling=False)

    assert sorted(loads(_read(written))) == [
        "@context",
        "meta",
        "timelines",
        "timetoshare",
    ]
    assert Timeline.from_file(written).length.value == 400


def test_an_object_built_in_code_is_written_minimal_styled_in_yaml(
    tmp_path: Path,
) -> None:
    """Minimal form, styling included, canonical YAML."""
    bundle = AlignmentBundle.from_dict(STYLED)
    written = _read(bundle.to_file(tmp_path / "bundle.yaml"))
    assert written == dumps(STYLED)
    assert written == dumps(bundle.to_dict())
    assert "graphical:" in written


def test_event_rows_are_left_out_on_request(tmp_path: Path) -> None:
    """``events=False`` writes a file read with event rows without them."""
    source = _write(tmp_path / "bundle.yaml", dumps(WITH_TABLES))
    target = AlignmentBundle.from_file(source).to_file(tmp_path / "out", events=False)
    assert _read(target) == dumps(_without(WITH_TABLES, "events"))


def test_an_object_built_in_code_writes_event_rows_on_request(
    tmp_path: Path,
) -> None:
    """Neither key by default; both with ``events`` and ``external_references``."""
    take = ContinuousPhysicalTimeline(length=64.5, uid="take_1")
    take.add_events(
        [
            {
                "id": "e1",
                "temporal_type": "instant",
                "event_type": "Beat",
                "instant": 0.0,
            },
            {
                "id": "e2",
                "name": "C4",
                "temporal_type": "interval",
                "event_type": "Note",
                "start": 1.0,
                "end": 2.5,
            },
        ]
    )
    take.add_external_references(REFERENCES)
    bundle = AlignmentBundle()
    bundle.add_timeline(ContinuousLogicalTimeline(length=Fraction(32), uid="score"))
    bundle.add_timeline(take)
    plain = bundle.to_file(tmp_path / "plain.yaml")
    assert _read(plain) == dumps(_without(WITH_TABLES, "events", "external_references"))
    tables = bundle.to_file(
        tmp_path / "tables.yaml", events=True, external_references=True
    )
    assert _read(tables) == dumps(WITH_TABLES)


def test_a_drawing_written_is_a_drawing_read(tmp_path: Path) -> None:
    """A child ending at its page in doubles only is written and read back."""
    page = ContinuousGraphicalTimeline(
        length=932.2326510101616, unit="points", uid="page"
    )
    system = ContinuousGraphicalTimeline(
        length=855.5289364103584, unit="points", uid="system"
    )
    page.add_child(system, 76.70371459980326)
    bundle = AlignmentBundle()
    bundle.add_timeline(page)
    path = bundle.to_file(tmp_path / "drawing.yaml")
    assert AlignmentBundle.from_file(path).to_dict() == bundle.to_dict()


def test_selection_passes_through(tmp_path: Path) -> None:
    """``from_file`` selects as ``from_dict`` does, and keeps the file's form."""
    path = _write(tmp_path / "group.json", dumps(GROUPED, serialization="json"))
    take = Timeline.from_file(path, timeline_id="take_1")
    assert take.id == "take_1"
    expected = dumps(
        {
            "@context": dict(FORMAT_CONTEXT),
            "timetoshare": "0.2.0",
            "timelines": [TAKE],
        },
        serialization="json",
    )
    assert _read(take.to_file(tmp_path / "take.json")) == expected
    group = TimelineGroup.from_file(path, group_id="g")
    assert group.timeline_ids == ["score", "take_1"]


# endregion


# region The editing workflow


EDITED_DIFF = """\
--- before.yaml
+++ after.yaml
@@ -63,4 +63,8 @@
     id: "#take_1"
     length: 64.5
     unit: seconds
+  -
+    id: "#take_2"
+    length: 70.25
+    unit: seconds
 timetoshare: "0.2.0"
"""


def test_a_timeline_added_changes_only_its_own_lines(tmp_path: Path) -> None:
    """Read, add a timeline, write back: the diff is the new timeline alone."""
    before = dumps(SIMPLE)
    path = _write(tmp_path / "study.yaml", before)
    bundle = AlignmentBundle.from_file(path)
    bundle.add_timeline(ContinuousPhysicalTimeline(length=70.25, uid="take_2"))
    bundle.to_file(path)
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        _read(path).splitlines(keepends=True),
        "before.yaml",
        "after.yaml",
    )
    assert "".join(diff) == EDITED_DIFF


def test_the_projection_table_is_read_from_what_parquet_reads() -> None:
    """A path string, a Path, and two file objects give the same Projections."""
    path = fixture_path("projections.parquet")
    manifest = loads(fixture_path("manifest.yaml").read_text(encoding="utf-8"))
    data = path.read_bytes()
    read = read_projection_table(str(path), manifest)
    assert {key: [p.id for p in value] for key, value in read.items()} == {
        "clt_2": ["projection_10"],
        "cpt_1": ["projection_11", "projection_12"],
        "dpt_1": [f"projection_{n}" for n in (1, 2, 3, 4, 5, 8, 9)],
    }
    for source in (path, io.BytesIO(data), pa.BufferReader(data)):
        assert read_projection_table(source, manifest) == read


# endregion
