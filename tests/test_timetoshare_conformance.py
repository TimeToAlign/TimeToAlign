"""Prove the library's manifests against the installed TimeToShare package."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from timetoshare import (
    FORMAT_CONTEXT,
    UNITS,
    SchemaValidationError,
    UnsupportedVersionError,
    canonical_order,
    complete,
    dumps,
    fixture_paths,
    load_catalogue,
    loads,
    minimal,
    rejection_case,
    rejection_paths,
    unknown_keys,
    validate,
    warning_paths,
    with_context,
)
from timetoshare.tables import (
    extraction_problems,
    non_event_rows,
    package_fixture_path,
    projection_problems,
    registry_problems,
)

from tests.helpers import filed
from timetoalign import (
    AlignmentBundle,
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
    Ms3Loader,
    Timeline,
    TimelineGroup,
)
from timetoalign.alignment import Package, resolve
from timetoalign.core import BoundingBox, Box, FlowManifestation, TimeUnit
from timetoalign.core.time import number_from_json, number_to_json
from timetoalign.loader.alignment import (
    Ieee1599Loader,
    ListenHereLoader,
    MatchfileLoader,
    MpmLoader,
    ParangonadaLoader,
)
from timetoalign.loader.alignment.tilia import TiliaJsonLoader
from timetoalign.loader.graphical.iiif import IIIFManifestLoader
from timetoalign.loader.graphical.paths import VerticalLinePath
from timetoalign.loader.midi.score import ScoreMidiLoader
from timetoalign.loader.physical.audio import AudioLoader
from timetoalign.maps import (
    BeatInMeasureMap,
    ChainMap,
    CombinationMap,
    ConstantMap,
    FloatingMeasureMap,
    InterpolationMap,
    IntervalToConstantMap,
    LinearMap,
    MeasureBeatMap,
    MeasureNumberMap,
    MetricalPositionMap,
    MetricMap,
    PiecewiseMap,
    QuartersToFloatingMeasures,
    QuartersToMeasureNumber,
    QuartersToTicks,
    SamplesToSeconds,
    ScalarMap,
    SecondsToSamples,
    ShiftMap,
    TableMap,
    TicksToQuarters,
)
from timetoalign.testdata import ensure_data
from timetoalign.timelines import MeasureMap
from timetoalign.timelines.flowcontrol import (
    Break,
    Jump,
    JumpDestination,
    JumpTarget,
)

SCORE = ensure_data("score")
MIDI = ensure_data("midi")
AUDIO = ensure_data("audio")
PARANGONAR = ensure_data("parangonar")
SUPRA = ensure_data("supra")
VIENNA = ensure_data("vienna_1x22")
MPM = ensure_data("mpm_toolbox")
IEEE1599 = ensure_data("ieee1599")
TILIA = SCORE / "bruckner5_scherzo/harnoncourt/Bruckner5_Scherzo.json"


def _with_structure(cls: type[Timeline], unit: str, length: Any) -> Timeline:
    """Build one direct timeline with every timeline-native structural member."""
    timeline = cls(length=length, unit=unit)
    timeline.add_child(cls(length=length / 3, unit=unit), length / 6)
    timeline.add_child(cls(length=length / 4, unit=unit), length / 2)
    timeline.create_region(length / 6, length / 2, id="theme", name="theme")
    timeline.flow_control.add(
        Break(
            timeline.make_coordinate(length / 2),
            manifestation=FlowManifestation.fine,
        )
    )
    start = timeline.flow_control.add(
        JumpTarget(
            timeline.make_coordinate(length / 6),
            name="start",
            manifestation=FlowManifestation.repeat_start,
        )
    )
    timeline.flow_control.add(
        Jump(
            timeline.make_coordinate(length / 2),
            [
                JumpDestination(
                    to=start.id,
                    repeat_count=2,
                    manifestation=FlowManifestation.repeat_end,
                )
            ],
        )
    )
    if unit != "quarters":
        return timeline
    timeline.add_measure_map(
        MeasureMap.from_rows(
            [
                {
                    "count": 1,
                    "qstamp": 0,
                    "number": 1,
                    "name": "1",
                    "time_signature": "4/4",
                    "actual_length": 4,
                },
                {
                    "count": 2,
                    "qstamp": 4,
                    "number": 2,
                    "name": "2",
                    "time_signature": "4/4",
                    "actual_length": 4,
                },
            ]
        )
    )
    return timeline


def _conversion_map_timelines() -> list[Timeline]:
    """Attach every registered conversion-map class to direct timelines."""
    meter_for_position = MetricMap(
        starts=[Fraction(0), Fraction(4)],
        mcs=[1, 2],
        mns=["1", "2"],
        lengths=[Fraction(4), Fraction(4)],
    )
    meter_for_beat = MetricMap(
        starts=[Fraction(0), Fraction(3)],
        mcs=[1, 2],
        mns=["1", "2"],
        lengths=[Fraction(3), Fraction(3)],
    )
    timeline = _with_structure(ContinuousLogicalTimeline, "quarters", Fraction(16))
    maps = [
        MetricalPositionMap(meter_for_position),
        BeatInMeasureMap(meter_for_beat),
        ChainMap([ScalarMap(Fraction(1, 2)), ScalarMap(60)]),
        CombinationMap({"label": ConstantMap(value="verse"), "x": LinearMap(scalar=2)}),
        ConstantMap(value="section-a"),
        InterpolationMap(
            source_coords=np.array([0.0, 8.0, 16.0]),
            target_coords=np.array([0.0, 4.25, 9.0]),
            source_id=timeline.id,
            target_id="seconds",
        ),
        IntervalToConstantMap(
            boundaries=[Fraction(0), Fraction(8)], values=["a", "b"], end=Fraction(16)
        ),
        LinearMap(scalar=Fraction(3, 2), offset=Fraction(1, 3)),
        MetricMap(starts=[Fraction(0)], mcs=[1], mns=["1"], lengths=[Fraction(4)]),
        PiecewiseMap(breaks=[Fraction(0), Fraction(8)], maps=[ShiftMap(offset=1)]),
        QuartersToFloatingMeasures(
            x_values=[Fraction(0), Fraction(4)], y_values=[1.0, 2.0]
        ),
        QuartersToMeasureNumber(boundaries=[Fraction(0)], mns=["1"]),
        QuartersToTicks(ppq=480),
        ScalarMap(Fraction(2, 3)),
        ShiftMap(offset=Fraction(1, 2)),
        TableMap(x_values=[Fraction(0), Fraction(4)], y_values=[0.0, 2.5]),
    ]
    for conversion_map in maps:
        timeline.add_conversion_map(conversion_map)
    for view in (MeasureNumberMap, FloatingMeasureMap, MeasureBeatMap):
        timeline.add_conversion_map(view(timeline.measure_map))
    ticks = DiscreteLogicalTimeline(length=960, unit="ticks")
    ticks.add_conversion_map(TicksToQuarters(ppq=480))
    samples = DiscretePhysicalTimeline(length=96000, unit="samples")
    samples.add_conversion_map(SamplesToSeconds(sample_rate=48000))
    seconds = ContinuousPhysicalTimeline(length=2.0, unit="seconds")
    seconds.add_conversion_map(SecondsToSamples(sample_rate=48000))
    return [timeline, ticks, samples, seconds]


def _ms3() -> Timeline:
    return (
        Ms3Loader(auto_discover=True)
        .load(
            SCORE
            / "flow_control/flow_only/out_of_the_flow_experience-flow_only.notes.tsv"
        )
        .create_timeline()
    )


def _midi() -> Timeline:
    return ScoreMidiLoader().load(MIDI / "score/beethoven_mtd.mid").create_timeline()


def _audio() -> list[Timeline]:
    """The samples timeline of an audio file and its seconds twin."""
    loader = AudioLoader().load(AUDIO / "hard_techno/Bass Kick.mp3")
    return [loader.create_timeline(), loader.create_continuous_timeline()]


def _parangonada() -> Timeline:
    return (
        ParangonadaLoader()
        .load(PARANGONAR / "Beethoven_Eroica_op35-cpjku")
        .create_timelines()[0]
    )


def _graphical() -> Timeline:
    """The SUPRA roll along its height, without the statement naming its canvas.

    Every manifest of the catalogue is one without statements.
    """
    (roll,) = (
        IIIFManifestLoader()
        .load(SUPRA / "image/ifff_manifest.json")
        .create_timelines(VerticalLinePath(x=2048, y0=0, y1=299400, tolerance=2048))
    )
    (canvas,) = roll.statements["dcterms:source"]
    return roll.remove_statement("dcterms:source", canvas)


def _direct_groups() -> list[TimelineGroup]:
    """A group across number types, and one whose member is a child."""
    score = ContinuousLogicalTimeline(length=Fraction(12), unit="quarters")
    performance = ContinuousPhysicalTimeline(length=12.5, unit="seconds")
    page = DiscreteGraphicalTimeline(length=Fraction(1920), unit="pixels")
    number_types = TimelineGroup(timelines=[score, page, performance])
    piece = ContinuousLogicalTimeline(length=Fraction(16), unit="quarters")
    second_half = ContinuousLogicalTimeline(length=Fraction(8), unit="quarters")
    piece.add_child(second_half, Fraction(8))
    recording = ContinuousPhysicalTimeline(length=10.0, unit="seconds")
    return [number_types, TimelineGroup(timelines=[second_half, recording])]


def _direct_bundle() -> list[AlignmentBundle]:
    """Two groups, an ungrouped root, and a grouped child whose root is not."""
    score = ContinuousLogicalTimeline(length=Fraction(12), unit="quarters")
    performance = ContinuousPhysicalTimeline(length=12.5, unit="seconds")
    page = DiscreteGraphicalTimeline(length=Fraction(1920), unit="pixels")
    recording = DiscretePhysicalTimeline(length=Fraction(441000), unit="samples")
    excerpt = DiscretePhysicalTimeline(length=Fraction(88200), unit="samples")
    recording.add_child(excerpt, Fraction(44100))
    bundle = AlignmentBundle(name="Mixed")
    bundle.add_timeline(score, as_group="Performance")
    bundle.add_timeline(performance, grouped_with=score.id)
    bundle.add_timeline(page, as_group="Facsimile")
    bundle.add_timeline(excerpt, grouped_with=page.id)
    bundle.add_timeline(ContinuousLogicalTimeline(length=Fraction(12), unit="quarters"))
    return [bundle]


LISTEN_HERE = {
    "header": {"ref": "rec-ref.mp3", "createdBy": "Listen Here! v0.20.0"},
    "body": {
        "audio": {
            "rec-a.mp3": {"times": [0.0, 0.02, 0.04], "peaks": [], "duration": 0.04},
            "rec-ref.mp3": {"times": [0.0, 0.025, 0.05], "peaks": [], "duration": 0.05},
        }
    },
}


def _listen_here() -> AlignmentBundle:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "alignment.json"
        path.write_text(json.dumps(LISTEN_HERE), encoding="utf-8")
        return ListenHereLoader.from_file(path).create_bundle()


CATALOGUE: list[tuple[str, Callable[[], list[Any]]]] = [
    (
        "continuous_logical",
        lambda: [_with_structure(ContinuousLogicalTimeline, "quarters", Fraction(12))],
    ),
    (
        "discrete_logical",
        lambda: [_with_structure(DiscreteLogicalTimeline, "ticks", Fraction(480))],
    ),
    (
        "continuous_physical",
        lambda: [_with_structure(ContinuousPhysicalTimeline, "seconds", 12.0)],
    ),
    (
        "discrete_physical",
        lambda: [_with_structure(DiscretePhysicalTimeline, "samples", Fraction(44100))],
    ),
    (
        "continuous_graphical",
        lambda: [_with_structure(ContinuousGraphicalTimeline, "points", 600.0)],
    ),
    (
        "css_pixels",
        lambda: [_with_structure(ContinuousGraphicalTimeline, "css_pixels", 600.0)],
    ),
    (
        "user_units",
        lambda: [_with_structure(ContinuousGraphicalTimeline, "user_units", 600.0)],
    ),
    (
        "discrete_graphical",
        lambda: [_with_structure(DiscreteGraphicalTimeline, "pixels", Fraction(1920))],
    ),
    ("conversion_maps", _conversion_map_timelines),
    ("ms3", lambda: [_ms3()]),
    ("midi", lambda: [_midi()]),
    ("audio", _audio),
    ("parangonada_timeline", lambda: [_parangonada()]),
    ("graphical", lambda: [_graphical()]),
    ("direct_groups", _direct_groups),
    ("direct_bundle", _direct_bundle),
    ("tilia_group", lambda: [TiliaJsonLoader.from_file(TILIA).create_group()]),
    (
        "parangonada",
        lambda: [
            ParangonadaLoader.from_file(
                PARANGONAR / "Beethoven_Eroica_op35-cpjku"
            ).create_bundle()
        ],
    ),
    (
        "matchfile",
        lambda: [
            MatchfileLoader.from_file(
                VIENNA / "Chopin_op10_no3_p01.match"
            ).create_bundle()
        ],
    ),
    (
        "mpm",
        lambda: [
            MpmLoader.from_file(
                MPM
                / "MPRproject_1971Curzon_VariationXIV"
                / "Beethoven_op35_1971Curzon_Var14only.mpr"
            ).create_bundle()
        ],
    ),
    (
        "ieee1599",
        lambda: [
            Ieee1599Loader.from_file(
                IEEE1599 / "Khorus Music/khomus.xml"
            ).create_bundle()
        ],
    ),
    ("listen_here", lambda: [_listen_here()]),
    ("tilia_bundle", lambda: [TiliaJsonLoader.from_file(TILIA).create_bundle()]),
]
"""Timelines, groups and bundles built directly and through the loaders."""


def _as_written(value: dict[str, Any]) -> str:
    """Return a manifest's JSON text with sorted keys and arrays as they stand."""
    return json.dumps(value, sort_keys=True)


@pytest.mark.parametrize(
    ("name", "builder"), CATALOGUE, ids=[name for name, _ in CATALOGUE]
)
def test_built_manifests_conform(name: str, builder: Callable[[], list[Any]]) -> None:
    """Every manifest the library builds conforms in both forms.

    Both forms validate without unknown keys, carry the inline context of a
    manifest without statements (the format's terms alone), ``minimal`` and
    ``complete`` map each onto the other, and the library already writes
    the canonical order the schema package defines.
    """
    del name
    for item in builder():
        written, written_minimal = item.to_dict(minimal=False), item.to_dict()
        for value in (written, written_minimal):
            assert value["@context"] == FORMAT_CONTEXT
            validate(value)
            assert unknown_keys(value) == []
            assert _as_written(canonical_order(value)) == _as_written(value)
        assert dumps(minimal(written)) == dumps(written_minimal)
        assert dumps(complete(written_minimal)) == dumps(written)


READERS: dict[str, type] = {
    ".": Timeline,
    "groups": TimelineGroup,
    "bundles": AlignmentBundle,
    "statements": AlignmentBundle,
    "flow": Timeline,
}
"""The class reading a fixture directory's manifests as a whole."""

FIXTURE_ROOT = Path(os.path.commonpath(fixture_paths()))
"""The package's ``fixtures/<version>/`` directory, which holds manifests itself."""
TWIN_SUFFIX = ".minimal.yaml"
FIXTURES = sorted(
    path.relative_to(FIXTURE_ROOT).as_posix()
    for path in fixture_paths()
    if not path.name.endswith(TWIN_SUFFIX)
)


def _twin(fixture: str) -> str:
    """Return the relative path of a fixture's minimal twin."""
    return fixture.removesuffix(".yaml") + TWIN_SUFFIX


def test_every_fixture_has_its_minimal_twin() -> None:
    """Fixtures and twins pair exactly, in the directories a reader is named for."""
    listed = {path.relative_to(FIXTURE_ROOT).as_posix() for path in fixture_paths()}
    twins = {path for path in listed if path.endswith(TWIN_SUFFIX)}
    assert twins == {_twin(fixture) for fixture in FIXTURES}
    assert len(listed) == 2 * len(FIXTURES)
    assert {Path(fixture).parent.as_posix() for fixture in FIXTURES} == set(READERS)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_fixture_and_twin_obey_the_laws(fixture: str) -> None:
    """A fixture and its twin are the package's two forms of one library object.

    Both validate without unknown keys; ``minimal`` of the fixture is the
    twin and ``complete`` of the twin the fixture; the directory's reader
    reads both to objects writing the same manifests, each written back byte
    for byte, its arrays already in canonical order.
    """
    text = (FIXTURE_ROOT / fixture).read_text(encoding="utf-8")
    twin_text = (FIXTURE_ROOT / _twin(fixture)).read_text(encoding="utf-8")
    value, twin = loads(text), loads(twin_text)
    for manifest in (value, twin):
        validate(manifest)
        assert unknown_keys(manifest) == []
    assert dumps(minimal(value)) == twin_text
    assert dumps(complete(twin)) == text

    reader = READERS[Path(fixture).parent.as_posix()]
    from_fixture, from_twin = reader.from_dict(value), reader.from_dict(twin)
    assert type(from_fixture) is type(from_twin)
    for read in (from_fixture, from_twin):
        written, written_minimal = read.to_dict(minimal=False), read.to_dict()
        assert dumps(written) == text
        assert dumps(written_minimal) == twin_text
        assert _as_written(written) == _as_written(value)
        assert _as_written(written_minimal) == _as_written(twin)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_a_bundle_reads_every_fixture(fixture: str) -> None:
    """``AlignmentBundle.from_dict`` reads any manifest and writes it back.

    The bundle's minimal form is the package's ``minimal`` of its complete
    form, whichever form was read.
    """
    for path, form in ((fixture, False), (_twin(fixture), True)):
        text = (FIXTURE_ROOT / path).read_text(encoding="utf-8")
        bundle = AlignmentBundle.from_dict(loads(text))
        assert dumps(bundle.to_dict(minimal=form)) == text
        written, written_complete = bundle.to_dict(), bundle.to_dict(minimal=False)
        assert written == minimal(written_complete)
        assert dumps(written) == dumps(minimal(written_complete))


REJECTIONS = {path.stem: path for path in rejection_paths()}


def _whole_reader(manifest: Any) -> Callable[[Any], Any]:
    """The class that reads *manifest* as a whole; the bundle for anything else."""
    if isinstance(manifest, dict) and isinstance(manifest.get("timelines"), list):
        groups = manifest.get("groups", [])
        if len(manifest["timelines"]) == 1 and not groups:
            return Timeline.from_dict
        if len(groups) == 1 and "id" not in manifest and "name" not in manifest:
            return TimelineGroup.from_dict
    return AlignmentBundle.from_dict


@pytest.mark.parametrize("case", sorted(REJECTIONS))
def test_the_library_refuses_every_rejection_case(case: str) -> None:
    """Both the whole reader and the bundle reader refuse each case as the validator.

    The library validates every manifest it reads with the package's
    validator and passes its refusal on unchanged, so each case is refused
    with the case's own code at the case's own path; the version case with
    the package's ``UnsupportedVersionError``.
    """
    rejection = rejection_case(REJECTIONS[case])
    for reader in {_whole_reader(rejection.manifest), AlignmentBundle.from_dict}:
        with pytest.raises(ValueError) as refused:
            reader(rejection.manifest)
        if rejection.error == "UnsupportedVersionError":
            assert type(refused.value) is UnsupportedVersionError
        else:
            assert isinstance(refused.value, SchemaValidationError)
            assert (refused.value.code, refused.value.path) == (
                rejection.code,
                rejection.path,
            )


SEVERAL_DEPTHS_ROW = {
    "actual_length": [4, 1],
    "count": 1,
    "end_repeat": False,
    "name": "1",
    "nominal_length": [4, 1],
    "number": 1,
    "qstamp": [0, 1],
    "start_repeat": False,
    "tempo": 120,
    "time_signature": "4/4",
}
"""The one measure-map row of ``several_depths.yaml`` as the library writes it."""


@pytest.mark.parametrize("path", warning_paths(), ids=lambda path: path.stem)
def test_the_library_writes_every_warning_case_back(path: Path) -> None:
    """Each case reads, and writes back in its own form with the same unknown keys.

    A case in the minimal form is written back as it stands, any other in
    the complete form as the package's ``complete`` writes it. The manifest
    written carries the inline context the package builds for the case: the
    format's terms and the prefixes its statements use, each standing for
    the namespace the case declares, else the catalogue's. A case without
    ``@context`` (``context_absent``) gains one; a prefix the case declares
    but does not use (``bundle_blank_*``) is not written.
    """
    case = loads(path.read_text(encoding="utf-8"))
    manifest = case["manifest"]
    is_minimal = dumps(manifest) == dumps(minimal(manifest))
    form = manifest if is_minimal else complete(manifest)
    expected = with_context(form, load_catalogue().prefixes)
    if path.stem == "several_depths":
        expected["measure_maps"][0]["rows"] = [SEVERAL_DEPTHS_ROW]

    written = AlignmentBundle.from_dict(manifest).to_dict(minimal=is_minimal)

    assert dumps(written) == dumps(expected)
    assert unknown_keys(written) == case["unknown_keys"]


def _quarters_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a valid manifest whose one root reads the measure map of *rows*."""
    result = ContinuousLogicalTimeline(length=Fraction(8), unit="quarters").to_dict()
    result["timelines"][0]["conversion_maps"] = [
        {
            "type": "MeasureRecordMap",
            "id": "#record",
            "source_unit": "quarters",
            "target_unit": None,
            "measure_map": "#bars",
        }
    ]
    result["measure_maps"] = [{"id": "#bars", "rows": rows}]
    return result


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [{"count": 1, "time_signature": "4/4"}, {"count": 3}, {"count": 2}],
            "count 2 does not increase",
        ),
        (
            [{"count": 1, "time_signature": "4/4", "next": [3]}, {"count": 2}],
            "next 3 at count 1 lies beyond the last row",
        ),
    ],
)
def test_row_rules_beyond_the_schema_are_the_library_s(
    rows: list[dict[str, Any]], message: str
) -> None:
    """The schema cannot express these rules; the library reader refuses them."""
    value = _quarters_manifest(rows)
    validate(value)
    with pytest.raises(ValueError, match=message):
        Timeline.from_dict(value)


def _refused_alike(value: dict[str, Any]) -> SchemaValidationError:
    """Return the validator's refusal of *value*, the library's being the same.

    The library reads a manifest through the package's validator, so its
    refusal is the validator's own: the same code, path and message.
    """
    with pytest.raises(SchemaValidationError) as refused:
        validate(value)
    with pytest.raises(SchemaValidationError) as library:
        Timeline.from_dict(value)
    assert (library.value.code, library.value.path, library.value.message) == (
        refused.value.code,
        refused.value.path,
        refused.value.message,
    )
    return refused.value


def _with_length(timeline: Timeline, length: Any) -> dict[str, Any]:
    """Return the timeline's manifest with its length replaced by *length*."""
    value = timeline.to_dict()
    value["timelines"][0]["length"] = length
    return value


def _length_read(value: dict[str, Any]) -> Any:
    """Return the length the library writes after reading *value*."""
    return Timeline.from_dict(value).to_dict()["timelines"][0]["length"]


def test_both_readers_agree_on_integral_json_numbers() -> None:
    """JSON writes 480.0 as 480; both readers take it as the integer it is."""
    ticks = DiscreteLogicalTimeline(length=960, unit="ticks", uid="ticks")
    value = _with_length(ticks, 480.0)
    validate(value)
    restored = _length_read(value)
    assert restored == 480
    assert type(restored) is int

    seconds = ContinuousPhysicalTimeline(length=4.0, unit="seconds", uid="seconds")
    value = _with_length(seconds, 2)
    validate(value)
    restored = _length_read(value)
    assert restored == 2.0
    assert type(restored) is float


@pytest.mark.parametrize(
    ("length", "message"),
    [
        (480.5, "length 480.5 of 'ticks' is a float on an int axis: write an int"),
        (
            [480, 1],
            "length [480, 1] of 'ticks' is a fraction on an int axis: write 480",
        ),
    ],
)
def test_both_readers_refuse_a_representation_alike(length: Any, message: str) -> None:
    """A value represented otherwise than its axis selects is refused by both alike."""
    value = _with_length(
        DiscreteLogicalTimeline(length=960, unit="ticks", uid="ticks"), length
    )
    refused = _refused_alike(value)
    assert refused.path == ("timelines", 0, "length")
    assert (refused.code, refused.message) == ("representation", message)


def test_both_readers_refuse_an_integer_no_double_holds() -> None:
    """2**53 + 1 on a float axis is no float: both readers refuse it alike."""
    value = _with_length(
        ContinuousPhysicalTimeline(length=4.0, unit="seconds", uid="take"),
        9007199254740993,
    )
    message = (
        "length 9007199254740993 of 'take' is an int on a float axis: write a float"
    )
    refused = _refused_alike(value)
    assert (refused.code, refused.message) == ("representation", message)


def _beyond(
    unit: str, length: Any, children: list[dict[str, Any]], **keys: Any
) -> dict[str, Any]:
    """A manifest file of one root holding *children*, written with bare IDs."""
    return filed(
        {
            "timetoshare": "0.2.0",
            "timelines": [
                {
                    "id": "r",
                    "unit": unit,
                    "length": length,
                    "children": children,
                    **keys,
                }
            ],
        }
    )


@pytest.mark.parametrize(
    ("value", "path", "message"),
    [
        (
            _beyond(
                "ticks", 10, [{"id": "c", "unit": "ticks", "offset": 8, "length": 4}]
            ),
            ("timelines", 0, "children", 0, "length"),
            "child 'c' ends at 12, beyond the length 10 of its parent 'r'",
        ),
        (
            _beyond(
                "seconds",
                10,
                [{"id": "c", "unit": "seconds", "offset": 8, "length": 4}],
            ),
            ("timelines", 0, "children", 0, "length"),
            "child 'c' ends at 12.0, beyond the length 10 of its parent 'r'",
        ),
        (
            _beyond(
                "quarters",
                [10, 3],
                [{"id": "c", "unit": "quarters", "offset": [1, 3], "length": [10, 3]}],
            ),
            ("timelines", 0, "children", 0, "length"),
            "child 'c' ends at [11, 3], beyond the length [10, 3] of its parent 'r'",
        ),
        (
            _beyond(
                "seconds",
                60.0,
                [{"id": "c", "unit": "seconds", "offset": 75.0, "length": None}],
            ),
            ("timelines", 0, "children", 0, "offset"),
            "offset 75.0 of 'c' is beyond the length 60.0 of its parent 'r'",
        ),
        (
            _beyond(
                "seconds",
                60.0,
                [
                    {"id": "a", "unit": "seconds", "offset": 0.0, "length": None},
                    {"id": "b", "unit": "seconds", "offset": 61.0, "length": 1.0},
                ],
                is_segment_line=True,
            ),
            ("timelines", 0, "children", 0, "length"),
            "child 'a' ends at 61.0, beyond the length 60.0 of its parent 'r'",
        ),
    ],
    ids=["int", "int_on_float", "fraction", "offset", "unknown_segment"],
)
def test_both_readers_refuse_a_child_beyond_its_parent_alike(
    value: dict[str, Any], path: tuple[Any, ...], message: str
) -> None:
    """A child beyond its parent's known length is refused in the same words."""
    refused = _refused_alike(value)
    assert (refused.path, refused.message) == (path, message)


def test_both_readers_refuse_a_string_length() -> None:
    """A length that is no number is refused by both readers."""
    value = _with_length(
        DiscreteLogicalTimeline(length=960, unit="ticks", uid="ticks"), "480"
    )
    refused = _refused_alike(value)
    assert (refused.code, refused.path, refused.message) == (
        "schema",
        ("timelines", 0, "length"),
        "'480' is not of type 'number'",
    )


def test_int64_boundary() -> None:
    """Integer coordinates preserve exact values through the JSON boundary."""
    assert number_to_json(2**63 - 1) == 2**63 - 1
    with pytest.raises(ValueError):
        number_to_json(2**63)
    assert number_from_json(9007199254740993) == 9007199254740993
    timeline = DiscreteLogicalTimeline(length=9007199254740993, unit="ticks")
    value = timeline.to_dict()
    assert value["timelines"][0]["length"] == 9007199254740993
    validate(value)


def test_the_unit_table_is_the_package_s() -> None:
    """Every unit, its domain, modality and number types are the package's."""
    assert {
        unit.value: (
            unit.domain.value,
            "discrete" if unit.is_discrete else "continuous",
            unit.default_number_type.name,
            {number_type.name for number_type in unit.allowed_number_types},
        )
        for unit in TimeUnit
    } == {
        unit: (
            entry.domain,
            entry.modality,
            entry.number_type,
            set(entry.number_types),
        )
        for unit, entry in UNITS.items()
    }


# region The package's tables


_ACCEPTANCES = package_fixture_path("acceptances")


@pytest.mark.parametrize(
    "case", sorted(path.name for path in _ACCEPTANCES.iterdir() if path.is_dir())
)
def test_every_acceptance_package_reads(case: str) -> None:
    root = _ACCEPTANCES / case
    package = Package.from_file(root / "package.yaml")
    for url in package.manifests:
        bundle = package.bundle(url)
        for timeline in bundle._known_timelines().values():
            timeline.events  # every pre-parsed content read
            assert timeline.n_events == len(
                timeline.registry.filter(within=timeline.id).table
            )
    expected = loads((root / "expected.yaml").read_text(encoding="utf-8"))
    for resolution in expected.get("resolutions", []):
        found = resolve(root / resolution["path"], resolution["id"])
        assert len(found) == len(resolution["found"])
        assert {getattr(item, "id", None) for item in found} <= {resolution["id"]}


def test_the_tables_written_pass_the_package_s_checks(tmp_path: Path) -> None:
    score = ContinuousLogicalTimeline(length=Fraction(8), uid="score")
    part = ContinuousLogicalTimeline(length=Fraction(2), uid="part")
    score.add_child(part, Fraction(4))
    part.add_events([{"id": "n1", "event_type": "Note", "start": Fraction(1, 2)}])
    page = DiscreteGraphicalTimeline(
        length=867,
        uid="page",
        meta={"box": {"x": 7, "y": 0, "width": 867, "height": 270}},
    )
    page.add_events(
        Box.from_shape(
            BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels"), origin=7, id="b1"
        )
    )
    bundle = AlignmentBundle(id="checked")
    bundle.add_timeline(score)
    bundle.add_timeline(page)
    bundle.create_match_claims([("n1", "part", {"start": 100}, "page")])

    path = bundle.to_file(tmp_path / "checked.yaml", events=True)

    manifest = loads(path.read_text(encoding="utf-8"))
    registry = pq.read_table(tmp_path / "registry.parquet")
    assert registry_problems(registry, manifest) == []
    assert extraction_problems(pq.read_table(tmp_path / "extractions.parquet")) == []
    assert (
        projection_problems(pq.read_table(tmp_path / "projections.parquet"), manifest)
        == []
    )
    copy = tmp_path / "as_package"
    copy.mkdir()
    for name in (
        "checked.yaml",
        "registry.parquet",
        "extractions.parquet",
        "match_claims.parquet",
    ):
        (copy / name).write_bytes((tmp_path / name).read_bytes())
    (copy / "package.yaml").write_text(
        dumps(
            {
                "timetoshare": "0.2.0",
                "manifests": [{"url": "checked.yaml"}],
                "tables": [
                    {"type": "Registry", "url": "registry.parquet"},
                    {"type": "Extraction", "url": "extractions.parquet"},
                    {"type": "MatchClaim", "url": "match_claims.parquet"},
                ],
            }
        ),
        encoding="utf-8",
    )

    def non_events(table: Any) -> list[tuple[Any, ...]]:
        return sorted(
            (row["id"], row["kind"], row["location"], row["timeline_id"])
            for row in table.to_pylist()
            if row["start"] is None
        )

    assert non_events(registry) == non_events(non_event_rows(copy))


# endregion
