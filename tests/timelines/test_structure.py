"""Validation of measure-anchored metrical readings."""

from __future__ import annotations

import json
import warnings
from fractions import Fraction as F
from numbers import Real
from pathlib import Path

import pydantic
import pytest
from pydantic import ValidationError

from timetoalign.core import (
    BeatPolicy,
    CadenzaMeasure,
    Coordinate,
)
from timetoalign.core import Duration as D
from timetoalign.core import (
    Interval,
    IrregularMeasure,
    Measure,
    MeasureConstituent,
    RegularMeasure,
    SplitIrregularMeasure,
    SplitRegularMeasure,
    Tempo,
    TimeUnit,
)
from timetoalign.core.time import number_to_json
from timetoalign.loader.score import Ms3Loader
from timetoalign.testdata import ensure_data
from timetoalign.timelines import (
    Expansion,
    MeasureMap,
    MetricHierarchy,
    MetricNode,
    Reading,
    SectionHierarchy,
    TempoEntry,
    TempoMap,
    Timeline,
)

SECTION_COUNTS = (78, 65, 60)
TOTAL_MEASURES = 203

SCORE_DATA = ensure_data("score")


def _score_measure_paths() -> list[Path]:
    """Return folded score measure tables used by the loader suite."""
    names = {
        "c05n05_musete.measures.tsv",
        "c11n08_Rondeau.measures.tsv",
        "Bruckner_WAB.105_3a_Scherzo.measures.tsv",
        "Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.measures.tsv",
        "WoO71.measures.tsv",
    }
    return sorted(
        path
        for path in SCORE_DATA.rglob("*.measures.tsv")
        if path.name in names and "unfolded" not in path.name
    )


def _mm_rows() -> list[list[dict[str, object]]]:
    """Return folded Measure Map specimen rows."""
    return [
        json.loads(path.read_text())
        for path in sorted(SCORE_DATA.rglob("*.mm.json"))
        if "unfolded" not in path.name
    ]


def _oracle_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return the reference compressed rows for one expanded library map."""
    from pymeasuremap.base import MeasureMap as OracleMeasureMap

    expanded = MeasureMap.expand_rows(rows)
    oracle_expanded = [
        {
            key: float(value) if isinstance(value, F) else value
            for key, value in row.items()
            if key not in {"ID", "volta", "tempo"}
        }
        for row in expanded
    ]
    return [
        {**row, "next": [] if row.get("next") == [-1] else row.get("next")}
        for row in OracleMeasureMap.from_dicts(oracle_expanded)
        .compress(ignore_ids=True)
        .to_dicts()
    ]


def _float_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Convert exact JSON row numbers to the reference numeric representation."""
    return [
        {
            key: (
                float(F(*value))
                if key in {"qstamp", "nominal_length", "actual_length"}
                and isinstance(value, list)
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]


def _abstract(count: int) -> list[Measure]:
    """Return a run of abstract measures."""
    return [Measure() for _ in range(count)]


def _tempi() -> tuple[Tempo, ...]:
    """Return the five exact sonata tempo facts."""
    return (
        Tempo(bpm=F(138), beat=D(F(1, 2), "whole_note"), text="Allegro"),
        Tempo(bpm=F(80), beat=D(F(3, 4), "whole_note"), text="Assai vivace"),
        Tempo(bpm=F(92), beat=D(F(1, 8), "whole_note"), text="Adagio sostenuto"),
        Tempo(bpm=F(76), beat=D(F(1, 16), "whole_note"), text="Largo"),
        Tempo(bpm=F(144), beat=D(F(1, 4), "whole_note"), text="Allegro risoluto"),
    )


def _rows(*, anacrusis: bool = False) -> list[dict[str, object]]:
    """Return compressed rows for the four movements."""
    allegro, scherzo, adagio, largo, risoluto = _tempi()
    return [
        {
            "number": 0 if anacrusis else 1,
            "time_signature": "2/2",
            **({"actual_length": F(1, 2)} if anacrusis else {}),
            "tempo": allegro,
        },
        {"number": 405},
        {
            "number": 0 if anacrusis else 1,
            "time_signature": "3/4",
            **({"actual_length": 1} if anacrusis else {}),
            "tempo": scherzo,
        },
        {"number": 175},
        {"number": 1, "time_signature": "6/8", "tempo": adagio},
        {"number": 187},
        {"number": 1, "time_signature": "4/4", "tempo": largo},
        {"number": 11, "tempo": risoluto},
        {"number": 400},
    ]


def test_rows_round_trip_fixpoint() -> None:
    """Compressed rows are a fixed point, proving no retained fact is lost."""
    maps = [
        MeasureMap.from_measure_data(Ms3Loader.from_file(path).store.measures)
        for path in _score_measure_paths()
    ]
    for measure_map in maps:
        rows = measure_map.to_rows()
        restored = MeasureMap.from_rows(rows)
        assert restored.to_rows() == rows


def test_compression_matches_the_reference_implementation() -> None:
    """The oracle proves which rows carry information beyond default succession."""
    pinned_rachmaninoff_counts = "[1, 8, 337, 374]"
    expected_counts = {
        "c05n05_musete": [1, 5, 6, 16, 17, 31, 32, 58],
        "c11n08_Rondeau": [1, 9, 10, 18, 19, 27, 28, 60],
        "Bruckner_WAB.105_3a_Scherzo.measures": [1, 382],
        "Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.measures": [
            1,
            8,
            337,
            374,
        ],
    }
    assert pinned_rachmaninoff_counts == str(
        expected_counts[
            "Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.measures"
        ]
    )
    specimen_rows = list(
        MeasureMap.from_measure_data(Ms3Loader.from_file(path).store.measures).to_rows(
            compress=False
        )
        for path in _score_measure_paths()
    )
    for rows in specimen_rows:
        library_rows = MeasureMap.from_rows(rows).to_rows()
        oracle_rows = _oracle_rows(rows)
        assert [row["count"] for row in library_rows] == [
            row["count"] for row in oracle_rows
        ]
        for library, oracle in zip(_float_rows(library_rows), oracle_rows, strict=True):
            assert {
                key: ([] if key == "next" and key not in library else library.get(key))
                for key in oracle
            } == oracle
    for path in _score_measure_paths():
        name = path.name.removesuffix(".tsv")
        if name in expected_counts:
            rows = MeasureMap.from_measure_data(
                Ms3Loader.from_file(path).store.measures
            ).to_rows()
            assert [row["count"] for row in rows] == expected_counts[name]
    woo_rows = next(
        MeasureMap.from_measure_data(Ms3Loader.from_file(path).store.measures).to_rows()
        for path in _score_measure_paths()
        if path.name == "WoO71.measures.tsv"
    )
    assert len(woo_rows) == 52


def test_standard_example_expands_and_compresses() -> None:
    """The standard example proves long omitted runs are reconstructed exactly."""
    rows = [{"time_signature": "2/2"}, {"number": 563}]
    measure_map = MeasureMap.from_rows(rows)
    assert len(measure_map) == 563
    assert measure_map.to_rows() == [
        {
            "count": 1,
            "qstamp": [0, 1],
            "number": 1,
            "name": "1",
            "time_signature": "2/2",
            "nominal_length": [4, 1],
            "actual_length": [4, 1],
            "start_repeat": False,
            "end_repeat": False,
            "next": [2],
        },
        {
            "count": 563,
            "qstamp": [2248, 1],
            "number": 563,
            "name": "563",
            "time_signature": "2/2",
            "nominal_length": [4, 1],
            "actual_length": [4, 1],
            "start_repeat": False,
            "end_repeat": False,
        },
    ]


def test_document_form_is_the_compressed_row_list() -> None:
    """The document list excludes derived ids and rejects the former wrapper."""
    measure_map = MeasureMap.from_rows([{"time_signature": "2/2"}, {"number": 563}])
    payload = measure_map.to_dict()
    assert isinstance(payload, list)
    assert {
        "count",
        "qstamp",
        "number",
        "name",
        "actual_length",
        "start_repeat",
        "end_repeat",
    } <= set(payload[0])
    assert "ID" not in payload[0]
    assert all(value is not None for row in payload for value in row.values())
    with pytest.raises(ValueError, match="list of Measure Map rows"):
        MeasureMap.from_dict({"measures": payload})  # type: ignore[arg-type]
    timeline = Timeline(length=F(2252), unit=TimeUnit.quarters, uid="row_document")
    timeline.add_measure_map(measure_map)
    assert Timeline.from_dict(timeline.to_dict()).to_dict() == timeline.to_dict()


def _span(start: int | F, end: int | F) -> Interval:
    """Construct an exact quarter-note test span."""
    return Interval(Coordinate(F(start), "quarters"), Coordinate(F(end), "quarters"))


def test_sonata_rows_anchor_exact_pulses_levels_and_policies() -> None:
    hierarchy = MetricHierarchy.from_rows(_rows())
    measure_map = hierarchy.measure_map
    selected = [measure_map[index - 1] for index in (1, 406, 581, 768, 778)]

    assert len(measure_map) == 1167
    assert all(type(measure) is RegularMeasure for measure in measure_map)
    assert [measure.id for measure in (measure_map[0], measure_map[-1])] == [
        "m1",
        "m1167",
    ]
    assert [measure.number for measure in selected] == [1, 1, 1, 1, 11]
    assert [measure.qstamp for measure in selected] == [0, 1620, 2145, 2706, 2746]
    assert measure_map.total_actual_length == 4306
    assert [entry.at.value for entry in hierarchy.tempo_indications] == [
        0,
        1620,
        2145,
        2706,
        2746,
    ]
    assert [entry.tempo for entry in hierarchy.tempo_indications] == list(_tempi())
    assert hierarchy.tempo_maps[0].kind == "tempo_indication"
    assert [
        hierarchy.pulse_at(at) for at in (0, 1620, 2145, 2706, 2746, 2707, 4305)
    ] == [
        D(F(2), "quarters"),
        D(F(3), "quarters"),
        D(F(1, 2), "quarters"),
        D(F(1, 4), "quarters"),
        D(F(1), "quarters"),
        D(F(1, 4), "quarters"),
        D(F(1), "quarters"),
    ]
    assert [
        hierarchy.node_at(at, level=level).level
        for at, level in ((0, 1), (1620, 0), (2145, 2), (2706, 2), (2746, 1))
    ] == [1, 0, 2, 2, 1]
    assert len(hierarchy.nodes(level=0)) == 3827
    assert len(hierarchy.forest) == 1167

    adagio_bar = hierarchy.node_at(2145, level=2)
    assert len(adagio_bar.expansions) == 1
    beats = adagio_bar.expansions[0].children
    assert [beat.span for beat in beats] == [
        _span(2145, F(4293, 2)),
        _span(F(4293, 2), 2148),
    ]
    assert all(len(beat.expansions[0].children) == 3 for beat in beats)
    assert all(
        child.span.duration == D(F(1, 2), "quarters")
        for beat in beats
        for child in beat.expansions[0].children
    )

    assert hierarchy.policy_at(0) == BeatPolicy.from_time_signature("2/2")
    assert hierarchy.policy_at(1620) == BeatPolicy(
        grouping=(3,), division=F(1), name="3/4"
    )
    assert hierarchy.policy_at(1620, level=-1) == BeatPolicy.from_time_signature("3/4")
    assert (
        hierarchy.policy_at(2145)
        == BeatPolicy.from_time_signature("6/8").as_divisions()
    )
    assert hierarchy.policy_at(2145, level=1) == BeatPolicy.from_time_signature("6/8")
    assert hierarchy.policy_at(2706) == BeatPolicy(
        grouping=(1,) * 16, division=F(1, 4), name="4/4"
    )
    assert hierarchy.policy_at(2706, level=1) == BeatPolicy(
        grouping=(4,) * 4, division=F(1, 4), name="4/4"
    )
    assert hierarchy.policy_at(2746) == BeatPolicy.from_time_signature("4/4")

    with pytest.raises(ValueError, match="MetricHierarchy.from_rows"):
        MeasureMap.from_rows(_rows())
    without_tempo = [
        {key: value for key, value in row.items() if key != "tempo"} for row in _rows()
    ]
    assert MeasureMap.from_rows(without_tempo) == measure_map


def test_anacrusis_clips_the_nominal_grid() -> None:
    hierarchy = MetricHierarchy.from_rows(_rows(anacrusis=True))
    assert len(hierarchy.measure_map) == 1169
    assert hierarchy.measure_map.total_actual_length == F(8615, 2)
    assert [entry.at.value for entry in hierarchy.tempo_indications] == [
        0,
        F(3241, 2),
        F(4293, 2),
        F(5415, 2),
        F(5495, 2),
    ]
    first, second = hierarchy.measure_map[0], hierarchy.measure_map[406]
    assert isinstance(first, MeasureConstituent) and first.offset_within_measure == F(
        7, 2
    )
    assert isinstance(second, MeasureConstituent) and second.offset_within_measure == 2
    assert hierarchy.pulse_at(0) == D(F(1, 2), "quarters")
    assert hierarchy.pulse_at(F(1, 2)) == D(F(2), "quarters")
    assert hierarchy.pulse_at(F(3241, 2)) == D(F(1), "quarters")
    assert hierarchy.pulse_at(F(3243, 2)) == D(F(3), "quarters")


def test_forest_equality_includes_only_structural_beat_effects() -> None:
    base = MetricHierarchy.from_rows(_rows())
    assert base == MetricHierarchy.from_rows(_rows())
    no_text = [
        {
            **row,
            **(
                {"tempo": row["tempo"].model_copy(update={"text": None})}
                if "tempo" in row
                else {}
            ),
        }
        for row in _rows()
    ]
    new_rates = [
        {
            **row,
            **(
                {"tempo": row["tempo"].model_copy(update={"bpm": F(1)})}
                if "tempo" in row
                else {}
            ),
        }
        for row in _rows()
    ]
    assert base == MetricHierarchy.from_rows(no_text)
    assert base == MetricHierarchy.from_rows(new_rates)
    changed = _rows()
    changed[2] = {
        **changed[2],
        "tempo": changed[2]["tempo"].model_copy(
            update={"beat": D(F(1, 4), "whole_note")}
        ),
    }
    assert base != MetricHierarchy.from_rows(changed)
    assert base != MetricHierarchy.from_rows(_rows(anacrusis=True))


def test_generated_readings_tile_complete_groups() -> None:
    tempo = Tempo(bpm=F(60), beat=D(F(1, 2), "whole_note"))
    hierarchy = MetricHierarchy.from_rows(
        [{"number": 1, "time_signature": "3/4", "tempo": tempo}, {"number": 4}]
    )
    assert len(hierarchy.readings) == 2
    generated = hierarchy.generated_readings[0]
    assert generated.reinterprets == "default"
    assert [node.span for node in hierarchy.nodes(reading=generated, level=1)] == [
        _span(0, 6),
        _span(6, 12),
    ]
    assert len(hierarchy.nodes(reading=generated, level=0)) == 6
    assert hierarchy.pulse_at(0) == D(F(1), "quarters")
    assert hierarchy.pulse_at(0, reading=generated) == D(F(2), "quarters")
    with pytest.raises(ValueError, match="remainder 3"):
        MetricHierarchy.from_rows(
            [{"number": 1, "time_signature": "3/4", "tempo": tempo}, {"number": 3}]
        )

    dotted = Tempo(bpm=F(60), beat=D(F(3, 8), "whole_note"))
    packed = MetricHierarchy.from_rows(
        [{"number": 1, "time_signature": "3/4", "tempo": dotted}, {"number": 4}]
    )
    assert len(packed.forest[0].expansions) == 2
    assert {expansion.readings for expansion in packed.forest[0].expansions} == {
        frozenset({"default"}),
        frozenset({packed.generated_readings[0].id}),
    }


def test_authored_hypermeasures_share_default_bars() -> None:
    hierarchy = MetricHierarchy.from_rows(
        [{"number": 1, "time_signature": "2/4"}, {"number": 4}]
    )
    roots = tuple(
        MetricNode(
            level=2,
            span=_span(start, start + 4),
            expansions=(
                Expansion(
                    children=tuple(
                        MetricNode(level=1, span=_span(bar, bar + 2))
                        for bar in (start, start + 2)
                    ),
                    readings=frozenset(),
                ),
            ),
        )
        for start in (0, 4)
    )
    merged = hierarchy.with_reading(Reading(id="mine", source="analysis"), roots)
    assert len(merged.forest) == 2
    assert len(merged.nodes(reading="mine", level=2)) == 2
    bars = tuple(
        child for root in merged.forest for child in root.expansions[0].children
    )
    assert all(bar.expansions[0].readings == frozenset({"default"}) for bar in bars)
    with pytest.raises(ValidationError, match="not covered"):
        MetricNode(
            level=1,
            span=_span(0, 2),
            expansions=(
                Expansion(
                    children=(MetricNode(level=0, span=_span(0, 1)),),
                    readings=frozenset(),
                ),
            ),
        )
    bad = MetricNode(
        level=1,
        span=_span(0, 2),
        expansions=(
            Expansion(
                children=(MetricNode(level=0, span=_span(0, 2)),),
                readings=frozenset({"default"}),
            ),
        ),
    )
    with pytest.raises(ValueError, match="invalid reading ids"):
        hierarchy.with_reading(Reading(id="bad", source="analysis"), (bad,))


def test_json_requirements_and_fixpoint() -> None:
    hierarchy = MetricHierarchy.from_rows(_rows())
    payload = json.loads(json.dumps(hierarchy.to_dict()))
    restored = MetricHierarchy.from_dict(payload, measure_map=hierarchy.measure_map)
    assert restored == hierarchy
    assert restored.to_dict() == payload
    with pytest.raises(ValidationError):
        TempoEntry(tempo=Tempo(bpm=F(60)))
    with pytest.raises(ValidationError):
        MetricNode(span=_span(0, 1))
    with pytest.raises(ValidationError):
        TempoMap(kind="indica" + "tion", entries=(), provenance={})


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"mystery": 1}], "mystery"),
        ([{"count": 2, "time_signature": "4/4"}], "count"),
        ([{"time_signature": "4/4", "qstamp": 1}], "qstamp"),
        ([{"time_signature": "nonsense"}], "time signature"),
        ([{"time_signature": "cadenza"}], "actual_length"),
        ([{"time_signature": "4/4", "tempo": 60}], "Tempo instance"),
    ],
)
def test_authoring_lane_rejects_contradictions(
    rows: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MeasureMap.expand_rows(rows)


def test_abstract_and_float_rows_keep_their_stated_semantics() -> None:
    abstract = MeasureMap.from_rows([{"number": 1}, {"number": 2}])
    assert all(
        measure.nominal_length is None and measure.actual_length is None
        for measure in abstract
    )
    expanded = MeasureMap.expand_rows(
        [{"number": 0, "nominal_length": 1, "actual_length": 0.5}]
    )
    assert expanded[0]["actual_length"] == F(1, 2)


class TestSectionHierarchyConstructionEquivalence:
    """The same partition spelled every accepted way compares equal."""

    def test_nested_measures_equal_measure_counts(self) -> None:
        nested = SectionHierarchy.from_measures([_abstract(n) for n in SECTION_COUNTS])
        counts = SectionHierarchy.from_measure_counts(list(SECTION_COUNTS))
        assert nested == counts
        assert counts.n_sections == 3
        assert [section.n_measures for section in counts.sections] == list(
            SECTION_COUNTS
        )
        assert counts.n_measures == TOTAL_MEASURES

    def test_int_form_equals_every_single_section_spelling(self) -> None:
        as_int = SectionHierarchy.from_measure_counts(TOTAL_MEASURES)
        alternatives = (
            SectionHierarchy.from_measure_counts([TOTAL_MEASURES]),
            SectionHierarchy.from_measure_counts({"whole": TOTAL_MEASURES}),
            SectionHierarchy.from_measure_counts([("whole", TOTAL_MEASURES)]),
            SectionHierarchy.from_measure_counts(
                zip(["whole"], [TOTAL_MEASURES], strict=True)
            ),
            SectionHierarchy.from_measures([_abstract(TOTAL_MEASURES)]),
        )
        assert as_int.n_sections == 1
        assert all(as_int == other for other in alternatives)

    def test_three_section_spellings_all_equal(self) -> None:
        as_list = SectionHierarchy.from_measure_counts(list(SECTION_COUNTS))
        alternatives = (
            SectionHierarchy.from_measure_counts({"I": 78, "II": 65, "III": 60}),
            SectionHierarchy.from_measure_counts([("I", 78), ("II", 65), ("III", 60)]),
            SectionHierarchy.from_measure_counts(
                zip(["I", "II", "III"], SECTION_COUNTS, strict=True)
            ),
        )
        assert all(as_list == other for other in alternatives)

    def test_consumed_once_zip_is_materialized(self) -> None:
        consumed = zip(["I", "II", "III"], SECTION_COUNTS, strict=True)
        from_zip = SectionHierarchy.from_measure_counts(consumed)
        assert list(consumed) == []
        assert from_zip == SectionHierarchy.from_measure_counts(
            [("I", 78), ("II", 65), ("III", 60)]
        )

    def test_measure_map_accepted_where_measures_expected(self) -> None:
        measure_map = MeasureMap(_abstract(TOTAL_MEASURES))
        from_map = SectionHierarchy.from_measures(measure_map)
        assert from_map.n_sections == 1
        assert from_map.measure_map is measure_map
        assert from_map == SectionHierarchy.from_measure_counts(TOTAL_MEASURES)

    def test_existing_map_is_partitioned_without_rebuilding(self) -> None:
        measure_map = MeasureMap(Measure(actual_length=F(1, 3)) for _ in range(6))
        hierarchy = SectionHierarchy.from_measure_map(measure_map, counts=[1, 2, 3])
        assert hierarchy.measure_map is measure_map
        assert [section.n_measures for section in hierarchy.sections] == [1, 2, 3]
        assert [
            tuple(measure.id for measure in section.measure_map)
            for section in hierarchy.sections
        ] == [("m1",), ("m2", "m3"), ("m4", "m5", "m6")]

    def test_existing_map_partition_must_cover_the_map_exactly(self) -> None:
        measure_map = MeasureMap(_abstract(6))
        with pytest.raises(
            ValueError,
            match=r"Section counts must sum to MeasureMap length 6, got \(2, 3\)",
        ):
            SectionHierarchy.from_measure_map(measure_map, counts=[2, 3])

    def test_display_names_excluded_from_equality(self) -> None:
        named = SectionHierarchy.from_measure_counts({"I": 78, "II": 65, "III": 60})
        differently_named = SectionHierarchy.from_measure_counts(
            [("intro", 78), ("middle", 65), ("finale", 60)]
        )
        anonymous = SectionHierarchy.from_measure_counts(list(SECTION_COUNTS))
        assert named == differently_named == anonymous
        assert [section.name for section in named.sections] == ["I", "II", "III"]
        assert [section.name for section in anonymous.sections] == [None, None, None]

    def test_different_section_shape_is_unequal(self) -> None:
        assert SectionHierarchy.from_measure_counts(
            list(SECTION_COUNTS)
        ) != SectionHierarchy.from_measure_counts([78, 60, 65])
        assert SectionHierarchy.from_measure_counts(
            [3, 2, 2]
        ) != SectionHierarchy.from_measure_counts([3, 3, 1])

    def test_concrete_leaf_quarter_spans(self) -> None:
        def bars(count: int) -> list[Measure]:
            return [
                Measure(actual_length=F(3), time_signature="3/4") for _ in range(count)
            ]

        hierarchy = SectionHierarchy.from_measures([bars(3), bars(2), bars(2)])
        assert [
            section.measure_map.total_actual_length for section in hierarchy.sections
        ] == [F(9), F(6), F(6)]
        assert hierarchy.measure_map.total_actual_length == F(21)


class TestMeasureMapImmutability:
    """A measure map exposes no mutators and a read-only tuple."""

    @pytest.mark.parametrize("mutator", ["append", "insert", "replace", "set"])
    def test_no_mutator_attributes(self, mutator: str) -> None:
        assert not hasattr(MeasureMap(_abstract(3)), mutator)

    def test_measures_tuple_is_not_writable(self) -> None:
        measure_map = MeasureMap(_abstract(3))
        assert isinstance(measure_map.measures, tuple)
        with pytest.raises(TypeError):
            measure_map.measures[0] = Measure()  # type: ignore[index]


class TestMeasureMapEqualityAndJson:
    """Map equality preserves concrete classes and exact dictionary values."""

    def test_maps_built_twice_compare_equal(self) -> None:
        assert MeasureMap([RegularMeasure(actual_length=F(1, 3))]) == MeasureMap(
            [RegularMeasure(actual_length=F(1, 3))]
        )

    def test_equal_fields_on_different_measure_classes_compare_unequal(self) -> None:
        regular = MeasureMap([RegularMeasure(actual_length=F(1, 3))])
        irregular = MeasureMap([IrregularMeasure(actual_length=F(1, 3))])
        assert regular != irregular

    def test_foreign_equality_operand_is_not_equal(self) -> None:
        assert MeasureMap([]) != ()

    def test_json_round_trip_is_a_row_fixpoint_and_emits_no_warning(self) -> None:
        measure_map = MeasureMap(
            [
                RegularMeasure(actual_length=F(1, 3)),
                IrregularMeasure(actual_length=F(2, 3)),
            ]
        )
        payload = json.loads(json.dumps(measure_map.to_dict()))
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            restored = MeasureMap.from_dict(payload)
        assert emitted == []
        assert restored.to_rows() == measure_map.to_rows()
        assert restored.to_dict() == payload


class TestTempoRealizationValues:
    """Tempo-map values preserve exact, distinct, immutable source evidence."""

    def test_observation_states_are_distinct_and_round_trip(self) -> None:
        at = Coordinate(F(0), "quarters")
        point = TempoEntry(at=at, observed=Coordinate(F(1, 3), "seconds"))
        interval = TempoEntry(
            at=at,
            observed=Interval(
                Coordinate(F(1, 3), "seconds"),
                Coordinate(F(2, 3), "seconds"),
            ),
        )
        undefined = TempoEntry(at=at, observation_undefined=True)
        unstated = TempoEntry(at=at)

        assert point.to_dict()["observed"]["type"] == "coordinate"
        assert interval.to_dict()["observed"]["type"] == "interval"
        assert undefined.to_dict()["observed"] is None
        assert undefined.to_dict()["observation_undefined"] is True
        assert unstated.to_dict()["observed"] is None
        assert unstated.to_dict()["observation_undefined"] is False
        assert len({point, interval, undefined, unstated}) == 4

        for entry in (point, interval, undefined, unstated):
            payload = json.loads(json.dumps(entry.to_dict()))
            restored = TempoEntry.from_dict(payload)
            assert restored == entry
            assert restored.to_dict() == payload

    def test_observation_cannot_be_present_and_explicitly_undefined(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="cannot be true"):
            TempoEntry(
                at=Coordinate(F(0), "quarters"),
                observed=Coordinate(F(1), "seconds"),
                observation_undefined=True,
            )

    def test_symbolic_position_rejects_physical_unit(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="seconds.*allowed units"):
            TempoEntry(at=Coordinate(F(3), "seconds"))

    def test_alternative_weights_are_exact_and_bounded(self) -> None:
        entry = TempoEntry(
            at=Coordinate(F(0), "quarters"),
            alternatives=((Coordinate(F(1), "seconds"), F(1, 3)),),
        )
        payload = json.loads(json.dumps(entry.to_dict()))
        assert entry.alternatives[0][1] == F(1, 3)
        assert type(entry.alternatives[0][1]) is F
        assert payload["alternatives"][0]["weight"] == number_to_json(F(1, 3))
        assert TempoEntry.from_dict(payload) == entry

    @pytest.mark.parametrize("weight", [float("nan"), float("inf"), F(0), F(-1), F(2)])
    def test_invalid_alternative_weight_is_rejected(self, weight: Real) -> None:
        with pytest.raises(
            pydantic.ValidationError,
            match="finite real number greater than zero and at most one",
        ):
            TempoEntry(
                at=Coordinate(F(0), "quarters"),
                alternatives=((Coordinate(F(1), "seconds"), weight),),
            )

    def test_provenance_and_level_names_are_immutable(self) -> None:
        tempo_map = TempoMap(kind="steady", entries=(), provenance={"nested": {"x": 1}})
        reading = Reading(id="analysis", source="analysis", level_names={-1: "groove"})
        with pytest.raises(TypeError):
            tempo_map.provenance["new"] = 2  # type: ignore[index]
        with pytest.raises(TypeError):
            tempo_map.provenance["nested"]["x"] = 2  # type: ignore[index]
        with pytest.raises(TypeError):
            reading.level_names[-1] = "changed"  # type: ignore[index]
        assert tempo_map.to_dict()["provenance"] == {"nested": {"x": 1}}
        assert reading.level_names[-1] == "groove"

    def test_non_json_provenance_is_rejected_at_construction(self) -> None:
        with pytest.raises(
            pydantic.ValidationError, match="key 'ratio'.*type Fraction"
        ):
            TempoMap(
                kind="tempo_indication",
                entries=(),
                provenance={"ratio": F(1, 3)},
            )

    def test_non_default_members_appear_in_representations(self) -> None:
        observed = Coordinate(F(2), "seconds")
        entry = TempoEntry(
            at=Coordinate(F(1), "quarters"),
            tempo=Tempo(bpm=90),
            observed=observed,
            alternatives=((Coordinate(F(3), "seconds"), F(1)),),
            is_interpolated=True,
        )
        tempo_map = TempoMap(
            kind="steady", entries=(entry,), provenance={"by": "probe"}
        )
        reading = Reading(
            id="performed",
            source="performance",
            by="A. Player",
            reinterprets="notation",
            level_names={-1: "groove"},
        )
        node = MetricNode(level=0, span=_span(0, F(3, 8)))

        assert "span=[0, 3/8) quarters" in repr(node)
        assert str(node) == repr(node)
        assert f"observed={observed!r}" in repr(entry)
        assert "alternatives=" in repr(entry)
        assert "is_interpolated=True" in repr(entry)
        assert "provenance={'by': 'probe'}" in repr(tempo_map)
        assert "by='A. Player'" in repr(reading)
        assert "reinterprets='notation'" in repr(reading)
        assert "level_names={-1: 'groove'}" in repr(reading)


class TestBeatPolicyScalarEquality:
    """Beat policies retain name-aware scalar equality."""

    def test_name_participates_in_scalar_equality(self) -> None:
        first = BeatPolicy(grouping=(1, 1, 1), division=F(1), name="a")
        second = BeatPolicy(grouping=(1, 1, 1), division=F(1), name="b")
        assert first != second

    def test_identical_policies_including_name_are_equal(self) -> None:
        first = BeatPolicy(grouping=(1, 1, 1), division=F(1), name="a")
        second = BeatPolicy(grouping=(1, 1, 1), division=F(1), name="a")
        assert first == second


class TestBeatPolicyAndMetricHierarchyJson:
    """Metric dictionary forms preserve authored units and complete forest values."""

    @pytest.mark.parametrize(
        "policy",
        [
            BeatPolicy(
                grouping=(1, 2),
                beat_size=D(F(1, 3), TimeUnit.whole_note),
                name="whole-note spelling",
            ),
            BeatPolicy(
                grouping=(2, 1),
                beat_size=D(F(1, 3), TimeUnit.quarters),
                name="quarters spelling",
            ),
            BeatPolicy(grouping=(3, 2), division=F(1, 3), name="division spelling"),
        ],
        ids=("whole-note", "quarters", "division"),
    )
    def test_policy_json_round_trip_preserves_full_scalar(
        self, policy: BeatPolicy
    ) -> None:
        payload = json.loads(json.dumps(policy.to_dict()))
        restored = BeatPolicy.from_dict(payload)
        assert restored == policy
        assert restored.to_dict() == payload
        assert payload["beat_size"]["number_type"] == "fraction"

    @staticmethod
    def _hierarchy() -> MetricHierarchy:
        measure_map = MeasureMap([Measure(actual_length=F(1))])
        root = MetricNode(
            level=1,
            span=_span(0, 1),
            expansions=(
                Expansion(
                    children=(
                        MetricNode(level=0, span=_span(0, F(1, 3))),
                        MetricNode(level=0, span=_span(F(1, 3), 1)),
                    ),
                    readings=frozenset({"performed"}),
                ),
            ),
        )
        return MetricHierarchy(
            measure_map=measure_map,
            authored_readings=(
                Reading(id="performed", source="performance", reinterprets="default"),
            ),
            authored_forest=(root,),
            tempo_maps=(
                TempoMap(
                    kind="tempo_indication",
                    entries=(
                        TempoEntry(
                            at=Coordinate(F(1, 3), "quarters"),
                            tempo=Tempo(bpm=F(90)),
                        ),
                    ),
                    provenance={"source": "score"},
                ),
            ),
        )

    def test_metric_hierarchy_round_trip_preserves_forest_and_maps(self) -> None:
        hierarchy = self._hierarchy()
        payload = json.loads(json.dumps(hierarchy.to_dict()))
        restored = MetricHierarchy.from_dict(payload, measure_map=hierarchy.measure_map)
        assert restored == hierarchy
        assert restored.to_dict() == payload
        assert restored.tempo_maps == hierarchy.tempo_maps
        assert restored.authored_forest[0].span == _span(0, 1)
        assert restored.authored_forest[0].expansions[0].children[0].span == _span(
            0, F(1, 3)
        )
        assert restored.authored_readings[0].reinterprets == "default"

    def test_timeline_payload_carries_the_same_hierarchy_json(self) -> None:
        hierarchy = self._hierarchy()
        timeline = Timeline(length=F(1), unit=TimeUnit.quarters, uid="metric_json")
        timeline.add_measure_map(hierarchy.measure_map)
        timeline.add_metric_hierarchy(hierarchy)
        payload = json.loads(json.dumps(timeline.to_dict(events=False)))
        restored = Timeline.from_dict(payload)

        assert restored.metric_hierarchy == hierarchy
        assert restored.metric_hierarchy.to_dict() == hierarchy.to_dict()
        assert restored.metric_hierarchy.authored_forest[0].expansions[0].children[
            0
        ].span == _span(0, F(1, 3))
        assert restored.metric_hierarchy.tempo_maps == hierarchy.tempo_maps
        assert restored.to_dict(events=False) == payload
        assert "events" not in payload


class TestMeasureScalar:
    """The measure scalar's field set and map-derived arithmetic."""

    def test_constructs_with_no_arguments(self) -> None:
        measure = Measure()
        assert measure.id is None
        assert measure.count is None
        assert measure.qstamp is None
        assert measure.actual_length is None

    def test_field_set_is_exactly_the_standard_plus_volta(self) -> None:
        assert set(Measure.model_fields) == {
            "id",
            "count",
            "qstamp",
            "number",
            "name",
            "time_signature",
            "nominal_length",
            "actual_length",
            "start_repeat",
            "end_repeat",
            "next",
            "volta",
        }

    def test_is_frozen(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            Measure().count = 5  # type: ignore[misc]

    def test_count_and_qstamp_are_prefix_sums_when_omitted(self) -> None:
        measure_map = MeasureMap(
            [
                Measure(actual_length=F(3)),
                Measure(actual_length=F(3)),
                Measure(actual_length=F(2)),
            ]
        )
        assert [measure.count for measure in measure_map] == [1, 2, 3]
        assert [measure.qstamp for measure in measure_map] == [F(0), F(3), F(6)]
        assert [measure.id for measure in measure_map] == ["m1", "m2", "m3"]

    def test_disagreeing_count_warns_and_is_corrected(self) -> None:
        with pytest.warns(UserWarning, match="supplies count 5"):
            measure_map = MeasureMap(
                [
                    Measure(count=5, actual_length=F(3)),
                    Measure(actual_length=F(3)),
                ]
            )
        assert [measure.count for measure in measure_map] == [1, 2]

    def test_disagreeing_qstamp_warns_and_is_corrected(self) -> None:
        with pytest.warns(UserWarning, match="supplies qstamp 99"):
            measure_map = MeasureMap(
                [
                    Measure(actual_length=F(3)),
                    Measure(qstamp=F(99), actual_length=F(3)),
                ]
            )
        assert [measure.qstamp for measure in measure_map] == [F(0), F(3)]


class TestMeasureJson:
    """Every concrete measure class survives JSON with exact fractions."""

    @pytest.mark.parametrize(
        "measure_class",
        [
            RegularMeasure,
            IrregularMeasure,
            SplitRegularMeasure,
            SplitIrregularMeasure,
            MeasureConstituent,
            CadenzaMeasure,
        ],
    )
    def test_concrete_class_and_all_fields_round_trip(
        self, measure_class: type[Measure]
    ) -> None:
        kwargs = {
            "id": "m7",
            "count": 7,
            "qstamp": F(2, 3),
            "number": 5,
            "name": "5a",
            "time_signature": "7/9",
            "nominal_length": F(7, 3),
            "actual_length": F(5, 3),
            "start_repeat": True,
            "end_repeat": True,
            "next": ("m8", "m11"),
            "volta": 2,
        }
        if measure_class is MeasureConstituent:
            kwargs["offset_within_measure"] = F(1, 3)
        measure = measure_class(**kwargs)
        payload = json.loads(json.dumps(measure.to_dict()))
        restored = Measure.from_dict(payload)
        assert type(restored) is measure_class
        assert restored == measure
        assert restored.qstamp == F(2, 3)
        assert restored.nominal_length == F(7, 3)
        assert restored.actual_length == F(5, 3)
        if isinstance(restored, MeasureConstituent):
            assert restored.offset_within_measure == F(1, 3)
        assert restored.to_dict() == payload

    def test_unknown_class_tag_names_known_classes(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"Unknown or missing Measure class 'UnknownMeasure'.*RegularMeasure",
        ):
            Measure.from_dict({"class": "UnknownMeasure"})

    def test_missing_class_tag_names_known_classes(self) -> None:
        payload = RegularMeasure(id="m1").to_dict()
        del payload["class"]
        with pytest.raises(
            ValueError, match=r"Unknown or missing Measure class None.*RegularMeasure"
        ):
            Measure.from_dict(payload)

    def test_undeclared_payload_key_is_rejected_by_name(self) -> None:
        payload = RegularMeasure(id="m1").to_dict()
        payload["offset_within_measure"] = number_to_json(F(1, 3))
        with pytest.raises(
            ValueError,
            match=(
                "Measure class 'RegularMeasure' does not declare the keys: "
                "offset_within_measure"
            ),
        ):
            Measure.from_dict(payload)

    def test_subclass_rejects_a_different_tag(self) -> None:
        with pytest.raises(
            ValueError, match="does not match receiving subclass 'RegularMeasure'"
        ):
            RegularMeasure.from_dict(IrregularMeasure().to_dict())


def test_abstract_measure_map_is_refused() -> None:
    with pytest.raises(ValueError, match="abstract measure map"):
        MetricHierarchy(measure_map=MeasureMap([Measure(), Measure()]))


def test_timeline_payload_without_measure_map_is_refused() -> None:
    payload = Timeline(length=F(4), unit=TimeUnit.quarters).to_dict()
    payload["measure_map"] = None
    payload["metric_hierarchy"] = {
        "authored_readings": [],
        "authored_forest": [],
        "tempo_maps": [],
    }
    with pytest.raises(ValueError, match="requires measure_map"):
        Timeline.from_dict(payload)


def test_default_reading_is_never_serialized() -> None:
    hierarchy = MetricHierarchy.from_rows(_rows())
    assert set(hierarchy.to_dict()) == {
        "authored_readings",
        "authored_forest",
        "tempo_maps",
    }
    assert "default" not in json.dumps(hierarchy.to_dict())


def test_signature_less_map_reads_bar_wise() -> None:
    hierarchy = MetricHierarchy(
        measure_map=MeasureMap(
            [
                Measure(actual_length=F(1)),
                Measure(actual_length=F(3, 2)),
                Measure(actual_length=F(2)),
            ]
        )
    )
    assert [(node.level, node.expansions) for node in hierarchy.forest] == [
        (0, ()),
        (0, ()),
        (0, ()),
    ]
    assert [hierarchy.pulse_at(at) for at in (0, 1, F(5, 2))] == [
        D(F(1), "quarters"),
        D(F(3, 2), "quarters"),
        D(F(2), "quarters"),
    ]


def test_bar_nodes_span_their_measures() -> None:
    hierarchy = MetricHierarchy.from_rows(_rows())
    assert [
        hierarchy.node_at(at, level=level).span
        for at, level in ((0, 1), (1620, 0), (2145, 2), (2706, 2), (2746, 1))
    ] == [
        _span(0, 4),
        _span(1620, 1623),
        _span(2145, 2148),
        _span(2706, 2710),
        _span(2746, 2750),
    ]


def test_reserved_reading_ids_and_default_tagged_expansions_are_refused() -> None:
    hierarchy = MetricHierarchy.from_rows([{"number": 1, "time_signature": "2/4"}])
    with pytest.raises(ValueError, match="Reserved"):
        MetricHierarchy(
            measure_map=hierarchy.measure_map,
            authored_readings=(Reading(id="default", source="analysis"),),
        )
    with pytest.raises(ValueError, match="reserved"):
        hierarchy.with_reading(Reading(id="generated_q_2_1", source="analysis"), ())
    tagged = MetricNode(
        level=1,
        span=_span(0, 2),
        expansions=(
            Expansion(
                children=(MetricNode(level=0, span=_span(0, 2)),),
                readings=frozenset({"default"}),
            ),
        ),
    )
    with pytest.raises(ValueError, match="reserved or unknown.*default"):
        MetricHierarchy(
            measure_map=hierarchy.measure_map,
            authored_readings=(Reading(id="mine", source="analysis"),),
            authored_forest=(tagged,),
        )


def test_simultaneous_tempo_indications_with_different_beats_are_refused() -> None:
    measure_map = MeasureMap.from_rows([{"number": 1, "time_signature": "4/4"}])
    tempo_map = TempoMap(
        kind="tempo_indication",
        entries=(
            TempoEntry(
                at=Coordinate(F(0), "quarters"),
                tempo=Tempo(bpm=F(60), beat=D(F(1), "quarters")),
            ),
            TempoEntry(
                at=Coordinate(F(0), "quarters"),
                tempo=Tempo(bpm=F(60), beat=D(F(2), "quarters")),
            ),
        ),
        provenance={},
    )
    with pytest.raises(ValueError, match="different beat units"):
        MetricHierarchy(measure_map=measure_map, tempo_maps=(tempo_map,))


def test_irregular_bar_without_window_is_unpartitioned() -> None:
    hierarchy = MetricHierarchy(
        measure_map=MeasureMap(
            [
                IrregularMeasure(
                    time_signature="4/4", nominal_length=F(4), actual_length=F(3)
                )
            ]
        )
    )
    assert len(hierarchy.forest) == 1
    assert hierarchy.forest[0].span == _span(0, 3)
    assert hierarchy.forest[0].expansions == ()


def test_overlapping_children_and_child_level_not_below_parent_are_refused() -> None:
    with pytest.raises(ValidationError, match="not covered"):
        MetricNode(
            level=2,
            span=_span(0, 2),
            expansions=(
                Expansion(
                    children=(
                        MetricNode(level=1, span=_span(0, F(3, 2))),
                        MetricNode(level=1, span=_span(1, 2)),
                    ),
                    readings=frozenset(),
                ),
            ),
        )
    with pytest.raises(ValidationError, match="not lower than parent"):
        MetricNode(
            level=1,
            span=_span(0, 1),
            expansions=(
                Expansion(
                    children=(MetricNode(level=1, span=_span(0, 1)),),
                    readings=frozenset(),
                ),
            ),
        )


def test_from_sections_is_gone() -> None:
    import timetoalign
    import timetoalign.timelines as timelines

    assert not hasattr(MetricHierarchy, "from_sections")
    assert "from_sections" not in timetoalign.__all__
    assert "from_sections" not in timelines.__all__


def test_construction_does_not_build_the_forest(monkeypatch) -> None:
    import timetoalign.timelines.structure as structure

    def refuse(*_args, **_kwargs):
        raise RuntimeError("packing called")

    monkeypatch.setattr(structure, "_pack_trees", refuse)
    hierarchy = MetricHierarchy.from_rows([{"number": 1, "time_signature": "4/4"}])
    with pytest.raises(RuntimeError, match="packing called"):
        _ = hierarchy.forest


def test_derived_forest_cannot_be_poisoned_through_the_instance_namespace() -> None:
    hierarchy = MetricHierarchy.from_rows([{"number": 1, "time_signature": "4/4"}])
    assert not hasattr(hierarchy, "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        hierarchy.forest = ()  # type: ignore[misc]


def test_repeated_queries_reuse_the_flattened_reading(monkeypatch) -> None:
    import timetoalign.timelines.structure as structure

    hierarchy = MetricHierarchy.from_rows([{"number": 1, "time_signature": "4/4"}])
    expected = hierarchy.nodes()

    def refuse(*_args, **_kwargs):
        raise RuntimeError("flattening called")

    monkeypatch.setattr(structure, "_flatten_nodes", refuse)
    assert hierarchy.nodes() == expected
    assert hierarchy.nodes(level=0) == tuple(
        node for node in expected if node.level == 0
    )


def test_rows_without_nominal_length_are_plain_measures() -> None:
    measure_map = MeasureMap.from_rows(
        [
            {"number": 0, "actual_length": F(1)},
            {"number": 1, "actual_length": F(2)},
        ]
    )
    assert [type(measure) for measure in measure_map] == [Measure, Measure]
    assert [measure.actual_length for measure in measure_map] == [F(1), F(2)]


def test_consecutive_zero_numbered_rows_are_refused() -> None:
    with pytest.raises(ValueError, match="split anacrusis.*explicit offset"):
        MeasureMap.expand_rows(
            [
                {"number": 0, "nominal_length": 4, "actual_length": 1},
                {"number": 0, "nominal_length": 4, "actual_length": 3},
            ]
        )


def test_count_keyed_anacrusis_row_without_count_is_position_one() -> None:
    expanded = MeasureMap.expand_rows(
        [{"time_signature": "4/4", "actual_length": 1}, {"count": 2}]
    )
    assert [(row["count"], row["number"]) for row in expanded] == [(1, 0), (2, 2)]


def test_nominal_length_contradicting_signature_is_refused() -> None:
    with pytest.raises(
        ValueError,
        match=r"row number 7.*'3/4'.*length 3.*nominal_length 2",
    ):
        MeasureMap.expand_rows(
            [
                {
                    "number": 7,
                    "time_signature": "3/4",
                    "nominal_length": 2,
                    "actual_length": 2,
                }
            ]
        )


def test_measure_constituent_beyond_nominal_window_names_the_measure() -> None:
    measure_map = MeasureMap(
        [
            MeasureConstituent(
                id="overflow",
                time_signature="4/4",
                nominal_length=F(4),
                actual_length=F(2),
                offset_within_measure=F(3),
            )
        ]
    )
    hierarchy = MetricHierarchy(measure_map=measure_map)
    with pytest.raises(
        ValueError,
        match=r"Measure 'overflow'.*offset 3.*actual length 2.*nominal length 4",
    ):
        _ = hierarchy.forest


def test_cadenza_states_no_pulse() -> None:
    hierarchy = MetricHierarchy(
        measure_map=MeasureMap(
            [CadenzaMeasure(time_signature="cadenza", actual_length=F(3))]
        )
    )
    with pytest.raises(ValueError, match="cadenza"):
        hierarchy.pulse_at(0)
    with pytest.raises(ValueError, match="cadenza"):
        hierarchy.policy_at(0)


def test_representations_name_what_they_hold() -> None:
    measure_map = MeasureMap.from_rows(
        [
            {"number": 1, "time_signature": "3/4"},
            {"number": 2},
            {"number": 3, "time_signature": "2/4"},
        ]
    )
    assert repr(measure_map) == (
        "MeasureMap(measures=3, actual_length=8, signatures=('3/4', '2/4'))"
    )
    assert repr(BeatPolicy.from_time_signature("3/4")) == "BeatPolicy(1+1+1 x 1, '3/4')"
    measure_repr = repr(measure_map[0])
    for fragment in (
        "number=1",
        "time_signature='3/4'",
        "qstamp=Fraction(0, 1)",
        "actual_length=Fraction(3, 1)",
    ):
        assert fragment in measure_repr
    node = MetricNode(level=0, span=_span(0, 1))
    assert str(node) == repr(node)


def test_authored_readings_coalesce_or_retain_their_partition() -> None:
    hierarchy = MetricHierarchy.from_rows([{"number": 1, "time_signature": "2/4"}])
    agreeing = MetricNode(
        level=1,
        span=_span(0, 2),
        expansions=(
            Expansion(
                children=(
                    MetricNode(level=0, span=_span(0, 1)),
                    MetricNode(level=0, span=_span(1, 2)),
                ),
                readings=frozenset(),
            ),
        ),
    )
    merged = hierarchy.with_reading(
        Reading(id="agreeing", source="analysis"), (agreeing,)
    )
    assert len(merged.forest[0].expansions) == 1
    assert merged.forest[0].expansions[0].readings == frozenset({"default", "agreeing"})

    divergent = MetricNode(
        level=1,
        span=_span(0, 2),
        expansions=(
            Expansion(
                children=(
                    MetricNode(level=0, span=_span(0, F(1, 2))),
                    MetricNode(level=0, span=_span(F(1, 2), 2)),
                ),
                readings=frozenset(),
            ),
        ),
    )
    split = hierarchy.with_reading(
        Reading(id="divergent", source="analysis"), (divergent,)
    )
    assert len(split.forest[0].expansions) == 2
    assert split.forest[0].expansions[1].readings == frozenset({"divergent"})


def test_policy_at_uses_only_the_named_authored_reading() -> None:
    hierarchy = MetricHierarchy(measure_map=MeasureMap([Measure(actual_length=F(1))]))

    def partition(points: tuple[F, ...]) -> MetricNode:
        boundaries = (F(0), *points, F(1))
        return MetricNode(
            level=1,
            span=_span(0, 1),
            expansions=(
                Expansion(
                    children=tuple(
                        MetricNode(level=0, span=_span(start, end))
                        for start, end in zip(
                            boundaries[:-1], boundaries[1:], strict=True
                        )
                    ),
                    readings=frozenset(),
                ),
            ),
        )

    hierarchy = hierarchy.with_reading(
        Reading(id="halves", source="analysis"),
        (partition((F(1, 2),)),),
    ).with_reading(
        Reading(id="thirds", source="analysis"),
        (partition((F(1, 3), F(2, 3))),),
    )
    assert hierarchy.policy_at(0, reading="halves") == BeatPolicy(
        grouping=(1, 1), division=F(1, 2)
    )
    assert hierarchy.policy_at(0, reading="thirds") == BeatPolicy(
        grouping=(1, 1, 1), division=F(1, 3)
    )
    with pytest.raises(ValueError, match="seconds.*quarters"):
        hierarchy.policy_at(Coordinate(F(1, 2), "seconds"), reading="halves")
    with pytest.raises(ValueError, match="Unknown.*known ids: default, halves, thirds"):
        hierarchy.policy_at(0, reading="missing")
