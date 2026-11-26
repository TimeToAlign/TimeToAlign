"""Score parsing matrix: every loader and format against the specimens.

Each loader reads each specimen in each format it has. The folded measure
counts are exact; the flow control a loader writes on its timeline is
played by the engine and compared, as ranges of measure counts, with the
``default`` rows of the specimen's ``.flow.csv`` file. What a source or a
parser does not state is a strict expected failure or a pinned difference,
each naming the reason. See ``README.md``, "Parsing matrix".
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from tests.loader.minting import named_flow_control
from timetoalign.timelines.flow import (
    flow_control_from_measures,
    measure_keys_from_flow_control,
)

from .conftest import (
    MAX_MUSICXML_SIZE_BYTES,
    SPECIMENS,
    TARGET_FLOWS_DIR,
    SpecimenConfig,
    atomic_ranges,
    get_flow_modes,
    get_specimen_path,
    gold_flows,
    musicxml_too_large,
    parse_flow_csv,
    played_ranges,
    repaired_stress_score,
)


def specimen_available(spec: SpecimenConfig) -> bool:
    """Check if the specimen's TSV file is available."""
    tsv_path = get_specimen_path(spec, "tsv")
    return tsv_path is not None and tsv_path.exists()


def format_available(spec: SpecimenConfig, file_type: str) -> bool:
    """Check if a specific format is available for a specimen."""
    path = get_specimen_path(spec, file_type)
    return path is not None and path.exists()


def _flow_control_counts(measures: Any) -> dict[str, int | bool]:
    """Return exact marker counts from a measure table.

    Args:
        measures: A ``MeasureData`` instance.

    Returns:
        Counts for the flow-control fields asserted by the parsing matrix.
    """
    table = measures._table
    repeats = (
        table.column("repeats").to_pylist() if "repeats" in table.column_names else []
    )
    bool_starts = (
        sum(value is True for value in table.column("start_repeat").to_pylist())
        if "start_repeat" in table.column_names
        else 0
    )
    bool_ends = (
        sum(value is True for value in table.column("end_repeat").to_pylist())
        if "end_repeat" in table.column_names
        else 0
    )
    repeat_starts = max(
        bool_starts, sum(value in {"start", "startend"} for value in repeats)
    )
    repeat_ends = max(bool_ends, sum(value in {"end", "startend"} for value in repeats))
    section_breaks = (
        sum(value == "section" for value in table.column("breaks").to_pylist())
        if "breaks" in table.column_names
        else 0
    )
    double_barlines = (
        sum(value == "double" for value in table.column("barline").to_pylist())
        if "barline" in table.column_names
        else 0
    )
    voltas = (
        sum(value in {1, 2} for value in table.column("volta").to_pylist())
        if "volta" in table.column_names
        else 0
    )
    return {
        "total_measures": len(table),
        "repeat_starts": repeat_starts,
        "repeat_ends": repeat_ends,
        "section_breaks": section_breaks,
        "double_barlines": double_barlines,
        "has_flow_control": any((repeat_starts, repeat_ends, section_breaks, voltas)),
    }


FLOW_CONTROL_PROFILES = [
    pytest.param(
        "WoO71",
        "tsv",
        {
            "total_measures": 397,
            "repeat_starts": 11,
            "repeat_ends": 11,
            "section_breaks": 12,
            "double_barlines": 4,
            "has_flow_control": True,
        },
        id="woo71-tsv",
    ),
    pytest.param(
        "WoO71",
        "measuremap",
        {"repeat_starts": 11, "repeat_ends": 11, "has_flow_control": True},
        id="woo71-measuremap",
    ),
    pytest.param(
        "WoO71",
        "partitura",
        {"has_flow_control": True},
        id="woo71-partitura",
    ),
    pytest.param(
        "WoO71",
        "music21",
        {"repeat_starts": 11, "repeat_ends": 11, "has_flow_control": True},
        id="woo71-music21",
    ),
    pytest.param(
        "flow_only",
        "tsv",
        {
            "total_measures": 15,
            "repeat_starts": 3,
            "repeat_ends": 6,
            "has_flow_control": True,
        },
        id="flow-only-tsv",
    ),
    pytest.param(
        "flow_only",
        "measuremap",
        {"repeat_starts": 3, "repeat_ends": 6, "has_flow_control": True},
        id="flow-only-measuremap",
    ),
    pytest.param(
        "flow_only",
        "partitura",
        {
            "total_measures": 15,
            "repeat_starts": 7,
            "repeat_ends": 7,
            "has_flow_control": True,
        },
        id="flow-only-partitura",
    ),
    pytest.param(
        "flow_only",
        "music21",
        {
            "total_measures": 15,
            "repeat_starts": 3,
            "repeat_ends": 6,
            "has_flow_control": True,
        },
        id="flow-only-music21",
    ),
]


# region Test Fixtures


@pytest.fixture(params=list(SPECIMENS.keys()))
def specimen_name(request) -> str:
    """Parametrized fixture providing each specimen name."""
    return request.param


@pytest.fixture
def specimen(specimen_name: str) -> SpecimenConfig:
    """Get the SpecimenConfig for the current specimen."""
    return SPECIMENS[specimen_name]


@pytest.fixture
def flow_csv_path(specimen: SpecimenConfig) -> Path:
    """Get the .flow.csv path for the current specimen."""
    return TARGET_FLOWS_DIR / specimen.flow_csv


# endregion

# region Test Classes


class TestFlowCSVValidation:
    """Validate .flow.csv files exist and are well-formed."""

    @pytest.mark.parametrize("specimen_name", SPECIMENS.keys())
    def test_flow_csv_exists(self, specimen_name: str) -> None:
        """Each specimen has a .flow.csv file."""
        spec = SPECIMENS[specimen_name]
        csv_path = TARGET_FLOWS_DIR / spec.flow_csv
        assert csv_path.exists(), f"Missing .flow.csv for {specimen_name}: {csv_path}"

    @pytest.mark.parametrize("specimen_name", SPECIMENS.keys())
    def test_flow_csv_has_atomic(self, specimen_name: str) -> None:
        """Each .flow.csv has atomic flow mode."""
        spec = SPECIMENS[specimen_name]
        csv_path = TARGET_FLOWS_DIR / spec.flow_csv
        if not csv_path.exists():
            pytest.skip(f"Flow CSV not found: {csv_path}")

        modes = get_flow_modes(csv_path)
        assert "atomic" in modes, f"{specimen_name}: Missing 'atomic' flow_mode"

    @pytest.mark.parametrize("specimen_name", SPECIMENS.keys())
    def test_flow_csv_has_default(self, specimen_name: str) -> None:
        """Each .flow.csv has default flow mode."""
        spec = SPECIMENS[specimen_name]
        csv_path = TARGET_FLOWS_DIR / spec.flow_csv
        if not csv_path.exists():
            pytest.skip(f"Flow CSV not found: {csv_path}")

        modes = get_flow_modes(csv_path)
        assert "default" in modes, f"{specimen_name}: Missing 'default' flow_mode"

    @pytest.mark.parametrize("specimen_name", SPECIMENS.keys())
    def test_atomic_section_count(self, specimen_name: str) -> None:
        """Atomic sections count matches expected."""
        spec = SPECIMENS[specimen_name]
        csv_path = TARGET_FLOWS_DIR / spec.flow_csv
        if not csv_path.exists():
            pytest.skip(f"Flow CSV not found: {csv_path}")

        entries = parse_flow_csv(csv_path)
        atomic_entries = [e for e in entries if e.flow_mode == "atomic"]
        actual = len(atomic_entries)
        expected = spec.expected_atomic_sections
        assert (
            actual == expected
        ), f"{specimen_name}: Expected {expected} atomic sections, got {actual}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestFlowControlProfiles:
    """Pin exact marker profiles and the Partitura region-model difference."""

    @pytest.mark.parametrize(
        "specimen_name,loader_name,expected", FLOW_CONTROL_PROFILES
    )
    def test_loader_flow_control_profile(
        self,
        specimen_name: str,
        loader_name: str,
        expected: dict[str, int | bool],
    ) -> None:
        """Each loader extracts its documented flow-control profile exactly."""
        from timetoalign.loader.score import (
            MeasureMapLoader,
            Ms3Loader,
            Music21Loader,
            PartituraLoader,
        )

        loader_classes = {
            "tsv": Ms3Loader,
            "measuremap": MeasureMapLoader,
            "partitura": PartituraLoader,
            "music21": Music21Loader,
        }
        file_types = {
            "tsv": "tsv",
            "measuremap": "mm_json",
            "partitura": "musicxml",
            "music21": "musicxml",
        }
        spec = SPECIMENS[specimen_name]
        source = get_specimen_path(spec, file_types[loader_name])
        if source is None or not source.exists():
            pytest.skip(f"Source not found for {specimen_name}: {loader_name}")
        if loader_name in {"partitura", "music21"} and musicxml_too_large(source):
            pytest.skip(f"Source too large for {loader_name}: {source}")

        loader = loader_classes[loader_name]()
        loader.load(source)
        counts = _flow_control_counts(loader.store.measures)
        if (
            specimen_name == "WoO71"
            and loader_name in {"partitura", "music21"}
            and counts["total_measures"] != 397
        ):
            pytest.skip(
                f"{loader_name} loaded {counts['total_measures']} multi-part measures"
            )

        for field, expected_value in expected.items():
            assert counts[field] == expected_value, (
                f"{specimen_name}/{loader_name} {field}: "
                f"got {counts[field]}, expected {expected_value}"
            )


class TestMs3LoaderValidation:
    """Validate Ms3Loader (gold standard) measure counts."""

    @pytest.mark.parametrize("specimen_name", SPECIMENS.keys())
    def test_tsv_folded_count(self, specimen_name: str) -> None:
        """Ms3Loader folded measure count is exact."""
        spec = SPECIMENS[specimen_name]
        tsv_path = get_specimen_path(spec, "tsv")
        if tsv_path is None or not tsv_path.exists():
            pytest.skip(f"TSV not found for {specimen_name}")

        from timetoalign.loader.score import Ms3Loader

        loader = Ms3Loader()
        loader.load(tsv_path)

        mc_values = loader.store.measures._table.column("mc").to_pylist()
        actual = len(set(mc_values))
        expected = spec.folded_measures

        assert actual == expected, (
            f"{specimen_name}: TSV folded count mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}"
        )


class TestMeasureMapLoaderValidation:
    """Validate MeasureMapLoader against gold standard."""

    @pytest.mark.parametrize("specimen_name", SPECIMENS.keys())
    def test_measuremap_folded_count(self, specimen_name: str) -> None:
        """MeasureMapLoader folded measure count matches TSV."""
        spec = SPECIMENS[specimen_name]
        mm_path = get_specimen_path(spec, "mm_json")
        if mm_path is None or not mm_path.exists():
            pytest.skip(f"MeasureMap not found for {specimen_name}")

        from timetoalign.loader.score import MeasureMapLoader

        loader = MeasureMapLoader()
        loader.load(mm_path)

        actual = len(loader.store.measures)
        expected = spec.folded_measures

        assert actual == expected, (
            f"{specimen_name}: MeasureMap folded count mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}"
        )

    @pytest.mark.parametrize("specimen_name", SPECIMENS.keys())
    def test_measuremap_repeat_signs_match_tsv(self, specimen_name: str) -> None:
        """A Measure Map states the repeat signs of the measures table.

        Both timelines' flow control is read back as row keys; the
        ``start_repeat`` and ``end_repeat`` of every measure must agree.
        """
        spec = SPECIMENS[specimen_name]
        from timetoalign.loader.score import MeasureMapLoader, Ms3Loader

        def repeat_signs(timeline: Any) -> list[tuple[int, bool, bool]]:
            return [
                (row["count"], row["start_repeat"], row["end_repeat"])
                for row in measure_keys_from_flow_control(
                    timeline.flow_control.elements, timeline.measure_map
                )
            ]

        from_map = MeasureMapLoader().load(get_specimen_path(spec, "mm_json"))
        from_table = Ms3Loader().load(get_specimen_path(spec, "tsv"))
        signs = repeat_signs(from_map.create_timeline())
        assert signs == repeat_signs(from_table.create_timeline())
        assert len(signs) == spec.folded_measures

    @pytest.mark.parametrize(
        ("specimen_name", "expected"),
        [
            ("c05n05_musete", [(1, 17), (1, 32), (17, 59), (32, 59)]),
            ("c11n08_Rondeau", [(1, 10), (1, 19), (10, 28), (19, 61), (28, 61)]),
        ],
    )
    def test_measuremap_states_repeats_only(
        self, specimen_name: str, expected: list[tuple[int, int]]
    ) -> None:
        """A Measure Map row has no key for a D.S., a D.C., a segno or a Fine.

        The timeline of a ``.mm.json`` file therefore plays the repeats and
        nothing else; what the file knows beyond them is its ``next``.
        """
        from timetoalign.loader.score import MeasureMapLoader

        loader = MeasureMapLoader(seed=1).load(
            get_specimen_path(SPECIMENS[specimen_name], "mm_json")
        )
        timeline = loader.create_timeline()
        ranges, result = played_ranges(timeline)
        assert [(row.start, row.end) for row in ranges] == expected
        assert result.findings == ()
        translated = [
            element.to_dict()
            for element in flow_control_from_measures(timeline.measure_map, seed=2)
        ]
        assert named_flow_control(
            timeline.flow_control.to_dict(), 1
        ) == named_flow_control(
            sorted(translated, key=lambda e: (Fraction(*e["coordinate"]), e["id"])), 2
        )


# A MusicXML score of several parts gives Music21Loader and PartituraLoader one
# measure row per part, which form no measure map: no timeline is created.
_ONE_MEASURE_ROW_PER_PART = pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="the measure table holds one row per part, which form no measure map",
)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestMusic21LoaderMusicXML:
    """Validate Music21Loader with MusicXML files.

    KNOWN DEVIATIONS:
    - music21's expandRepeats() ignores D.S., D.C., and Fine markers
    - For specimens with D.S./D.C. (c05n05_musete, c11n08_Rondeau, flow_only),
      music21 produces different (often shorter) flows
    """

    @pytest.mark.parametrize("specimen_name", SPECIMENS.keys())
    def test_music21_musicxml_folded_count(self, specimen_name: str) -> None:
        """Music21Loader MusicXML folded measure count is correct."""
        spec = SPECIMENS[specimen_name]
        xml_path = get_specimen_path(spec, "musicxml")
        if xml_path is None or not xml_path.exists():
            pytest.skip(f"MusicXML not found for {specimen_name}")

        # Skip large files to avoid test timeouts
        if musicxml_too_large(xml_path):
            file_size_mb = xml_path.stat().st_size / 1_000_000
            pytest.skip(
                f"MusicXML file too large ({file_size_mb:.1f}MB) for Music21 test. "
                f"Max size: {MAX_MUSICXML_SIZE_BYTES / 1_000_000:.1f}MB"
            )

        try:
            from timetoalign.loader.score import Music21Loader
        except ImportError:
            pytest.skip("Music21Loader not available")

        loader = Music21Loader()
        loader.load(xml_path)

        mc_values = loader.store.measures._table.column("mc").to_pylist()
        actual = len(set(mc_values))
        expected = spec.folded_measures

        assert actual == expected, (
            f"{specimen_name}: Music21 MusicXML folded count mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}"
        )

    @pytest.mark.parametrize(
        "specimen_name",
        [
            (
                pytest.param(name, marks=_ONE_MEASURE_ROW_PER_PART)
                if name in {"polyrhythm_only", "c05n05_musete", "c11n08_Rondeau"}
                else name
            )
            for name in SPECIMENS
            if name != "flow_only"
        ],
    )
    def test_music21_musicxml_default_flow(self, specimen_name: str) -> None:
        """Music21Loader's timeline plays the ``default`` flow of ``.flow.csv``.

        ``flow_only`` is pinned in TestDocumentedDeviations: music21 reads no
        D.S., D.C., segno, coda or Fine.
        """
        spec = SPECIMENS[specimen_name]
        xml_path = get_specimen_path(spec, "musicxml")
        if xml_path is None or not xml_path.exists():
            pytest.skip(f"MusicXML not found for {specimen_name}")
        if musicxml_too_large(xml_path):
            pytest.skip(f"MusicXML file too large for Music21: {xml_path}")

        from timetoalign.loader.score import Music21Loader

        timeline = Music21Loader().load(xml_path).create_timeline()
        ranges, result = played_ranges(timeline)
        assert ranges == gold_flows(specimen_name)["default"]
        assert result.findings == ()


# music21's MEI reader keeps only the measures that state an ``@n``.
_MUSIC21_MEI_DROPS_MEASURES = pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="music21's MEI reader drops the 5 measures without @n (10 of 15)",
)

# music21 gives the skeleton's empty measures no length, so the load raises.
_MUSIC21_MEI_MEASURES_WITHOUT_LENGTH = pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="music21 gives 9 of the skeleton's 21 measures no length",
)


def _music21_mei_marks(name: str) -> Any:
    """Return the known differences of Music21's MEI reading of a specimen."""
    if name == "flow_only":
        return pytest.param(name, marks=_MUSIC21_MEI_DROPS_MEASURES)
    if name == "op18_no4_mov4_flow":
        return pytest.param(name, marks=_MUSIC21_MEI_MEASURES_WITHOUT_LENGTH)
    return name


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestMusic21LoaderMEI:
    """Validate Music21Loader with MEI files.

    MEI files are exported from MuseScore and should produce identical results
    to MusicXML. The same D.S./D.C. limitations apply.
    """

    @pytest.mark.parametrize(
        "specimen_name", [_music21_mei_marks(name) for name in SPECIMENS]
    )
    def test_music21_mei_folded_count(self, specimen_name: str) -> None:
        """Music21Loader MEI folded measure count is correct."""
        spec = SPECIMENS[specimen_name]
        mei_path = get_specimen_path(spec, "mei")
        if mei_path is None or not mei_path.exists():
            pytest.skip(f"MEI not found for {specimen_name}")

        # Skip large files to avoid test timeouts
        if musicxml_too_large(mei_path):
            file_size_mb = mei_path.stat().st_size / 1_000_000
            pytest.skip(
                f"MEI file too large ({file_size_mb:.1f}MB) for Music21 test. "
                f"Max size: {MAX_MUSICXML_SIZE_BYTES / 1_000_000:.1f}MB"
            )

        try:
            from timetoalign.loader.score import Music21Loader
        except ImportError:
            pytest.skip("Music21Loader not available")

        loader = Music21Loader()
        try:
            loader.load(mei_path)
        except (SyntaxError, KeyError) as e:
            # Music21 MEI parser raises:
            # - music21.exceptions21.Music21Exception subclasses (MeiElementError, etc.)
            #   which inherit from Exception directly — caught separately below
            # - xml.etree.ElementTree.ParseError (subclass of SyntaxError)
            # - KeyError for missing elements
            pytest.skip(f"Music21 failed to load MEI: {e}")
        except Exception as e:
            if "music21" in type(e).__module__:
                pytest.skip(f"Music21 failed to load MEI: {e}")
            raise

        mc_values = loader.store.measures._table.column("mc").to_pylist()
        actual = len(set(mc_values))
        expected = spec.folded_measures

        assert actual == expected, (
            f"{specimen_name}: Music21 MEI folded count mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}"
        )

    @pytest.mark.parametrize(
        "specimen_name",
        [
            (
                pytest.param(name, marks=pytest.mark.slow)
                if name == "c05n05_musete"
                else _music21_mei_marks(name)
            )
            for name in SPECIMENS
        ],
    )
    def test_music21_mei_matches_musicxml(self, specimen_name: str) -> None:
        """Music21Loader MEI produces same flow as MusicXML."""
        spec = SPECIMENS[specimen_name]
        mei_path = get_specimen_path(spec, "mei")
        xml_path = get_specimen_path(spec, "musicxml")

        if mei_path is None or not mei_path.exists():
            pytest.skip(f"MEI not found for {specimen_name}")
        if xml_path is None or not xml_path.exists():
            pytest.skip(f"MusicXML not found for {specimen_name}")

        # Skip large files to avoid test timeouts
        if musicxml_too_large(mei_path) or musicxml_too_large(xml_path):
            pytest.skip(
                f"MEI or MusicXML file too large for Music21 test. "
                f"Max size: {MAX_MUSICXML_SIZE_BYTES / 1_000_000:.1f}MB"
            )

        try:
            from timetoalign.loader.score import Music21Loader
        except ImportError:
            pytest.skip("Music21Loader not available")

        # Load both formats
        mei_loader = Music21Loader()
        xml_loader = Music21Loader()

        try:
            mei_loader.load(mei_path)
        except (SyntaxError, KeyError) as e:
            # xml.etree.ElementTree.ParseError (subclass of SyntaxError),
            # KeyError for missing elements
            pytest.skip(f"Music21 failed to load MEI: {e}")
        except Exception as e:
            if "music21" in type(e).__module__:
                pytest.skip(f"Music21 failed to load MEI: {e}")
            raise

        xml_loader.load(xml_path)

        # Music21 reads repeat signs and endings and no other sign, so the
        # three columns the flow control is written from are the whole of it.
        def signs(loader: Any) -> list[tuple[Any, ...]]:
            table = loader.store.measures._table
            return list(
                zip(
                    table.column("mc").to_pylist(),
                    table.column("start_repeat").to_pylist(),
                    table.column("end_repeat").to_pylist(),
                    table.column("volta").to_pylist(),
                    strict=True,
                )
            )

        assert signs(mei_loader) == signs(xml_loader)
        assert len(signs(xml_loader)) >= spec.folded_measures

    def test_measures_without_length_are_refused(self) -> None:
        """A measure music21 gives no length states no position after it.

        The ``op18_no4_mov4_flow`` skeleton's empty measures have no timed
        content, and music21 leaves out its endings, which hold nothing but
        measure rests. The load raises rather than write measures of length
        zero, on which two runs of brackets would fall on one coordinate.
        """
        pytest.importorskip("music21")
        from timetoalign.loader.score import Music21Loader

        path = get_specimen_path(SPECIMENS["op18_no4_mov4_flow"], "mei")
        with pytest.raises(ValueError) as raised:
            Music21Loader().load(path)
        assert str(raised.value) == (
            "music21 gives 9 of the 21 measures it reads from op18_no4_mov4_flow.mei no "
            "length (a measure without timed content has none): measure numbers "
            "[2, 11, 20, 29, 46, 79, 86, 95, 104]. Neither their lengths nor the "
            "positions of the measures after them are stated, so no measure table is "
            "written"
        )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestPartituraLoaderMusicXML:
    """Validate PartituraLoader with MusicXML files.

    KNOWN DEVIATIONS:
    - Partitura uses a region-based repeat model (start/end pairs)
    - Missing repeat starts are inferred, causing different counts
    - For flow_only specimen, partitura infers 7 starts vs 3 in TSV
    """

    @pytest.mark.parametrize("specimen_name", SPECIMENS.keys())
    def test_partitura_musicxml_folded_count(self, specimen_name: str) -> None:
        """PartituraLoader MusicXML folded measure count is correct."""
        spec = SPECIMENS[specimen_name]
        xml_path = get_specimen_path(spec, "musicxml")
        if xml_path is None or not xml_path.exists():
            pytest.skip(f"MusicXML not found for {specimen_name}")

        # Skip large files to avoid test timeouts
        if musicxml_too_large(xml_path):
            file_size_mb = xml_path.stat().st_size / 1_000_000
            pytest.skip(
                f"MusicXML file too large ({file_size_mb:.1f}MB) for Partitura test. "
                f"Max size: {MAX_MUSICXML_SIZE_BYTES / 1_000_000:.1f}MB"
            )

        try:
            from timetoalign.loader.score import PartituraLoader
        except ImportError:
            pytest.skip("PartituraLoader not available")

        loader = PartituraLoader()
        loader.load(xml_path)

        mc_values = loader.store.measures._table.column("mc").to_pylist()
        actual = len(set(mc_values))
        expected = spec.folded_measures

        assert actual == expected, (
            f"{specimen_name}: Partitura MusicXML folded count mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}\n"
            f"Note: Multi-part scores may have different counts"
        )

    @pytest.mark.parametrize(
        "specimen_name",
        [
            (
                pytest.param(name, marks=_ONE_MEASURE_ROW_PER_PART)
                if name == "polyrhythm_only"
                else name
            )
            for name in SPECIMENS
            if name != "flow_only"
        ],
    )
    def test_partitura_musicxml_default_flow(self, specimen_name: str) -> None:
        """PartituraLoader's timeline plays the ``default`` flow of ``.flow.csv``.

        The D.S. al Fine of ``c05n05_musete`` and the D.C. of
        ``c11n08_Rondeau`` are read from the MusicXML directions. The
        inferred repeat boundaries of ``flow_only`` are pinned in
        TestDocumentedDeviations.
        """
        spec = SPECIMENS[specimen_name]
        xml_path = get_specimen_path(spec, "musicxml")
        if xml_path is None or not xml_path.exists():
            pytest.skip(f"MusicXML not found for {specimen_name}")
        if musicxml_too_large(xml_path):
            pytest.skip(f"MusicXML file too large for Partitura: {xml_path}")

        from timetoalign.loader.score import PartituraLoader

        timeline = PartituraLoader().load(xml_path).create_timeline()
        gold = gold_flows(specimen_name)
        ranges, result = played_ranges(timeline)
        assert ranges == gold["default"]
        assert result.findings == ()
        assert atomic_ranges(timeline) == gold["atomic"]


def _strict_xfail(reason: str, raises: type[Exception] = AssertionError) -> Any:
    return pytest.mark.xfail(raises=raises, strict=True, reason=reason)


# Partitura's MEI importer raises before the loader sees a measure.
_PARTITURA_MEI_DIR_WITHOUT_TSTAMP = _strict_xfail(
    "Partitura's MEI importer raises in _find_dir_positions: a <dir> with @type "
    "is placed by @startid and states no @tstamp",
    raises=TypeError,
)
_PARTITURA_MEI_IMPORT_FAILS = {
    "polyrhythm_only": _strict_xfail(
        "Partitura's MEI importer raises in _handle_metersig: staff 3 states its meter "
        "only as meter.sym='cut'",
        raises=Exception,
    ),
    "c05n05_musete": _PARTITURA_MEI_DIR_WITHOUT_TSTAMP,
    "c11n08_Rondeau": _PARTITURA_MEI_DIR_WITHOUT_TSTAMP,
}


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestPartituraLoaderMEI:
    """Validate PartituraLoader with MEI files."""

    @pytest.mark.parametrize(
        "specimen_name",
        [
            pytest.param(name, marks=_PARTITURA_MEI_IMPORT_FAILS.get(name, ()))
            for name in SPECIMENS
        ],
    )
    def test_partitura_mei_folded_count(self, specimen_name: str) -> None:
        """PartituraLoader MEI folded measure count is correct."""
        spec = SPECIMENS[specimen_name]
        mei_path = get_specimen_path(spec, "mei")
        if mei_path is None or not mei_path.exists():
            pytest.skip(f"MEI not found for {specimen_name}")

        # Skip large files to avoid test timeouts
        if musicxml_too_large(mei_path):
            file_size_mb = mei_path.stat().st_size / 1_000_000
            pytest.skip(
                f"MEI file too large ({file_size_mb:.1f}MB) for Partitura test. "
                f"Max size: {MAX_MUSICXML_SIZE_BYTES / 1_000_000:.1f}MB"
            )

        try:
            from timetoalign.loader.score import PartituraLoader
        except ImportError:
            pytest.skip("PartituraLoader not available")

        loader = PartituraLoader()
        try:
            loader.load(mei_path)
        except (SyntaxError, KeyError) as e:
            # Partitura MEI parser raises:
            # - lxml.etree.XMLSyntaxError (subclass of SyntaxError) for malformed XML
            # - KeyError for missing elements in the score structure
            pytest.skip(f"Partitura failed to load MEI: {e}")

        mc_values = loader.store.measures._table.column("mc").to_pylist()
        actual = len(set(mc_values))
        expected = spec.folded_measures

        assert actual == expected, (
            f"{specimen_name}: Partitura MEI folded count mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}"
        )


def _verovio_timeline(piece: str) -> tuple[Any, Any]:
    """Return the Verovio loader of a specimen's MEI and its timeline.

    The MEI is read with ``expandNever``: the measure table is the printed
    one either way, and Verovio 6.4.1 crashes the process when it expands
    the repeats of the flow-control scores.
    """
    pytest.importorskip("verovio")
    from timetoalign.loader.score import VerovioLoader

    loader = VerovioLoader(options={"expandNever": True}).load(
        get_specimen_path(SPECIMENS[piece], "mei")
    )
    return loader, loader.create_timeline()


# Rachmaninoff is not loaded: Verovio 6.4.1 crashes the process on its MEI.
# flow_only has a test of its own: its source ends the flow at a bracket.
VEROVIO_MATRIX = [
    pytest.param("polyrhythm_only", id="polyrhythm_only"),
    pytest.param(
        "c05n05_musete",
        id="c05n05_musete",
        marks=_strict_xfail(
            "the MEI states a D.C. at MC 31 where the target flow jumps to the segno"
        ),
    ),
    pytest.param(
        "c11n08_Rondeau",
        id="c11n08_Rondeau",
        marks=_strict_xfail(
            "the MEI states its D.C.s at MCs 18 and 27 without where they end"
        ),
    ),
    pytest.param(
        "WoO71",
        id="WoO71",
        marks=[
            pytest.mark.slow,
            _strict_xfail(
                "the MEI states no section breaks, which cut the target's sections"
            ),
        ],
    ),
    pytest.param(
        "op18_no4_mov4_flow",
        id="op18_no4_mov4_flow",
        marks=_strict_xfail(
            "the skeleton's measures have no content, so no timemap entry",
            raises=ValueError,
        ),
    ),
]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestVerovioLoaderMEI:
    """The flow control of VerovioLoader's timeline plays the target default flow."""

    @pytest.mark.parametrize("piece", VEROVIO_MATRIX)
    def test_verovio_mei_default_flow(self, piece: str) -> None:
        loader, timeline = _verovio_timeline(piece)
        assert len(loader.store.measures) == SPECIMENS[piece].folded_measures
        ranges, result = played_ranges(timeline)
        assert ranges == gold_flows(piece)["default"]
        assert result.findings == ()

    def test_flow_only_plays_as_the_measures_table(self) -> None:
        """The MEI's marks give the flow the ms3 table gives, sign for sign.

        As the source stands the flow ends at the second of three brackets;
        with the same repair it is the target's ``default`` flow.
        """
        from timetoalign.loader.score import Ms3Loader

        loader, timeline = _verovio_timeline("flow_only")
        assert len(loader.store.measures) == SPECIMENS["flow_only"].folded_measures
        table = Ms3Loader().load(get_specimen_path(SPECIMENS["flow_only"], "tsv"))
        from_table = table.create_timeline()
        gold = gold_flows("flow_only")

        ranges, result = played_ranges(timeline)
        expected, expected_result = played_ranges(from_table)
        assert ranges == expected == gold["default"][:4]

        def findings(found: Any, held: Any) -> list[tuple[Any, ...]]:
            """The findings, each element ID in a message replaced by its role."""
            parts = []
            for finding in found.findings:
                message = finding.message
                for element in held.flow_control.elements:
                    message = message.replace(f"'{element.id}'", element.role)
                parts.append((finding.kind, finding.coordinate, finding.end, message))
            return parts

        assert findings(result, timeline) == findings(expected_result, from_table)
        assert atomic_ranges(timeline) == gold["atomic"]

        def signs(held: Any) -> list[tuple[Any, ...]]:
            return sorted(
                (element.role, element.coordinate.value, str(element.manifestation))
                for element in held.flow_control.elements
                if element.role in {"jump_target", "break", "section_break"}
            )

        assert signs(timeline) == signs(from_table)

        def targets(held: Any) -> list[tuple[Any, ...]]:
            return sorted(
                (target.coordinate.value, str(target.manifestation), target.name or "")
                for target in held.flow_control.jump_targets
            )

        # Verovio counts a second coda; the measures table names MuseScore's.
        assert [
            (target[2], named[2])
            for target, named in zip(
                targets(timeline), targets(from_table), strict=True
            )
            if target != named
        ] == [("coda2", "codab")]
        repaired, result = played_ranges(
            timeline, registry=repaired_stress_score(timeline)
        )
        assert repaired == gold["default"]
        assert result.findings == ()

    @pytest.mark.slow
    def test_woo71_mc_sequence(self) -> None:
        """WoO71's flow differs only in its cuts: the MC sequence is the target's."""
        _, timeline = _verovio_timeline("WoO71")
        ranges, _ = played_ranges(timeline)

        def sequence(rows: Any) -> list[int]:
            return [mc for row in rows for mc in range(row.start, row.end)]

        played = sequence(ranges)
        assert played == sequence(gold_flows("WoO71")["default"])
        assert len(played) == SPECIMENS["WoO71"].unfolded_measures


class TestCrossLoaderParity:
    """Validate that marker-based loaders (TSV, MeasureMap, Music21) produce identical counts."""

    @pytest.mark.parametrize("specimen_name", SPECIMENS.keys())
    def test_tsv_vs_measuremap_folded_count(self, specimen_name: str) -> None:
        """TSV and MeasureMap have identical folded measure counts."""
        spec = SPECIMENS[specimen_name]
        tsv_path = get_specimen_path(spec, "tsv")
        mm_path = get_specimen_path(spec, "mm_json")

        if tsv_path is None or not tsv_path.exists():
            pytest.skip(f"TSV not found for {specimen_name}")
        if mm_path is None or not mm_path.exists():
            pytest.skip(f"MeasureMap not found for {specimen_name}")

        from timetoalign.loader.score import MeasureMapLoader, Ms3Loader

        tsv_loader = Ms3Loader()
        tsv_loader.load(tsv_path)
        tsv_count = len(set(tsv_loader.store.measures._table.column("mc").to_pylist()))

        mm_loader = MeasureMapLoader()
        mm_loader.load(mm_path)
        mm_count = len(mm_loader.store.measures)

        assert tsv_count == mm_count, (
            f"{specimen_name}: TSV vs MeasureMap count mismatch.\n"
            f"TSV: {tsv_count}\n"
            f"MeasureMap: {mm_count}"
        )


# endregion

# region Deviation Documentation Tests


class TestDocumentedDeviations:
    """Document known parser deviations for the README.

    These tests capture expected differences between parsers with exact output
    assertions, so a parser change requires an intentional test update.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("specimen_name", "starts", "ends"),
        [
            ("c05n05_musete", [17, 32], [16, 31, 58]),
            ("c11n08_Rondeau", [10, 19, 28], [9, 18, 27, 60]),
        ],
    )
    def test_music21_reads_no_navigation_marks(
        self, specimen_name: str, starts: list[int], ends: list[int]
    ) -> None:
        """Document: music21 reads repeat signs and no D.S., D.C., segno or Fine.

        The measure table states the repeats of the score and leaves the
        four navigation columns empty, so no leap can be written.
        """
        from timetoalign.loader.score import Music21Loader

        loader = Music21Loader()
        loader.load(get_specimen_path(SPECIMENS[specimen_name], "musicxml"))
        rows = loader.store.measures._table.to_pylist()

        assert sorted({row["mc"] for row in rows if row["start_repeat"]}) == starts
        assert sorted({row["mc"] for row in rows if row["end_repeat"]}) == ends
        for column in ("markers", "jump_bwd", "jump_fwd", "play_until"):
            assert {row[column] for row in rows} == {None}

    def test_flow_only_music21_deviation(self) -> None:
        """Document: music21 states no D.S., D.C., segno, coda or Fine of flow_only.

        Its timeline holds the repeats and brackets alone. The flow ends at
        the second of three brackets, as the score's own source has it, and
        the atomic sections lack the boundaries the segno and the two codas
        give (MCs 8 and 13).
        """
        from timetoalign.loader.score import Music21Loader

        loader = Music21Loader()
        loader.load(get_specimen_path(SPECIMENS["flow_only"], "musicxml"))
        timeline = loader.create_timeline()

        assert {
            manifestation.value
            for jump in timeline.flow_control.jumps
            for manifestation in jump.manifestations
        } == {"repeat_end", "volta"}
        ranges, _ = played_ranges(timeline)
        assert [(row.start, row.end) for row in ranges] == [
            (1, 4),
            (1, 6),
            (4, 5),
            (6, 7),
        ]
        assert [(row.start, row.end) for row in atomic_ranges(timeline)] == [
            (1, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 14),
            (14, 15),
            (15, 16),
        ]

    def test_flow_only_partitura_deviation(self) -> None:
        """Document: Partitura's flow_only repeat structure is malformed.

        Partitura discards the segno and coda directions and infers missing
        repeat boundaries on this specimen. Its default unfolding is
        therefore not a meaningful observable; this table pins exactly what
        the parser delivers at the source.
        """
        from timetoalign.loader.score import PartituraLoader

        loader = PartituraLoader()
        loader.load(get_specimen_path(SPECIMENS["flow_only"], "musicxml"))
        rows = loader.store.measures._table.to_pylist()

        assert [row["mc"] for row in rows if row["start_repeat"]] == [
            1,
            4,
            6,
            9,
            10,
            11,
            12,
        ]
        assert [row["mc"] for row in rows if row["end_repeat"]] == [
            3,
            5,
            8,
            9,
            10,
            11,
            14,
        ]
        assert [
            (row["mc"], row["volta"]) for row in rows if row["volta"] is not None
        ] == [
            (5, 1),
            (6, 2),
            (7, 3),
            (14, 1),
            (15, 2),
        ]

    def test_flow_only_ms3_deviation(self) -> None:
        """Document: ms3 diverges from canonical in flow_only due to ambiguous encoding.

        The score has ambiguous repeat encoding that ms3 interprets differently:
        - Canonical: 31 MCs (humans would play it this way)
        - ms3: Different sequence due to repeat end inference

        Additionally, music21 fails entirely on this specimen.
        """
        spec = SPECIMENS["flow_only"]
        csv_path = TARGET_FLOWS_DIR / spec.flow_csv
        if not csv_path.exists():
            pytest.skip("Flow CSV not found")

        modes = get_flow_modes(csv_path)

        # The CSV documents multiple valid interpretations
        assert "default" in modes, "default mode required"
        assert "ms3" in modes, "ms3 deviation should be documented in CSV"


# endregion
