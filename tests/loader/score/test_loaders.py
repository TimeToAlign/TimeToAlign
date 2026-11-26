"""Tests for symbolic score loaders (ScoreStore architecture)."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from timetoalign.core import TimeUnit
from timetoalign.loader.score.ms3 import Ms3Loader
from timetoalign.loader.score.music21 import Music21Loader
from timetoalign.loader.score.partitura import PartituraLoader
from timetoalign.loader.score.store import ScoreStore
from timetoalign.maps import ScalarMap
from timetoalign.testdata import ensure_data

DATA_DIR = ensure_data("vienna_1x22")
MIDI_SCORE_DIR = ensure_data("midi") / "score"
MS3_DIR = DATA_DIR / "ms3"
WOO71_MEASURES = ensure_data("score") / "beethoven_woo71" / "WoO71.measures.tsv"
WOO71_NOTES = WOO71_MEASURES.with_name("WoO71.notes.tsv")


def _write_synthetic_notes(
    path: Path, duration_qb: str, duration: str | None = None
) -> Path:
    """Write one minimal MS3 notes row for coordinate validation."""
    columns = ["mc", "mn", "quarterbeats", "duration_qb", "midi", "name", "octave"]
    values = ["1", "1", "0", duration_qb, "60", "C4", "4"]
    if duration is not None:
        columns.append("duration")
        values.append(duration)
    path.write_text(
        "\t".join(columns) + "\n" + "\t".join(values) + "\n", encoding="utf-8"
    )
    return path


def _write_synthetic_annotation_facet(
    path: Path, *, label_column: str, quarterbeats: str, duration_qb: str, duration: str
) -> Path:
    """Write one minimal MS3 chords/harmonies row carrying both duration columns."""
    columns = ["mc", "mn", "quarterbeats", "duration_qb", "duration", label_column]
    values = ["44", "44", quarterbeats, duration_qb, duration, "i"]
    path.write_text(
        "\t".join(columns) + "\n" + "\t".join(values) + "\n", encoding="utf-8"
    )
    return path


def _write_label_facet_with_measures(
    directory: Path, *, labels: list[tuple[str, str]], measure_end: tuple[str, str]
) -> Path:
    """Write a harmonies facet with no duration column, beside its measures.

    Mirrors the DCML shape the derivation exists for: label positions stated
    exactly, spans stated only as ms3's float ``duration_qb``, and the end of
    the piece knowable only from the companion measures facet.
    """
    harmonies = directory / "piece.harmonies.tsv"
    harmonies.write_text(
        "\t".join(["mc", "mn", "quarterbeats", "duration_qb", "label"])
        + "\n"
        + "".join(
            f"1\t1\t{quarterbeats}\t{duration_qb}\ti\n"
            for quarterbeats, duration_qb in labels
        ),
        encoding="utf-8",
    )
    quarterbeats, act_dur = measure_end
    (directory / "piece.measures.tsv").write_text(
        "\t".join(["mc", "mn", "quarterbeats", "act_dur"])
        + "\n"
        + f"1\t1\t{quarterbeats}\t{act_dur}\n",
        encoding="utf-8",
    )
    return harmonies


def _exact_pair(coordinate: dict[str, object]) -> tuple[int, int]:
    """Return the exact numerator and denominator from a coordinate struct."""
    numerator = coordinate["numerator"]
    denominator = coordinate["denominator"]
    assert isinstance(numerator, int)
    assert isinstance(denominator, int)
    return numerator, denominator


@pytest.fixture
def chopin_xml():
    return DATA_DIR / "Chopin_op10_no3.musicxml"


@pytest.fixture
def chopin_tsv_notes():
    return MS3_DIR / "chopin_op10_no3.notes.tsv"


class TestMs3Loader:
    """Tests for Ms3Loader."""

    def test_measure_coordinates_preserve_exact_fractions(self):
        """Known MS3 measures retain exact starts, durations, and ends."""
        loader = Ms3Loader.from_file(WOO71_MEASURES)
        rows = {row["mc"]: row for row in loader.store.measures}
        expected = {
            1: (Fraction(0), Fraction(1), Fraction(1)),
            2: (Fraction(1), Fraction(2), Fraction(3)),
            3: (Fraction(3), Fraction(2), Fraction(5)),
        }

        for mc, (start, duration, end) in expected.items():
            row = rows[mc]
            assert (
                Fraction(row["start"]["numerator"], row["start"]["denominator"])
                == start
            )
            assert (
                Fraction(row["duration"]["numerator"], row["duration"]["denominator"])
                == duration
            )
            assert Fraction(row["end"]["numerator"], row["end"]["denominator"]) == end

    def test_triplet_duration_uses_symbolic_fraction(self, tmp_path):
        """A triplet duration keeps its exact one-third quarter value."""
        path = _write_synthetic_notes(
            tmp_path / "triplet.notes.tsv",
            duration_qb="0.3333333333333333",
            duration="1/12",
        )

        row = list(Ms3Loader.from_file(path).store.notes)[0]

        assert _exact_pair(row["duration"]) == (1, 3)
        assert _exact_pair(row["end"]) == (1, 3)

    def test_binary_rational_duration_without_symbolic_source(self, tmp_path):
        """A compact native binary duration remains exact without duration."""
        path = _write_synthetic_notes(
            tmp_path / "binary_duration.notes.tsv", duration_qb="0.5"
        )

        row = list(Ms3Loader.from_file(path).store.notes)[0]

        assert _exact_pair(row["duration"]) == (1, 2)
        assert _exact_pair(row["end"]) == (1, 2)

    def test_pathological_derived_duration_mirrors_its_double(self, tmp_path):
        """A derived decimal keeps its double, mirrored by its exact dyadic.

        There is no such thing as a cell without a pair: the ratio side is
        always populated, and for a value with no symbolic source it holds
        the double's exact dyadic rather than a tidied-up ratio. The double
        is numerically identical to that dyadic, which is why mirroring it is
        not fabrication -- inventing ``1/3`` here would be.
        """
        path = _write_synthetic_notes(
            tmp_path / "derived_decimal.notes.tsv",
            duration_qb="0.3333333333333333",
        )

        row = list(Ms3Loader.from_file(path).store.notes)[0]
        duration = row["duration"]

        assert duration["value"] == 0.3333333333333333
        assert _exact_pair(duration) == (6004799503160661, 18014398509481984)
        assert Fraction(*_exact_pair(duration)) == Fraction(0.3333333333333333)
        assert Fraction(*_exact_pair(duration)) != Fraction(1, 3)

    @pytest.mark.parametrize(
        ("facet", "label_column", "store_attribute"),
        [("chords", "chord", "controls"), ("harmonies", "label", "annotations")],
    )
    def test_annotation_facets_read_the_symbolic_duration_column(
        self, tmp_path, facet, label_column, store_attribute
    ):
        """Chords and harmonies take the exact duration, like notes and measures.

        The row is the shape that used to fail: a triplet position stated
        exactly as ``695/4`` with a duration the source gives both ways. Read
        from ``duration_qb`` the sum needs a 65-bit numerator and the load
        raises; read from ``duration`` it is ``1043/6``.
        """
        path = _write_synthetic_annotation_facet(
            tmp_path / f"triplet.{facet}.tsv",
            label_column=label_column,
            quarterbeats="695/4",
            duration_qb="0.0833333333333333",
            duration="1/48",
        )

        loader = Ms3Loader.from_file(path)
        row = list(getattr(loader.store, store_attribute))[0]

        assert _exact_pair(row["start"]) == (695, 4)
        assert _exact_pair(row["duration"]) == (1, 12)
        assert _exact_pair(row["end"]) == (1043, 6)

    def test_harmony_label_spans_come_from_the_next_label(self, tmp_path):
        """A label runs to the next one, exactly, and the last to the piece end.

        A harmonies facet states no symbolic duration. Its ``duration_qb`` is
        this same subtraction done in float, residue and all --
        ``0.16666666666668561`` is not even the nearest double to ``1/6``.
        Redoing it on the exact ``quarterbeats`` cells reads the value the
        source already encodes rather than guessing one.
        """
        path = _write_label_facet_with_measures(
            tmp_path,
            labels=[
                ("1003/2", "0.16666666666668561"),
                ("1505/3", "0.3333333333333144"),
                ("502", "2.0"),
            ],
            measure_end=("500", "1"),
        )

        rows = list(Ms3Loader.from_file(path).store.annotations)

        # Each span is the next label's own position, minus this one's.
        assert _exact_pair(rows[0]["duration"]) == (1, 6)
        assert _exact_pair(rows[0]["end"]) == (1505, 3)
        assert _exact_pair(rows[1]["duration"]) == (1, 3)
        assert _exact_pair(rows[1]["end"]) == (502, 1)

        # The float column would have given a 55-bit dyadic instead.
        assert Fraction(*_exact_pair(rows[0]["duration"])) != Fraction(
            0.16666666666668561
        )

        # The last label has no successor: 500 + 1x4 = 504 from the measures.
        assert _exact_pair(rows[2]["duration"]) == (2, 1)
        assert _exact_pair(rows[2]["end"]) == (504, 1)

    def test_harmony_label_spans_fall_back_without_a_measures_facet(self, tmp_path):
        """With no companion measures, the last label keeps the derived column.

        The span of the final label is unknowable from the harmonies facet
        alone, so the loader falls back rather than inventing an end. Earlier
        labels still take their exact spans from their successors.
        """
        path = tmp_path / "lonely.harmonies.tsv"
        path.write_text(
            "mc\tmn\tquarterbeats\tduration_qb\tlabel\n"
            "1\t1\t1003/2\t0.16666666666668561\ti\n"
            "1\t1\t1505/3\t0.5\ti\n",
            encoding="utf-8",
        )

        rows = list(Ms3Loader.from_file(path).store.annotations)

        assert _exact_pair(rows[0]["duration"]) == (1, 6)
        assert _exact_pair(rows[1]["duration"]) == (1, 2)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("corpus_name", "expected_facets"), [("score", 46), ("supra", 4)]
    )
    def test_every_corpus_annotation_facet_loads(self, corpus_name, expected_facets):
        """All chords and harmonies files in a corpus load.

        A sweep rather than a specimen on purpose: the exact-duration defect
        survived a green suite because the files it broke were in no test's
        load set, and any hand-picked replacement would leave the same gap.
        ``supra`` carries the only files whose label spans are large enough
        for the float column to overflow the coordinate struct.
        """
        corpus = Path(ensure_data(corpus_name))
        facets = sorted(
            path
            for path in corpus.rglob("*.tsv")
            if ".chords." in path.name or ".harmonies." in path.name
        )

        assert len(facets) == expected_facets

        failures = []
        for path in facets:
            try:
                Ms3Loader.from_file(path)
            except Exception as error:  # noqa: BLE001 - reported, not swallowed
                failures.append(f"{path.relative_to(corpus)}: {error}")

        assert failures == []

    def test_woo71_note_coordinates_are_exact(self):
        """Populated WoO71 note coordinates carry exact pairs."""
        rows = list(Ms3Loader.from_file(WOO71_NOTES).store.notes)

        assert len(rows) == 4753
        for row in rows:
            if row["temporal_type"] == "instant":
                assert row["end"] is None
                assert row["duration"] is None
                continue
            for field in ("start", "duration", "end"):
                _exact_pair(row[field])

    def test_pitch_schema(self, chopin_tsv_notes):
        """Pitch is represented once: specific_pitch default + raw midi int.

        A spelled score faithfully supports SpecificPitch, so
        ``specific_pitch`` ({step, alter, octave, cents}) is the sole
        default semantic pitch field.  The source MIDI number survives
        only as a non-default raw ``midi`` int column, which affords an
        EnharmonicPitch view on request.  No ``midi_pitch`` struct exists.
        """
        from timetoalign.core.events import EnharmonicPitch, EnharmonicPitchField

        loader = Ms3Loader()
        loader.load(chopin_tsv_notes)
        store = loader.store
        first = list(store.notes)[0]

        # Represent-once: the redundant midi_pitch struct is gone.
        assert "midi_pitch" not in store.notes.table.column_names

        # SpecificPitch is the default semantic pitch field.
        sp = first.get("specific_pitch")
        assert sp is not None
        assert sp.get("step") == "B"  # B3

        # The source MIDI number is a raw int column.
        assert first.get("midi") == 59  # B3

        # EnharmonicPitch is afforded over the raw ``midi`` column.
        ep = store.notes.get_field(EnharmonicPitch)
        assert isinstance(ep, EnharmonicPitchField)
        assert ep.name == "midi"
        assert ep[0] == EnharmonicPitch(midi_number=59)
        # The convenience property routes through the same affordance.
        assert store.notes.enharmonic_pitch_field[0] == EnharmonicPitch(midi_number=59)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestPartituraLoader:
    """Tests for PartituraLoader."""

    def test_measure_count(self, chopin_xml):
        """Partitura extracts measures."""
        loader = PartituraLoader()
        loader.load(chopin_xml)
        store = loader.store
        assert len(store.measures) == 22

    @pytest.mark.slow
    def test_midi_loads_without_type_error(self):
        """Regression: MIDI files must load without TypeError (#18).

        Loading beethoven_op18.mid previously raised
        ``TypeError: Cannot add coordinates with different units: ticks vs seconds``
        due to the use of ``beat_map`` instead of ``quarter_map``.
        """
        midi_path = MIDI_SCORE_DIR / "beethoven_op18.mid"
        loader = PartituraLoader()
        loader.load(midi_path)
        assert len(loader.store.notes) == 4186


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestCrossValidation:
    """Cross-validation tests comparing all loaders.

    Note: note_count and pitch cross-validation are handled by the dedicated
    test_cross_validation.py module (TestNoteCountConsistency, TestMidiPitchExact).
    """

    def test_mc_onset_populated(self, chopin_xml, chopin_tsv_notes):
        """mc_onset is populated for all loaders."""
        l1 = Ms3Loader()
        l1.load(chopin_tsv_notes)
        tsv = l1.store

        l2 = PartituraLoader()
        l2.load(chopin_xml)
        pt = l2.store

        l3 = Music21Loader()
        l3.load(chopin_xml)
        m21 = l3.store

        for store, name in [(tsv, "TSV"), (pt, "Partitura"), (m21, "Music21")]:
            first = list(store.notes)[0]
            mc_onset = first.get("mc_onset")
            assert mc_onset is not None, f"{name}: mc_onset is None"
            assert "numerator" in mc_onset, f"{name}: mc_onset missing numerator"


# ---------------------------------------------------------------------------
# Anacrusis offset tests
# ---------------------------------------------------------------------------


def _raw_min_onset_partitura(xml_path: Path) -> float:
    """Return the minimum raw quarter-beat onset from partitura directly."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import partitura as pt

        score = pt.load_score(str(xml_path), force_note_ids=True)
        parts = score.parts if hasattr(score, "parts") else [score]
        min_onset = float("inf")
        for part in parts:
            bm = part.quarter_map
            for obj in part.iter_all(include_subclasses=True):
                if hasattr(obj, "midi_pitch"):
                    onset = float(bm(obj.start.t))
                    if onset < min_onset:
                        min_onset = onset
        return min_onset if min_onset != float("inf") else 0.0


def _first_note_duration_float_partitura(store: ScoreStore) -> float | None:
    """Return the duration float value of the first note row (proxy for coord presence)."""
    first = list(store.notes)[0] if store.notes else None
    if first is None:
        return None
    dur = first.get("duration")
    if dur is None:
        return None
    return dur.get("value") if isinstance(dur, dict) else None


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestAnacrusisOffset:
    """anacrusis_offset property and coordinate normalisation.

    Chopin Op. 10 No. 3 has a single anacrusis note with raw partitura onset
    of -0.5 quarter beats.  After normalisation:
    - ``loader.anacrusis_offset`` must equal ``-min(raw_onset)``
    - ``store.anacrusis_offset`` must equal the loader property
    - ``store.metadata["anacrusis_offset"]`` must equal the loader property

    A freshly constructed (not-yet-loaded) loader must report offset == 0.0.
    An empty ScoreStore must report offset == 0.0.
    """

    def test_partitura_offset_equals_minus_raw_min(self, chopin_xml):
        """PartituraLoader.anacrusis_offset == -min(raw partitura onset)."""
        loader = PartituraLoader()
        loader.load(chopin_xml)
        raw_min = _raw_min_onset_partitura(chopin_xml)
        assert raw_min == -0.5
        assert loader.anacrusis_offset == 0.5

    def test_partitura_offset_nonzero_for_chopin(self, chopin_xml):
        """Chopin Op.10/3 has an anacrusis: offset must be > 0."""
        loader = PartituraLoader()
        loader.load(chopin_xml)
        assert (
            loader.anacrusis_offset > 0.0
        ), "Expected anacrusis for Chopin Op.10/3 but offset == 0"

    def test_partitura_anacrusis_offset_in_metadata(self, chopin_xml):
        """anacrusis_offset is stored in ScoreStore.metadata."""
        loader = PartituraLoader()
        loader.load(chopin_xml)
        store = loader.store
        assert "anacrusis_offset" in store.metadata
        assert store.metadata["anacrusis_offset"] == 0.5

    def test_partitura_store_property_matches_loader(self, chopin_xml):
        """ScoreStore.anacrusis_offset reads from metadata and matches loader."""
        loader = PartituraLoader()
        loader.load(chopin_xml)
        assert loader.store.anacrusis_offset == 0.5

    def test_partitura_offset_is_exact(self, chopin_xml):
        """The offset is the exact ratio of the exact onsets, everywhere it is read."""
        loader = PartituraLoader()
        loader.load(chopin_xml)
        for offset in (
            loader.anacrusis_offset,
            loader.store.anacrusis_offset,
            loader.store.metadata["anacrusis_offset"],
        ):
            assert offset == Fraction(1, 2)
            assert type(offset) is Fraction

    def test_raw_quarters_shift_answers_exactly(self, chopin_xml):
        """The anacrusis ShiftMap takes an exact offset, so it answers exactly."""
        loader = PartituraLoader()
        loader.load(chopin_xml)
        shift = loader.store.create_timeline().get_conversion_map("raw_quarters")
        assert shift.offset == Fraction(-1, 2)
        first, second = shift(Fraction(1, 2)), shift(Fraction(3, 2))
        assert (first, second) == (Fraction(0), Fraction(1))
        assert type(first) is Fraction and type(second) is Fraction

    def test_exact_offset_survives_parquet(self, chopin_xml, tmp_path):
        """metadata.json writes the offset as a Fraction pair and reads it back."""
        loader = PartituraLoader()
        loader.load(chopin_xml)
        loader.store.to_parquet(tmp_path)
        written = json.loads((tmp_path / "metadata.json").read_text())
        assert written["anacrusis_offset"] == [1, 2]
        offset = ScoreStore.from_parquet(tmp_path).anacrusis_offset
        assert offset == Fraction(1, 2)
        assert type(offset) is Fraction

    def test_partitura_raw_plus_offset_is_zero(self, chopin_xml):
        """raw_min_onset + anacrusis_offset == 0 (the contract)."""
        loader = PartituraLoader()
        loader.load(chopin_xml)
        raw_min = _raw_min_onset_partitura(chopin_xml)
        assert raw_min + loader.anacrusis_offset == 0.0

    def test_music21_anacrusis_offset_in_metadata(self, chopin_xml):
        """Music21 stores anacrusis_offset in ScoreStore.metadata."""
        loader = Music21Loader()
        loader.load(chopin_xml)
        store = loader.store
        assert "anacrusis_offset" in store.metadata
        assert store.metadata["anacrusis_offset"] == 0.0

    def test_music21_store_property_matches_loader(self, chopin_xml):
        """Music21 store.anacrusis_offset reads from metadata and matches loader."""
        loader = Music21Loader()
        loader.load(chopin_xml)
        assert loader.store.anacrusis_offset == 0.0

    def test_music21_zero_offset_for_chopin(self, chopin_xml):
        """Music21 reports 0.0 offset for Chopin Op.10/3.

        Music21 already places the anacrusis note at offset 0.0 (it treats
        the start of the anacrusis beat as the timeline origin, not the
        downbeat of measure 1).  Therefore no shift is needed and
        anacrusis_offset == 0.0 is correct for music21.

        Note: this means PartituraLoader and Music21Loader produce different
        TTA coordinates for anacrusis notes (partitura: 0.5; music21: 0.0).
        ``MatchfileLoader`` must always use the score timeline's own
        ``anacrusis_offset`` — obtained from the same loader that built it —
        rather than assuming a common origin.
        """
        loader = Music21Loader()
        loader.load(chopin_xml)
        assert loader.anacrusis_offset == 0.0
        assert type(loader.anacrusis_offset) is Fraction

    def test_default_offset_before_loading(self):
        """Freshly constructed loaders report anacrusis_offset == 0.0."""
        assert PartituraLoader().anacrusis_offset == 0.0
        assert Music21Loader().anacrusis_offset == 0.0

    def test_store_default_offset_zero(self):
        """An empty ScoreStore reports anacrusis_offset == 0.0."""
        assert ScoreStore.empty().anacrusis_offset == 0.0

    def test_offset_propagates_to_measure_coordinates(self, chopin_xml):
        """Measures are shifted by the same offset as the notes.

        The MusicXML's measure 1 holds one note of 8 divisions at 16 per
        quarter: an anacrusis of 1/2 quarter. Shifted by the offset 1/2, it
        spans [0, 1/2), and measure 2 starts on the first downbeat at 1/2.
        """
        loader = PartituraLoader()
        loader.load(chopin_xml)

        starts = loader.store.measures.table.column("start").to_pylist()[:2]
        assert [
            Fraction(cell["numerator"], cell["denominator"]) for cell in starts
        ] == [Fraction(0), Fraction(1, 2)]


# region Parquet Serialization Tests


class TestTicksResolution:
    """A score states no tick resolution; a ticks C-Map comes only from ``ppq``."""

    def test_ms3_measures_table_has_no_ticks_map(self):
        """A DCML measures table counts in quarters and names no PPQ."""
        timeline = Ms3Loader.from_file(WOO71_MEASURES).create_timeline()
        assert timeline.get_conversion_map(TimeUnit.ticks) is None

    def test_partitura_musicxml_has_no_ticks_map(self, chopin_xml):
        """A MusicXML file's divisions are no ticks axis: the store adds none."""
        loader = PartituraLoader()
        loader.load(chopin_xml)
        timeline = loader.store.create_timeline()
        assert timeline.get_conversion_map(TimeUnit.ticks) is None

    def test_empty_store_derives_no_map(self):
        """Nothing is derived from an empty store, ticks included."""
        assert ScoreStore.empty().get_cmaps() == {}
        assert list(ScoreStore.empty().get_cmaps(ppq=960)) == ["ticks"]

    def test_named_resolution_is_exact(self):
        """``ppq=960`` attaches exactly the quarters -> ticks scalar 960."""
        store = Ms3Loader.from_file(WOO71_MEASURES).store
        ticks = store.create_timeline(ppq=960).get_conversion_map(TimeUnit.ticks)
        assert type(ticks) is ScalarMap
        assert (ticks.scalar, ticks.source_unit, ticks.target_unit) == (
            960,
            TimeUnit.quarters,
            TimeUnit.ticks,
        )
        two_quarters, a_third = ticks(2), ticks(Fraction(1, 3))
        assert (two_quarters, a_third) == (1920, 320)
        assert type(two_quarters) is int and type(a_third) is int

    @pytest.mark.parametrize("source", ["measures", "notes"])
    def test_the_loader_passes_ppq_on(self, source, chopin_tsv_notes):
        """Both branches of ``Ms3Loader.create_timeline`` honour ``ppq``."""
        path = WOO71_MEASURES if source == "measures" else chopin_tsv_notes
        timeline = Ms3Loader.from_file(path).create_timeline(ppq=960)
        maps = timeline._get_unit_maps(TimeUnit.ticks)
        assert len(maps) == 1
        ticks = maps[0]
        assert type(ticks) is ScalarMap
        assert (ticks.scalar, ticks.source_unit) == (960, TimeUnit.quarters)

    @pytest.mark.parametrize("source", ["measures", "notes"])
    def test_the_loader_passes_attach_cmaps_on(self, source, chopin_tsv_notes):
        """``attach_cmaps=False`` attaches no map, even with a resolution named."""
        path = WOO71_MEASURES if source == "measures" else chopin_tsv_notes
        loader = Ms3Loader.from_file(path)
        timeline = loader.create_timeline(ppq=960, attach_cmaps=False)
        assert timeline.n_conversion_maps == 0
        if source == "measures":
            assert Ms3Loader.from_file(path).create_timeline().n_conversion_maps == 1


class TestScoreStoreParquet:
    """Round-trip tests for ScoreStore parquet serialisation."""

    def test_to_parquet_creates_directory(self, chopin_tsv_notes, tmp_path):
        """to_parquet creates the output directory and writes facet files."""
        loader = Ms3Loader()
        loader.load(chopin_tsv_notes)
        store = loader.store

        out_dir = tmp_path / "score_out"
        store.to_parquet(out_dir)

        assert out_dir.is_dir()
        assert (out_dir / "notes.parquet").exists()
        assert (out_dir / "metadata.json").exists()

    def test_round_trip_note_count(self, chopin_tsv_notes, tmp_path):
        """Notes survive a round-trip through parquet."""
        loader = Ms3Loader()
        loader.load(chopin_tsv_notes)
        original = loader.store

        out_dir = tmp_path / "score_rt"
        original.to_parquet(out_dir)
        restored = ScoreStore.from_parquet(out_dir)

        assert len(restored.notes) == len(original.notes)

    def test_round_trip_measure_count(self, chopin_tsv_notes, tmp_path):
        """Measures survive a round-trip through parquet."""
        loader = Ms3Loader()
        loader.load(chopin_tsv_notes)
        original = loader.store

        out_dir = tmp_path / "score_rt"
        original.to_parquet(out_dir)
        restored = ScoreStore.from_parquet(out_dir)

        assert len(restored.measures) == len(original.measures)

    def test_round_trip_metadata(self, chopin_tsv_notes, tmp_path):
        """Store-level metadata survives a round-trip."""
        loader = Ms3Loader()
        loader.load(chopin_tsv_notes)
        original = loader.store

        out_dir = tmp_path / "score_rt"
        original.to_parquet(out_dir)
        restored = ScoreStore.from_parquet(out_dir)

        assert restored.metadata.get("parser") == original.metadata.get("parser")

    def test_round_trip_unit_preserved(self, chopin_tsv_notes, tmp_path):
        """Unit metadata is preserved through the round-trip."""
        loader = Ms3Loader()
        loader.load(chopin_tsv_notes)
        original = loader.store

        out_dir = tmp_path / "score_rt"
        original.to_parquet(out_dir)
        restored = ScoreStore.from_parquet(out_dir)

        assert restored.notes.unit == original.notes.unit

    def test_empty_facets_not_written(self, chopin_tsv_notes, tmp_path):
        """Empty facets are omitted from the directory."""
        loader = Ms3Loader()
        loader.load(chopin_tsv_notes)
        store = loader.store

        # Ms3Loader loading only notes should have no controls/annotations
        out_dir = tmp_path / "sparse_out"
        store.to_parquet(out_dir)

        # Controls/annotations are empty, so their files should not exist
        if len(store.controls) == 0:
            assert not (out_dir / "controls.parquet").exists()
        if len(store.annotations) == 0:
            assert not (out_dir / "annotations.parquet").exists()

    def test_from_parquet_missing_dir_raises(self, tmp_path):
        """from_parquet raises FileNotFoundError for a missing directory."""
        with pytest.raises(FileNotFoundError):
            ScoreStore.from_parquet(tmp_path / "nonexistent")

    def test_from_parquet_missing_facets_returns_empty(self, tmp_path):
        """from_parquet returns empty facets when files are absent."""
        import json

        out_dir = tmp_path / "partial"
        out_dir.mkdir()
        (out_dir / "metadata.json").write_text(json.dumps({}))

        store = ScoreStore.from_parquet(out_dir)
        assert len(store.notes) == 0
        assert len(store.measures) == 0


class TestScoreLoaderParquet:
    """Tests for ScoreLoader.to_parquet / from_parquet overrides."""

    def test_loader_to_parquet_non_empty(self, chopin_tsv_notes, tmp_path):
        """ScoreLoader.to_parquet writes non-empty files (the original bug)."""
        loader = Ms3Loader()
        loader.load(chopin_tsv_notes)

        out_dir = tmp_path / "loader_out"
        loader.to_parquet(out_dir)

        # The file must hold every note (the bug was that it was empty):
        # one row per row of the notes TSV.
        assert pq.read_table(out_dir / "notes.parquet").num_rows == 498

    def test_loader_round_trip(self, chopin_tsv_notes, tmp_path):
        """ScoreLoader.from_parquet restores a usable loader."""
        loader = Ms3Loader()
        loader.load(chopin_tsv_notes)
        original_count = len(loader.store.notes)

        out_dir = tmp_path / "loader_rt"
        loader.to_parquet(out_dir)
        restored = Ms3Loader.from_parquet(out_dir)

        assert isinstance(restored, Ms3Loader)
        assert len(restored.store.notes) == original_count
        assert len(restored.events) == original_count

    def test_loader_round_trip_events_property(self, chopin_tsv_notes, tmp_path):
        """Restored loader's .events property returns notes (not empty)."""
        loader = Ms3Loader()
        loader.load(chopin_tsv_notes)

        out_dir = tmp_path / "events_rt"
        loader.to_parquet(out_dir)
        restored = Ms3Loader.from_parquet(out_dir)

        # This is the critical assertion: .events must NOT be empty
        assert len(restored.events) == 498  # Chopin Op.10 No.3 gold standard


# endregion
