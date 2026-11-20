"""Tests for Step 2 (``field_specs``) blueprint resolution.

Validation logic is documented in ``tests/loader/README.md`` (the
``test_step2_field_specs.py`` row).  Summary:

* A purpose-built loader subclass loads a tiny in-memory CSV and
  declares one or more blueprint-mode :class:`SemanticField` instances
  via ``field_specs``.  We verify:

  - Each blueprint's resulting column carries ``b"timetoalign"``
    metadata with the correct ``field_type``.
  - Atomic source columns are packed into the target ``pa_schema``
    shape (e.g. ``midi_pitch: int64`` → ``{midi_number: int64}``).
  - ``EventData.get_field(<ScalarClass>)`` round-trips the resulting
    semantic field — element-wise materialisation yields the
    paired scalar.

* Error paths: live-mode SemanticField raises ``TypeError``; multi-
  source dict blueprint raises ``NotImplementedError``; unresolvable
  references raise ``KeyError``; a target class that cannot be built
  from a name alone raises a ``TypeError`` naming that target and its
  storage shape.

* The three ``source_fields=`` shorthands accepted by SemanticField
  (string / single-element list / explicit dict) are exercised at
  blueprint construction time (the loader currently only consumes
  the string form; the list and dict forms are documented for
  consistency).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from timetoalign.core import (
    CoordinateField,
    EnharmonicPitch,
    EnharmonicPitchField,
    Id,
    IdField,
    MeasureNumber,
    MeasureNumberField,
)
from timetoalign.loader.tabular import CsvLoader

# ---------------------------------------------------------------------------
# Fixture loader subclasses (declared inline so column_specs / field_specs
# come from the test class itself).
# ---------------------------------------------------------------------------


@pytest.fixture
def csv_file_pitches(tmp_path: Path) -> Path:
    """Tiny CSV with start, duration, midi_pitch, note_id columns."""
    path = tmp_path / "events.csv"
    path.write_text(
        "id,start,end,midi_pitch,note_id\n"
        "n1,0.0,1.0,60,n_a\n"
        "n2,1.0,2.0,62,n_b\n"
        "n3,2.0,3.0,64,n_c\n"
    )
    return path


@pytest.fixture
def csv_file_no_field_specs(tmp_path: Path) -> Path:
    """Tiny CSV with no Step-2 promotion candidates — for negative tests."""
    path = tmp_path / "events.csv"
    path.write_text("id,start,end\n" "n1,0.0,1.0\n" "n2,1.0,2.0\n")
    return path


# ---------------------------------------------------------------------------
# Happy path: string-shorthand source_fields
# ---------------------------------------------------------------------------


class TestFieldSpecsHappyPath:

    def test_atomic_pitch_promoted_to_enharmonic_pitch_field(
        self, csv_file_pitches: Path
    ) -> None:

        class PitchLoader(CsvLoader):
            id_column = "id"
            start_column = "start"
            end_column = "end"
            column_specs = {"midi_pitch": int, "note_id": str}
            field_specs = [EnharmonicPitchField(source_fields="midi_pitch")]

        loader = PitchLoader.from_file(csv_file_pitches)
        events = loader.events
        assert events.count == 3
        # Column dtype is now the target struct.
        import pyarrow as pa

        assert pa.types.is_struct(events.table["midi_pitch"].type)
        # Metadata carries the paired-class field_type.
        meta = events.table.schema.field("midi_pitch").metadata
        assert meta is not None
        assert b"timetoalign" in meta
        assert b"EnharmonicPitchField" in meta[b"timetoalign"]

    def test_get_field_round_trip(self, csv_file_pitches: Path) -> None:

        class PitchLoader(CsvLoader):
            id_column = "id"
            start_column = "start"
            end_column = "end"
            column_specs = {"midi_pitch": int, "note_id": str}
            field_specs = [
                EnharmonicPitchField(source_fields="midi_pitch"),
                IdField(source_fields="note_id"),
            ]

        loader = PitchLoader.from_file(csv_file_pitches)
        events = loader.events
        ep_field = events.get_field(EnharmonicPitch)
        assert isinstance(ep_field, EnharmonicPitchField)
        assert ep_field[0] == EnharmonicPitch(midi_number=60)
        assert ep_field[2] == EnharmonicPitch(midi_number=64)
        id_field = events.get_field(Id)
        assert isinstance(id_field, IdField)
        assert id_field[0] == Id(value="n_a")
        assert id_field[2] == Id(value="n_c")

    def test_measure_number_promotion(self, csv_file_pitches: Path) -> None:

        class MeasureNumberLoader(CsvLoader):
            id_column = "id"
            start_column = "start"
            end_column = "end"
            column_specs = {"midi_pitch": int}
            field_specs = [
                # Reuse "midi_pitch" as a measure-number for this synthetic test.
                MeasureNumberField(source_fields="midi_pitch"),
            ]

        loader = MeasureNumberLoader.from_file(csv_file_pitches)
        events = loader.events
        mn = events.get_field(MeasureNumber)
        assert mn[0] == MeasureNumber(mn="60")
        assert mn[2] == MeasureNumber(mn="64")


# ---------------------------------------------------------------------------
# Blueprint construction: three syntactic-sugar shorthands
# ---------------------------------------------------------------------------


class TestBlueprintShorthands:
    """Verify SemanticField's three accepted ``source_fields=`` shapes."""

    def test_string_shorthand(self) -> None:
        bp = EnharmonicPitchField(source_fields="midi_pitch")
        assert bp.is_blueprint is True
        # Stored as normalised dict on the blueprint.
        # (resolve_source_fields normalises a top-level string into a
        # bare string; we just confirm presence here.)
        assert bp._blueprint_source_fields == "midi_pitch"

    def test_list_shorthand_not_yet_supported(self) -> None:
        # The list-of-one shorthand is reserved for future expansion
        # (see the loader-spec).  Today, only the string and dict
        # forms are accepted by ``resolve_source_fields``.
        with pytest.raises(TypeError):
            EnharmonicPitchField(source_fields=["midi_pitch"])

    def test_dict_shorthand(self) -> None:
        bp = EnharmonicPitchField(source_fields={"midi_number": "midi_pitch"})
        assert bp.is_blueprint is True
        assert bp._blueprint_source_fields == {"midi_number": "midi_pitch"}


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


class TestFieldSpecsNegativePaths:

    def test_unknown_source_raises_key_error(self, csv_file_pitches: Path) -> None:

        class BadLoader(CsvLoader):
            id_column = "id"
            start_column = "start"
            end_column = "end"
            column_specs = {"midi_pitch": int}
            field_specs = [EnharmonicPitchField(source_fields="not_a_column")]

        loader = BadLoader()
        with pytest.raises(KeyError, match="not_a_column"):
            loader.load(csv_file_pitches)

    def test_live_mode_blueprint_rejected(self, csv_file_pitches: Path) -> None:
        # A live-mode (non-blueprint) SemanticField is not a valid Step-2
        # entry — the loader rejects it.
        import pyarrow as pa

        # Construct a live-mode EnharmonicPitchField (with data).
        struct_type = EnharmonicPitchField.pa_schema
        struct_arr = pa.StructArray.from_arrays(
            [pa.array([60, 62, 64], type=pa.int64())],
            fields=list(struct_type),
        )
        live = EnharmonicPitchField.from_field(
            (struct_arr, pa.field("midi_pitch", struct_type))
        )

        class BadLoader(CsvLoader):
            id_column = "id"
            start_column = "start"
            end_column = "end"
            column_specs = {"midi_pitch": int}
            field_specs = [live]

        loader = BadLoader()
        with pytest.raises(TypeError, match="blueprint"):
            loader.load(csv_file_pitches)

    def test_non_semantic_field_entry_rejected(self, csv_file_pitches: Path) -> None:

        class BadLoader(CsvLoader):
            id_column = "id"
            start_column = "start"
            end_column = "end"
            column_specs = {"midi_pitch": int}
            field_specs = ["not_a_semantic_field"]  # type: ignore[list-item]

        loader = BadLoader()
        with pytest.raises(TypeError, match="SemanticField"):
            loader.load(csv_file_pitches)

    def test_dict_blueprint_not_yet_implemented(self, csv_file_pitches: Path) -> None:
        # The loader currently supports only the string shorthand.
        # A blueprint constructed with a dict spec should raise on
        # materialisation.
        bp = EnharmonicPitchField(source_fields={"midi_number": "midi_pitch"})

        class LoaderWithDictBlueprint(CsvLoader):
            id_column = "id"
            start_column = "start"
            end_column = "end"
            column_specs = {"midi_pitch": int}
            field_specs = [bp]

        loader = LoaderWithDictBlueprint()
        with pytest.raises(NotImplementedError, match="source_fields"):
            loader.load(csv_file_pitches)

    def test_target_without_a_packing_rule_names_itself(
        self, csv_file_pitches: Path
    ) -> None:
        """A plain column cannot fill a target that needs more than a name.

        ``CoordinateField`` carries a bound time unit, so there is no way
        to ask it to read a bare source column. The loader says which
        target it could not pack into and what shape that target stores,
        instead of letting the field's own constructor fail on an
        argument the caller never passed.
        """

        class BadLoader(CsvLoader):
            id_column = "id"
            start_column = "start"
            end_column = "end"
            column_specs = {"midi_pitch": int}
            field_specs = [CoordinateField(source_fields="midi_pitch")]

        loader = BadLoader()
        with pytest.raises(TypeError, match="CoordinateField.*no packing rule"):
            loader.load(csv_file_pitches)


# ---------------------------------------------------------------------------
# Field-specs sequence vs dict form
# ---------------------------------------------------------------------------


class TestFieldSpecsContainerShapes:

    def test_sequence_form(self, csv_file_pitches: Path) -> None:

        class L(CsvLoader):
            id_column = "id"
            start_column = "start"
            end_column = "end"
            column_specs = {"midi_pitch": int, "note_id": str}
            field_specs = [
                EnharmonicPitchField(source_fields="midi_pitch"),
                IdField(source_fields="note_id"),
            ]

        loader = L.from_file(csv_file_pitches)
        events = loader.events
        assert events.get_field(EnharmonicPitch) is not None
        assert events.get_field(Id) is not None

    def test_dict_form(self, csv_file_pitches: Path) -> None:

        class L(CsvLoader):
            id_column = "id"
            start_column = "start"
            end_column = "end"
            column_specs = {"midi_pitch": int, "note_id": str}
            field_specs = {
                "pitch_slot": EnharmonicPitchField(source_fields="midi_pitch"),
                "id_slot": IdField(source_fields="note_id"),
            }

        loader = L.from_file(csv_file_pitches)
        events = loader.events
        assert events.get_field(EnharmonicPitch) is not None
        assert events.get_field(Id) is not None
