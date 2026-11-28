"""Tests for :class:`SoloLoader` against the Chopin Nocturne ``.solo`` specimen.

Validation logic is documented in ``tests/loader/tabular/README.md`` —
section *SoloLoader*.  Summary:

* Exact event count is 2494.
* Three sentinel rows are asserted exactly (rows 0, 1, 12 — pairing of
  note-on/note-off and a multi-voice chord).
* The composite ``position`` column splits into ``measure_number`` +
  ``mn_onset`` (default name + explicit name).
* ``pitch`` is decorated with ``EnharmonicPitchField`` metadata; the
  atomic int column is packed into the field's struct shape.
* ``note_id`` is decorated with ``IdField`` metadata; the string
  column is packed into ``{value: string}``.
* ``get_field(EnharmonicPitch)`` and ``get_field(Id)`` round-trip the
  paired scalar.
"""

from __future__ import annotations

import pytest

from timetoalign.core import (
    EnharmonicPitch,
    EnharmonicPitchField,
    Id,
    IdField,
)
from timetoalign.loader.tabular import SoloLoader
from timetoalign.testdata import ensure_data

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


SOLO_FILE = ensure_data("performance_precision") / "Chopin Nocturne Op. 9 No. 2.solo"


@pytest.fixture(scope="module")
def loader() -> SoloLoader:
    """Load the canonical Chopin Nocturne .solo specimen once per session."""
    return SoloLoader.from_file(SOLO_FILE)


# ---------------------------------------------------------------------------
# Row count + sentinel-row content
# ---------------------------------------------------------------------------


class TestSoloLoaderCounts:

    def test_event_count_is_exact(self, loader: SoloLoader) -> None:
        assert loader.events.count == 2494

    def test_the_locators(self, loader: SoloLoader) -> None:
        locators = loader.events.table.column("locator").to_pylist()
        assert [locators[i] for i in (0, 1, 12)] == [
            "row:0 key:n1b8xktz",
            "row:1 key:n1b8xktz",
            "row:12 key:n1fst1u3",
        ]
        assert len(set(locators)) == 2494

    def test_row_0_content(self, loader: SoloLoader) -> None:
        table = loader.events.table
        # measure_number: the printed label "0"; a .solo file states no
        # measure count and no volta, so both stay null.
        assert table["measure_number"][0].as_py() == {
            "rendition": None,
            "mc": None,
            "mn": "0",
            "volta": None,
            "section": None,
        }
        # mn_onset: 11/8
        mn = table["mn_onset"][0].as_py()
        assert mn["numerator"] == 11
        assert mn["denominator"] == 8
        # duration: 0/1 (note-off marker)
        dur = table["duration"][0].as_py()
        assert dur["numerator"] == 0
        assert dur["denominator"] == 1
        # channel, pitch, velocity, note_id
        assert table["channel"][0].as_py() == 90
        # pitch is now a {midi_number: int64} struct (field_specs packing).
        assert table["pitch"][0].as_py() == {"midi_number": 70}
        assert table["velocity"][0].as_py() == 80
        # note_id is now a {value: string} struct.
        assert table["note_id"][0].as_py() == {"value": "n1b8xktz"}

    def test_row_1_content(self, loader: SoloLoader) -> None:
        table = loader.events.table
        assert table["measure_number"][1].as_py() == {
            "rendition": None,
            "mc": None,
            "mn": "1",
            "volta": None,
            "section": None,
        }
        mn = table["mn_onset"][1].as_py()
        assert mn["numerator"] == 0
        assert mn["denominator"] == 1
        dur = table["duration"][1].as_py()
        assert dur["numerator"] == 1
        assert dur["denominator"] == 8
        assert table["pitch"][1].as_py() == {"midi_number": 70}
        assert table["velocity"][1].as_py() == 0
        assert table["note_id"][1].as_py() == {"value": "n1b8xktz"}

    def test_row_12_content(self, loader: SoloLoader) -> None:
        table = loader.events.table
        # Row 12 (zero-indexed): "1+3/8\t1/2\t90\t58\t0\tn1fst1u3"
        assert table["measure_number"][12].as_py() == {
            "rendition": None,
            "mc": None,
            "mn": "1",
            "volta": None,
            "section": None,
        }
        mn = table["mn_onset"][12].as_py()
        assert mn["numerator"] == 3
        assert mn["denominator"] == 8
        dur = table["duration"][12].as_py()
        assert dur["numerator"] == 1
        assert dur["denominator"] == 2
        assert table["pitch"][12].as_py() == {"midi_number": 58}
        assert table["velocity"][12].as_py() == 0
        assert table["note_id"][12].as_py() == {"value": "n1fst1u3"}


# ---------------------------------------------------------------------------
# column_specs structure: composite-part names + opaque struct
# ---------------------------------------------------------------------------


class TestSoloLoaderColumnSpecs:

    def test_top_level_columns_present(self, loader: SoloLoader) -> None:
        column_names = set(loader.events.table.column_names)
        # Canonical + step-1 fields all appear.
        assert {
            "id",
            "name",
            "temporal_type",
            "event_type",
            "start",
            "end",
            "duration",
            "channel",
            "pitch",
            "velocity",
            "note_id",
            "measure_number",
            "mn_onset",
            "position",
        }.issubset(column_names)

    def test_composite_position_struct(self, loader: SoloLoader) -> None:
        import pyarrow as pa

        table = loader.events.table
        # position remains as an opaque struct of {measure_number, mn_onset}
        assert pa.types.is_struct(table["position"].type)
        sub_field_names = [
            table["position"].type.field(i).name
            for i in range(table["position"].type.num_fields)
        ]
        assert sub_field_names == ["measure_number", "mn_onset"]


# ---------------------------------------------------------------------------
# field_specs metadata + get_field round-trip
# ---------------------------------------------------------------------------


class TestSoloLoaderFieldSpecs:

    def test_pitch_metadata_advertises_enharmonic_pitch_field(
        self, loader: SoloLoader
    ) -> None:
        meta = loader.events.table.schema.field("pitch").metadata
        assert meta is not None
        assert b"timetoalign" in meta
        assert b"EnharmonicPitchField" in meta[b"timetoalign"]

    def test_note_id_metadata_advertises_id_field(self, loader: SoloLoader) -> None:
        meta = loader.events.table.schema.field("note_id").metadata
        assert meta is not None
        assert b"timetoalign" in meta
        assert b"IdField" in meta[b"timetoalign"]

    def test_get_field_enharmonic_pitch(self, loader: SoloLoader) -> None:
        field = loader.events.get_field(EnharmonicPitch)
        assert isinstance(field, EnharmonicPitchField)
        assert len(field) == 2494
        # Row 0: MIDI 70 → A♯/B♭4
        assert field[0] == EnharmonicPitch(midi_number=70)
        # Row 12: MIDI 58 → A♯/B♭3
        assert field[12] == EnharmonicPitch(midi_number=58)

    def test_get_field_id(self, loader: SoloLoader) -> None:
        field = loader.events.get_field(Id)
        assert isinstance(field, IdField)
        assert len(field) == 2494
        assert field[0] == Id(value="n1b8xktz")
        assert field[1] == Id(value="n1b8xktz")
        assert field[2] == Id(value="nh9xux4")

    def test_get_field_measure_number_not_promoted_by_default(
        self, loader: SoloLoader
    ) -> None:
        # MeasureNumberField is NOT in field_specs on SoloLoader — the
        # measure_number column is produced by column_specs via the
        # paired-class emission helper, which already carries no semantic
        # decoration.  Verify the column shape is the lossless measure-label
        # struct: a source that states only the printed label fills only it.
        import pyarrow as pa

        table = loader.events.table
        mn_col = table["measure_number"]
        assert pa.types.is_struct(mn_col.type)
        sub_names = [mn_col.type.field(i).name for i in range(mn_col.type.num_fields)]
        assert sub_names == ["rendition", "mc", "mn", "volta", "section"]
