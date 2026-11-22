"""Tests for ``SemanticFieldAccessMixin`` and domain mixins.

The paired Object/ObjectField API uses one paired
``XField(SemanticField[X])`` per scalar. These tests verify the mixin dispatch
on the paired classes.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from timetoalign.core.events import (
    DcmlHarmonyField,
    EnharmonicPitchField,
    SpecificPitch,
    SpecificPitchField,
    WesternTertianHarmonyField,
)
from timetoalign.core.fields import (
    TIMETOALIGN_METADATA_KEY,
    metadata_blob_from_dict,
)
from timetoalign.storage.mixins import (
    HarmonyAccessMixin,
    SemanticFieldAccessMixin,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inject_metadata(pa_field: pa.Field, field_type: str, **extra: str) -> pa.Field:
    """Return a copy of *pa_field* with timetoalign metadata injected."""
    blob = metadata_blob_from_dict({"field_type": field_type, **extra})
    existing = pa_field.metadata or {}
    merged = {**existing, TIMETOALIGN_METADATA_KEY: blob}
    return pa_field.with_metadata(merged)


def _make_midi_pitch_table() -> pa.Table:
    """Build a table with a midi_pitch column (EnharmonicPitchField metadata)."""
    schema = EnharmonicPitchField.pa_schema
    arr = pa.array([{"midi_number": 60}, {"midi_number": 64}], type=schema)
    col_field = _inject_metadata(pa.field("midi_pitch", schema), "EnharmonicPitchField")
    return pa.table({"midi_pitch": arr}, schema=pa.schema([col_field]))


def _make_specific_pitch_table() -> pa.Table:
    """Build a table with a specific_pitch column (SpecificPitchField metadata)."""
    schema = SpecificPitchField.pa_schema
    arr = pa.array(
        [{"step": "C", "alter": 0, "octave": 4, "cents": 0.0}],
        type=schema,
    )
    col_field = _inject_metadata(
        pa.field("specific_pitch", schema), "SpecificPitchField"
    )
    return pa.table({"specific_pitch": arr}, schema=pa.schema([col_field]))


def _make_both_pitch_table() -> pa.Table:
    """Build a table with both midi_pitch and specific_pitch columns."""
    ep_schema = EnharmonicPitchField.pa_schema
    midi_arr = pa.array([{"midi_number": 60}], type=ep_schema)
    midi_field = _inject_metadata(
        pa.field("midi_pitch", ep_schema), "EnharmonicPitchField"
    )

    sp_schema = SpecificPitchField.pa_schema
    specific_arr = pa.array(
        [{"step": "C", "alter": 0, "octave": 4, "cents": 0.0}], type=sp_schema
    )
    specific_field = _inject_metadata(
        pa.field("specific_pitch", sp_schema), "SpecificPitchField"
    )

    schema = pa.schema([midi_field, specific_field])
    return pa.table(
        {"midi_pitch": midi_arr, "specific_pitch": specific_arr}, schema=schema
    )


def _make_dcml_harmony_table() -> pa.Table:
    """Build a table with a harmony column (DcmlHarmonyField metadata)."""
    schema = DcmlHarmonyField.pa_schema
    arr = pa.array(
        [
            {
                "label": "V65",
                "standard": "dcml",
                "root": 7,
                "bass": 11,
                "chord_type": "Mm7",
                "inversion": 1,
                "numeral": "V",
                "localkey": "I",
                "globalkey": "C",
                "tonicized_key": None,
                "pedal": None,
            }
        ],
        type=schema,
    )
    col_field = _inject_metadata(
        pa.field("harmony", schema), "DcmlHarmonyField", standard="dcml"
    )
    return pa.table({"harmony": arr}, schema=pa.schema([col_field]))


def _make_western_harmony_table() -> pa.Table:
    """Build a table with a Western tertian harmony column."""
    schema = WesternTertianHarmonyField.pa_schema
    arr = pa.array(
        [
            {
                "label": "CM",
                "standard": "chord_symbol",
                "root": 0,
                "bass": 0,
                "chord_type": "M",
                "inversion": 0,
            }
        ],
        type=schema,
    )
    col_field = _inject_metadata(
        pa.field("harmony", schema), "WesternTertianHarmonyField"
    )
    return pa.table({"harmony": arr}, schema=pa.schema([col_field]))


class _MixinHost(SemanticFieldAccessMixin):
    def __init__(self, table: pa.Table) -> None:
        self._table = table


class _HarmonyHost(HarmonyAccessMixin):
    def __init__(self, table: pa.Table) -> None:
        self._table = table


# ---------------------------------------------------------------------------
# SemanticFieldAccessMixin tests
# ---------------------------------------------------------------------------


class TestSemanticFieldAccessMixin:
    def test_get_field_returns_enharmonic(self) -> None:
        host = _MixinHost(_make_midi_pitch_table())
        result = host.get_field(EnharmonicPitchField)
        assert isinstance(result, EnharmonicPitchField)

    def test_get_field_returns_specific(self) -> None:
        host = _MixinHost(_make_specific_pitch_table())
        result = host.get_field(SpecificPitchField)
        assert isinstance(result, SpecificPitchField)

    def test_get_field_raises_on_no_match(self) -> None:
        table = pa.table({"x": pa.array([1, 2, 3])})
        host = _MixinHost(table)
        with pytest.raises(KeyError, match="No field matching"):
            host.get_field(EnharmonicPitchField)

    def test_get_fields_returns_all_matches(self) -> None:
        host = _MixinHost(_make_midi_pitch_table())
        results = host.get_fields(EnharmonicPitchField)
        assert len(results) == 1
        assert isinstance(results[0], EnharmonicPitchField)

    def test_get_fields_empty_on_no_match(self) -> None:
        table = pa.table({"x": pa.array([1, 2, 3])})
        host = _MixinHost(table)
        results = host.get_fields(EnharmonicPitchField)
        assert results == []

    def test_has_field_true(self) -> None:
        host = _MixinHost(_make_midi_pitch_table())
        assert host.has_field(EnharmonicPitchField) is True

    def test_has_field_false(self) -> None:
        table = pa.table({"x": pa.array([1, 2, 3])})
        host = _MixinHost(table)
        assert host.has_field(EnharmonicPitchField) is False

    def test_get_field_harmony(self) -> None:
        host = _MixinHost(_make_dcml_harmony_table())
        result = host.get_field(DcmlHarmonyField)
        assert isinstance(result, DcmlHarmonyField)

    def test_get_field_data_access(self) -> None:
        host = _MixinHost(_make_midi_pitch_table())
        field = host.get_field(EnharmonicPitchField)
        assert len(field) == 2
        pitch = field[0]
        assert pitch is not None
        assert pitch.midi_number == 60
        assert pitch.pitch_class == 0


# ---------------------------------------------------------------------------
# SemanticFieldAccessMixin pitch-access tests
# ---------------------------------------------------------------------------


class TestSemanticFieldPitchAccess:
    def test_get_pitch_field_with_type(self) -> None:
        host = _MixinHost(_make_midi_pitch_table())
        result = host.get_pitch_field(EnharmonicPitchField)
        assert isinstance(result, EnharmonicPitchField)

    def test_get_pitch_field_default_priority_sp(self) -> None:
        """When both EP and SP are present, default picks SP (most informative)."""
        host = _MixinHost(_make_both_pitch_table())
        result = host.get_pitch_field()
        assert isinstance(result, SpecificPitchField)

    def test_get_pitch_field_default_only_ep(self) -> None:
        host = _MixinHost(_make_midi_pitch_table())
        result = host.get_pitch_field()
        assert isinstance(result, EnharmonicPitchField)

    def test_get_pitch_field_default_only_sp(self) -> None:
        host = _MixinHost(_make_specific_pitch_table())
        result = host.get_pitch_field()
        assert isinstance(result, SpecificPitchField)

    def test_get_pitch_field_raises_no_pitch(self) -> None:
        table = pa.table({"x": pa.array([1])})
        host = _MixinHost(table)
        with pytest.raises(KeyError, match="No pitch field"):
            host.get_pitch_field()


# ---------------------------------------------------------------------------
# HarmonyAccessMixin tests
# ---------------------------------------------------------------------------


class TestHarmonyAccessMixin:
    def test_get_harmony_field_with_type(self) -> None:
        host = _HarmonyHost(_make_dcml_harmony_table())
        result = host.get_harmony_field(DcmlHarmonyField)
        assert isinstance(result, DcmlHarmonyField)

    def test_get_harmony_field_default_dcml(self) -> None:
        host = _HarmonyHost(_make_dcml_harmony_table())
        result = host.get_harmony_field()
        assert isinstance(result, DcmlHarmonyField)

    def test_get_harmony_field_default_western(self) -> None:
        host = _HarmonyHost(_make_western_harmony_table())
        result = host.get_harmony_field()
        assert isinstance(result, WesternTertianHarmonyField)

    def test_get_harmony_field_raises_no_harmony(self) -> None:
        table = pa.table({"x": pa.array([1])})
        host = _HarmonyHost(table)
        with pytest.raises(KeyError, match="No harmony field"):
            host.get_harmony_field()

    def test_get_harmony_field_data_access(self) -> None:
        host = _HarmonyHost(_make_dcml_harmony_table())
        field = host.get_harmony_field()
        assert len(field) == 1
        harmony = field[0]
        assert harmony is not None
        assert harmony.label == "V65"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Integration with EventData
# ---------------------------------------------------------------------------


class TestEventDataComposition:
    def test_note_event_data_has_pitch_access(self) -> None:
        from timetoalign.loader.score.stores.notes import NoteEventData

        assert hasattr(NoteEventData, "get_pitch_field")
        assert hasattr(NoteEventData, "has_field")
        assert hasattr(NoteEventData, "get_field")
        assert hasattr(NoteEventData, "get_fields")

    def test_annotation_event_data_has_harmony_access(self) -> None:
        from timetoalign.loader.score.stores.annotations import AnnotationEventData

        assert hasattr(AnnotationEventData, "get_harmony_field")
        assert hasattr(AnnotationEventData, "has_field")

    def test_note_event_data_enharmonic_pitch_field(self) -> None:
        from timetoalign.loader.score.stores.notes import NoteEventData

        # Represent once: NoteEventData affords EnharmonicPitch over the raw
        # ``midi`` int column (the source MIDI number), reached via the
        # ``enharmonic_pitch_field`` accessor.
        store = NoteEventData.from_dicts(
            [
                {
                    "event_type": "Note",
                    "start": 0.0,
                    "duration": 1.0,
                    "midi": 60,
                }
            ],
        )
        pf = store.enharmonic_pitch_field
        assert isinstance(pf, EnharmonicPitchField)
        assert pf[0] is not None
        assert pf[0].midi_number == 60

    def test_note_event_data_specific_pitch_field(self) -> None:
        from timetoalign.loader.score.stores.notes import NoteEventData

        store = NoteEventData.from_dicts(
            [
                {
                    "event_type": "Note",
                    "start": 0.0,
                    "duration": 1.0,
                    "specific_pitch": {
                        "step": "C",
                        "alter": 0,
                        "octave": 4,
                        "cents": 0.0,
                    },
                }
            ],
        )
        spf = store.specific_pitch_field
        assert isinstance(spf, SpecificPitchField)
        assert spf[0] == SpecificPitch(step="C", alter=0, octave=4, cents=0.0)

    def test_plain_event_data_has_field_access(self) -> None:
        from timetoalign.storage.events import EventData

        assert hasattr(EventData, "get_field")
        assert hasattr(EventData, "get_fields")
        assert hasattr(EventData, "has_field")

    def test_plain_event_data_get_field_with_metadata(self) -> None:
        from timetoalign.core import TimeUnit
        from timetoalign.storage.events import EventData

        store = EventData.from_dicts(
            [
                {
                    "event_type": "Note",
                    "start": 0.0,
                    "duration": 1.0,
                }
            ],
            unit=TimeUnit.seconds,
        )
        schema = EnharmonicPitchField.pa_schema
        pitch_arr = pa.array([{"midi_number": 60}], type=schema)
        col_field = _inject_metadata(
            pa.field("midi_pitch", schema), "EnharmonicPitchField"
        )
        new_table = store._table.append_column(col_field, pitch_arr)
        store._table = new_table

        result = store.get_field(EnharmonicPitchField)
        assert isinstance(result, EnharmonicPitchField)
        assert result[0] is not None
        assert result[0].midi_number == 60

    def test_plain_event_data_has_field_false_without_metadata(self) -> None:
        from timetoalign.core import TimeUnit
        from timetoalign.storage.events import EventData

        store = EventData.from_dicts(
            [
                {
                    "event_type": "Beat",
                    "start": 0.0,
                }
            ],
            unit=TimeUnit.seconds,
        )
        assert store.has_field(EnharmonicPitchField) is False


# ---------------------------------------------------------------------------
# Field cache tests
# ---------------------------------------------------------------------------


class TestFieldCache:
    def test_cache_returns_same_object(self) -> None:
        host = _MixinHost(_make_midi_pitch_table())
        result1 = host.get_field(EnharmonicPitchField)
        result2 = host.get_field(EnharmonicPitchField)
        assert result1 is result2

    def test_cache_independent_per_instance(self) -> None:
        table = _make_midi_pitch_table()
        host_a = _MixinHost(table)
        host_b = _MixinHost(table)

        field_a = host_a.get_field(EnharmonicPitchField)
        field_b = host_b.get_field(EnharmonicPitchField)

        assert isinstance(field_a, EnharmonicPitchField)
        assert isinstance(field_b, EnharmonicPitchField)
        assert field_a is not field_b

    def test_get_fields_populates_cache(self) -> None:
        host = _MixinHost(_make_both_pitch_table())
        first_call = host.get_fields(EnharmonicPitchField)
        second_call = host.get_fields(EnharmonicPitchField)
        assert len(first_call) == 1
        assert len(second_call) == 1
        assert first_call[0] is second_call[0]

    def test_has_field_after_get_field_uses_cache(self) -> None:
        host = _MixinHost(_make_midi_pitch_table())
        host.get_field(EnharmonicPitchField)
        assert host.has_field(EnharmonicPitchField) is True
