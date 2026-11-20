"""Represent-pitch-once contract across EventData and loaders.

Pitch is a property of a note event and is **represented exactly once** —
by the single most-expressive semantic pitch type the source *faithfully*
supports.  Poorer / derived views are reached on request (conversion from
the expressive field, or a non-default raw column promoted via the
``_afforded_fields`` mechanism), never stored a second time.

These tests pin, with exact expected values:

1. the keystone ``from_dicts`` / ``add_events`` struct-preservation fix
   (carried struct-dict columns become real ``pa.struct`` columns with
   ``field_type`` metadata, never JSON strings);
2. the uniform ``_afforded_fields`` promotion of a raw atomic column to
   its semantic view on request;
3. the per-source default pitch type via ``get_pitch_field()``;
4. represent-once (no redundant *default* pitch struct);
5. the scalar↔EventData contract for the MIDI EventData classes;
6. on-request EnharmonicPitch from a SpecificPitch field via both routes;
7. the multi-batch concat re-affordance — the afforded pitch view
   survives ``EventData.extend`` / ``Timeline.add_events`` schema
   promotion across batches (raw column stays ``int64``; the field cache
   is dropped so the affordance re-attaches over the concatenated table).
8. blueprint resolution uses the same raw-column promotion as class lookup,
   including the Vienna Chopin notes fixture and its exact first MIDI value.

The validation logic is documented in ``tests/loader/README.md`` under
"Represent pitch once".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

from timetoalign.core import NumberType, TimeUnit
from timetoalign.core.events import (
    EnharmonicPitch,
    EnharmonicPitchClass,
    EnharmonicPitchField,
    MidiPitch,
    MidiPitchField,
    SpecificPitch,
    SpecificPitchClass,
    SpecificPitchClassField,
    SpecificPitchField,
)
from timetoalign.core.fields import TIMETOALIGN_METADATA_KEY, parse_metadata_blob
from timetoalign.loader.midi.events import MidiEventData, ScoreMidiEventData
from timetoalign.loader.score.ms3 import Ms3Loader
from timetoalign.loader.score.stores.notes import NoteEventData
from timetoalign.storage.events import EventData
from timetoalign.testdata import ensure_data

if TYPE_CHECKING:
    from timetoalign.timelines.base import Timeline


def _field_type_meta(table: pa.Table, name: str) -> str | None:
    """Return the ``field_type`` recorded in a column's timetoalign blob."""
    meta = table.schema.field(name).metadata
    if not meta or TIMETOALIGN_METADATA_KEY not in meta:
        return None
    return parse_metadata_blob(meta[TIMETOALIGN_METADATA_KEY]).get("field_type")


# ---------------------------------------------------------------------------
# 1. Keystone: from_dicts / add_events preserve carried struct affordance
# ---------------------------------------------------------------------------


class TestKeystoneStructPreservation:
    """``from_dicts`` must rebuild struct-dict columns as real structs."""

    def test_pitch_dict_becomes_struct_not_string(self) -> None:
        data = EventData.from_dicts(
            [
                {"event_type": "Note", "start": 0.0, "pitch": {"midi_number": 60}},
                {"event_type": "Note", "start": 1.0, "pitch": {"midi_number": 62}},
            ],
            TimeUnit.seconds,
        )
        assert data.schema.field("pitch").type == pa.struct(
            [pa.field("midi_number", pa.int64())]
        )
        assert data.table.column("pitch").to_pylist() == [
            {"midi_number": 60},
            {"midi_number": 62},
        ]

    def test_known_struct_shape_gets_field_type_metadata(self) -> None:
        data = EventData.from_dicts(
            [{"event_type": "Note", "start": 0.0, "pitch": {"midi_number": 60}}],
            TimeUnit.seconds,
        )
        assert _field_type_meta(data.table, "pitch") == "EnharmonicPitchField"

    def test_carried_struct_keys_reordered_to_canonical(self) -> None:
        # pa.array infers struct fields alphabetically ({alter, step}); the
        # loader must reorder to the paired class's canonical {step, alter}.
        data = EventData.from_dicts(
            [{"event_type": "Note", "start": 0.0, "spc": {"alter": -1, "step": "E"}}],
            TimeUnit.seconds,
        )
        assert data.schema.field("spc").type == pa.struct(
            [pa.field("step", pa.string()), pa.field("alter", pa.int64())]
        )
        assert _field_type_meta(data.table, "spc") == "SpecificPitchClassField"
        assert data.get_field(SpecificPitchClass)[0] == SpecificPitchClass(
            step="E", alter=-1
        )

    def test_struct_column_round_trips_to_scalar(self) -> None:
        data = EventData.from_dicts(
            [{"event_type": "Note", "start": 0.0, "pitch": {"midi_number": 67}}],
            TimeUnit.seconds,
        )
        assert data.get_field(EnharmonicPitch)[0] == EnharmonicPitch(midi_number=67)

    def test_unknown_struct_shape_stays_plain_struct(self) -> None:
        # A struct that matches no paired SemanticField is preserved as a
        # plain struct (no field_type metadata, no masquerading).
        data = EventData.from_dicts(
            [{"event_type": "Note", "start": 0.0, "blob": {"a": 1, "b": "x"}}],
            TimeUnit.seconds,
        )
        assert pa.types.is_struct(data.schema.field("blob").type)
        assert _field_type_meta(data.table, "blob") is None

    def test_coordinate_shaped_extra_stays_plain_struct(self) -> None:
        # The generic {value, numerator, denominator} struct must NOT be
        # mistaken for a CoordinateField in an extra column.
        data = EventData.from_dicts(
            [
                {
                    "event_type": "Note",
                    "start": 0.0,
                    "other": {"value": 1.0, "numerator": 1, "denominator": 1},
                }
            ],
            TimeUnit.seconds,
        )
        assert _field_type_meta(data.table, "other") is None

    def test_scalar_int_column_stays_numeric(self) -> None:
        # A bare int extra column must stay numeric, not be stringified.
        data = EventData.from_dicts(
            [{"event_type": "Note", "start": 0.0, "midi": 60}],
            TimeUnit.seconds,
        )
        assert pa.types.is_integer(data.schema.field("midi").type)
        assert data.table.column("midi").to_pylist() == [60]


class TestAddEventsRoundTrip:
    """A Timeline.add_events round-trip preserves the pitch affordance."""

    def test_add_events_preserves_struct_affordance(self) -> None:
        from timetoalign.timelines import ContinuousPhysicalTimeline

        tl = ContinuousPhysicalTimeline(length=10.0, unit=TimeUnit.seconds, uid="cpt_1")
        tl.add_events(
            [
                {"event_type": "Note", "start": 0.0, "pitch": {"midi_number": 60}},
                {"event_type": "Note", "start": 1.0, "pitch": {"midi_number": 64}},
            ]
        )
        events = tl.events
        assert events.schema.field("pitch").type == pa.struct(
            [pa.field("midi_number", pa.int64())]
        )
        assert events.get_field(EnharmonicPitch)[0] == EnharmonicPitch(midi_number=60)
        assert events.get_field(EnharmonicPitch)[1] == EnharmonicPitch(midi_number=64)

    def test_heterogeneous_batches_preserve_affordance(self) -> None:
        # A Note batch (with pitch struct) followed by a markup batch
        # (without it) must keep the struct column and null-fill.
        from timetoalign.timelines import DiscreteLogicalTimeline

        tl = DiscreteLogicalTimeline(length=1000, unit=TimeUnit.ticks, uid="dlt_1")
        tl.add_events(
            [{"event_type": "Note", "start": 0, "pitch": {"midi_number": 60}}],
            allow_expansion=True,
        )
        tl.add_events(
            [{"event_type": "Tempo", "start": 0, "bpm": "120"}],
            allow_expansion=True,
        )
        events = tl.events
        assert pa.types.is_struct(events.schema.field("pitch").type)
        assert events.table.column("pitch").to_pylist() == [{"midi_number": 60}, None]
        assert events.get_field(EnharmonicPitch)[0] == EnharmonicPitch(midi_number=60)


# ---------------------------------------------------------------------------
# 2. Uniform _afforded_fields mechanism over a raw atomic column
# ---------------------------------------------------------------------------


class TestAffordedFieldsMechanism:
    """A declared raw atomic column is promoted to its view on request."""

    def _midi_events(self) -> MidiEventData:
        return MidiEventData.from_dicts(
            [
                {
                    "event_type": "Note",
                    "start": 0,
                    "end": 480,
                    "pitch": 60,
                    "velocity": 90,
                    "channel": 0,
                    "track": 0,
                },
                {
                    "event_type": "ControlChange",
                    "instant": 0,
                    "control": 64,
                    "value": 127,
                    "channel": 0,
                    "track": 0,
                },
            ],
            TimeUnit.ticks,
        )

    def test_raw_pitch_column_is_int64(self) -> None:
        events = self._midi_events()
        assert events.schema.field("pitch").type == pa.int64()

    def test_raw_pitch_column_has_no_field_type_metadata(self) -> None:
        # The affordance is materialised on request — the raw column itself
        # carries no semantic metadata.
        events = self._midi_events()
        assert _field_type_meta(events.table, "pitch") is None

    def test_get_field_promotes_raw_column(self) -> None:
        events = self._midi_events()
        ep = events.get_field(EnharmonicPitch)
        assert isinstance(ep, EnharmonicPitchField)
        assert ep[0] == EnharmonicPitch(midi_number=60)

    def test_both_midi_number_views_share_raw_column(self) -> None:
        events = self._midi_events()
        ep = events.get_field(EnharmonicPitch)
        mp = events.get_field(MidiPitch)
        assert [ep[i].midi_number if ep[i] else None for i in range(len(ep))] == [
            60,
            None,
        ]
        assert [mp[i].midi_number if mp[i] else None for i in range(len(mp))] == [
            60,
            None,
        ]

    def test_has_field_sees_afforded(self) -> None:
        events = self._midi_events()
        assert events.has_field(EnharmonicPitchField) is True

    def test_afforded_field_is_cached(self) -> None:
        events = self._midi_events()
        first = events.get_field(EnharmonicPitch)
        second = events.get_field(EnharmonicPitch)
        assert first is second


@pytest.fixture
def vienna_chopin_notes() -> NoteEventData:
    """Load the 498-row Vienna Chopin notes table used by the tutorial."""
    data_dir = ensure_data("vienna_1x22")
    path = data_dir / "ms3" / "chopin_op10_no3.notes.tsv"
    return Ms3Loader.from_file(path).get_events()


class TestBlueprintAffordance:
    """Blueprints share declared raw-column promotion with class lookup."""

    def test_enharmonic_blueprint_matches_class_path(
        self, vienna_chopin_notes: NoteEventData
    ) -> None:
        events = vienna_chopin_notes
        class_path = events.get_field(EnharmonicPitch)
        blueprint_path = events.get_field(EnharmonicPitchField(source_fields="midi"))

        assert len(blueprint_path) == 498
        assert blueprint_path[0].midi_number == 59  # first pitch is B3
        assert [blueprint_path[i] for i in range(len(blueprint_path))] == [
            class_path[i] for i in range(len(class_path))
        ]

    def test_midi_blueprint_matches_class_path(
        self, vienna_chopin_notes: NoteEventData
    ) -> None:
        events = vienna_chopin_notes
        class_path = events.get_field(MidiPitch)
        blueprint_path = events.get_field(MidiPitchField(source_fields="midi"))

        assert len(blueprint_path) == 498
        assert blueprint_path[0].midi_number == 59
        assert [blueprint_path[i] for i in range(len(blueprint_path))] == [
            class_path[i] for i in range(len(class_path))
        ]

    def test_unpromotable_blueprint_source_stays_type_error(self) -> None:
        events = NoteEventData(
            pa.table({"midi": pa.array(["not a MIDI number"])}),
            TimeUnit.quarters,
        )

        with pytest.raises(TypeError):
            events.get_field(EnharmonicPitchField(source_fields="midi"))


# ---------------------------------------------------------------------------
# 3 + 5. MIDI EventData: scalar↔EventData contract, get_pitch_field == EP
# ---------------------------------------------------------------------------


class TestMidiEventDataAffordsEP:
    """MidiEvent.pitch: EnharmonicPitch — the EventData must afford it."""

    def test_get_pitch_field_returns_ep(self) -> None:
        events = MidiEventData.from_dicts(
            [
                {"event_type": "Note", "start": 0, "end": 480, "pitch": 60},
            ],
            TimeUnit.ticks,
        )
        field = events.get_pitch_field()
        assert isinstance(field, EnharmonicPitchField)
        assert field[0] == EnharmonicPitch(midi_number=60)

    def test_score_midi_event_data_affords_ep(self) -> None:
        events = ScoreMidiEventData.from_dicts(
            [
                {
                    "event_type": "Note",
                    "start": 0,
                    "end": 480,
                    "pitch": 72,
                    "voice": 1,
                    "staff": 1,
                    "part_id": "P1",
                },
            ],
            TimeUnit.ticks,
        )
        assert isinstance(events.get_pitch_field(), EnharmonicPitchField)
        assert events.get_field(EnharmonicPitch)[0] == EnharmonicPitch(midi_number=72)

    def test_midi_pitch_display_view_on_request(self) -> None:
        events = MidiEventData.from_dicts(
            [{"event_type": "Note", "start": 0, "end": 480, "pitch": 60}],
            TimeUnit.ticks,
        )
        mp = events.get_pitch_field().convert_to(MidiPitch)
        assert mp[0] == MidiPitch(midi_number=60)

    def test_no_specific_pitch_afforded(self) -> None:
        # A bare MIDI number never affords SpecificPitch (would be inference).
        events = MidiEventData.from_dicts(
            [{"event_type": "Note", "start": 0, "end": 480, "pitch": 60}],
            TimeUnit.ticks,
        )
        with pytest.raises(KeyError):
            events.get_field(SpecificPitchField)


# ---------------------------------------------------------------------------
# 4 + 6. Score NoteEventData: represent-once, SP default, on-request EP
# ---------------------------------------------------------------------------


def _note_events() -> NoteEventData:
    """Two spelled notes (C4=60, F#4=66) + one rest, via from_dicts."""
    rows = [
        {
            "event_type": "Note",
            "quarterbeats": 0,
            "duration_qb": 1,
            "specific_pitch": {"step": "C", "alter": 0, "octave": 4, "cents": 0.0},
            "midi": 60,
        },
        {
            "event_type": "Note",
            "quarterbeats": 1,
            "duration_qb": 1,
            "specific_pitch": {"step": "F", "alter": 1, "octave": 4, "cents": 0.0},
            "midi": 66,
        },
        {
            "event_type": "Rest",
            "quarterbeats": 2,
            "duration_qb": 1,
            "specific_pitch": None,
            "midi": None,
        },
    ]
    return NoteEventData.from_dicts(
        rows, unit=TimeUnit.quarters, number_type=NumberType.fraction
    )


class TestScoreRepresentOnce:
    """SpecificPitch is the sole default; EnharmonicPitch is afforded."""

    def test_get_pitch_field_returns_specific_pitch(self) -> None:
        events = _note_events()
        field = events.get_pitch_field()
        assert isinstance(field, SpecificPitchField)
        assert field[0] == SpecificPitch(step="C", alter=0, octave=4, cents=0.0)
        assert field[1] == SpecificPitch(step="F", alter=1, octave=4, cents=0.0)

    def test_no_midi_pitch_struct_column(self) -> None:
        # Represent-once: the redundant default EnharmonicPitch struct is gone.
        events = _note_events()
        assert "midi_pitch" not in events.table.column_names

    def test_specific_pitch_is_the_only_default_pitch_struct(self) -> None:
        # Exactly one default semantic pitch field (the SP struct); resolving
        # SpecificPitchField yields it by name.
        events = _note_events()
        sp = events.get_field(SpecificPitchField)
        assert sp.name == "specific_pitch"

    def test_midi_is_a_raw_int_column(self) -> None:
        events = _note_events()
        assert events.schema.field("midi").type == pa.int64()
        # raw column: no semantic field_type metadata
        assert _field_type_meta(events.table, "midi") is None

    def test_enharmonic_pitch_afforded_from_raw_midi(self) -> None:
        events = _note_events()
        ep = events.get_field(EnharmonicPitch)
        assert isinstance(ep, EnharmonicPitchField)
        assert ep.name == "midi"
        assert ep[0] == EnharmonicPitch(midi_number=60)
        assert ep[1] == EnharmonicPitch(midi_number=66)

    def test_midi_pitch_afforded_from_raw_midi(self) -> None:
        events = _note_events()
        mp = events.get_field(MidiPitch)
        assert mp.name == "midi"
        assert mp[0] == MidiPitch(midi_number=60)
        assert mp[1] == MidiPitch(midi_number=66)

    def test_enharmonic_pitch_field_property(self) -> None:
        # The convenience property routes through the affordance.
        events = _note_events()
        assert events.enharmonic_pitch_field[0] == EnharmonicPitch(midi_number=60)

    def test_on_request_ep_two_routes_agree(self) -> None:
        # Conversion route (SP field -> MidiPitch) and raw-column route
        # (afforded EnharmonicPitch over `midi`) agree element-wise.
        events = _note_events()
        sp_field = events.get_pitch_field()
        converted = sp_field.convert_to(MidiPitch)  # data-shaped conversion
        afforded = events.get_field(EnharmonicPitch)
        # Notes 0 and 1 (the rest is null and excluded from the comparison).
        assert converted[0].midi_number == afforded[0].midi_number == 60
        assert converted[1].midi_number == afforded[1].midi_number == 66

    def test_scalar_to_does_not_support_ep_directly(self) -> None:
        # Documented: SpecificPitch.to(EnharmonicPitch) is unsupported; the
        # MidiPitch thin alias is the path.  (Guards the README claim.)
        sp = SpecificPitch(step="C", alter=0, octave=4)
        with pytest.raises(TypeError):
            sp.to(EnharmonicPitch)
        assert sp.to(MidiPitch) == MidiPitch(midi_number=60)


# ---------------------------------------------------------------------------
# MSM-shaped table: EP default, SPC additionally afforded, no SP
# ---------------------------------------------------------------------------


class TestMsmShapedAffordsEpAndSpc:
    """A number-only-plus-inconsistent-spelling source affords EP (+SPC)."""

    def _msm_events(self) -> EventData:
        # Mirror MpmLoader._msm_note_row output shape: EP struct default +
        # SPC struct additional + verbatim raw spelling columns.
        return EventData.from_dicts(
            [
                {
                    "event_type": "Note",
                    "start": 0,
                    "pitch": {"midi_number": 75},
                    "specific_pitch_class": {"step": "E", "alter": -1},
                    "pitchname": "e",
                    "accidentals": -1,
                    "octave": 4,
                },
            ],
            TimeUnit.ticks,
        )

    def test_default_pitch_is_enharmonic(self) -> None:
        events = self._msm_events()
        assert isinstance(events.get_pitch_field(), EnharmonicPitchField)
        assert events.get_pitch_field()[0] == EnharmonicPitch(midi_number=75)

    def test_spc_additionally_afforded(self) -> None:
        events = self._msm_events()
        spc = events.get_field(SpecificPitchClass)
        assert isinstance(spc, SpecificPitchClassField)
        assert spc[0] == SpecificPitchClass(step="E", alter=-1)

    def test_specific_pitch_never_built(self) -> None:
        events = self._msm_events()
        with pytest.raises(KeyError):
            events.get_field(SpecificPitchField)

    def test_raw_spelling_kept(self) -> None:
        events = self._msm_events()
        assert events.table.column("pitchname").to_pylist() == ["e"]
        assert events.table.column("accidentals").to_pylist() == [-1]
        assert events.table.column("octave").to_pylist() == [4]

    def test_epc_reachable_by_conversion_from_ep(self) -> None:
        # Downhill derivation EP -> EPC (drop spelling -> pitch class).
        events = self._msm_events()
        epc = events.get_pitch_field().convert_to(EnharmonicPitchClass)
        assert epc[0] == EnharmonicPitchClass(pitch_class=75 % 12)


# ---------------------------------------------------------------------------
# 7. Multi-batch concat re-affordance (the key represent-once risk)
# ---------------------------------------------------------------------------


def _midi_note(start: int, pitch: int) -> dict[str, object]:
    """A minimal MIDI note row with a bare-int pitch."""
    return {
        "event_type": "Note",
        "start": start,
        "end": start + 480,
        "pitch": pitch,
        "velocity": 90,
        "channel": 0,
        "track": 0,
    }


def _ep_numbers(field: EnharmonicPitchField) -> list[int | None]:
    """Materialise a field's midi_number values (None for null rows)."""
    out: list[int | None] = []
    for i in range(len(field)):
        scalar = field[i]
        out.append(None if scalar is None else scalar.midi_number)
    return out


class TestMultiBatchConcatAffordance:
    """The bare-int pitch affordance survives ``extend`` / ``add_events``.

    Represent-once stores a number-only pitch as a raw ``int64`` and
    affords ``EnharmonicPitch`` over it on demand.  The multi-batch
    ingestion path replaces the table in place with a
    ``pa.concat_tables(..., promote_options="default")`` result, so the
    affordance must re-attach over the concatenated table — this is the
    specific risk a materialised pitch struct would have sidestepped.
    """

    # -- EventData.extend level ---------------------------------------------

    def test_extend_grows_afforded_field_with_interleaved_query(self) -> None:
        # Afford -> extend -> query again: the field must span both batches
        # (a stale cache would expose only the pre-extend rows).
        data = MidiEventData.from_dicts(
            [_midi_note(0, 60), _midi_note(480, 62)], TimeUnit.ticks
        )
        first = data.get_field(EnharmonicPitch)
        assert _ep_numbers(first) == [60, 62]

        data.extend(MidiEventData.from_dicts([_midi_note(960, 64)], TimeUnit.ticks))
        second = data.get_field(EnharmonicPitch)
        assert _ep_numbers(second) == [60, 62, 64]
        assert [s.midi_number for s in (second[0], second[1], second[2])] == [
            60,
            62,
            64,
        ]

    def test_third_heterogeneous_batch_null_fills_and_affords(self) -> None:
        # A third batch with NO pitch column (a Control-Change row) plus a
        # brand-new column exercises promote_options="default" null-fill.
        data = MidiEventData.from_dicts(
            [_midi_note(0, 60), _midi_note(480, 62)], TimeUnit.ticks
        )
        data.get_field(EnharmonicPitch)  # prime the cache
        data.extend(MidiEventData.from_dicts([_midi_note(960, 64)], TimeUnit.ticks))
        data.get_field(EnharmonicPitch)  # prime again over the 3-row table
        data.extend(
            MidiEventData.from_dicts(
                [
                    {
                        "event_type": "ControlChange",
                        "instant": 1440,
                        "control": 64,
                        "value": 127,
                        "channel": 0,
                        "track": 0,
                    }
                ],
                TimeUnit.ticks,
            )
        )
        field = data.get_field(EnharmonicPitch)
        assert _ep_numbers(field) == [60, 62, 64, None]
        assert isinstance(data.get_pitch_field(), EnharmonicPitchField)

    def test_raw_pitch_column_stays_int64_across_concats(self) -> None:
        # Robustness / negative: the raw column is never promoted to a
        # struct nor stringified, and carries no field_type metadata.
        data = MidiEventData.from_dicts([_midi_note(0, 60)], TimeUnit.ticks)
        for start, pitch in ((480, 62), (960, 64)):
            data.extend(
                MidiEventData.from_dicts([_midi_note(start, pitch)], TimeUnit.ticks)
            )
            assert data.schema.field("pitch").type == pa.int64()
            assert _field_type_meta(data.table, "pitch") is None
        assert _ep_numbers(data.get_field(EnharmonicPitch)) == [60, 62, 64]

    def test_concat_returns_new_eventdata_that_affords(self) -> None:
        # The non-mutating sibling `concat` returns a fresh EventData that
        # also affords the view (no shared stale cache).
        a = MidiEventData.from_dicts([_midi_note(0, 60)], TimeUnit.ticks)
        a.get_field(EnharmonicPitch)  # cache on the source object
        b = MidiEventData.from_dicts([_midi_note(480, 62)], TimeUnit.ticks)
        merged = a.concat(b)
        assert _ep_numbers(merged.get_field(EnharmonicPitch)) == [60, 62]
        # The source object is untouched by the non-mutating concat.
        assert _ep_numbers(a.get_field(EnharmonicPitch)) == [60]

    # -- Timeline.add_events level ------------------------------------------

    def _midi_backed_timeline(self) -> "Timeline":
        # A timeline whose events store is a MidiEventData.  SingleStore ->
        # create_timeline keeps the concrete class through prefix_ids.
        from timetoalign.storage.store import SingleStore

        data = MidiEventData.from_dicts([_midi_note(0, 60)], TimeUnit.ticks)
        timeline = SingleStore(data, name="notes").create_timeline()
        assert isinstance(timeline.events, MidiEventData)
        return timeline

    def test_timeline_add_events_affords_across_batches(self) -> None:
        timeline = self._midi_backed_timeline()
        assert _ep_numbers(timeline.events.get_field(EnharmonicPitch)) == [60]

        timeline.add_events([_midi_note(480, 62)], allow_expansion=True)
        assert _ep_numbers(timeline.events.get_field(EnharmonicPitch)) == [60, 62]

        timeline.add_events([_midi_note(960, 64)], allow_expansion=True)
        events = timeline.events
        assert _ep_numbers(events.get_field(EnharmonicPitch)) == [60, 62, 64]
        assert isinstance(events.get_pitch_field(), EnharmonicPitchField)
        assert events.schema.field("pitch").type == pa.int64()
