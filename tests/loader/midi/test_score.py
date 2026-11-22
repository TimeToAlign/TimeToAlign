"""Tests for ScoreMidiLoader."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from timetoalign.loader.midi import (
    MidiEventData,
    MidiEventType,
    PerformanceMidiLoader,
    ScoreMidiEventData,
    ScoreMidiLoader,
)

SUPRA_RAW_NOTE_COUNT = 30092
SUPRA_RAW_FIRST_NOTE_START = 741.0
SUPRA_RAW_LAST_NOTE_START = 277754.0
SUPRA_RAW_PITCH_SUM = 1938603
BEETHOVEN_NOTE_COUNT = 3751
BEETHOVEN_MIDO_FIRST_NOTE_COORDINATES = (0.0, 252.0, 252.0)
BEETHOVEN_PARTITURA_FIRST_NOTE_COORDINATES = (0.0, 252.0, 252.0)
MIDO_BEETHOVEN_SORTED_PITCH_MULTISETS = (
    (36, 22),
    (38, 10),
    (39, 8),
    (40, 1),
    (41, 21),
    (42, 1),
    (43, 109),
    (44, 46),
    (45, 8),
    (46, 33),
    (47, 18),
    (48, 191),
    (49, 17),
    (50, 46),
    (51, 50),
    (52, 20),
    (53, 39),
    (54, 17),
    (55, 283),
    (56, 37),
    (57, 31),
    (58, 71),
    (59, 80),
    (60, 306),
    (61, 29),
    (62, 172),
    (63, 211),
    (64, 83),
    (65, 145),
    (66, 32),
    (67, 200),
    (68, 71),
    (69, 41),
    (70, 49),
    (71, 126),
    (72, 201),
    (73, 18),
    (74, 193),
    (75, 139),
    (76, 69),
    (77, 135),
    (78, 31),
    (79, 151),
    (80, 51),
    (81, 20),
    (82, 26),
    (83, 5),
    (84, 26),
    (85, 4),
    (86, 14),
    (87, 7),
    (88, 9),
    (89, 7),
    (90, 1),
    (91, 14),
    (92, 1),
    (96, 5),
)
PARTITURA_BEETHOVEN_SORTED_PITCH_MULTISETS = (
    (36, 22),
    (38, 10),
    (39, 8),
    (40, 1),
    (41, 21),
    (42, 1),
    (43, 109),
    (44, 46),
    (45, 8),
    (46, 33),
    (47, 18),
    (48, 191),
    (49, 17),
    (50, 46),
    (51, 50),
    (52, 20),
    (53, 39),
    (54, 17),
    (55, 283),
    (56, 37),
    (57, 31),
    (58, 71),
    (59, 80),
    (60, 306),
    (61, 29),
    (62, 172),
    (63, 211),
    (64, 83),
    (65, 145),
    (66, 32),
    (67, 200),
    (68, 71),
    (69, 41),
    (70, 49),
    (71, 126),
    (72, 201),
    (73, 18),
    (74, 193),
    (75, 139),
    (76, 69),
    (77, 135),
    (78, 31),
    (79, 151),
    (80, 51),
    (81, 20),
    (82, 26),
    (83, 5),
    (84, 26),
    (85, 4),
    (86, 14),
    (87, 7),
    (88, 9),
    (89, 7),
    (90, 1),
    (91, 14),
    (92, 1),
    (96, 5),
)


class TestScoreMidiLoader:
    """Tests for loading score MIDI files."""

    @pytest.mark.slow
    def test_load_beethoven_quartet(self, beethoven_score_path: Path) -> None:
        """Can load multi-part score MIDI."""
        loader = ScoreMidiLoader(part_voice_assign_mode=0)
        loader.load(beethoven_score_path)

        assert len(loader) == 3751  # beethoven_op18.mid has 3751 events
        assert loader.ticks_per_beat is not None

        # Check metadata
        meta = loader.metadata["sources"][0]
        assert meta["parser"] == "partitura"
        assert meta["parts"] == 4  # String quartet: 4 parts

        # Check events have score info
        df = loader.events.to_dataframe()
        assert not df["pitch"].isnull().all()
        # A score states no velocity.
        assert loader.events.table.column("velocity").null_count == 3751

    def test_load_empty_raises(self, tmp_path: Path) -> None:
        """Loading invalid/empty file raises error."""
        # ScoreMidiLoader uses partitura which raises its own errors or returns empty
        empty_file = tmp_path / "empty.mid"
        empty_file.touch()

        loader = ScoreMidiLoader()
        # ScoreMidiLoader uses partitura which raises EOFError on empty file
        # (no exception wrapping in score.py, raw partitura error escapes)
        with pytest.raises(EOFError):
            loader.load(empty_file)

    @pytest.mark.slow
    def test_score_emits_wide_schema(self, beethoven_score_path: Path) -> None:
        """Score MIDI emits the wider 10-extra-column schema.

        Partitura supplies ``voice``, ``staff`` and ``part_id`` per
        note, so the storage class is :class:`ScoreMidiEventData`
        (not the narrower performance-side :class:`MidiEventData`).
        """
        loader = ScoreMidiLoader(part_voice_assign_mode=0)
        loader.load(beethoven_score_path)

        # Concrete class is the wider subclass.
        assert type(loader.events) is ScoreMidiEventData
        # Subclass relationship is preserved.
        assert isinstance(loader.events, MidiEventData)

        columns = set(loader.events.table.column_names)
        for required in (
            "pitch",
            "velocity",
            "channel",
            "track",
            "control",
            "value",
            "program",
            "voice",
            "staff",
            "part_id",
        ):
            assert required in columns, f"missing {required} column"

    @pytest.mark.slow
    def test_mido_fast_path_matches_performance_on_supra_raw(
        self, supra_raw_path: Path
    ) -> None:
        """The score mido path preserves raw note pairing exactly."""
        score_loader = ScoreMidiLoader(parser="mido")
        performance_loader = PerformanceMidiLoader(
            include_controls=False,
            include_program_changes=False,
        )
        score_loader.load(supra_raw_path)
        performance_loader.load(supra_raw_path)

        score_notes = [
            event
            for event in score_loader.events
            if event["event_type"] == MidiEventType.NOTE
        ]
        performance_notes = [
            event
            for event in performance_loader.events
            if event["event_type"] == MidiEventType.NOTE
        ]

        score_metrics = (
            len(score_notes),
            score_notes[0]["start"]["value"],
            score_notes[-1]["start"]["value"],
            sum(event["pitch"] for event in score_notes),
        )
        performance_metrics = (
            len(performance_notes),
            performance_notes[0]["start"]["value"],
            performance_notes[-1]["start"]["value"],
            sum(event["pitch"] for event in performance_notes),
        )

        assert score_metrics == (
            SUPRA_RAW_NOTE_COUNT,
            SUPRA_RAW_FIRST_NOTE_START,
            SUPRA_RAW_LAST_NOTE_START,
            SUPRA_RAW_PITCH_SUM,
        )
        assert performance_metrics == (
            SUPRA_RAW_NOTE_COUNT,
            SUPRA_RAW_FIRST_NOTE_START,
            SUPRA_RAW_LAST_NOTE_START,
            SUPRA_RAW_PITCH_SUM,
        )
        assert score_metrics == performance_metrics

    def test_mido_fast_path_matches_partitura_pitch_content(
        self, beethoven_score_path: Path
    ) -> None:
        """The fast path preserves the quantized score's note pitch multiset."""
        mido_loader = ScoreMidiLoader(parser="mido")
        partitura_loader = ScoreMidiLoader(parser="partitura")
        mido_loader.load(beethoven_score_path)
        partitura_loader.load(beethoven_score_path)

        mido_notes = [
            event
            for event in mido_loader.events
            if event["event_type"] == MidiEventType.NOTE
        ]
        partitura_notes = [
            event
            for event in partitura_loader.events
            if event["event_type"] == MidiEventType.NOTE
        ]
        mido_pitch_multiset = tuple(
            sorted(Counter(event["pitch"] for event in mido_notes).items())
        )
        partitura_pitch_multiset = tuple(
            sorted(Counter(event["pitch"] for event in partitura_notes).items())
        )
        mido_coordinates = tuple(
            mido_notes[0][field]["value"] for field in ("start", "end", "duration")
        )
        partitura_coordinates = tuple(
            partitura_notes[0][field]["value"] for field in ("start", "end", "duration")
        )

        assert len(mido_notes) == BEETHOVEN_NOTE_COUNT
        assert len(partitura_notes) == BEETHOVEN_NOTE_COUNT
        assert mido_pitch_multiset == MIDO_BEETHOVEN_SORTED_PITCH_MULTISETS
        assert partitura_pitch_multiset == PARTITURA_BEETHOVEN_SORTED_PITCH_MULTISETS
        assert mido_pitch_multiset == partitura_pitch_multiset
        assert mido_coordinates == BEETHOVEN_MIDO_FIRST_NOTE_COORDINATES
        assert partitura_coordinates == BEETHOVEN_PARTITURA_FIRST_NOTE_COORDINATES

    def test_mido_fast_path_emits_narrow_schema(
        self, beethoven_score_path: Path
    ) -> None:
        """The fast path does not invent score-structure columns."""
        mido_loader = ScoreMidiLoader(parser="mido")
        partitura_loader = ScoreMidiLoader(parser="partitura")
        mido_loader.load(beethoven_score_path)
        partitura_loader.load(beethoven_score_path)

        assert isinstance(mido_loader.events, MidiEventData)
        assert not isinstance(mido_loader.events, ScoreMidiEventData)
        assert isinstance(partitura_loader.events, ScoreMidiEventData)
        assert mido_loader.events.to_dataframe().columns.tolist() == [
            "id",
            "name",
            "temporal_type",
            "event_type",
            "start",
            "end",
            "duration",
            "pitch",
            "velocity",
            "channel",
            "track",
            "control",
            "value",
            "program",
        ]
        assert "voice" not in mido_loader.events.to_dataframe().columns
        assert "staff" not in mido_loader.events.to_dataframe().columns
        assert "part_id" not in mido_loader.events.to_dataframe().columns

    def test_mido_fast_path_pins_ticks_per_beat(
        self, beethoven_score_path: Path
    ) -> None:
        """The mido path exposes the source file's exact PPQ."""
        loader = ScoreMidiLoader(parser="mido")
        loader.load(beethoven_score_path)

        assert loader.ticks_per_beat == 480

    @pytest.mark.parametrize(
        "options",
        [
            {"part_voice_assign_mode": 1},
            {"quantization_unit": 16},
            {"estimate_voice_info": True},
            {"estimate_key": True},
        ],
    )
    def test_mido_fast_path_rejects_structural_options(self, options: dict) -> None:
        """Structural score options require partitura's parser."""
        with pytest.raises(ValueError, match="need partitura"):
            ScoreMidiLoader(parser="mido", **options)

    def test_unknown_parser_raises(self) -> None:
        """Only the two supported parser names are accepted."""
        with pytest.raises(ValueError, match="parser must be"):
            ScoreMidiLoader(parser="nonsense")


if __name__ == "__main__":
    pytest.main([__file__])
