"""Tests for Ieee1599Loader — one IEEE 1599 document → AlignmentBundle.

This module tests ``Ieee1599Loader`` against the six specimens of the
``ieee1599`` corpus.  An IEEE 1599 document encodes one work in several
layers, all of which point back at a single ``<spine>`` through ``event_ref``;
the loader reads that as an alignment whose hub is the spine.

It verifies:

- the spine's **relative** ``timing`` / ``hpos`` deltas accumulating into
  absolute VTU coordinates (see ``tests/loader/alignment/README.md``), with
  source event ids verbatim;
- one LOS timeline carrying notes (one event per notehead), rests and lyric
  syllables at the VTU coordinate of the spine event they reference;
- one integer-pixel graphical SegmentLine per edition, with discrete graphical
  accolade children and raw bounding-box structs, and one seconds physical
  timeline per track (instants at ``start_time``);
- exact timeline uids, whose role component encodes the sanitisation rule;
- the columnar claim topology — one ``MatchClaimField`` for the whole document,
  reached through ``loader.get_field(MatchClaim)``, carrying a unit per row and
  leaving the bundle's per-claim Python list empty;
- the ``from_graph`` cross-section: one row per connected component, spine
  column non-null and unique, asserted cell by cell on the animals specimen;
- fidelity spot-checks — verbatim ``num``/``den`` durations, ``<undefined/>``
  accidentals, ties, tuplets, lyrics, ``<track_general>`` descriptions, and
  media references recorded whether or not the file exists; and
- the loader contract (one document per loader through every ingest method,
  cached timeline building, uid filtering, singleton timeline groups,
  ``get_field`` selector behaviour).

The ``<structural>`` layer has its own module, ``test_ieee1599_structural.py``.

All counts and coordinates are exact per the Zero Tolerance Validation
Policy.  Validation logic is documented in
``tests/loader/alignment/README.md``.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from timetoalign.alignment.claims import MatchClaim, MatchClaimField
from timetoalign.core import IdCoordinateField, NumberType, TimeUnit
from timetoalign.loader.alignment import Ieee1599Loader
from timetoalign.maps.interval import IntervalToConstantMap
from timetoalign.timelines.types import (
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    SegmentLine,
)

# region Helpers


def _rows(table: pa.Table, **equals: Any) -> list[dict[str, Any]]:
    """Return the rows of *table* whose columns equal the given values."""
    mask = None
    for column, value in equals.items():
        comparison = pc.equal(table.column(column), value)
        mask = comparison if mask is None else pc.and_(mask, comparison)
    if mask is None:
        return table.to_pylist()
    return table.filter(mask).to_pylist()


def _coordinate(event: dict[str, Any], key: str = "start") -> float:
    """Return an event's stored coordinate value."""
    return event[key]["value"]


def _coordinates(timeline: Any, key: str = "start") -> list[float]:
    """Return every event's coordinate on *timeline*, in storage order."""
    return [entry["value"] for entry in timeline.events.table.column(key).to_pylist()]


def _graphical_events(timeline: SegmentLine[DiscreteGraphicalTimeline]) -> pa.Table:
    """Return graphical child events in their parent SegmentLine coordinates."""
    return timeline.get_events().table


def _projection_ids(loader: Ieee1599Loader) -> list[str]:
    """Return the side-A timeline id of every claim, in field order."""
    column = loader.get_field(MatchClaim).table.column("match_claim")
    return column.combine_chunks().field("timeline_a_id").to_pylist()


def _layer_uids(loader: Ieee1599Loader) -> dict[str, set[str]]:
    """Map each projection layer onto the timeline uids that belong to it."""
    return {
        "los": {loader.los_uid},
        "notational": set(loader.edition_uids),
        "audio": set(loader.track_uids),
    }


def _claims_per_layer(loader: Ieee1599Loader) -> dict[str, int]:
    """Count the single claim field's rows per projection layer."""
    ids = _projection_ids(loader)
    return {
        layer: sum(1 for timeline_id in ids if timeline_id in uids)
        for layer, uids in _layer_uids(loader).items()
    }


def _first_claim_index(loader: Ieee1599Loader, layer: str) -> int:
    """Return the row index of the first claim projecting from *layer*."""
    uids = _layer_uids(loader)[layer]
    return next(
        index for index, tid in enumerate(_projection_ids(loader)) if tid in uids
    )


# endregion


# region Spine accumulation


class TestSpineAccumulation:
    """The spine's relative deltas accumulate into absolute VTU coordinates."""

    def test_animals_deltas_accumulate(self, ieee1599_loader) -> None:
        """Eight deltas of 0, 1, 1, ... become coordinates 0 ... 7."""
        spine = ieee1599_loader("animals").create_timeline("spine:dlt_1")

        assert spine.n_events == 8
        assert _coordinates(spine) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    def test_animals_source_ids_are_verbatim(self, ieee1599_loader) -> None:
        """Semantic spine ids survive untouched — ``event_ref`` targets them."""
        spine = ieee1599_loader("animals").create_timeline("spine:dlt_1")

        assert spine.events.table.column("id").to_pylist() == [
            "event_cow",
            "event_pig",
            "event_dog",
            "event_sheep",
            "event_horse",
            "event_cat",
            "event_rooster",
            "event_duck",
        ]

    def test_khomus_first_ten_coordinates(self, ieee1599_loader) -> None:
        """Deltas 0,0,0,0,1024,1024,2048,1024,2048,2048 accumulate as expected."""
        spine = ieee1599_loader("khomus").create_timeline("spine:dlt_1")

        assert _coordinates(spine)[:10] == [
            0.0,
            0.0,
            0.0,
            0.0,
            1024.0,
            2048.0,
            4096.0,
            5120.0,
            7168.0,
            9216.0,
        ]

    def test_gymnopedie_simultaneous_events_share_a_coordinate(
        self, ieee1599_loader
    ) -> None:
        """Nine leading ``timing="0"`` events all sit at coordinate 0."""
        spine = ieee1599_loader("gymnopedie").create_timeline("spine:dlt_1")

        assert _coordinates(spine)[:10] == [0.0] * 9 + [1024.0]
        assert spine.events.table.column("id").to_pylist()[:7] == [
            "Clef_part_1_1",
            "Clef_part_2_1",
            "KeySignature_part_1_1",
            "KeySignature_part_2_1",
            "TimeSignature_part_1_1",
            "TimeSignature_part_2_1",
            "part_1_voice0_measure1_ev0",
        ]

    @pytest.mark.parametrize(
        "specimen, n_events, n_distinct, length",
        [
            ("gymnopedie", 382, 188, 236544.0),
            ("animals", 8, 8, 7.0),
            ("khomus", 41, 38, 48128.0),
        ],
    )
    def test_spine_shape(
        self,
        ieee1599_loader,
        specimen: str,
        n_events: int,
        n_distinct: int,
        length: float,
    ) -> None:
        """Distinct coordinates are fewer than events; length is the delta sum."""
        spine = ieee1599_loader(specimen).create_timeline("spine:dlt_1")

        assert spine.n_events == n_events
        assert len(set(_coordinates(spine))) == n_distinct
        assert spine.length.value == length

    def test_hpos_is_cumulative_and_deltas_are_not_stored(
        self, ieee1599_loader
    ) -> None:
        """``hpos`` is one cumulative int field; the raw deltas are dropped."""
        spine = ieee1599_loader("khomus").create_timeline("spine:dlt_1")
        columns = spine.events.table.schema.names

        assert "hpos" in columns
        assert "timing" not in columns
        assert spine.events.table.column("hpos").to_pylist()[:6] == [
            0,
            0,
            0,
            0,
            1024,
            2048,
        ]

    def test_spine_events_stay_interval_free(self, ieee1599_loader) -> None:
        """Spine events are instants: no ``end``, no ``duration``."""
        table = ieee1599_loader("animals").create_timeline("spine:dlt_1").events.table

        assert table.column("temporal_type").to_pylist() == ["instant"] * 8
        assert table.column("end").null_count == 8
        assert table.column("duration").null_count == 8


# endregion


# region Timeline identity


class TestTimelineIdentity:
    """Timeline types, units and uid roles."""

    def test_gymnopedie_uids(self, ieee1599_loader) -> None:
        """Roles fold accents and collapse runs outside ``[a-z0-9_]``."""
        assert ieee1599_loader("gymnopedie").timeline_uids == [
            "spine:dlt_1",
            "los:dlt_2",
            "eng_montreal_les_editions_outremontaises_2006:dgt_1",
            "eng_transcription_2012:dgt_2",
            "satie_gymnopedie1_coleman:cpt_1",
            "satie_gymnopedie1_pfaul:cpt_2",
        ]

    def test_animals_uids(self, ieee1599_loader) -> None:
        """Three editions and two tracks, numbered per timeline type."""
        assert ieee1599_loader("animals").timeline_uids == [
            "spine:dlt_1",
            "los:dlt_2",
            "farm_picture:dgt_1",
            "coloring_page:dgt_2",
            "animal_shapes:dgt_3",
            "animals_eng:cpt_1",
            "animals_ita:cpt_2",
        ]

    def test_timeline_types_and_units(self, ieee1599_loader) -> None:
        """Each layer gets the timeline type its coordinates call for."""
        loader = ieee1599_loader("animals")
        spine = loader.create_timeline("spine:dlt_1")
        los = loader.create_timeline("los:dlt_2")
        edition = loader.create_timeline("farm_picture:dgt_1")
        track = loader.create_timeline("animals_eng:cpt_1")

        _, page = edition.get_segment_by_index(0)

        assert isinstance(spine, DiscreteLogicalTimeline)
        assert isinstance(los, DiscreteLogicalTimeline)
        assert isinstance(edition, SegmentLine)
        assert edition.segment_type is SegmentLine[DiscreteGraphicalTimeline]
        assert page.segment_type is DiscreteGraphicalTimeline
        assert isinstance(track, ContinuousPhysicalTimeline)
        assert (spine.unit, spine.number_type) == (TimeUnit.ticks, NumberType.int)
        assert (los.unit, los.number_type) == (TimeUnit.ticks, NumberType.int)
        assert (edition.unit, edition.number_type) == (TimeUnit.pixels, NumberType.int)
        assert (track.unit, track.number_type) == (TimeUnit.seconds, NumberType.float)

    def test_graphical_timeline_rounds_pixels_and_keeps_source_bbox(
        self, ieee1599_loader
    ) -> None:
        """Discrete coordinates round while fractional source values remain."""
        edition = ieee1599_loader("animals").create_timeline("farm_picture:dgt_1")
        cow = _rows(_graphical_events(edition), event_ref="event_cow")[0]

        assert edition.unit == TimeUnit.pixels
        assert edition.number_type == NumberType.int
        assert _coordinate(cow) == 993.0
        assert cow["source_bbox"] == {
            "ul": {"x": 992.96, "y": 263.32},
            "lr": {"x": 1206.4, "y": 448.92},
        }
        assert "source_upper_left_x" not in _graphical_events(edition).column_names

    def test_graphical_timeline_records_the_declared_measurement_unit(
        self, ieee1599_loader
    ) -> None:
        """The document says ``pixels``; that wording is kept per page."""
        edition = ieee1599_loader("animals").create_timeline("farm_picture:dgt_1")

        assert [page["measurement_unit"] for page in edition.meta["pages"]] == [
            "pixels"
        ]

    def test_bundle_holds_every_timeline_in_its_own_group(
        self, ieee1599_loader, ieee1599_bundle
    ) -> None:
        """Claims relate singleton groups rather than extending one group."""
        loader = ieee1599_loader("gymnopedie")
        bundle = ieee1599_bundle("gymnopedie")

        assert list(bundle.timelines) == loader.timeline_uids
        group_ids = [f"group_{n}" for n in range(1, len(loader.timeline_uids) + 1)]
        assert bundle.group_ids == group_ids
        assert [group.name for group in bundle.groups.values()] == loader.timeline_uids
        assert all(group.n_timelines == 1 for group in bundle.groups.values())
        assert bundle.timeline_to_groups == {
            uid: (group_id,) for uid, group_id in zip(loader.timeline_uids, group_ids)
        }
        assert loader.create_group() is None


# endregion


# region LOS layer


class TestLosLayer:
    """Notes, rests and lyric syllables on the VTU axis."""

    @pytest.mark.parametrize(
        "specimen, n_notes, n_rests, n_syllables",
        [
            ("gymnopedie", 469, 88, 0),
            ("animals", 0, 8, 8),
            ("khomus", 36, 2, 0),
        ],
    )
    def test_event_census(
        self,
        ieee1599_loader,
        specimen: str,
        n_notes: int,
        n_rests: int,
        n_syllables: int,
    ) -> None:
        """One event per notehead, per rest and per lyric syllable."""
        table = ieee1599_loader(specimen).store["los"].table
        census = {
            event_type: table.filter(
                pc.equal(table.column("event_type"), event_type)
            ).num_rows
            for event_type in ("Note", "Rest", "Syllable")
        }

        assert census == {"Note": n_notes, "Rest": n_rests, "Syllable": n_syllables}
        assert table.num_rows == n_notes + n_rests + n_syllables

    def test_gymnopedie_first_note(self, ieee1599_loader) -> None:
        """A chord's notehead lands at the VTU of the spine event it names."""
        los = ieee1599_loader("gymnopedie").create_timeline("los:dlt_2")
        note = _rows(
            los.events.table, event_ref="part_1_voice0_measure5_ev1", event_type="Note"
        )

        assert len(note) == 1
        assert _coordinate(note[0]) == 13312.0
        assert note[0]["step"] == "F"
        assert note[0]["octave"] == 6
        assert note[0]["actual_accidental"] == "sharp"
        assert note[0]["notehead_index"] == 0
        assert (note[0]["duration_num"], note[0]["duration_den"]) == (1, 4)

    def test_chords_are_recoverable_from_notehead_index(self, ieee1599_loader) -> None:
        """469 noteheads come from 288 chords; grouping recovers them."""
        table = ieee1599_loader("gymnopedie").store["los"].table
        notes = table.filter(pc.equal(table.column("event_type"), "Note"))
        chords = set(notes.column("event_ref").to_pylist())

        assert notes.num_rows == 469
        assert len(chords) == 288
        # The thickest chord carries four noteheads (indices 0..3).
        assert max(notes.column("notehead_index").to_pylist()) == 3

    def test_duration_is_the_verbatim_num_den_pair(self, ieee1599_loader) -> None:
        """``4/4`` stays ``4/4``; a reduced Fraction would have lost the form."""
        table = ieee1599_loader("animals").store["los"].table
        rest = _rows(table, event_ref="event_cow", event_type="Rest")[0]

        assert (rest["duration_num"], rest["duration_den"]) == (4, 4)
        assert Fraction(rest["duration_num"], rest["duration_den"]) == Fraction(1)

    def test_los_events_stay_interval_free(self, ieee1599_loader) -> None:
        """A notated duration is not a tick length: LOS events are instants."""
        table = ieee1599_loader("khomus").create_timeline("los:dlt_2").events.table

        assert set(table.column("temporal_type").to_pylist()) == {"instant"}
        assert table.column("end").null_count == table.num_rows

    def test_animals_lyrics(self, ieee1599_loader) -> None:
        """Syllables sit at the VTU of their ``start_event_ref``."""
        table = ieee1599_loader("animals").store["los"].table
        syllables = _rows(table, event_type="Syllable")

        assert len(syllables) == 8
        assert syllables[0]["event_ref"] == "event_cow"
        assert syllables[0]["instant"] == 0
        assert syllables[0]["text"] == (
            "The black and white cow / La mucca bianca e nera"
        )
        assert syllables[-1]["instant"] == 7

    def test_columns_are_pruned_to_what_the_specimen_states(
        self, ieee1599_loader
    ) -> None:
        """No all-null columns: gymnopédie has no lyrics and no tuplets."""
        columns = set(ieee1599_loader("gymnopedie").store["los"].table.schema.names)

        assert {"step", "octave", "actual_accidental", "duration_num"} <= columns
        assert "text" not in columns
        assert "hyphen" not in columns
        assert "tuplet_enter_num" not in columns

    def test_staff_attributes_are_kept_out_of_the_timeline(
        self, ieee1599_loader
    ) -> None:
        """Clefs / key / time signatures are curated data, not LOS events."""
        loader = ieee1599_loader("gymnopedie")
        staff_list = loader.store["staff_list"].table
        los_refs = set(loader.store["los"].table.column("event_ref").to_pylist())

        assert staff_list.num_rows == 6
        assert staff_list.column("kind").to_pylist() == [
            "time_signature",
            "key_signature",
            "clef",
            "time_signature",
            "key_signature",
            "clef",
        ]
        assert _rows(staff_list, kind="clef")[0]["shape"] == "G"
        assert _rows(staff_list, kind="time_signature")[0]["num"] == 3
        assert "Clef_part_1_1" not in los_refs


# endregion


# region Notational and audio layers


class TestNotationalLayer:
    """One graphical SegmentLine per edition; boxes as interval events."""

    @pytest.mark.parametrize(
        "specimen, uid, n_events, length",
        [
            (
                "gymnopedie",
                "eng_montreal_les_editions_outremontaises_2006:dgt_1",
                382,
                7686.0,
            ),
            ("gymnopedie", "eng_transcription_2012:dgt_2", 382, 7356.0),
            ("animals", "farm_picture:dgt_1", 8, 2807.0),
            ("animals", "coloring_page:dgt_2", 6, 968.0),
            ("animals", "animal_shapes:dgt_3", 13, 991.0),
            ("khomus", "original_score:dgt_1", 39, 7418.0),
            ("khomus", "finale_transcription:dgt_2", 39, 16926.0),
        ],
    )
    def test_edition_shape(
        self,
        ieee1599_loader,
        specimen: str,
        uid: str,
        n_events: int,
        length: float,
    ) -> None:
        """Every graphical event remains on its edition's SegmentLine."""
        edition = ieee1599_loader(specimen).create_timeline(uid)

        assert _graphical_events(edition).num_rows == n_events
        assert edition.length.value == length

    def test_box_becomes_an_interval_event(self, ieee1599_loader) -> None:
        """The raw integer box is one nested struct, not flat box columns."""
        edition = ieee1599_loader("animals").create_timeline("farm_picture:dgt_1")
        events = _graphical_events(edition)
        cow = _rows(events, event_ref="event_cow")[0]

        assert cow["temporal_type"] == "interval"
        assert _coordinate(cow, "start") == 993.0
        assert _coordinate(cow, "end") == 1206.0
        assert cow["bbox"] == {"ul": {"x": 993, "y": 263}, "lr": {"x": 1206, "y": 449}}
        assert "upper_left_y" not in events.column_names
        assert "lower_right_y" not in events.column_names
        assert cow["file_name"] == "score_files/farm_picture/animal_colors.jpg"

    def test_integral_boxes_omit_the_source_bbox(self, ieee1599_loader) -> None:
        """Verbatim boxes are stored only when rounding would lose precision."""
        edition = ieee1599_loader("gymnopedie").create_timeline(
            "eng_montreal_les_editions_outremontaises_2006:dgt_1"
        )

        assert "source_bbox" not in _graphical_events(edition).column_names

    def test_pages_are_identified_on_every_box(self, ieee1599_loader) -> None:
        """Four pages per gymnopédie edition, recoverable per box."""
        loader = ieee1599_loader("gymnopedie")
        edition = loader.create_timeline(
            "eng_montreal_les_editions_outremontaises_2006:dgt_1"
        )
        pages = edition.meta["pages"]

        assert [page["position_in_group"] for page in pages] == [1, 2, 3, 4]
        assert pages[0]["file_name"] == (
            "score/Montreal/IMSLP01599-Satie_Gymnopedies-1.png"
        )
        assert set(
            _graphical_events(edition).column("position_in_group").to_pylist()
        ) == {
            1,
            2,
            3,
            4,
        }

    def test_gymnopedie_accolades_are_pinned_to_their_pages(
        self, ieee1599_loader
    ) -> None:
        """The half-span rule finds 4, 5, 5, 4 systems on every edition."""
        loader = ieee1599_loader("gymnopedie")
        counts = []
        for uid in loader.edition_uids:
            edition = loader.create_timeline(uid)
            counts.append(
                (
                    edition.n_segments,
                    [page.n_segments for _, _, page in edition.iter_segments()],
                )
            )

        assert counts == [(4, [4, 5, 5, 4]), (4, [4, 5, 5, 4])]

    def test_page_children_name_their_own_image(self, ieee1599_loader) -> None:
        """Each page child is one ``<graphic_instance>``: one file, one origin."""
        edition = ieee1599_loader("gymnopedie").create_timeline(
            "eng_montreal_les_editions_outremontaises_2006:dgt_1"
        )
        pages = [page for _, _, page in edition.iter_segments()]

        # A page is named by its position; its id is minted in the edition.
        assert [page.name for page in pages] == [
            f"page_{index}" for index in (1, 2, 3, 4)
        ]
        assert [page.id for page in pages] == ["dgt_1", "dgt_7", "dgt_12", "dgt_18"]
        assert [page.meta["page"]["position_in_group"] for page in pages] == [
            1,
            2,
            3,
            4,
        ]
        assert [page.meta["page"]["file_name"] for page in pages] == [
            f"score/Montreal/IMSLP01599-Satie_Gymnopedies-{index}.png"
            for index in (1, 2, 3, 4)
        ]
        assert [page.length.value for page in pages] == [1694, 2152, 2105, 1735]
        accolades = [accolade for _, _, accolade in pages[0].iter_segments()]
        assert [accolade.name for accolade in accolades] == [
            f"accolade_{index}" for index in (1, 2, 3, 4)
        ]
        assert [accolade.id for accolade in accolades] == [
            "dgt_2",
            "dgt_3",
            "dgt_4",
            "dgt_5",
        ]

    def test_gymnopedie_first_drop_starts_the_second_accolade(
        self, ieee1599_loader
    ) -> None:
        """The first page's first >half-span reset ends the first accolade."""
        edition = ieee1599_loader("gymnopedie").create_timeline(
            "eng_montreal_les_editions_outremontaises_2006:dgt_1"
        )
        _, page = edition.get_segment_by_index(0)
        _, first = page.get_segment_by_index(0)
        _, second = page.get_segment_by_index(1)
        first_rows = first.events.table.to_pylist()
        second_rows = second.events.table.to_pylist()

        assert first.length.value == 426
        assert len(first_rows) == 22
        assert first_rows[-1]["start"]["value"] == 411.0
        assert second_rows[0]["start"]["value"] == 53.0
        assert 411 - 53 > (422 - 49) / 2

    def test_page_file_map_resolves_each_gymnopedie_page(self, ieee1599_loader) -> None:
        """Every page's first box maps to its verbatim image file name."""
        edition = ieee1599_loader("gymnopedie").create_timeline(
            "eng_montreal_les_editions_outremontaises_2006:dgt_1"
        )
        page_starts = [
            offset.value + page.get_events().table.to_pylist()[0]["start"]["value"]
            for _, offset, page in edition.iter_segments()
        ]
        cmap = edition.get_conversion_map("file_name")

        assert isinstance(cmap, IntervalToConstantMap)
        assert cmap.end == edition.length.value
        assert cmap.to_dict()["type"] == "PiecewiseMap"
        assert [(start, cmap(start)) for start in page_starts] == [
            (50.0, "score/Montreal/IMSLP01599-Satie_Gymnopedies-1.png"),
            (1749.0, "score/Montreal/IMSLP01599-Satie_Gymnopedies-2.png"),
            (3943.0, "score/Montreal/IMSLP01599-Satie_Gymnopedies-3.png"),
            (6007.0, "score/Montreal/IMSLP01599-Satie_Gymnopedies-4.png"),
        ]

    @pytest.mark.parametrize(
        "specimen, page_accolade_counts",
        [
            ("gymnopedie", [(4, 18), (4, 18)]),
            ("animals", [(1, 3), (1, 1), (1, 1)]),
            ("khomus", [(1, 4), (1, 4)]),
            ("pazzariello", [(3, 34), (2, 14)]),
            ("serie", [(28, 463)]),
            ("bach", [(3, 48), (3, 48), (2, 24)]),
        ],
    )
    def test_every_specimen_loads_its_paged_editions(
        self,
        ieee1599_loader,
        specimen: str,
        page_accolade_counts: list[tuple[int, int]],
    ) -> None:
        """All six specimens load with their observed pages and accolades."""
        loader = ieee1599_loader(specimen)
        bundle = loader.create_bundle()
        editions = [loader.create_timeline(uid) for uid in loader.edition_uids]

        assert [
            (
                edition.n_segments,
                sum(page.n_segments for _, _, page in edition.iter_segments()),
            )
            for edition in editions
        ] == page_accolade_counts
        assert len(bundle.groups) == len(loader.timeline_uids)


class TestAudioLayer:
    """One seconds timeline per track; track events as instants."""

    @pytest.mark.parametrize(
        "specimen, uid, n_events, length",
        [
            ("gymnopedie", "satie_gymnopedie1_coleman:cpt_1", 382, 188.77),
            ("gymnopedie", "satie_gymnopedie1_pfaul:cpt_2", 382, 192.22),
            ("animals", "animals_eng:cpt_1", 8, 39.0),
            ("animals", "animals_ita:cpt_2", 8, 37.5),
            ("khomus", "khomus_audio:cpt_1", 41, 46.98),
            ("khomus", "khomus_video:cpt_2", 41, 45.88),
        ],
    )
    def test_track_shape(
        self,
        ieee1599_loader,
        specimen: str,
        uid: str,
        n_events: int,
        length: float,
    ) -> None:
        """Track indexing is complete per track, and lengths are exact."""
        track = ieee1599_loader(specimen).create_timeline(uid)

        assert track.n_events == n_events
        assert track.length.value == length

    def test_track_event_is_an_instant(self, ieee1599_loader) -> None:
        """Start times are seconds; the file name travels with the event."""
        track = ieee1599_loader("animals").create_timeline("animals_eng:cpt_1")
        pig = _rows(track.events.table, event_ref="event_pig")[0]

        assert pig["temporal_type"] == "instant"
        assert _coordinate(pig) == 6.5
        assert pig["file_name"] == "audio_files/animals_eng.mp3"

    def test_track_general_is_recorded(self, ieee1599_loader) -> None:
        """``<track_general>`` reaches the timeline whole and verbatim."""
        track = ieee1599_loader("gymnopedie").create_timeline(
            "satie_gymnopedie1_coleman:cpt_1"
        )

        assert track.meta["performers"] == [{"name": "Chase Coleman", "type": "piano"}]
        assert track.meta["notes"] == "Chase Coleman"
        # The specimen states no ``<recordings>`` for this track, so no key.
        assert "recordings" not in track.meta

    def test_track_without_a_general_section(self, ieee1599_loader) -> None:
        """Nothing is invented for a track that describes itself no further."""
        track = ieee1599_loader("khomus").create_timeline("khomus_audio:cpt_1")

        assert track.meta["performers"] == [
            {"name": "Audio performance", "type": "khomus"}
        ]
        assert "notes" not in track.meta
        assert "recordings" not in track.meta

    def test_declared_format_may_contradict_the_extension(
        self, ieee1599_loader, ieee1599_dir
    ) -> None:
        """``video_avi`` naming a ``.mp4`` is recorded verbatim, not corrected."""
        track = ieee1599_loader("khomus").create_timeline("khomus_video:cpt_2")

        assert track.meta["file_name"] == "audio_files/khomus_video.mp4"
        assert track.meta["file_format"] == "video_avi"
        assert (ieee1599_dir / "Khorus Music" / track.meta["file_name"]).exists()


# endregion


# region Claims


class TestClaims:
    """The columnar spine-hub claim topology."""

    @pytest.mark.parametrize(
        "specimen, los, notational, audio",
        [
            ("gymnopedie", 557, 764, 764),
            ("animals", 16, 27, 16),
            ("khomus", 38, 78, 82),
        ],
    )
    def test_claim_counts(
        self,
        ieee1599_loader,
        ieee1599_bundle,
        specimen: str,
        los: int,
        notational: int,
        audio: int,
    ) -> None:
        """One claim per projected event: LOS + graphic + track."""
        loader = ieee1599_loader(specimen)
        bundle = ieee1599_bundle(specimen)

        assert len(loader.get_field(MatchClaim)) == los + notational + audio
        assert len(loader) == los + notational + audio
        assert bundle.n_cross_group_claims == los + notational + audio
        assert _claims_per_layer(loader) == {
            "los": los,
            "notational": notational,
            "audio": audio,
        }

    def test_claims_stay_columnar(self, ieee1599_bundle) -> None:
        """No ``MatchClaim`` object is materialised into the bundle."""
        bundle = ieee1599_bundle("gymnopedie")

        assert bundle.cross_group_claims == []
        assert len(bundle.cross_group_claim_fields) == 1

    def test_get_field_returns_the_one_field(self, ieee1599_loader) -> None:
        """The uniform field API: both selectors resolve to one field."""
        loader = ieee1599_loader("animals")
        field = loader.get_field(MatchClaim)

        assert isinstance(field, MatchClaimField)
        assert len(field) == 59
        assert loader.get_field(MatchClaimField) is field
        assert loader.claim_field is field

    def test_get_field_rejects_other_selectors(self, ieee1599_loader) -> None:
        """Non-claim selectors are refused."""
        loader = ieee1599_loader("animals")

        with pytest.raises(TypeError, match="MatchClaim"):
            loader.get_field(str)

    @pytest.mark.parametrize(
        "layer, unit",
        [
            ("los", TimeUnit.ticks),
            ("notational", TimeUnit.pixels),
            ("audio", TimeUnit.seconds),
        ],
    )
    def test_claim_units_are_per_row(
        self, ieee1599_loader, layer: str, unit: TimeUnit
    ) -> None:
        """One field, but each row carries its own layer's unit on side A."""
        loader = ieee1599_loader("animals")
        claim = loader.get_field(MatchClaim)[_first_claim_index(loader, layer)]

        assert claim.start_anchor.coordinate_a.unit == unit
        assert claim.start_anchor.coordinate_b.unit == TimeUnit.ticks
        assert claim.timeline_b_id == "spine:dlt_1"
        assert claim.is_synchronous

    def test_first_graphic_claim_is_exact(self, ieee1599_loader) -> None:
        """The first gymnopédie box: edition x=50 against spine VTU 0."""
        loader = ieee1599_loader("gymnopedie")
        claim = loader.get_field(MatchClaim)[_first_claim_index(loader, "notational")]

        assert claim.timeline_a_id == (
            "eng_montreal_les_editions_outremontaises_2006:dgt_1"
        )
        assert claim.start_anchor.coordinate_a.value == 50.0
        assert claim.start_anchor.coordinate_b.value == 0

    def test_claim_provenance(self, ieee1599_loader) -> None:
        """The document's ``creator`` is the agent; the alignment is read, not run."""
        field = ieee1599_loader("gymnopedie").get_field(MatchClaim)

        assert field.metadata is not None
        assert field.metadata.agent.name == "Finale Plugin"
        assert field.metadata.agent.identifier == "ieee1599_event_ref"
        assert field.metadata.certainty == 1.0


# endregion


# region Cross-section


class TestCrossSection:
    """``get_matchstamp_table(from_graph=True)`` — the aligned instants."""

    @pytest.mark.parametrize(
        "specimen, n_rows, n_spine_coordinates",
        [
            ("gymnopedie", 188, 188),
            ("animals", 8, 8),
            ("khomus", 38, 38),
        ],
    )
    def test_row_count(
        self,
        ieee1599_loader,
        ieee1599_bundle,
        specimen: str,
        n_rows: int,
        n_spine_coordinates: int,
    ) -> None:
        """One row per component, never more than one per spine coordinate."""
        table = ieee1599_bundle(specimen).get_matchstamp_table(from_graph=True)
        spine_column = IdCoordinateField.from_table(table, "spine:dlt_1")

        assert table.num_rows == n_rows
        assert all(spine_column[position] is not None for position in range(n_rows))
        assert (
            len({spine_column[position].value for position in range(n_rows)}) == n_rows
        )
        assert n_rows <= n_spine_coordinates
        spine = ieee1599_loader(specimen).create_timeline("spine:dlt_1")
        assert len(set(_coordinates(spine))) == n_spine_coordinates

    def test_every_timeline_participates(
        self, ieee1599_loader, ieee1599_bundle
    ) -> None:
        """The cross-section spans all six gymnopédie timelines."""
        loader = ieee1599_loader("gymnopedie")
        table = ieee1599_bundle("gymnopedie").get_matchstamp_table(from_graph=True)

        assert sorted(table.column_names) == sorted(loader.timeline_uids)

    def test_gymnopedie_origin_row(self, ieee1599_bundle) -> None:
        """The work's origin, as every layer states it."""
        table = ieee1599_bundle("gymnopedie").get_matchstamp_table(from_graph=True)
        rows = [
            {
                name: (
                    None
                    if (
                        coordinate := IdCoordinateField.from_table(table, name)[
                            position
                        ]
                    )
                    is None
                    else coordinate.value
                )
                for name in table.column_names
            }
            for position in range(table.num_rows)
        ]
        origin = [row for row in rows if row["spine:dlt_1"] == 0.0]

        assert origin == [
            {
                "spine:dlt_1": 0.0,
                "los:dlt_2": 0.0,
                "eng_montreal_les_editions_outremontaises_2006:dgt_1": 49.0,
                "eng_transcription_2012:dgt_2": 75.0,
                "satie_gymnopedie1_coleman:cpt_1": 2.46,
                "satie_gymnopedie1_pfaul:cpt_2": 0.5,
            }
        ]

    def test_animals_cross_section_cell_by_cell(self, ieee1599_bundle) -> None:
        """The one specimen with no merging: eight animals, every cell exact."""
        table = ieee1599_bundle("animals").get_matchstamp_table(from_graph=True)
        rows = [
            {
                name: (
                    None
                    if (
                        coordinate := IdCoordinateField.from_table(table, name)[
                            position
                        ]
                    )
                    is None
                    else coordinate.value
                )
                for name in table.column_names
            }
            for position in range(table.num_rows)
        ]
        by_spine = {row["spine:dlt_1"]: row for row in rows}

        assert sorted(by_spine) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        # The cow: depicted twice in "Animal shapes", the collapse keeps the
        # smaller rounded box coordinate (14 rather than 159).
        assert by_spine[0.0] == {
            "spine:dlt_1": 0.0,
            "los:dlt_2": 0.0,
            "farm_picture:dgt_1": 993.0,
            "coloring_page:dgt_2": 234.0,
            "animal_shapes:dgt_3": 14.0,
            "animals_eng:cpt_1": 0.0,
            "animals_ita:cpt_2": 0.0,
        }
        # The dog: absent from "Animal shapes"? No — absent from the colouring
        # page, which does not depict it.
        assert by_spine[2.0]["coloring_page:dgt_2"] is None
        assert by_spine[2.0]["animal_shapes:dgt_3"] == 704.0
        # The cat: absent from the colouring page too.
        assert by_spine[5.0]["coloring_page:dgt_2"] is None
        # The pig and the sheep are missing from "Animal shapes".
        assert by_spine[1.0]["animal_shapes:dgt_3"] is None
        assert by_spine[3.0]["animal_shapes:dgt_3"] is None
        # The duck, last event: both recordings and all three pictures.
        assert by_spine[7.0] == {
            "spine:dlt_1": 7.0,
            "los:dlt_2": 7.0,
            "farm_picture:dgt_1": 2656.0,
            "coloring_page:dgt_2": 387.0,
            "animal_shapes:dgt_3": 22.0,
            "animals_eng:cpt_1": 39.0,
            "animals_ita:cpt_2": 37.5,
        }


# endregion


# region Loader contract


class TestLoaderContract:
    """Two-phase lifecycle, caching and error paths."""

    def test_curated_store_tables(self, ieee1599_loader) -> None:
        """Curated layer tables, not the generic tag auto-flatten."""
        loader = ieee1599_loader("gymnopedie")

        assert loader.keys() == [
            "spine",
            "los",
            "staff_list",
            "notational",
            "audio",
            "structural",
        ]
        assert loader.store["notational"].table.num_rows == 764
        assert loader.store["audio"].table.num_rows == 764

    def test_file_metadata(self, ieee1599_loader) -> None:
        """Root attributes plus ``<general>`` bibliography."""
        metadata = ieee1599_loader("gymnopedie").file_metadata

        assert metadata["version"] == "1.0"
        assert metadata["creator"] == "Finale Plugin"
        assert metadata["title"] == "Gymnopédie No. 1"
        assert metadata["work_title"] == "3 Gymnopédies"
        assert metadata["authors"] == [{"name": "Erik Satie", "type": "composer"}]

    def test_bom_and_doctype_are_handled(self, ieee1599_loader, ieee1599_path) -> None:
        """Three specimens carry a UTF-8 BOM and all declare a DOCTYPE."""
        raw = ieee1599_path("animals").read_bytes()

        assert raw.startswith(b"\xef\xbb\xbf")
        assert b"<!DOCTYPE ieee1599" in raw
        assert ieee1599_loader("animals").file_metadata["version"] == "1.0"

    def test_create_timeline_caches(self, ieee1599_loader) -> None:
        """A timeline is built once and handed out again."""
        loader = ieee1599_loader("animals")

        assert loader.create_timeline("los:dlt_2") is loader.create_timeline(
            "los:dlt_2"
        )
        assert loader.create_timeline() is loader.create_timeline("spine:dlt_1")

    def test_create_timeline_rejects_unknown_uid(self, ieee1599_loader) -> None:
        """An unknown uid names the available ones."""
        with pytest.raises(KeyError, match="spine:dlt_1"):
            ieee1599_loader("animals").create_timeline("nope:cpt_9")

    def test_create_timelines_id_pattern(self, ieee1599_loader) -> None:
        """Timelines are filterable by a uid regex, matching and not."""
        loader = ieee1599_loader("animals")

        assert [tl.id for tl in loader.create_timelines(":cpt")] == [
            "animals_eng:cpt_1",
            "animals_ita:cpt_2",
        ]
        assert loader.create_timelines("no-such-timeline") == []

    def test_one_document_per_loader(self, ieee1599_path) -> None:
        """One spine, one bundle: a second document needs a second loader."""
        loader = Ieee1599Loader.from_file(ieee1599_path("animals"))

        with pytest.raises(ValueError, match="already holds"):
            loader.load(ieee1599_path("khomus"))
        with pytest.raises(ValueError, match="exactly one"):
            Ieee1599Loader().load(ieee1599_path("animals"), ieee1599_path("khomus"))

    def test_string_and_element_ingest_share_the_guard(self, ieee1599_path) -> None:
        """Every door into the parse is one document wide, not just ``load()``."""
        xml = ieee1599_path("khomus").read_text(encoding="utf-8-sig")
        loader = Ieee1599Loader.from_file(ieee1599_path("animals"))

        with pytest.raises(ValueError, match="already holds"):
            loader.load_string(xml)
        with pytest.raises(ValueError, match="already holds"):
            loader.load_element(ElementTree.fromstring(xml))
        assert loader.timeline_uids == [
            "spine:dlt_1",
            "los:dlt_2",
            "farm_picture:dgt_1",
            "coloring_page:dgt_2",
            "animal_shapes:dgt_3",
            "animals_eng:cpt_1",
            "animals_ita:cpt_2",
        ]

    def test_string_ingest_accepts_a_first_document(self, ieee1599_path) -> None:
        """The guard refuses a second document, not the first one."""
        xml = ieee1599_path("animals").read_text(encoding="utf-8-sig")
        loader = Ieee1599Loader().load_string(xml)

        assert loader.spine_uid == "spine:dlt_1"
        assert len(loader) == 59

    def test_empty_curated_layers_have_stable_tables(self) -> None:
        """The four primary layers exist even when their XML content is empty."""
        loader = Ieee1599Loader().load_string(
            "<ieee1599><logic><spine /></logic><notational /><audio /></ieee1599>"
        )

        assert loader.keys() == ["spine", "los", "notational", "audio"]
        assert {name: table.table.num_rows for name, table in loader.store.items()} == {
            "spine": 0,
            "los": 0,
            "notational": 0,
            "audio": 0,
        }
        assert loader.store["spine"].table.schema.names == [
            "id",
            "event_type",
            "instant",
            "hpos",
        ]
        assert loader.store["los"].table.schema.names == [
            "event_type",
            "instant",
            "event_ref",
            "part",
            "staff",
            "voice",
            "measure",
            "notehead_index",
            "duration_num",
            "duration_den",
            "tuplet_enter_num",
            "tuplet_enter_den",
            "tuplet_in_num",
            "tuplet_in_den",
            "augmentation_dots",
            "step",
            "octave",
            "actual_accidental",
            "printed_accidental",
            "tie",
            "text",
            "hyphen",
        ]
        assert loader.store["notational"].table.schema.names == [
            "timeline_uid",
            "edition",
            "event_type",
            "start",
            "end",
            "event_ref",
            "bbox",
            "source_bbox",
            "file_name",
            "position_in_group",
        ]
        assert loader.store["audio"].table.schema.names == [
            "timeline_uid",
            "track",
            "event_type",
            "instant",
            "event_ref",
            "file_name",
        ]

    def test_methods_before_load(self) -> None:
        """The two-phase contract is enforced, not assumed."""
        loader = Ieee1599Loader()

        with pytest.raises(RuntimeError, match="load"):
            loader.get_field(MatchClaim)
        with pytest.raises(RuntimeError, match="load"):
            loader.create_timeline()
        with pytest.raises(RuntimeError, match="load"):
            loader.create_bundle()

    def test_clear_resets_everything(self, ieee1599_path) -> None:
        """``clear()`` returns the loader to its unloaded state."""
        loader = Ieee1599Loader.from_file(ieee1599_path("animals"))
        loader.clear()

        assert loader.timeline_uids == []
        assert loader.claim_field is None
        assert len(loader) == 0
        assert repr(loader) == "Ieee1599Loader(not loaded)"

    def test_repr(self, ieee1599_loader) -> None:
        """The repr states the layer shape."""
        assert repr(ieee1599_loader("animals")) == (
            "Ieee1599Loader(spine=8, los=16, editions=3, tracks=2, claims=59)"
        )


# endregion


# region Large specimens


@pytest.mark.slow
class TestLargeSpecimens:
    """The three specimens whose full parse is too slow for the fast lane."""

    def test_pazzariello_shape(self, ieee1599_loader, ieee1599_bundle) -> None:
        """616 spine events, two editions, one recording, 129 syllables."""
        loader = ieee1599_loader("pazzariello")
        bundle = ieee1599_bundle("pazzariello")

        assert loader.create_timeline("spine:dlt_1").n_events == 616
        assert loader.timeline_uids == [
            "spine:dlt_1",
            "los:dlt_2",
            "musescore_transcription:dgt_1",
            "partitura_autografa:dgt_2",
            "recording_01:cpt_1",
        ]
        assert _claims_per_layer(loader) == {
            "los": 1070,
            "notational": 1162,
            "audio": 616,
        }
        assert bundle.n_cross_group_claims == 2848
        assert bundle.get_matchstamp_table(from_graph=True).num_rows == 294

    def test_pazzariello_undefined_accidentals(self, ieee1599_loader) -> None:
        """``<undefined/>`` is recorded as such — never dropped, never inferred."""
        table = ieee1599_loader("pazzariello").store["los"].table
        undefined = _rows(table, printed_accidental="undefined")

        assert len(undefined) == 2
        assert [row["event_ref"] for row in undefined] == [
            "baritono1_meas19_voice1_ev3",
            "baritono1_meas19_voice1_ev4",
        ]
        assert [row["actual_accidental"] for row in undefined] == [
            "double_sharp",
            "natural",
        ]
        assert undefined[0]["instant"] == 34080

    def test_pazzariello_tuplets_and_lyrics(self, ieee1599_loader) -> None:
        """All four tuplet integers and the syllable hyphenation survive."""
        table = ieee1599_loader("pazzariello").store["los"].table
        tuplets = table.filter(pc.is_valid(table.column("tuplet_enter_num")))
        syllables = _rows(table, event_type="Syllable")

        assert tuplets.num_rows == 51
        assert tuplets.to_pylist()[0]["tuplet_enter_num"] == 3
        assert tuplets.to_pylist()[0]["tuplet_enter_den"] == 16
        assert tuplets.to_pylist()[0]["tuplet_in_num"] == 1
        assert tuplets.to_pylist()[0]["tuplet_in_den"] == 8
        assert len(syllables) == 129
        assert syllables[0]["text"] == "Mo',"
        assert syllables[0]["hyphen"] == "no"
        assert syllables[1]["hyphen"] == "yes"

    def test_serie_shape(self, ieee1599_loader, ieee1599_bundle) -> None:
        """One edition of 28 pages, six tracks of unequal completeness."""
        loader = ieee1599_loader("serie")
        bundle = ieee1599_bundle("serie")

        assert loader.create_timeline("spine:dlt_1").n_events == 3509
        assert loader.edition_uids == ["serie_in_9_8:dgt_1"]
        assert len(loader.create_timeline("serie_in_9_8:dgt_1").meta["pages"]) == 28
        assert [loader.create_timeline(uid).n_events for uid in loader.track_uids] == [
            3509,
            3509,
            3509,
            3509,
            3507,
            852,
        ]
        assert _claims_per_layer(loader) == {
            "los": 3443,
            "notational": 3442,
            "audio": 18395,
        }
        assert bundle.n_cross_group_claims == 25280
        assert bundle.get_matchstamp_table(from_graph=True).num_rows == 1154

    def test_serie_track_recordings(self, ieee1599_loader) -> None:
        """The only specimen with ``<recordings>`` keeps their attributes."""
        track = ieee1599_loader("serie").create_timeline("serie_9_8_sito:cpt_1")

        assert track.meta["recordings"] == [{"date": "1988", "studio_name": "LIM"}]
        assert track.meta["notes"] == (
            "Computer music master produced at LIM by Antonio José Rodriguez "
            "Selles and Goffredo Haus (1988)"
        )

    def test_bach_shape(self, ieee1599_loader, ieee1599_bundle) -> None:
        """Three editions, four recordings, 140 ties."""
        loader = ieee1599_loader("bach")
        bundle = ieee1599_bundle("bach")
        los = loader.store["los"].table

        assert loader.create_timeline("spine:dlt_1").n_events == 1144
        assert loader.edition_uids == [
            "transcription:dgt_1",
            "breitkopf_und_hartel_1878:dgt_2",
            "leipzig_ca_1750:dgt_3",
        ]
        assert los.filter(pc.equal(los.column("tie"), True)).num_rows == 140
        assert _claims_per_layer(loader) == {
            "los": 1132,
            "notational": 3397,
            "audio": 4576,
        }
        assert bundle.n_cross_group_claims == 9105
        assert bundle.get_matchstamp_table(from_graph=True).num_rows == 571

    def test_bach_media_references_are_dangling(
        self, ieee1599_loader, ieee1599_path
    ) -> None:
        """Every referenced file is absent from disk, yet recorded verbatim."""
        loader = ieee1599_loader("bach")
        directory = ieee1599_path("bach").parent
        names = [
            loader.create_timeline(uid).meta["file_name"] for uid in loader.track_uids
        ]

        assert names == [
            "audio_files/bach_artefuga_01_orchestra.mp3",
            "audio_files/zoltan_kocsis_keyboard.mp3",
            "audio_files/emerson_string_quartet.mp3",
            "audio_files/christoph_lahme_harmonium.mp3",
        ]
        assert not any((directory / Path(name)).exists() for name in names)


# endregion
