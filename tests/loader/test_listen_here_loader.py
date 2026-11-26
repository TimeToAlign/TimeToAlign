"""Tests for ListenHereLoader — Listen Here! alignment JSON → AlignmentBundle.

This module tests ``ListenHereLoader`` against an inline **synthetic**
Listen Here! alignment JSON written to ``tmp_path`` (no pooch corpus,
parallel-safe).  A Listen Here! export describes many recordings of one
work warped onto a shared equidistant reference grid: per recording, a
``times`` array whose ``i``-th entry is that recording's clock-time
(seconds) at reference-grid column ``i``.

It verifies:

- the dense complete-topology pairwise claim field (C(R, 2) × columns
  synchronous instant claims) held columnar in a ``MatchClaimField``,
  reached through the uniform ``loader.get_field(MatchClaim)`` API (the
  ``loader.claim_field`` property is gone);
- the loader repr names the reference recording (``header.ref``);
- one seconds timeline per recording (float), each in its own group,
  holding no events and no C-Map, its ``length`` the recording's stored
  ``duration``;
- faithful preservation of negative pre-onset warp coordinates (never
  clamped or dropped);
- claims that are CHUNK on both sides (the file states grid positions,
  no events), each anchored at its column's two positions in seconds, the
  file's numbers exactly; no recording beside the file is read;
- a ``get_matchstamp_at`` headline read returning all recordings'
  coordinates at one reference instant via the columnar query path;
- every other bundle claim reader answering the columnar store with
  exact values (claim counts, filtered queries, the vectorized
  ``get_claim_field``, both matchstamp-table modes, commensurability,
  and a ``MatchLine`` → ``WarpMap`` transfer);
- the bare-array ``body.audio`` value form (``length == max(times)``);
- the field-level provenance (method ``imported``, a note naming
  ``header.createdBy`` and ``dtw_chroma_alignment``); and
- the ``ValueError`` paths (unequal ``times`` lengths, ``header.ref``
  absent from ``body.audio``, fewer than two recordings).

All counts and coordinates are exact per the Zero Tolerance Validation
Policy.  Validation logic is documented in ``tests/loader/README.md``.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pytest

from tests.loader.minting import minted
from timetoalign.alignment.claims import MatchClaim, MatchClaimField
from timetoalign.core import (
    CHUNK,
    Coordinate,
    IdCoordinateField,
    NumberType,
    Provenance,
    TimeUnit,
)
from timetoalign.loader.alignment import ListenHereLoader
from timetoalign.timelines import ContinuousPhysicalTimeline

# region Fixtures


#: The canonical synthetic specimen (mirrors tests/loader/README.md).
_CANONICAL_SPEC = {
    "header": {"ref": "rec-ref.mp3", "createdBy": "Listen Here! v0.20.0"},
    "body": {
        "audio": {
            "rec-a.mp3": {
                "times": [0.00, 0.02, 0.04, 0.06, 0.08],
                "peaks": [0.1, 0.2],
                "duration": 0.08,
            },
            "rec-b.mp3": {
                "times": [-0.01, 0.01, 0.03, 0.05, 0.07],
                "peaks": [0.1, 0.2],
                "duration": 0.10,
            },
            "rec-ref.mp3": {
                "times": [0.00, 0.025, 0.045, 0.065, 0.085],
                "peaks": [0.1, 0.2],
                "duration": 0.085,
            },
        }
    },
}


def _write_spec(directory: Path, spec: dict) -> Path:
    """Write ``spec`` as ``alignment.json`` under ``directory``; return path."""
    path = directory / "alignment.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def _write_silent_wav(path: Path, sample_rate: int) -> None:
    """Write a minimal WAV file whose metadata pins ``sample_rate``."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * sample_rate)


#: What the loader (seed 0) mints, in order: one timeline per recording in
#: sorted key order, then the 15 claims.
_MINTED = minted(0, *["cpt"] * 3, *["claim"] * 15)
REC = dict(zip(["rec-a", "rec-b", "rec-ref"], _MINTED[:3], strict=True))
CLAIMS = _MINTED[3:]

#: Each recording's grid times in seconds, as the file states them.
SECONDS = {
    stem: _CANONICAL_SPEC["body"]["audio"][f"{stem}.mp3"]["times"]
    for stem in ("rec-a", "rec-b", "rec-ref")
}


@pytest.fixture()
def canonical_path(tmp_path: Path) -> Path:
    """Path to the canonical synthetic specimen in ``tmp_path``."""
    return _write_spec(tmp_path, _CANONICAL_SPEC)


@pytest.fixture()
def loader(canonical_path: Path) -> ListenHereLoader:
    """A ``ListenHereLoader`` loaded from the canonical specimen."""
    return ListenHereLoader.from_file(canonical_path, seed=0)


# endregion


# region After-load counts


def test_claim_count(loader: ListenHereLoader) -> None:
    # C(3, 2) = 3 pairs × 5 grid columns = 15 synchronous claims.
    assert len(loader) == 15
    assert len(loader.get_field(MatchClaim)) == 15


def test_recording_keys(loader: ListenHereLoader) -> None:
    assert loader.recording_keys == ["rec-a", "rec-b", "rec-ref"]


def test_reference(loader: ListenHereLoader) -> None:
    assert loader.reference == "rec-ref.mp3"


def test_get_field_returns_match_claim_field(loader: ListenHereLoader) -> None:
    assert isinstance(loader.get_field(MatchClaim), MatchClaimField)


def test_get_field_by_field_class_is_same(loader: ListenHereLoader) -> None:
    # The paired Field class resolves to the same field as the scalar class.
    assert loader.get_field(MatchClaimField) is loader.get_field(MatchClaim)


def test_get_field_rejects_other_selectors(loader: ListenHereLoader) -> None:
    with pytest.raises(TypeError, match="MatchClaim"):
        loader.get_field(str)


def test_claim_field_property_is_gone(loader: ListenHereLoader) -> None:
    # The old hard-coded property was deleted in favour of get_field().
    assert not hasattr(loader, "claim_field")


def test_timeline_ids(loader: ListenHereLoader) -> None:
    assert loader.get_field(MatchClaim).timeline_ids == {
        REC["rec-a"],
        REC["rec-b"],
        REC["rec-ref"],
    }


def test_repr(loader: ListenHereLoader) -> None:
    assert repr(loader) == (
        "ListenHereLoader(recordings=3, reference='rec-ref.mp3', claims=15)"
    )


def test_repr_html_names_reference(loader: ListenHereLoader) -> None:
    html = loader._repr_html_()
    assert "rec-ref.mp3" in html
    assert "Reference" in html


def test_repr_html_shows_claims_not_zero_events(loader: ListenHereLoader) -> None:
    html = loader._repr_html_()
    # The payload count is named "Claims" (15), never the base "Events: 0".
    assert "<tr><td><b>Claims</b></td><td>15</td></tr>" in html
    assert "Events" not in html
    # Listen Here!-specific rows are present.
    assert "<tr><td><b>Recordings</b></td><td>3</td></tr>" in html
    assert "<b>File</b>" in html
    assert "in 3 group(s)" in html


# endregion


# region Faithfulness — negative coordinates kept


def test_negative_coordinate_kept(loader: ListenHereLoader) -> None:
    field = loader.get_field(MatchClaim)
    anchor = field.table.column("start_anchor").combine_chunks()
    minimum = min(
        pc.min(anchor.field("coordinate_a").field("value")).as_py(),
        pc.min(anchor.field("coordinate_b").field("value")).as_py(),
    )
    # rec-b's first grid column is -0.01 (pre-onset extrapolation); the
    # loader keeps it, neither clamped nor dropped.
    assert minimum == -0.01


# endregion


# region Bundle structure


def test_bundle_timeline_and_group_counts(loader: ListenHereLoader) -> None:
    bundle = loader.create_bundle()
    assert bundle.n_timelines == 3
    assert len(bundle.groups) == 3


def test_create_timelines_id_pattern_filters(loader: ListenHereLoader) -> None:
    """The pinned three-recording shape filters to the rec-b timeline."""
    timelines = loader.create_timelines(id_pattern=REC["rec-b"])
    assert len(timelines) == 1
    assert timelines[0].id == REC["rec-b"]
    assert timelines[0].name == "rec b"


def test_create_timelines_id_pattern_no_match(loader: ListenHereLoader) -> None:
    """An unmatched ID pattern returns no timelines."""
    assert loader.create_timelines(id_pattern=r"^missing:") == []


def test_bundle_timeline_uids(loader: ListenHereLoader) -> None:
    bundle = loader.create_bundle()
    assert set(bundle.timelines.keys()) == {
        REC["rec-a"],
        REC["rec-b"],
        REC["rec-ref"],
    }


def test_recordings_hold_no_events(loader: ListenHereLoader) -> None:
    # The file states grid positions, no events: no point, no Projection.
    bundle = loader.create_bundle()
    assert [len(timeline.events) for timeline in bundle.timelines.values()] == [
        0,
        0,
        0,
    ]


def test_recording_timelines_are_seconds(loader: ListenHereLoader) -> None:
    # The file gives seconds: each recording is a float seconds timeline,
    # without a samples C-Map or a sample rate.
    bundle = loader.create_bundle()
    assert [
        (
            type(timeline),
            timeline.unit,
            timeline.number_type,
            timeline.n_conversion_maps,
            timeline.meta,
        )
        for timeline in bundle.timelines.values()
    ] == [(ContinuousPhysicalTimeline, TimeUnit.seconds, NumberType.float, 0, {})] * 3


def test_recording_timeline_length(loader: ListenHereLoader) -> None:
    # Each length is the recording's stored duration, in seconds.
    bundle = loader.create_bundle()
    assert [bundle.timelines[REC[stem]].length for stem in REC] == [
        Coordinate(0.08, TimeUnit.seconds),
        Coordinate(0.10, TimeUnit.seconds),
        Coordinate(0.085, TimeUnit.seconds),
    ]


def _two_recordings(times: list[float]) -> dict[str, Any]:
    """A two-recording export (WAV keys) whose recordings share *times*."""
    return {
        "header": {"ref": "rec-a.wav"},
        "body": {
            "audio": {
                "rec-a.wav": {"times": times, "duration": times[-1]},
                "rec-b.wav": {"times": times, "duration": times[-1]},
            }
        },
    }


def test_recordings_beside_the_file_are_not_read(tmp_path: Path) -> None:
    # The loader represents what the file gives: recordings beside it change
    # nothing, so the bundle with them is the bundle without them.
    spec = _two_recordings([0.0, 0.02, 0.04])
    without = tmp_path / "without"
    without.mkdir()
    beside = tmp_path / "beside"
    beside.mkdir()
    for name in ("rec-a.wav", "rec-b.wav"):
        _write_silent_wav(beside / name, sample_rate=48000)
    read = [
        ListenHereLoader.from_file(_write_spec(directory, spec), seed=0)
        .create_bundle()
        .to_dict()
        for directory in (without, beside)
    ]
    assert read[0] == read[1]


def test_grid_seconds_are_kept_exactly(tmp_path: Path) -> None:
    # 0.001 s and 0.1 s stay the file's doubles: no nearest sample, no
    # guessed ratio.
    path = _write_spec(tmp_path, _two_recordings([0.0, 0.001, 0.1]))
    loader = ListenHereLoader.from_file(path)
    claims = list(loader.get_field(MatchClaim))
    assert [c.start_anchor.coordinates for c in claims] == [
        (Coordinate(value, TimeUnit.seconds), Coordinate(value, TimeUnit.seconds))
        for value in (0.0, 0.001, 0.1)
    ]
    assert [type(c.start_anchor.coordinate_a.value) for c in claims] == [float] * 3
    bundle = loader.create_bundle()
    assert [len(timeline.events) for timeline in bundle.timelines.values()] == [0, 0]


# endregion


# region Headline read — get_matchstamp_at


def test_matchstamp_spans_all_recordings(loader: ListenHereLoader) -> None:
    bundle = loader.create_bundle()
    stamp = bundle.get_matchstamp_at(0.045, REC["rec-ref"])
    assert stamp.n_timelines == 3
    # Reference-grid column index 2: rec-a=0.04, rec-b=0.03, rec-ref=0.045.
    assert {
        timeline_id: stamp.get_coordinate(timeline_id).to_coordinate()
        for timeline_id in REC.values()
    } == {
        REC["rec-a"]: Coordinate(0.04, TimeUnit.seconds),
        REC["rec-b"]: Coordinate(0.03, TimeUnit.seconds),
        REC["rec-ref"]: Coordinate(0.045, TimeUnit.seconds),
    }


def test_matchstamp_at_first_grid_coordinate(loader: ListenHereLoader) -> None:
    # Query an exact grid coordinate read off one recording: rec-a column 0
    # is 0.0, where rec-b is -0.01 and rec-ref is 0.0. Same instant in all 3.
    bundle = loader.create_bundle()
    stamp = bundle.get_matchstamp_at(0.0, REC["rec-a"])
    assert stamp.n_timelines == 3
    assert {
        timeline_id: stamp.get_coordinate(timeline_id).to_coordinate()
        for timeline_id in REC.values()
    } == {
        REC["rec-a"]: Coordinate(0.0, TimeUnit.seconds),
        REC["rec-b"]: Coordinate(-0.01, TimeUnit.seconds),
        REC["rec-ref"]: Coordinate(0.0, TimeUnit.seconds),
    }


# endregion


# region Columnar bundle answers every claim query


def test_bundle_claim_count(loader: ListenHereLoader) -> None:
    # The field's rows are the bundle's MatchClaim table, ids minted by the loader.
    table = loader.create_bundle().match_claim_table
    assert len(table) == 15
    assert table.table.column("id").to_pylist() == CLAIMS


def test_bundle_get_match_claims(loader: ListenHereLoader) -> None:
    bundle = loader.create_bundle()
    claims = bundle.get_match_claims()
    assert len(claims) == 15
    assert all(claim.is_synchronous for claim in claims)


def test_bundle_get_match_claims_filtered(loader: ListenHereLoader) -> None:
    bundle = loader.create_bundle()
    # rec-a pairs with the two other recordings at all 5 grid columns.
    assert len(bundle.get_match_claims(timeline_id=REC["rec-a"])) == 10
    assert len(bundle.get_match_claims(between=(REC["rec-a"], REC["rec-b"]))) == 5


def test_bundle_get_claim_field(loader: ListenHereLoader) -> None:
    # The vectorized accessor answers the same query without materialising.
    bundle = loader.create_bundle()
    assert len(bundle.get_claim_field()) == 15
    assert len(bundle.get_claim_field(timeline_id=REC["rec-a"])) == 10
    assert len(bundle.get_claim_field(timeline_id="missing:cpt_1")) == 0


def test_bundle_matchstamp_table_per_claim(loader: ListenHereLoader) -> None:
    bundle = loader.create_bundle()
    table = bundle.get_matchstamp_table()
    assert table.num_rows == 15
    # The columns follow the bundle's registration order, the loader's.
    assert list(bundle.timelines) == list(REC.values())
    assert table.column_names == list(REC.values())
    # Every row is one pairwise claim: exactly two filled cells.
    for row in table.to_pylist():
        assert sum(value is not None for value in row.values()) == 2


def test_bundle_matchstamp_table_from_graph(loader: ListenHereLoader) -> None:
    # The 15 pairwise rows collapse into the 5 reference-grid cross-sections.
    bundle = loader.create_bundle()
    table = bundle.get_matchstamp_table(from_graph=True)
    assert table.num_rows == 5
    assert list(bundle.timelines) == list(REC.values())
    assert table.column_names == list(REC.values())
    semantic_rows = [
        {
            name: (
                None
                if (coordinate := IdCoordinateField.from_table(table, name)[position])
                is None
                else coordinate.value
            )
            for name in table.column_names
        }
        for position in range(table.num_rows)
    ]
    assert semantic_rows == [
        {REC[stem]: SECONDS[stem][column] for stem in REC} for column in range(5)
    ]


def test_bundle_are_commensurable(loader: ListenHereLoader) -> None:
    bundle = loader.create_bundle()
    assert bundle.are_commensurable(REC["rec-a"], REC["rec-b"]) is True
    assert bundle.are_commensurable(REC["rec-b"], REC["rec-ref"]) is True
    assert bundle.are_commensurable(REC["rec-a"], "missing:cpt_1") is False


def test_bundle_transfer(loader: ListenHereLoader) -> None:
    # The MatchLine -> WarpMap path over the columnar store: grid column 1.
    bundle = loader.create_bundle()
    assert bundle.transfer(0.02, REC["rec-a"], REC["rec-b"]) == 0.01


def test_bundle_diagram_reports_claim_count(loader: ListenHereLoader) -> None:
    assert "MatchClaims: 15" in str(loader.create_bundle().diagram())


# endregion


# region Metadata


def test_claim_provenance(loader: ListenHereLoader) -> None:
    # Provenance is field-level, injected on read; pull a materialised claim
    # from the columnar field.
    claim = loader.get_field(MatchClaim)[0]
    assert claim.provenance == Provenance(
        method="imported", note="Listen Here! v0.20.0 dtw_chroma_alignment"
    )
    assert loader.get_field(MatchClaim).provenance == claim.provenance


def test_claims_are_chunk_on_both_sides(loader: ListenHereLoader) -> None:
    # The file names no event: every claim is CHUNK on both sides, in field
    # order the three pairs over the five grid columns, each anchored at its
    # column's two positions in seconds, the file's numbers.
    claims = list(loader.get_field(MatchClaim))
    pairs = [("rec-a", "rec-b"), ("rec-a", "rec-ref"), ("rec-b", "rec-ref")]
    assert [
        (c.timeline_a_id, c.event_a_id, c.timeline_b_id, c.event_b_id) for c in claims
    ] == [(REC[a], CHUNK, REC[b], CHUNK) for a, b in pairs for _ in range(5)]
    assert [c.chunk_sides for c in claims] == [
        (REC[a], REC[b]) for a, b in pairs for _ in range(5)
    ]
    assert [c.start_anchor.coordinates for c in claims] == [
        (
            Coordinate(SECONDS[a][column], TimeUnit.seconds),
            Coordinate(SECONDS[b][column], TimeUnit.seconds),
        )
        for a, b in pairs
        for column in range(5)
    ]
    assert all(c.end_anchor is None for c in claims)


# endregion


# region Bare-array audio value form


def test_bare_array_value_form(tmp_path: Path) -> None:
    spec = {
        "header": {"ref": "rec-x.mp3", "createdBy": "Listen Here! v0.20.0"},
        "body": {
            "audio": {
                # Object form for one, bare-array form for the other.
                "rec-x.mp3": {"times": [0.0, 0.02, 0.04], "duration": 0.04},
                "rec-y.mp3": [0.0, 0.02, 0.04],
            }
        },
    }
    path = _write_spec(tmp_path, spec)
    bundle = ListenHereLoader.from_file(path, seed=0).create_bundle()
    # The bare-array recording's length is the max of its times.
    assert bundle.timelines[minted(0, "cpt", "cpt")[1]].length == Coordinate(
        0.04, TimeUnit.seconds
    )


# endregion


# region Error cases


def test_unequal_times_lengths_raise(tmp_path: Path) -> None:
    spec = {
        "header": {"ref": "rec-a.mp3", "createdBy": "Listen Here!"},
        "body": {
            "audio": {
                "rec-a.mp3": {"times": [0.0, 0.02, 0.04], "duration": 0.04},
                "rec-b.mp3": {"times": [0.0, 0.02], "duration": 0.02},
            }
        },
    }
    path = _write_spec(tmp_path, spec)
    with pytest.raises(ValueError, match="same length"):
        ListenHereLoader.from_file(path)


def test_ref_not_in_audio_raises(tmp_path: Path) -> None:
    spec = {
        "header": {"ref": "missing.mp3", "createdBy": "Listen Here!"},
        "body": {
            "audio": {
                "rec-a.mp3": {"times": [0.0, 0.02], "duration": 0.02},
                "rec-b.mp3": {"times": [0.0, 0.02], "duration": 0.02},
            }
        },
    }
    path = _write_spec(tmp_path, spec)
    with pytest.raises(ValueError, match="header.ref"):
        ListenHereLoader.from_file(path)


def test_fewer_than_two_recordings_raises(tmp_path: Path) -> None:
    spec = {
        "header": {"ref": "rec-a.mp3", "createdBy": "Listen Here!"},
        "body": {
            "audio": {
                "rec-a.mp3": {"times": [0.0, 0.02], "duration": 0.02},
            }
        },
    }
    path = _write_spec(tmp_path, spec)
    with pytest.raises(ValueError, match="at least 2 recordings"):
        ListenHereLoader.from_file(path)


# endregion
