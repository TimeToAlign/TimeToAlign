"""Tests for the IEEE 1599 ``<structural>`` layer → spine external references.

The gymnopédie specimen is the only one of the six that carries a
``<structural>`` layer.  It states two analyses of the same piece, each as a
``<segmentation>`` of spine events plus a ``<petri_nets>`` block whose
``<place>`` elements bind Petri-net nodes to those segments.  Joining the two
halves on the segment id answers "which Petri-net node models this spine
event", which the loader carries as ``external_references`` on the spine.

It verifies:

- the exact row count and the arithmetic behind it — one row per
  ``(segment_event, place)`` pair, one fallback row per event of a segment no
  place names;
- exact resolutions from both analyses, access-point uris verbatim and
  unresolved;
- that the rows reach the spine identically through ``create_timeline()`` and
  through ``create_bundle()``, and that ``to_dict`` renders them only when
  asked; and
- that a specimen without a ``<structural>`` layer yields an empty table.

All counts and resolutions are exact per the Zero Tolerance Validation Policy.
Validation logic is documented in ``tests/loader/alignment/README.md``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from timetoalign.timelines.engines.external_references import (
    EXTERNAL_REFERENCE_SCHEMA,
)

#: One row per ``<segment_event>``: no segment of the specimen is named by two
#: places, so every per-segment multiplier is exactly one (README).
GYMNOPEDIE_ROWS = 1795

#: The one segment no ``<place>`` names, and the single event it covers.
UNMAPPED_SEGMENT = "Analisi_1_L3_RS_I_23"
UNMAPPED_COMMENT = "segment without petri-net node"


# region Helpers


def _references(loader: Any) -> pa.Table:
    """Return the external-reference table of a specimen's spine timeline."""
    return loader.create_timeline("spine:dlt_1").external_references


def _by_comment(table: pa.Table, comment: str) -> list[dict[str, Any]]:
    """Return the rows whose ``comment`` is exactly *comment*."""
    return table.filter(pc.equal(table.column("comment"), comment)).to_pylist()


def _access_uris(row: dict[str, Any]) -> list[str]:
    """Return the access-point uris of one row, in order."""
    return [point["uri"] for point in row["access_points"]]


# endregion


# region Resolution


class TestStructuralResolution:
    """Segment ↔ Petri-net-place resolution, exactly as the document states it."""

    def test_row_count_is_one_per_segment_event(self, ieee1599_loader) -> None:
        """1795 ``<segment_event>`` elements, 1795 rows: 892 + 903."""
        table = _references(ieee1599_loader("gymnopedie"))

        assert table.num_rows == GYMNOPEDIE_ROWS
        assert table.schema == EXTERNAL_REFERENCE_SCHEMA

    def test_both_analyses_contribute(self, ieee1599_loader) -> None:
        """The two analyses are independent and both are represented."""
        rows = _references(ieee1599_loader("gymnopedie")).to_pylist()

        per_analysis = Counter(
            uri.split("/")[0] for row in rows for uri in _access_uris(row)
        )
        assert per_analysis == {"Analisi_1": 891, "Analisi_2": 903}
        # The 892nd Analisi_1 row is the unmapped segment's, which has no uri.
        assert GYMNOPEDIE_ROWS == 891 + 903 + 1

    def test_analisi_1_l1_a_resolves_to_p2(self, ieee1599_loader) -> None:
        """The 32 events of ``Analisi_1_L1_A`` all map to place ``p2``."""
        rows = _by_comment(_references(ieee1599_loader("gymnopedie")), "Analisi_1_L1_A")

        assert len(rows) == 32
        assert {row["external_id"] for row in rows} == {"p2"}
        assert {tuple(_access_uris(row)) for row in rows} == {("Analisi_1/L1.pnml",)}
        assert rows[0] == {
            "event_id": "part_1_voice0_measure1_ev0",
            "external_id": "p2",
            "access_points": [{"uri": "Analisi_1/L1.pnml", "kind": "relative_path"}],
            "comment": "Analisi_1_L1_A",
        }

    def test_analisi_2_l2_rs_b_resolves_to_p4(self, ieee1599_loader) -> None:
        """The second analysis resolves the same way, into its own nets."""
        rows = _by_comment(
            _references(ieee1599_loader("gymnopedie")), "Analisi_2_L2_RS_B"
        )

        assert len(rows) == 10
        assert {row["external_id"] for row in rows} == {"p4"}
        assert {tuple(_access_uris(row)) for row in rows} == {("Analisi_2/L2-RS.pnml",)}
        assert rows[0]["event_id"] == "part_1_voice0_measure17_ev0"

    def test_unmapped_segment_keeps_its_events(self, ieee1599_loader) -> None:
        """The one segment no place names states itself instead."""
        rows = _by_comment(_references(ieee1599_loader("gymnopedie")), UNMAPPED_COMMENT)

        assert rows == [
            {
                "event_id": "part_1_voice0_measure72_ev0",
                "external_id": UNMAPPED_SEGMENT,
                "access_points": [],
                "comment": UNMAPPED_COMMENT,
            }
        ]

    def test_every_other_row_has_exactly_one_access_point(
        self, ieee1599_loader
    ) -> None:
        """One place per segment throughout, so one access point per row."""
        rows = _references(ieee1599_loader("gymnopedie")).to_pylist()

        assert Counter(len(row["access_points"]) for row in rows) == {
            1: GYMNOPEDIE_ROWS - 1,
            0: 1,
        }

    def test_rows_cover_every_segmented_spine_event(self, ieee1599_loader) -> None:
        """376 of the 382 spine events are segmented; six are not."""
        loader = ieee1599_loader("gymnopedie")
        spine = loader.create_timeline("spine:dlt_1")
        referenced = set(spine.external_references.column("event_id").to_pylist())
        spine_ids = spine.events.table.column("id").to_pylist()

        assert len(referenced) == 376
        assert sorted(set(spine_ids) - referenced) == [
            "Clef_part_1_1",
            "Clef_part_2_1",
            "KeySignature_part_1_1",
            "KeySignature_part_2_1",
            "TimeSignature_part_1_1",
            "TimeSignature_part_2_1",
        ]

    def test_access_point_uris_are_verbatim_and_unresolved(
        self, ieee1599_loader, ieee1599_dir
    ) -> None:
        """29 nets, two spellings, and no path resolved against disk."""
        table = _references(ieee1599_loader("gymnopedie"))
        uris = {uri for row in table.to_pylist() for uri in _access_uris(row)}

        assert len(uris) == 29
        assert sorted(uris) == [
            "Analisi_1/L1.pnml",
            "Analisi_1/L2_A.pnml",
            "Analisi_1/L2_B.pnml",
            "Analisi_1/L2_C.pnml",
            "Analisi_1/L2_D.pnml",
            "Analisi_1/L2_E.pnml",
            "Analisi_1/L2_F.pnml",
            "Analisi_1/L2_G.pnml",
            "Analisi_1/L2_H.pnml",
            "Analisi_1/L2_I.pnml",
            "Analisi_1/L2_L.pnml",
            "Analisi_1/L3_RI_A.pnml",
            "Analisi_1/L3_RI_B.pnml",
            "Analisi_1/L3_RI_C.pnml",
            "Analisi_1/L3_RS_A.pnml",
            "Analisi_1/L3_RS_C.pnml",
            "Analisi_1/L3_RS_D.pnml",
            "Analisi_1/L3_RS_G.pnml",
            "Analisi_1/L3_RS_I.pnml",
            "Analisi_2/L1.pnml",
            "Analisi_2/L2-RI.pnml",
            "Analisi_2/L2-RS.pnml",
            "Analisi_2/L3-RI-A.pnml",
            "Analisi_2/L3-RI-C.pnml",
            "Analisi_2/L3-RS-A.pnml",
            "Analisi_2/L3-RS-B.pnml",
            "Analisi_2/L3-RS-C.pnml",
            "Analisi_2/L3-RS-D.pnml",
            "Analisi_2/L3-RS-F.pnml",
        ]
        # The document's own directory holds no ``Analisi_1``; the net lives in
        # a sibling package directory, so the uri is stated, never resolved.
        satie = ieee1599_dir / "SatiePetriNets"
        assert not (satie / "ieee1599" / "Analisi_1" / "L1.pnml").exists()
        assert (satie / "petri" / "Analisi_1" / "L1.pnml").exists()

    def test_kinds_are_relative_paths(self, ieee1599_loader) -> None:
        """Every access point is a path relative to the document."""
        rows = _references(ieee1599_loader("gymnopedie")).to_pylist()

        assert {point["kind"] for row in rows for point in row["access_points"]} == {
            "relative_path"
        }


# endregion


# region Delivery


class TestStructuralDelivery:
    """Where the rows surface: the store, the bundle, and ``to_dict``."""

    def test_curated_structural_store_table(self, ieee1599_loader) -> None:
        """The layer gets its own curated table, one row per reference."""
        table = ieee1599_loader("gymnopedie").store["structural"].table

        assert table.num_rows == GYMNOPEDIE_ROWS
        assert table.column_names == [
            "event_id",
            "external_id",
            "access_points",
            "comment",
        ]

    def test_bundle_spine_carries_the_rows(
        self, ieee1599_loader, ieee1599_bundle
    ) -> None:
        """``create_bundle()`` and ``create_timeline()`` deliver the same spine."""
        loader = ieee1599_loader("gymnopedie")
        bundle_spine = ieee1599_bundle("gymnopedie").timelines["spine:dlt_1"]

        assert bundle_spine is loader.create_timeline("spine:dlt_1")
        assert bundle_spine.external_references.num_rows == GYMNOPEDIE_ROWS

    def test_to_dict_renders_the_rows_only_when_asked(self, ieee1599_loader) -> None:
        """The default serialization stays slim; the flag renders every row."""
        spine = ieee1599_loader("gymnopedie").create_timeline("spine:dlt_1")

        assert "external_references" not in spine.to_dict()
        rendered = spine.to_dict(external_references=True)["external_references"]
        assert len(rendered) == GYMNOPEDIE_ROWS
        assert rendered[0] == {
            "event_id": "part_1_voice0_measure1_ev0",
            "external_id": "p2",
            "access_points": [{"uri": "Analisi_1/L1.pnml", "kind": "relative_path"}],
            "comment": "Analisi_1_L1_A",
        }

    @pytest.mark.parametrize(
        "specimen, expected_tables",
        [
            ("animals", ["spine", "los", "notational", "audio"]),
            ("khomus", ["spine", "los", "staff_list", "notational", "audio"]),
        ],
    )
    def test_specimen_without_structural_layer(
        self, ieee1599_loader, specimen: str, expected_tables: list[str]
    ) -> None:
        """No ``<structural>`` layer, no table and no rows — never a null."""
        loader = ieee1599_loader(specimen)
        table = _references(loader)

        assert loader.keys() == expected_tables
        assert table.num_rows == 0
        assert table.schema == EXTERNAL_REFERENCE_SCHEMA


# endregion
