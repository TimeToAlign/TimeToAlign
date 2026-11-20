"""Exact timeline extents derived from MS3 measure structure."""

from fractions import Fraction

from timetoalign.loader.score.ms3 import Ms3Loader


def _write_score_facets(tmp_path):
    measures_path = tmp_path / "miniature.measures.tsv"
    measures_path.write_text(
        "mc\tmn\tquarterbeats\tact_dur\ttimesig\n"
        "1\t1\t0\t1\t4/4\n"
        "2\t2\t4\t1\t4/4\n"
        "3\t3\t8\t1\t4/4\n",
        encoding="utf-8",
    )
    notes_path = tmp_path / "miniature.notes.tsv"
    notes_path.write_text(
        "mc\tmn\tquarterbeats\tduration_qb\tduration\tmidi\tname\toctave\n"
        "3\t3\t9\t1\t1/4\t60\tC\t4\n",
        encoding="utf-8",
    )
    return notes_path, measures_path


def test_full_score_timeline_uses_exact_notated_extent(tmp_path):
    notes_path, _ = _write_score_facets(tmp_path)

    timeline = Ms3Loader.from_file(notes_path, auto_discover=True).create_timeline()

    assert isinstance(timeline.length.value, Fraction)
    assert timeline.length.value == Fraction(12)


def test_measures_only_timeline_keeps_exact_notated_extent(tmp_path):
    _, measures_path = _write_score_facets(tmp_path)

    timeline = Ms3Loader.from_file(measures_path).create_timeline()

    assert isinstance(timeline.length.value, Fraction)
    assert timeline.length.value == Fraction(12)
