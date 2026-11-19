"""Tests for ``SegmentNameGenerator`` and its integration in the controller.

Two layers, both following the project's ZERO TOLERANCE policy (exact string
and exact count comparisons, no ranges):

1. Unit tests for the standalone ``SegmentNameGenerator`` — the alphabet walk,
   the numeric-overflow tail, the volta-suffix rule, the legacy sequential
   mode, custom alphabets, and the edge cases (leading volta, counter reset).
2. An integration test that builds a ``ScoreFlowController`` over the Op.18
   No.4 iv specimen (which carries three volta groups) and pins the exact
   atomic-section ids plus the ``to[]`` graph edges that reference them.

See ``README.md`` for the full validity rationale.
"""

from __future__ import annotations

from timetoalign.loader.score import Ms3Loader
from timetoalign.timelines import ScoreFlowController, SegmentNameGenerator

from .conftest import SCORE_DIR

# Folded measures TSV for Beethoven Op.18 No.4 iv (repeats + volta brackets).
SCORE_DATA_DIR = SCORE_DIR
OP18_MEASURES = (
    SCORE_DATA_DIR
    / "beethoven_op18-4iv_multimodal"
    / "op18_no4_mov4_flow"
    / "op18_no4_mov4_flow.measures.tsv"
)


# region Unit Tests: SegmentNameGenerator


class TestSegmentNameGenerator:
    """Exact-value tests for the standalone naming strategy."""

    def test_default_sequential(self) -> None:
        """No voltas -> plain alphabet walk."""
        gen = SegmentNameGenerator()
        assert gen.generate([False, False, False]) == ["A", "B", "C"]

    def test_alphabet_overflow_numeric_tail(self) -> None:
        """Beyond the alphabet length, labels repeat with a numeric suffix."""
        # A two-symbol alphabet: indices 0,1 -> X,Y; index 2 -> X2.
        gen = SegmentNameGenerator(alphabet=["X", "Y"])
        assert gen.generate([False, False, False]) == ["X", "Y", "X2"]

    def test_volta_suffix(self) -> None:
        """A base followed by two voltas -> base, base1, base2."""
        gen = SegmentNameGenerator()
        assert gen.generate([False, True, True]) == ["A", "A1", "A2"]

    def test_volta_suffix_disabled_is_sequential(self) -> None:
        """volta_suffix=False -> voltas consume the next base labels."""
        gen = SegmentNameGenerator(volta_suffix=False)
        assert gen.generate([False, True, True]) == ["A", "B", "C"]

    def test_custom_alphabet(self) -> None:
        """A caller-supplied alphabet drives the base labels."""
        gen = SegmentNameGenerator(alphabet=["X", "Y", "Z"])
        assert gen.generate([False, False, False]) == ["X", "Y", "Z"]

    def test_first_section_volta_fallback(self) -> None:
        """A leading volta with no preceding base falls back to a letter."""
        gen = SegmentNameGenerator()
        # First flag True -> no last_base yet -> base branch (never "None1").
        assert gen.generate([True, False, True]) == ["A", "B", "B1"]

    def test_two_volta_groups_reset(self) -> None:
        """A non-volta section resets the positional volta counter."""
        gen = SegmentNameGenerator()
        assert gen.generate([False, True, True, False, True]) == [
            "A",
            "A1",
            "A2",
            "B",
            "B1",
        ]

    def test_empty_flags(self) -> None:
        """No sections -> no labels."""
        assert SegmentNameGenerator().generate([]) == []


# endregion

# region Integration: ScoreFlowController labelling


class TestControllerVoltaLabelling:
    """The controller labels its atomic sections via the generator."""

    EXPECTED_IDS = [
        "A",
        "B",
        "C",
        "D",
        "D1",
        "D2",
        "E",
        "F",
        "F1",
        "F2",
        "G",
        "G1",
        "G2",
    ]
    # to[] edges, in section order, after the volta-suffix relabelling.
    EXPECTED_EDGES = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D1", "D2"),
        ("D",),
        ("E",),
        ("E", "F"),
        ("F1", "F2"),
        ("F",),
        ("G",),
        ("G1", "G2"),
        ("G",),
        (),
    ]
    LEGACY_IDS = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
    ]

    def _controller(
        self, name_generator: SegmentNameGenerator | None = None
    ) -> ScoreFlowController:
        loader = Ms3Loader.from_file(OP18_MEASURES)
        return ScoreFlowController(loader.store.measures, name_generator=name_generator)

    def test_default_volta_suffix_ids(self) -> None:
        """Default naming yields volta-suffixed ids (B-style: D, D1, D2)."""
        sections = self._controller().get_sections()
        assert [s.id for s in sections] == self.EXPECTED_IDS

    def test_section_count_unchanged(self) -> None:
        """Relabelling does not change the number of atomic sections."""
        assert len(self._controller().get_sections()) == 13

    def test_to_edges_carry_new_labels(self) -> None:
        """to[] graph edges reference the same volta-suffixed labels."""
        sections = self._controller().get_sections()
        assert [s.to for s in sections] == self.EXPECTED_EDGES

    def test_volta_suffix_disabled_is_legacy_sequential(self) -> None:
        """volta_suffix=False reproduces the pure-sequential ids."""
        gen = SegmentNameGenerator(volta_suffix=False)
        sections = self._controller(gen).get_sections()
        assert [s.id for s in sections] == self.LEGACY_IDS


# endregion
