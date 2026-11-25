"""The format's expected flows (see README, *The format's expected flows*).

TimeToShare states, for each of its flow fixtures with flow control, the
flow that follows from it. The files were derived by hand from the rules the
engine implements; the engine is compared with them exactly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from timetoshare import fixture_paths, loads

from timetoalign import AlignmentBundle
from timetoalign.core.time import number_from_json, number_to_json
from timetoalign.timelines.flow.engine import AtomicSection, atomic_sections, play

FIXTURES = Path(os.path.commonpath(fixture_paths()))
"""The package's ``fixtures/<version>/`` directory."""
EXPECTED = FIXTURES.parent / "flows" / FIXTURES.name
"""The package's ``fixtures/flows/<version>/`` directory."""
CASES = sorted(path.stem for path in EXPECTED.glob("*.yaml"))

KNOWN_DIFFERENCES: dict[str, str] = {}
"""Cases whose file differs from the rules, each with the exact difference."""


def test_the_nine_cases_are_found() -> None:
    assert CASES == [
        "da_capo_al_fine",
        "dal_segno_al_coda",
        "loop",
        "ossia",
        "repairs",
        "repeat_end_and_dal_segno",
        "sections",
        "sonata_exposition",
        "volta_passes",
    ]
    assert set(KNOWN_DIFFERENCES) <= set(CASES)


def _joined(runs: list[list[Any]]) -> list[tuple[Any, Any]]:
    """The file's runs, each joined to the one before that it goes on from."""
    joined: list[tuple[Any, Any]] = []
    for start, end in runs:
        first, last = number_from_json(start), number_from_json(end)
        if joined and joined[-1][1] == first:
            joined[-1] = (joined[-1][0], last)
        else:
            joined.append((first, last))
    return joined


def _covered(
    runs: list[tuple[Any, Any]], sections: tuple[AtomicSection, ...]
) -> list[str]:
    """The letters of the atomic sections the runs cover, in the order played."""
    return [
        section.name
        for start, end in runs
        for section in sections
        if start <= section.start.value and section.end.value <= end
    ]


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            marks=(
                [pytest.mark.xfail(reason=KNOWN_DIFFERENCES[case], strict=True)]
                if case in KNOWN_DIFFERENCES
                else []
            ),
        )
        for case in CASES
    ],
)
def test_the_engine_plays_the_expected_flow(case: str) -> None:
    expected = loads((EXPECTED / f"{case}.yaml").read_text(encoding="utf-8"))
    manifest = loads(
        (FIXTURES / f"{expected['manifest']}.yaml").read_text(encoding="utf-8")
    )
    timeline = AlignmentBundle.from_dict(manifest).get_timeline(expected["timeline"])
    registry, length = timeline.flow_control, timeline.length

    sections = atomic_sections(registry, length)
    assert [
        {
            "name": section.name,
            "start": number_to_json(section.start.value),
            "end": number_to_json(section.end.value),
        }
        for section in sections
    ] == expected["atomic_sections"]

    assert set(expected["conventions"]) == {"default", "single", "printed"}
    for convention, stated in expected["conventions"].items():
        result = play(registry, length, convention=convention)
        runs = [(run.start.value, run.end.value) for run in result.runs]
        assert runs == _joined(stated["runs"]), convention
        assert [type(bound) for run in runs for bound in run] == [
            type(bound) for run in _joined(stated["runs"]) for bound in run
        ], convention
        assert _covered(runs, sections) == stated["sections"], convention
        if "findings" in stated:
            assert [
                {
                    "coordinate": number_to_json(finding.coordinate.value),
                    "kind": str(finding.kind),
                }
                for finding in result.findings
            ] == stated["findings"], convention
    assert "findings" in expected["conventions"]["default"]
