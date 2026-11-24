"""The field-type and scalar-to-field maps that stored columns are read back through.

Validation logic is documented in ``tests/storage/README.md`` (the
``test_field_type_map.py`` section). Each check runs in a fresh interpreter,
since the pytest process has always imported every subpackage already.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_FIELD_TYPE_MAP = [
    "CoordinateField timetoalign.core.time.CoordinateField",
    "DurationField timetoalign.core.time.DurationField",
    "IdCoordinateField timetoalign.core.time.IdCoordinateField",
    "IdDurationField timetoalign.core.time.IdDurationField",
    "EnharmonicPitchField timetoalign.core.events.EnharmonicPitchField",
    "EnharmonicPitchClassField timetoalign.core.events.EnharmonicPitchClassField",
    "GenericPitchField timetoalign.core.events.GenericPitchField",
    "GenericPitchClassField timetoalign.core.events.GenericPitchClassField",
    "MidiPitchField timetoalign.core.events.MidiPitchField",
    "SpecificPitchField timetoalign.core.events.SpecificPitchField",
    "SpecificPitchClassField timetoalign.core.events.SpecificPitchClassField",
    "DcmlHarmonyField timetoalign.core.events.DcmlHarmonyField",
    "HarmonyLabelField timetoalign.core.events.HarmonyLabelField",
    "PitchBasedHarmonyField timetoalign.core.events.PitchBasedHarmonyField",
    "RomanNumeralHarmonyField timetoalign.core.events.RomanNumeralHarmonyField",
    "WesternTertianHarmonyField timetoalign.core.events.WesternTertianHarmonyField",
    "BeatField timetoalign.core.events.BeatField",
    "BeatPolicyField timetoalign.core.events.BeatPolicyField",
    "IdField timetoalign.core.events.IdField",
    "MeasureField timetoalign.core.events.MeasureField",
    "MeasureNumberField timetoalign.core.events.MeasureNumberField",
    "NoteField timetoalign.core.events.NoteField",
    "IntervalField timetoalign.core.time.IntervalField",
    "PointField timetoalign.core.events.PointField",
    "BoundingBoxField timetoalign.core.events.BoundingBoxField",
    "QuadrilateralField timetoalign.core.events.QuadrilateralField",
    "ScoreNoteField timetoalign.core.events.ScoreNoteField",
    "RegularMeasureField timetoalign.core.events.RegularMeasureField",
    "IrregularMeasureField timetoalign.core.events.IrregularMeasureField",
    "SplitRegularMeasureField timetoalign.core.events.SplitRegularMeasureField",
    "SplitIrregularMeasureField timetoalign.core.events.SplitIrregularMeasureField",
    "MeasureConstituentField timetoalign.core.events.MeasureConstituentField",
    "CadenzaMeasureField timetoalign.core.events.CadenzaMeasureField",
    "TempoField timetoalign.core.events.TempoField",
    "EventReferenceField timetoalign.core.events.EventReferenceField",
    "EventField timetoalign.core.events.EventField",
    "ProjectionField timetoalign.core.events.ProjectionField",
    "BoxField timetoalign.core.events.BoxField",
    "MidiEventField timetoalign.core.events.MidiEventField",
    "ScoreMidiEventField timetoalign.core.events.ScoreMidiEventField",
    "BeliefField timetoalign.core.provenance.BeliefField",
    "ProvenanceField timetoalign.core.provenance.ProvenanceField",
    "MatchClaimField timetoalign.alignment.claims.MatchClaimField",
]
"""Every entry as ``name module.Class``, in iteration order."""

_SCALAR_TO_FIELD_MAP = [
    f"{field.removesuffix('Field')} {field}"
    for field in (entry.split()[1] for entry in _FIELD_TYPE_MAP)
]
"""Every entry as ``module.Scalar module.Field``, in iteration order: each field
class of the field-type map, keyed by its scalar, which lives beside it under
the field's name without ``Field``."""

_PRINT_MAPS = """
from timetoalign.storage.mixins import _get_field_type_map, _get_scalar_to_field_map

def _name(cls):
    return f"{cls.__module__}.{cls.__qualname__}"

def print_maps(label):
    mapping = _get_field_type_map()
    for name, field_cls in mapping.items():
        print(f"{label}-type", name, _name(field_cls))
    for scalar_cls, field_cls in _get_scalar_to_field_map().items():
        print(f"{label}-scalar", _name(scalar_cls), _name(field_cls))
    return mapping
"""


def _run(*parts: str) -> list[str]:
    """Run *parts* as one script in a fresh interpreter; return its output lines."""
    script = "\n".join(textwrap.dedent(part) for part in parts)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.splitlines()


def _entries(lines: list[str], label: str) -> list[str]:
    """The map entries printed under *label*, without the label."""
    return [
        line.removeprefix(f"{label} ") for line in lines if line.startswith(f"{label} ")
    ]


def test_maps_after_import_timetoalign_grow_when_alignment_loads() -> None:
    lines = _run(
        """
        import sys

        import timetoalign

        assert "timetoalign.alignment" not in sys.modules, "alignment already imported"
        """,
        _PRINT_MAPS,
        """
        print_maps("bare")
        import timetoalign.alignment
        from timetoalign.alignment.claims import MatchClaim
        from timetoalign.storage.mixins import _get_field_class_for_scalar

        print_maps("alignment")
        print("MatchClaim", _get_field_class_for_scalar(MatchClaim).__qualname__)
        """,
    )
    assert _entries(lines, "bare-type") == _FIELD_TYPE_MAP[:-1]
    assert _entries(lines, "bare-scalar") == _SCALAR_TO_FIELD_MAP[:-1]
    assert _entries(lines, "alignment-type") == _FIELD_TYPE_MAP
    assert _entries(lines, "alignment-scalar") == _SCALAR_TO_FIELD_MAP
    assert lines[-1] == "MatchClaim MatchClaimField"


@pytest.mark.parametrize(
    "early_read", [True, False], ids=["early-read", "late-read-only"]
)
@pytest.mark.parametrize(
    "first_import",
    [
        "timetoalign",
        "timetoalign.storage",
        "timetoalign.storage.mixins",
        "timetoalign.core.fields",
        "timetoalign.core.events",
        "timetoalign.alignment.claims",
        "timetoalign.timelines",
    ],
)
def test_maps_do_not_depend_on_import_order(
    first_import: str, early_read: bool
) -> None:
    lines = _run(
        f"""
        import importlib
        import pkgutil
        import sys

        import {first_import}
        """,
        _PRINT_MAPS,
        f"""
        early = None
        if {early_read}:
            print("alignment loaded", "timetoalign.alignment" in sys.modules)
            early = print_maps("early")
        import timetoalign

        subpackages = [m.name for m in pkgutil.iter_modules(timetoalign.__path__) if m.ispkg]
        for name in reversed(subpackages):
            importlib.import_module(f"timetoalign.{{name}}")
        late = print_maps("late")
        print("same object", early is None or late is early)
        """,
    )
    assert _entries(lines, "late-type") == _FIELD_TYPE_MAP
    assert _entries(lines, "late-scalar") == _SCALAR_TO_FIELD_MAP
    assert lines[-1] == "same object True"
    if not early_read:
        assert _entries(lines, "early-type") == []
        return
    complete = {"alignment loaded True": True, "alignment loaded False": False}[
        lines[0]
    ]
    end = None if complete else -1
    assert _entries(lines, "early-type") == _FIELD_TYPE_MAP[:end]
    assert _entries(lines, "early-scalar") == _SCALAR_TO_FIELD_MAP[:end]
