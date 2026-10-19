"""Event-level scalars and their paired SemanticField wrappers.

This is the catch-all module for TTA scalar types that are not the
fundamental Coordinate / Duration pair.  Each scalar is followed
immediately by its ``XField(SemanticField[X])`` class in the same file
— the paired Object + ObjectField is the unit of code organisation
and is never split across files (see Contributing → §2.4
Architectural decision log).

Sections (in file order):

* Graphical scalars + paired Fields (1 pair)
* Pitch scalars + paired Fields (7 pairs)
* Harmony scalars + paired Fields (5 pairs) — plus the DCML import
  schemas (``DcmlStorageSchema`` etc.) and the ``Inversion`` enum.
* Event scalars + paired Fields (Note, Measure, MeasureNumber, Id)
* MIDI event scalars + paired Fields (``MidiEvent`` base +
  ``ScoreMidiEvent`` subclass).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from typing import Any, ClassVar, Literal

import pyarrow as pa
import pyarrow.compute as pc
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from .fields import (
    ScalarVocabulary,
    SemanticField,
    build_struct_array,
    data_shaped,
    install_paired_field_registry,
    register_value_projector,
)
from .protocols import TwelveTETPitchMixin
from .time import (
    Coordinate,
    Duration,
    number_from_json,
    number_to_json,
    struct_to_rational,
)


def _pc_mod(arr: pa.Array, n: int) -> pa.Array:
    """Vectorized Python-semantics integer modulo (``a mod n``, n > 0).

    PyArrow does not expose a ``compute.mod`` function as of v23, and
    ``pc.divide`` on integers truncates toward zero (C-style) rather
    than flooring (Python-style).  To match Python's ``a % n``
    semantics on negative inputs:

    1. Truncated remainder: ``r = a - trunc(a / n) * n``
    2. Adjust to non-negative: ``((r + n) % n)`` via one more subtract.
       Equivalent: ``r_pos = r if r >= 0 else r + n``.

    Implemented as: ``r + n*(r < 0)`` via ``pc.add`` + ``pc.if_else``.
    """
    # TODO(pyarrow): replace with pc.mod once it lands — tracked upstream at
    # https://github.com/apache/arrow/issues/46901 (open as of pyarrow v23).
    quot = pc.divide(arr, n)
    r = pc.subtract(arr, pc.multiply(quot, n))
    # Adjust to non-negative when r is negative.
    return pc.if_else(pc.less(r, 0), pc.add(r, n), r)


# ═══════════════════════════════════════════════════════════════════════════
# 1. PITCH — scalars and paired Fields
# ═══════════════════════════════════════════════════════════════════════════

# Step-name to semitone offset from C (C=0, D=2, E=4, F=5, G=7, A=9, B=11)
_STEP_TO_SEMITONE: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

# Pitch-class to step name (for reverse mapping — natural notes only)
_PC_TO_STEP: dict[int, str] = {v: k for k, v in _STEP_TO_SEMITONE.items()}

# All 12 chromatic pitch class labels (for EPC display)
_PC_TO_LABEL: tuple[str, ...] = (
    "C",
    "C♯/D♭",
    "D",
    "D♯/E♭",
    "E",
    "F",
    "F♯/G♭",
    "G",
    "G♯/A♭",
    "A",
    "A♯/B♭",
    "B",
)

# Step-name to diatonic index (C=0, D=1, E=2, F=3, G=4, A=5, B=6)
_STEP_TO_GPC: dict[str, int] = {
    "C": 0,
    "D": 1,
    "E": 2,
    "F": 3,
    "G": 4,
    "A": 5,
    "B": 6,
}

# Base fifths for each step (unaltered)
_BASE_FIFTHS: dict[str, int] = {
    "F": -1,
    "C": 0,
    "G": 1,
    "D": 2,
    "A": 3,
    "E": 4,
    "B": 5,
}

# Regex for parsing pitch labels like "C#4", "Bb3", "D♯5", "F♭-1"
_PITCH_LABEL_RE = re.compile(
    r"^([A-Ga-g])"  # step letter
    r"([#♯]*|[b♭]*)"  # accidentals
    r"(-?\d+)?$"  # optional octave
)

_SPECIFIC_PITCH_STEPS = ("C", "D", "E", "F", "G", "A", "B")
_StepLiteral = Literal["C", "D", "E", "F", "G", "A", "B"]


def _step_alter_to_fifths(step: str, alter: int) -> int:
    """Compute line-of-fifths position from step and alter."""
    return _BASE_FIFTHS.get(step, 0) + (7 * alter)


def _parse_pitch_label(label: str) -> tuple[str, int, int | None]:
    """Parse a pitch label string into (step, alter, octave_or_None)."""
    m = _PITCH_LABEL_RE.match(label.strip())
    if m is None:
        raise ValueError(f"Cannot parse pitch label: {label!r}")
    step = m.group(1).upper()
    acc_str = m.group(2)
    octave_str = m.group(3)

    if not acc_str:
        alter = 0
    elif acc_str[0] in ("#", "♯"):
        alter = len(acc_str)
    else:
        alter = -len(acc_str)

    octave = int(octave_str) if octave_str is not None else None
    return step, alter, octave


def _parse_octave_from_sp(sp: str | None, step: str) -> int:
    """Extract octave number from a specific-pitch string like ``"C4"``."""
    if not sp:
        return 4
    try:
        idx = len(sp)
        while idx > 0 and (sp[idx - 1].isdigit() or sp[idx - 1] == "-"):
            idx -= 1
        if idx < len(sp):
            return int(sp[idx:])
    except (ValueError, IndexError):
        pass
    return 4


# ---------------------------------------------------------------------------
# BoundingBox + BoundingBoxField
# ---------------------------------------------------------------------------


class BoundingBox(ScalarVocabulary, BaseModel):
    """Axis-aligned rectangular extent in image coordinates.

    ``ul`` and ``lr`` are the upper-left and lower-right corners. Image
    coordinates are assumed: x grows rightward and y grows downward.
    Storage struct (derived): ``{ul: {x: float64, y: float64}, lr: {x:
    float64, y: float64}}``.
    """

    model_config = ConfigDict(frozen=True)

    class Point(BaseModel):
        """A two-dimensional point in image coordinates."""

        model_config = ConfigDict(frozen=True)

        x: float
        y: float

    ul: Point
    lr: Point

    @model_validator(mode="after")
    def _validate_corners(self) -> BoundingBox:
        """Require lower-right coordinates to be at or beyond upper-left."""
        coordinates = (self.ul.x, self.ul.y, self.lr.x, self.lr.y)
        if not all(math.isfinite(coordinate) for coordinate in coordinates):
            raise ValueError("BoundingBox coordinates must be finite")
        if self.lr.x < self.ul.x:
            raise ValueError("BoundingBox.lr.x must be greater than or equal to ul.x")
        if self.lr.y < self.ul.y:
            raise ValueError("BoundingBox.lr.y must be greater than or equal to ul.y")
        return self

    @classmethod
    def from_corners(
        cls,
        ulx: int | float,
        uly: int | float,
        lrx: int | float,
        lry: int | float,
    ) -> BoundingBox:
        """Construct a bounding box from upper-left and lower-right corners."""
        return cls(ul={"x": ulx, "y": uly}, lr={"x": lrx, "y": lry})

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> BoundingBox | None:
        """Materialize a bounding box scalar from an Arrow struct row."""
        if row.get("ul") is None or row.get("lr") is None:
            return None
        return cls.model_validate(row)


class BoundingBoxField(SemanticField[BoundingBox]):
    """Columnar wrapper for ``BoundingBox`` (paired Field)."""

    @classmethod
    def from_field(cls, source: Any, **kw: Any) -> BoundingBoxField:
        """Construct from standard field sources or bounding-box scalars."""
        if isinstance(source, list):
            if any(
                value is not None and not isinstance(value, BoundingBox)
                for value in source
            ):
                raise TypeError(
                    "BoundingBoxField list ingestion requires BoundingBox values"
                )
            array = build_struct_array(BoundingBox, source)
            field = pa.field(kw.pop("name", "bounding_box"), cls.pa_schema)
            return super().from_field((array, field), **kw)
        return super().from_field(source, **kw)

    @classmethod
    def matches_pa_field(cls, pa_field: pa.Field) -> bool:
        """Accept canonical boxes and raw integer/float corner structs."""
        if super().matches_pa_field(pa_field):
            return True
        if not pa.types.is_struct(pa_field.type):
            return False
        corners = {field.name: field.type for field in pa_field.type}
        if set(corners) != {"ul", "lr"}:
            return False
        for corner in corners.values():
            if not pa.types.is_struct(corner):
                return False
            coordinates = {field.name: field.type for field in corner}
            if set(coordinates) != {"x", "y"}:
                return False
            if not all(
                pa.types.is_integer(coordinate) or pa.types.is_floating(coordinate)
                for coordinate in coordinates.values()
            ):
                return False
        return True


install_paired_field_registry()


# ---------------------------------------------------------------------------
# EnharmonicPitchClass + EnharmonicPitchClassField
# ---------------------------------------------------------------------------


class EnharmonicPitchClass(TwelveTETPitchMixin, BaseModel):
    """Enharmonic pitch class scalar -- chromatic pitch class (0-11).

    Pydantic v2 ``BaseModel``, frozen.  Storage struct (derived):
    ``{pitch_class: int64}``.
    """

    model_config = ConfigDict(frozen=True)

    _REPR_ABBR: ClassVar[str] = "EPC"

    pitch_class: int

    def __init__(
        self,
        pitch_class: int | None = None,
        /,
        **data: Any,
    ) -> None:
        if pitch_class is not None:
            if "pitch_class" in data:
                raise TypeError(
                    "EnharmonicPitchClass received conflicting positional and "
                    "keyword arguments"
                )
            data = {"pitch_class": pitch_class, **data}
        super().__init__(**data)

    @property
    def semantic_type(self) -> str:
        return "EnharmonicPitchClass"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "pitch_type": "epc",
        }

    def to(
        self, target_type: type, *, format: str | None = None
    ) -> "EnharmonicPitchClass":
        if target_type is EnharmonicPitchClass or target_type is type(self):
            return self
        raise TypeError(
            f"Cannot convert EnharmonicPitchClass to {target_type.__name__} "
            f"(octave and/or spelling information required)"
        )

    @data_shaped
    def get(self, *, format: str | None = None) -> str:
        return str(self.pitch_class)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> EnharmonicPitchClass | None:
        pc = row.get("pitch_class")
        if pc is None:
            return None
        return cls(pitch_class=int(pc))

    def to_dict(self) -> dict[str, object]:
        return self.model_dump()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.pitch_class == other
        if isinstance(other, EnharmonicPitchClass):
            return self.pitch_class == other.pitch_class
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("EnharmonicPitchClass", self.pitch_class))

    def __str__(self) -> str:
        # Pretty label override: ``get()`` returns the bare pitch-class
        # integer (untouched -- its Field-vectorized mirror depends on it),
        # so ``str()`` mirrors the repr's inner token instead.
        if 0 <= self.pitch_class < 12:
            return _PC_TO_LABEL[self.pitch_class]
        return str(self.pitch_class)

    def __repr__(self) -> str:
        if 0 <= self.pitch_class < 12:
            return f"EPC({_PC_TO_LABEL[self.pitch_class]})"
        return f"EPC({self.pitch_class})"


class EnharmonicPitchClassField(SemanticField[EnharmonicPitchClass]):
    """Columnar wrapper for ``EnharmonicPitchClass`` (paired Field)."""

    def _pc_array(self) -> pa.Array:
        return self.to_pyarrow().field("pitch_class")

    def get(self, *, format: str | None = None) -> pa.Array:
        """Vectorized cast of ``pitch_class`` to string."""
        return pc.cast(self._pc_array(), pa.string())


# ---------------------------------------------------------------------------
# GenericPitchClass + GenericPitchClassField
# ---------------------------------------------------------------------------

_GPC_NAMES: tuple[str, ...] = ("C", "D", "E", "F", "G", "A", "B")


class GenericPitchClass(TwelveTETPitchMixin, BaseModel):
    """Generic pitch class scalar -- diatonic step (0-6).

    Pydantic v2 ``BaseModel``, frozen.  Storage struct (derived):
    ``{step: int64}``.
    """

    model_config = ConfigDict(frozen=True)

    _REPR_ABBR: ClassVar[str] = "GPC"

    step: int

    def __init__(
        self,
        step: int | None = None,
        /,
        **data: Any,
    ) -> None:
        if step is not None:
            if "step" in data:
                raise TypeError(
                    "GenericPitchClass received conflicting positional and "
                    "keyword arguments"
                )
            data = {"step": step, **data}
        super().__init__(**data)

    @property
    def pitch_class(self) -> int:  # type: ignore[override]
        return self.step

    @property
    def semantic_type(self) -> str:
        return "GenericPitchClass"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "pitch_type": "gpc",
        }

    def to(
        self, target_type: type, *, format: str | None = None
    ) -> "GenericPitchClass":
        if target_type is GenericPitchClass or target_type is type(self):
            return self
        raise TypeError(
            f"Cannot convert GenericPitchClass to {target_type.__name__} "
            f"(spelling or octave information required)"
        )

    @data_shaped
    def get(self, *, format: str | None = None) -> str:
        if 0 <= self.step < 7:
            return _GPC_NAMES[self.step]
        return str(self.step)

    def to_dict(self) -> dict[str, object]:
        return {"pitch_class": self.step}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> GenericPitchClass | None:
        pc = row.get("pitch_class")
        if pc is None:
            pc = row.get("step")
        if pc is None:
            return None
        return cls(step=int(pc))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.step == other
        if isinstance(other, GenericPitchClass):
            return self.step == other.step
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("GenericPitchClass", self.step))


class GenericPitchClassField(SemanticField[GenericPitchClass]):
    """Columnar wrapper for ``GenericPitchClass`` (paired Field)."""

    def _step_array(self) -> pa.Array:
        return self.to_pyarrow().field("step")

    def get(self, *, format: str | None = None) -> pa.Array:
        """Vectorized step-index → step-name lookup via a take table."""
        step = self._step_array()
        lookup = pa.array(list(_GPC_NAMES), type=pa.string())
        return pc.take(lookup, pc.cast(step, pa.int64()))


# ---------------------------------------------------------------------------
# GenericPitch + GenericPitchField
# ---------------------------------------------------------------------------


class GenericPitch(TwelveTETPitchMixin, BaseModel):
    """Generic pitch scalar -- diatonic step + octave.

    Pydantic v2 ``BaseModel``, frozen.  Storage struct (derived):
    ``{step: int64, octave: int64}``.
    """

    model_config = ConfigDict(frozen=True)

    _REPR_ABBR: ClassVar[str] = "GP"

    step: int
    octave: int

    def __init__(
        self,
        step: int | None = None,
        octave: int | None = None,
        /,
        **data: Any,
    ) -> None:
        positional = {
            k: v for k, v in (("step", step), ("octave", octave)) if v is not None
        }
        if positional and any(k in data for k in positional):
            raise TypeError(
                "GenericPitch received conflicting positional and keyword arguments"
            )
        data = {**positional, **data}
        super().__init__(**data)

    @property
    def pitch_class(self) -> int:  # type: ignore[override]
        return self.step

    @property
    def semantic_type(self) -> str:
        return "GenericPitch"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "pitch_type": "gp",
        }

    @data_shaped
    def to(
        self, target_type: type, *, format: str | None = None
    ) -> "TwelveTETPitchMixin":
        if target_type is GenericPitch or target_type is type(self):
            return self
        if target_type is GenericPitchClass:
            return GenericPitchClass(step=self.step)
        raise TypeError(
            f"Cannot convert GenericPitch to {target_type.__name__} "
            f"(accidental information required)"
        )

    @data_shaped
    def get(self, *, format: str | None = None) -> str:
        if 0 <= self.step < 7:
            return f"{_GPC_NAMES[self.step]}{self.octave}"
        return f"step={self.step}, oct={self.octave}"

    def to_dict(self) -> dict[str, object]:
        return {"pitch_class": self.step, "octave": self.octave}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> GenericPitch | None:
        pc = row.get("pitch_class")
        if pc is None:
            pc = row.get("step")
        if pc is None:
            return None
        octave = row.get("octave", 4)
        return cls(step=int(pc), octave=int(octave))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GenericPitch):
            return self.step == other.step and self.octave == other.octave
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("GenericPitch", self.step, self.octave))


class GenericPitchField(SemanticField[GenericPitch]):
    """Columnar wrapper for ``GenericPitch`` (paired Field)."""

    def _step_array(self) -> pa.Array:
        return self.to_pyarrow().field("step")

    def _octave_array(self) -> pa.Array:
        return self.to_pyarrow().field("octave")

    def convert_to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Vectorized mirror of ``GenericPitch.to(target_scalar_cls)``.

        Supports ``GenericPitch`` (identity) and ``GenericPitchClass``
        (drops the ``octave`` sub-field).
        """
        if target_scalar_cls is GenericPitch:
            return self
        if target_scalar_cls is GenericPitchClass:
            step = self._step_array()
            new_struct = pa.StructArray.from_arrays(
                [pc.cast(step, pa.int64())],
                fields=[pa.field("step", pa.int64(), nullable=True)],
            )
            pa_field = pa.field("step", new_struct.type)
            return GenericPitchClassField.from_field((new_struct, pa_field))
        raise TypeError(
            f"Cannot convert GenericPitch field to {target_scalar_cls.__name__}"
        )

    def to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Parity alias for :meth:`convert_to`."""
        return self.convert_to(target_scalar_cls)

    def get(self, *, format: str | None = None) -> pa.Array:
        """Vectorized step-name + octave string concatenation."""
        step = self._step_array()
        octave = self._octave_array()
        lookup = pa.array(list(_GPC_NAMES), type=pa.string())
        step_str = pc.take(lookup, pc.cast(step, pa.int64()))
        octave_str = pc.cast(octave, pa.string())
        return pc.binary_join_element_wise(step_str, octave_str, "")


# ---------------------------------------------------------------------------
# SpecificPitchClass + SpecificPitchClassField
# ---------------------------------------------------------------------------


class SpecificPitchClass(TwelveTETPitchMixin, BaseModel):
    """Pitch class with spelling.

    Pydantic v2 ``BaseModel``, frozen.  Storage struct (derived):
    ``{step: string, alter: int64}``.  ``fifths`` and ``pitch_class`` are
    derived (not stored).
    """

    model_config = ConfigDict(frozen=True)

    _REPR_ABBR: ClassVar[str] = "SPC"

    step: _StepLiteral
    alter: int = 0

    def __init__(
        self,
        step: str | None = None,
        alter: int | None = None,
        /,
        **data: Any,
    ) -> None:
        positional = {
            k: v for k, v in (("step", step), ("alter", alter)) if v is not None
        }
        if positional and any(k in data for k in positional):
            raise TypeError(
                "SpecificPitchClass received conflicting positional and "
                "keyword arguments"
            )
        data = {**positional, **data}
        super().__init__(**data)

    @field_validator("step", mode="before")
    @classmethod
    def _normalise_step(cls, v: object) -> str:
        if not isinstance(v, str):
            raise TypeError(f"step must be a string, got {type(v).__name__}")
        upper = v.upper()
        if upper not in _SPECIFIC_PITCH_STEPS:
            raise ValueError(f"step must be one of {_SPECIFIC_PITCH_STEPS}, got {v!r}")
        return upper

    @computed_field  # type: ignore[prop-decorator]
    @property
    @data_shaped
    def fifths(self) -> int:
        """Position on the line of fifths (derived from step + alter)."""
        return _step_alter_to_fifths(self.step, self.alter)

    @property
    @data_shaped
    def pitch_class(self) -> int:  # type: ignore[override]
        base = _STEP_TO_SEMITONE.get(self.step, 0)
        return (base + self.alter) % 12

    @property
    def semantic_type(self) -> str:
        return "SpecificPitchClass"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "pitch_type": "spc",
        }

    @data_shaped
    def to(
        self, target_type: type, *, format: str | None = None
    ) -> "SpecificPitchClass | EnharmonicPitchClass":
        if target_type is SpecificPitchClass or target_type is type(self):
            return self
        if target_type is EnharmonicPitchClass:
            return EnharmonicPitchClass(pitch_class=self.pitch_class)
        raise TypeError(
            f"Cannot convert SpecificPitchClass to {target_type.__name__} "
            f"(octave information required)"
        )

    @data_shaped
    def get(self, *, format: str | None = None) -> str:
        alter_str = ""
        if self.alter > 0:
            alter_str = "♯" * self.alter
        elif self.alter < 0:
            alter_str = "♭" * abs(self.alter)
        return f"{self.step}{alter_str}"

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "alter": self.alter,
            "pitch_class": self.pitch_class,
            "fifths": self.fifths,
            "label": self.get(),
        }

    @classmethod
    def from_string(cls, value: str) -> SpecificPitchClass:
        """Construct a spelled pitch class from its written string.

        Args:
            value: Pitch-class string such as ``"C♯"`` or ``"D♭"``.

        Returns:
            The parsed specific pitch class.
        """
        step, alter, _ = _parse_pitch_label(value)
        return cls(step=step, alter=alter)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> SpecificPitchClass | None:
        step = row.get("step") or row.get("gpc_str")
        if step is None:
            return None
        alter = row.get("alter")
        if alter is None:
            alter = row.get("acc", 0)
        return cls(
            step=str(step),
            alter=int(alter or 0),
        )


class SpecificPitchClassField(SemanticField[SpecificPitchClass]):
    """Columnar wrapper for ``SpecificPitchClass`` (paired Field)."""

    def _step_array(self) -> pa.Array:
        return self.to_pyarrow().field("step")

    def _alter_array(self) -> pa.Array:
        return self.to_pyarrow().field("alter")

    def _step_to_semitone(self) -> pa.Array:
        """Vectorized ``step (string) → semitone offset (int8)`` via take table.

        Encodes the mapping ``C=0, D=2, E=4, F=5, G=7, A=9, B=11``.
        Unknown steps yield 0 (matches scalar behaviour).
        """
        # Build a parallel pa.Table of (step, semitone) and look up the step.
        steps = self._step_array()
        # _SPECIFIC_PITCH_STEPS = ("C","D","E","F","G","A","B")
        step_to_semi = {s: _STEP_TO_SEMITONE[s] for s in _SPECIFIC_PITCH_STEPS}
        # pa.compute does not provide a dict lookup; emulate with a chained
        # ``replace_substring_regex``-style approach via ``pc.cast`` after
        # mapping through a struct.  Simpler: list of (case, output) via
        # successive ``pc.if_else`` calls.
        result = pa.array([0] * len(steps), type=pa.int64())
        for name, semi in step_to_semi.items():
            mask = pc.equal(steps, name)
            result = pc.if_else(mask, pa.scalar(semi, type=pa.int64()), result)
        return result

    def _step_to_base_fifths(self) -> pa.Array:
        """Vectorized step → base-fifths lookup."""
        steps = self._step_array()
        base = {s: _BASE_FIFTHS[s] for s in _SPECIFIC_PITCH_STEPS}
        result = pa.array([0] * len(steps), type=pa.int64())
        for name, f in base.items():
            mask = pc.equal(steps, name)
            result = pc.if_else(mask, pa.scalar(f, type=pa.int64()), result)
        return result

    @property
    def pitch_class(self) -> pa.Array:
        """Vectorized ``(step_semi + alter) mod 12``."""
        semi = self._step_to_semitone()
        alter = pc.cast(self._alter_array(), pa.int64())
        return _pc_mod(pc.add(semi, alter), 12)

    @property
    def fifths(self) -> pa.Array:
        """Vectorized line-of-fifths position (``base_fifths + 7*alter``)."""
        base = self._step_to_base_fifths()
        alter = pc.cast(self._alter_array(), pa.int64())
        return pc.add(base, pc.multiply(alter, 7))

    def convert_to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Vectorized mirror of ``SpecificPitchClass.to(target_scalar_cls)``.

        Supports identity, and ``EnharmonicPitchClass`` (collapses spelling
        to the chromatic pitch-class).
        """
        if target_scalar_cls is SpecificPitchClass:
            return self
        if target_scalar_cls is EnharmonicPitchClass:
            pc_arr = self.pitch_class
            new_struct = pa.StructArray.from_arrays(
                [pc.cast(pc_arr, pa.int64())],
                fields=[pa.field("pitch_class", pa.int64(), nullable=True)],
            )
            pa_field = pa.field("pitch_class", new_struct.type)
            return EnharmonicPitchClassField.from_field((new_struct, pa_field))
        raise TypeError(
            f"Cannot convert SpecificPitchClass field to {target_scalar_cls.__name__}"
        )

    def to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Parity alias for :meth:`convert_to`."""
        return self.convert_to(target_scalar_cls)

    def get(self, *, format: str | None = None) -> pa.Array:
        """Vectorized step + accidental concatenation.

        Implementation note: per-element accidental construction via
        ``pc.compute`` (no scalar materialisation loop) using
        ``pc.if_else`` over the small fixed alter range observed in
        musical practice (``-2..+2``).
        """
        step = self._step_array()
        alter = pc.cast(self._alter_array(), pa.int64())
        # Build accidental string per row: positive → "♯" * alter; negative
        # → "♭" * |alter|; zero → "".  Bounded alter range allows a
        # case-by-case if_else chain.
        out = pa.array([""] * len(step), type=pa.string())
        for n in (-2, -1, 1, 2):
            if n > 0:
                marker = "♯" * n
            else:
                marker = "♭" * abs(n)
            mask = pc.equal(alter, n)
            out = pc.if_else(mask, pa.scalar(marker, type=pa.string()), out)
        return pc.binary_join_element_wise(step, out, "")


# ---------------------------------------------------------------------------
# EnharmonicPitch + EnharmonicPitchField
# ---------------------------------------------------------------------------

# Note name labels for enharmonic display (sharp spelling for black keys)
_EP_LABELS: tuple[str, ...] = (
    "C",
    "C♯",
    "D",
    "E♭",
    "E",
    "F",
    "F♯",
    "G",
    "A♭",
    "A",
    "B♭",
    "B",
)


class EnharmonicPitch(TwelveTETPitchMixin, BaseModel):
    """Enharmonic pitch scalar -- MIDI number with pitch-name display.

    Pydantic v2 ``BaseModel``, frozen.  Storage struct (derived):
    ``{midi_number: int64}``.  ``pitch_class`` and ``octave`` are
    properties, not stored.
    """

    model_config = ConfigDict(frozen=True)

    _REPR_ABBR: ClassVar[str] = "EP"

    midi_number: int

    def __init__(
        self,
        midi_number: int | None = None,
        /,
        **data: Any,
    ) -> None:
        if midi_number is not None:
            if "midi_number" in data:
                raise TypeError(
                    f"{type(self).__name__} received conflicting positional and "
                    "keyword arguments"
                )
            data = {"midi_number": midi_number, **data}
        super().__init__(**data)

    @property
    @data_shaped
    def pitch_class(self) -> int:  # type: ignore[override]
        return self.midi_number % 12

    @property
    @data_shaped
    def octave(self) -> int:
        return (self.midi_number // 12) - 1

    @property
    def semantic_type(self) -> str:
        return "EnharmonicPitch"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "pitch_type": "ep",
        }

    @data_shaped
    def to(
        self, target_type: type, *, format: str | None = None
    ) -> "TwelveTETPitchMixin":
        if target_type is EnharmonicPitch or target_type is type(self):
            return self
        if target_type is MidiPitch:
            return MidiPitch(midi_number=self.midi_number)
        if target_type is EnharmonicPitchClass:
            return EnharmonicPitchClass(pitch_class=self.pitch_class)
        raise TypeError(
            f"Cannot convert {type(self).__name__} to {target_type.__name__} "
            f"(spelling information required for specific pitch types)"
        )

    @data_shaped
    def get(self, *, format: str | None = None) -> str:
        if format == "midi":
            return str(self.midi_number)
        return f"{_EP_LABELS[self.pitch_class]}{self.octave}"

    def to_dict(self) -> dict[str, object]:
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> EnharmonicPitch | None:
        if "midi_number" in row and row.get("midi_number") is not None:
            return cls(midi_number=int(row["midi_number"]))
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EnharmonicPitch):
            return self.midi_number == other.midi_number
        return NotImplemented

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.midi_number))

    def __str__(self) -> str:
        return f"{_PC_TO_LABEL[self.pitch_class]}{self.octave}"

    def __repr__(self) -> str:
        return f"EP({_PC_TO_LABEL[self.pitch_class]}{self.octave})"


class EnharmonicPitchField(SemanticField[EnharmonicPitch]):
    """Columnar wrapper for ``EnharmonicPitch`` (paired Field).

    The on-disk struct shape is the pydantic-derived
    ``{midi_number: int64}``; cached on the class as ``pa_schema``.
    """

    def _midi_array(self) -> pa.Array:
        return self.to_pyarrow().field("midi_number")

    @property
    def pitch_class(self) -> pa.Array:
        """Vectorized ``midi_number mod 12``."""
        return _pc_mod(pc.cast(self._midi_array(), pa.int64()), 12)

    @property
    def octave(self) -> pa.Array:
        """Vectorized ``(midi_number // 12) - 1``."""
        return pc.subtract(pc.divide(pc.cast(self._midi_array(), pa.int64()), 12), 1)

    def convert_to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Vectorized mirror of ``EnharmonicPitch.to(target_scalar_cls)``.

        Supports identity, ``MidiPitch`` (metadata-only retype), and
        ``EnharmonicPitchClass`` (``pc.mod(midi_number, 12)``).
        """
        if target_scalar_cls is EnharmonicPitch:
            return self
        if target_scalar_cls is MidiPitch:
            # Metadata-only retype: same struct, different scalar_cls.
            arr = self.to_pyarrow()
            pa_field = self.field.with_metadata(None) if self.field else None
            new_pa_field = pa.field(self.name, arr.type)
            return MidiPitchField.from_field((arr, new_pa_field))
        if target_scalar_cls is EnharmonicPitchClass:
            pc_arr = self.pitch_class
            new_struct = pa.StructArray.from_arrays(
                [pc.cast(pc_arr, pa.int64())],
                fields=[pa.field("pitch_class", pa.int64(), nullable=True)],
            )
            pa_field = pa.field("pitch_class", new_struct.type)
            return EnharmonicPitchClassField.from_field((new_struct, pa_field))
        raise TypeError(
            f"Cannot convert EnharmonicPitch field to {target_scalar_cls.__name__}"
        )

    def to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Parity alias for :meth:`convert_to`."""
        return self.convert_to(target_scalar_cls)

    def get(self, *, format: str | None = None) -> pa.Array:
        """Vectorized label + octave string concatenation.

        Implementation: pitch-class → label via ``pc.take`` from the
        12-element ``_EP_LABELS`` lookup table, then concat with the
        string-cast octave.
        """
        if format == "midi":
            return pc.cast(self._midi_array(), pa.string())
        pc_arr = self.pitch_class
        labels = pa.array(list(_EP_LABELS), type=pa.string())
        label_str = pc.take(labels, pc.cast(pc_arr, pa.int64()))
        octave_str = pc.cast(self.octave, pa.string())
        return pc.binary_join_element_wise(label_str, octave_str, "")


# ---------------------------------------------------------------------------
# MidiPitch + MidiPitchField
# ---------------------------------------------------------------------------


class MidiPitch(EnharmonicPitch):
    """Display alias of :class:`EnharmonicPitch` reserved for ``MidiPitchField``.

    Identical data, storage struct, conversions, and protocol conformance.
    Differs only in ``__repr__``, ``semantic_type``, and the default
    ``get()`` format.
    """

    _REPR_ABBR: ClassVar[str] = "MP"

    @property
    def semantic_type(self) -> str:
        return "MidiPitch"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "pitch_type": "ep",
        }

    @data_shaped
    def get(self, *, format: str | None = None) -> str:
        if format is None or format == "midi":
            return str(self.midi_number)
        return super().get(format=format)

    def __str__(self) -> str:
        return str(self.midi_number)

    def __repr__(self) -> str:
        return f"MP({self.midi_number})"


class MidiPitchField(SemanticField[MidiPitch]):
    """Columnar wrapper for ``MidiPitch`` (paired Field).

    Same struct as ``EnharmonicPitchField`` (``MidiPitch`` is a subclass
    of ``EnharmonicPitch``), but with distinct ``scalar_cls`` so
    ``__getitem__`` yields ``MidiPitch`` instances.
    """

    def _midi_array(self) -> pa.Array:
        return self.to_pyarrow().field("midi_number")

    @property
    def pitch_class(self) -> pa.Array:
        return _pc_mod(pc.cast(self._midi_array(), pa.int64()), 12)

    @property
    def octave(self) -> pa.Array:
        return pc.subtract(pc.divide(pc.cast(self._midi_array(), pa.int64()), 12), 1)

    def convert_to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Vectorized mirror of ``EnharmonicPitch.to`` for the MidiPitch view."""
        if target_scalar_cls is MidiPitch:
            return self
        if target_scalar_cls is EnharmonicPitch:
            arr = self.to_pyarrow()
            new_pa_field = pa.field(self.name, arr.type)
            return EnharmonicPitchField.from_field((arr, new_pa_field))
        if target_scalar_cls is EnharmonicPitchClass:
            pc_arr = self.pitch_class
            new_struct = pa.StructArray.from_arrays(
                [pc.cast(pc_arr, pa.int64())],
                fields=[pa.field("pitch_class", pa.int64(), nullable=True)],
            )
            pa_field = pa.field("pitch_class", new_struct.type)
            return EnharmonicPitchClassField.from_field((new_struct, pa_field))
        raise TypeError(
            f"Cannot convert MidiPitch field to {target_scalar_cls.__name__}"
        )

    def to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Parity alias for :meth:`convert_to`."""
        return self.convert_to(target_scalar_cls)

    def get(self, *, format: str | None = None) -> pa.Array:
        """Vectorized cast of ``midi_number`` to string (default format)."""
        if format is None or format == "midi":
            return pc.cast(self._midi_array(), pa.string())
        # Delegate to EP-style formatting when an explicit non-midi format is asked
        pc_arr = self.pitch_class
        labels = pa.array(list(_EP_LABELS), type=pa.string())
        label_str = pc.take(labels, pc.cast(pc_arr, pa.int64()))
        octave_str = pc.cast(self.octave, pa.string())
        return pc.binary_join_element_wise(label_str, octave_str, "")


# ---------------------------------------------------------------------------
# SpecificPitch + SpecificPitchField
# ---------------------------------------------------------------------------


class SpecificPitch(TwelveTETPitchMixin, BaseModel):
    """Specific pitch scalar with full enharmonic identity.

    Storage struct (derived): ``{step: string, alter: int64, octave: int64,
    cents: float64?}``.  ``fifths``, ``midi_number``, ``pitch_class`` are
    derived (not stored).
    """

    model_config = ConfigDict(frozen=True)

    _REPR_ABBR: ClassVar[str] = "SP"

    step: _StepLiteral
    alter: int = 0
    octave: int
    cents: float | None = None

    @field_validator("step", mode="before")
    @classmethod
    def _normalise_step(cls, v: object) -> str:
        if not isinstance(v, str):
            raise TypeError(f"step must be a string, got {type(v).__name__}")
        upper = v.upper()
        if upper not in _SPECIFIC_PITCH_STEPS:
            raise ValueError(f"step must be one of {_SPECIFIC_PITCH_STEPS}, got {v!r}")
        return upper

    @computed_field  # type: ignore[prop-decorator]
    @property
    @data_shaped
    def fifths(self) -> int:
        return _step_alter_to_fifths(self.step, self.alter)

    @property
    @data_shaped
    def midi_number(self) -> int:
        base = _STEP_TO_SEMITONE.get(self.step, 0)
        return (self.octave + 1) * 12 + base + self.alter

    @property
    @data_shaped
    def pitch_class(self) -> int:  # type: ignore[override]
        base = _STEP_TO_SEMITONE.get(self.step, 0)
        return (base + self.alter) % 12

    @property
    def semantic_type(self) -> str:
        return "SpecificPitch"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "pitch_type": "sp",
        }

    @data_shaped
    def to(
        self, target_type: type, *, format: str | None = None
    ) -> "TwelveTETPitchMixin":
        if target_type is SpecificPitch or target_type is type(self):
            return self
        if target_type is MidiPitch:
            return MidiPitch(midi_number=self.midi_number)
        if target_type is EnharmonicPitchClass:
            return EnharmonicPitchClass(pitch_class=self.pitch_class)
        if target_type is SpecificPitchClass:
            return SpecificPitchClass(step=self.step, alter=self.alter)
        raise TypeError(f"Cannot convert SpecificPitch to {target_type.__name__}")

    @data_shaped
    def get(self, *, format: str | None = None) -> str:
        if format == "midi":
            return str(self.midi_number)
        alter_str = ""
        if self.alter > 0:
            alter_str = "♯" * self.alter
        elif self.alter < 0:
            alter_str = "♭" * abs(self.alter)
        return f"{self.step}{alter_str}{self.octave}"

    def to_dict(self) -> dict[str, object]:
        """Return the declared and computed fields plus the derived labels."""
        return {
            **self.model_dump(),
            "midi_number": self.midi_number,
            "pitch_class": self.pitch_class,
            "label": self.get(),
        }

    @classmethod
    def from_string(cls, value: str) -> SpecificPitch:
        """Construct a spelled pitch from its written string.

        Args:
            value: Pitch string including its octave, such as ``"C♯4"``.

        Returns:
            The parsed specific pitch.

        Raises:
            ValueError: If the pitch string has no octave.
        """
        step, alter, octave = _parse_pitch_label(value)
        if octave is None:
            raise ValueError(
                f"Octave required for SpecificPitch, got {value!r}. "
                f"Use SpecificPitchClass.from_string() for octave-free pitches."
            )
        return cls(step=step, alter=alter, octave=octave)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SpecificPitch | None":
        """Construct from a PyArrow struct row dict (trust-boundary regime).

        Accepts the canonical struct ``{step, alter, octave, cents}``.
        """
        if "step" in row and row.get("step") is not None:
            cents_raw = row.get("cents")
            cents = float(cents_raw) if cents_raw is not None else None
            return cls(
                step=str(row["step"]),
                alter=int(row.get("alter", 0) or 0),
                octave=int(row["octave"]),
                cents=cents,
            )
        return None


class SpecificPitchField(SemanticField[SpecificPitch]):
    """Columnar wrapper for ``SpecificPitch`` (paired Field).

    Pa schema: ``{step: string, alter: int64, octave: int64, cents: float64}``.
    """

    def _step_array(self) -> pa.Array:
        return self.to_pyarrow().field("step")

    def _alter_array(self) -> pa.Array:
        return self.to_pyarrow().field("alter")

    def _octave_array(self) -> pa.Array:
        return self.to_pyarrow().field("octave")

    def _step_to_semitone(self) -> pa.Array:
        """Vectorized ``step`` (string) → semitone offset (int64) lookup."""
        steps = self._step_array()
        result = pa.array([0] * len(steps), type=pa.int64())
        for name, semi in _STEP_TO_SEMITONE.items():
            mask = pc.equal(steps, name)
            result = pc.if_else(mask, pa.scalar(semi, type=pa.int64()), result)
        return result

    def _step_to_base_fifths(self) -> pa.Array:
        steps = self._step_array()
        result = pa.array([0] * len(steps), type=pa.int64())
        for name, f in _BASE_FIFTHS.items():
            mask = pc.equal(steps, name)
            result = pc.if_else(mask, pa.scalar(f, type=pa.int64()), result)
        return result

    @property
    def pitch_class(self) -> pa.Array:
        """Vectorized ``(step_semi + alter) mod 12``."""
        semi = self._step_to_semitone()
        alter = pc.cast(self._alter_array(), pa.int64())
        return _pc_mod(pc.add(semi, alter), 12)

    @property
    def midi_number(self) -> pa.Array:
        """Vectorized ``(octave + 1) * 12 + step_semi + alter``."""
        semi = self._step_to_semitone()
        alter = pc.cast(self._alter_array(), pa.int64())
        octave = pc.cast(self._octave_array(), pa.int64())
        return pc.add(pc.add(pc.multiply(pc.add(octave, 1), 12), semi), alter)

    @property
    def fifths(self) -> pa.Array:
        """Vectorized line-of-fifths position (``base_fifths + 7*alter``)."""
        base = self._step_to_base_fifths()
        alter = pc.cast(self._alter_array(), pa.int64())
        return pc.add(base, pc.multiply(alter, 7))

    def convert_to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Vectorized mirror of ``SpecificPitch.to(target_scalar_cls)``.

        Supports identity, ``MidiPitch``, ``EnharmonicPitchClass``,
        ``SpecificPitchClass`` — matching the scalar method's supported
        targets.  ``EnharmonicPitch`` is intentionally NOT supported
        here (the scalar ``to`` does not support it either; users go
        through ``MidiPitch`` as a thin alias).
        """
        if target_scalar_cls is SpecificPitch:
            return self
        if target_scalar_cls is MidiPitch:
            midi_arr = pc.cast(self.midi_number, pa.int64())
            new_struct = pa.StructArray.from_arrays(
                [midi_arr],
                fields=[pa.field("midi_number", pa.int64(), nullable=True)],
            )
            pa_field = pa.field("midi_number", new_struct.type)
            return MidiPitchField.from_field((new_struct, pa_field))
        if target_scalar_cls is EnharmonicPitchClass:
            pc_arr = pc.cast(self.pitch_class, pa.int64())
            new_struct = pa.StructArray.from_arrays(
                [pc_arr],
                fields=[pa.field("pitch_class", pa.int64(), nullable=True)],
            )
            pa_field = pa.field("pitch_class", new_struct.type)
            return EnharmonicPitchClassField.from_field((new_struct, pa_field))
        if target_scalar_cls is SpecificPitchClass:
            step = self._step_array()
            alter = pc.cast(self._alter_array(), pa.int64())
            new_struct = pa.StructArray.from_arrays(
                [step, alter],
                fields=[
                    pa.field("step", pa.string(), nullable=True),
                    pa.field("alter", pa.int64(), nullable=True),
                ],
            )
            pa_field = pa.field("spc", new_struct.type)
            return SpecificPitchClassField.from_field((new_struct, pa_field))
        raise TypeError(
            f"Cannot convert SpecificPitch field to {target_scalar_cls.__name__}"
        )

    def to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Parity alias for :meth:`convert_to`."""
        return self.convert_to(target_scalar_cls)

    def get(self, *, format: str | None = None) -> pa.Array:
        """Vectorized ``step + accidental + octave`` string concatenation."""
        if format == "midi":
            return pc.cast(self.midi_number, pa.string())
        step = self._step_array()
        alter = pc.cast(self._alter_array(), pa.int64())
        octave_str = pc.cast(self._octave_array(), pa.string())
        out = pa.array([""] * len(step), type=pa.string())
        for n in (-2, -1, 1, 2):
            if n > 0:
                marker = "♯" * n
            else:
                marker = "♭" * abs(n)
            mask = pc.equal(alter, n)
            out = pc.if_else(mask, pa.scalar(marker, type=pa.string()), out)
        return pc.binary_join_element_wise(
            pc.binary_join_element_wise(step, out, ""), octave_str, ""
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. HARMONY — scalars, paired Fields, plus DCML import schemas
# ═══════════════════════════════════════════════════════════════════════════


class Inversion(IntEnum):
    """Chord inversion as an enum.

    The DCML ``figbass`` string maps to these values.
    """

    ROOT = 0
    FIRST = 1
    SECOND = 2
    THIRD = 3


# DCML figbass -> Inversion enum mapping
FIGBASS_TO_INVERSION: dict[str, Inversion] = {
    "": Inversion.ROOT,
    "7": Inversion.ROOT,
    "6": Inversion.FIRST,
    "64": Inversion.SECOND,
    "2": Inversion.THIRD,
    "65": Inversion.FIRST,
    "43": Inversion.SECOND,
    "42": Inversion.THIRD,
}


def figbass_to_inversion(figbass: str) -> Inversion | None:
    """Convert a DCML ``figbass`` string to an ``Inversion`` enum member.

    Args:
        figbass: The DCML figured bass string.

    Returns:
        The corresponding ``Inversion`` enum member, or ``None`` if
        the string is not a recognised figured bass.
    """
    if not figbass:
        return Inversion.ROOT
    return FIGBASS_TO_INVERSION.get(figbass)


# ---------------------------------------------------------------------------
# DCML import schemas — distinct from the canonical paired XField.pa_schema.
# These are the *external* DCML TSV column layouts used at the import edge.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarmonyBaseSchema:
    """Minimal harmony schema: label and codec standard.

    Storage struct::

        {label: string, standard: string}
    """

    label: str = "label"
    standard: str = "standard"

    schema: ClassVar[pa.StructType] = pa.struct(
        [
            pa.field("label", pa.string(), nullable=True),
            pa.field("standard", pa.string(), nullable=True),
        ]
    )


@dataclass(frozen=True)
class WesternTertianSchema:
    """Schema for Western tertian harmony import: root, bass, chord type, inversion."""

    label: str = "label"
    standard: str = "standard"
    root: str = "root"
    bass: str = "bass"
    chord_type: str = "chord_type"
    inversion: str = "inversion"

    schema: ClassVar[pa.StructType] = pa.struct(
        [
            pa.field("label", pa.string(), nullable=True),
            pa.field("standard", pa.string(), nullable=True),
            pa.field("root", pa.int64(), nullable=True),
            pa.field("bass", pa.int64(), nullable=True),
            pa.field("chord_type", pa.string(), nullable=True),
            pa.field("inversion", pa.int64(), nullable=True),
        ]
    )


@dataclass(frozen=True)
class RomanNumeralSchema:
    """Schema for Roman numeral harmony import: adds numeral, localkey, globalkey."""

    label: str = "label"
    standard: str = "standard"
    root: str = "root"
    bass: str = "bass"
    chord_type: str = "chord_type"
    inversion: str = "inversion"
    numeral: str = "numeral"
    localkey: str = "localkey"
    globalkey: str = "globalkey"
    key_context: str = "key_context"

    schema: ClassVar[pa.StructType] = pa.struct(
        [
            pa.field("label", pa.string(), nullable=True),
            pa.field("standard", pa.string(), nullable=True),
            pa.field("root", pa.int64(), nullable=True),
            pa.field("bass", pa.int64(), nullable=True),
            pa.field("chord_type", pa.string(), nullable=True),
            pa.field("inversion", pa.int64(), nullable=True),
            pa.field("numeral", pa.string(), nullable=True),
            pa.field("localkey", pa.string(), nullable=True),
            pa.field("globalkey", pa.string(), nullable=True),
            pa.field("key_context", pa.string(), nullable=True),
        ]
    )


@dataclass(frozen=True)
class DcmlStorageSchema:
    """Storage schema for DCML harmony labels (the raw DCML TSV columns).

    Import-edge schema, NOT the canonical ``DcmlHarmonyField.pa_schema``.
    Used by external DCML reads which then map to internal model via
    ``DcmlHarmony.from_row``.
    """

    label: str = "label"
    globalkey: str = "globalkey"
    localkey: str = "localkey"
    numeral: str = "numeral"
    form: str = "form"
    figbass: str = "figbass"
    chord_type: str = "chord_type"
    root: str = "root"
    bass_note: str = "bass_note"

    schema: ClassVar[pa.StructType] = pa.struct(
        [
            pa.field("label", pa.string(), nullable=True),
            pa.field("globalkey", pa.string(), nullable=True),
            pa.field("localkey", pa.string(), nullable=True),
            pa.field("numeral", pa.string(), nullable=True),
            pa.field("form", pa.string(), nullable=True),
            pa.field("figbass", pa.string(), nullable=True),
            pa.field("chord_type", pa.string(), nullable=True),
            pa.field("root", pa.int64(), nullable=True),
            pa.field("bass_note", pa.int64(), nullable=True),
        ]
    )


# ---------------------------------------------------------------------------
# Harmony scalars + paired Fields
# ---------------------------------------------------------------------------


class HarmonyLabel(ScalarVocabulary, BaseModel):
    """Root harmony scalar.

    Pydantic v2 ``BaseModel``, frozen.  Minimal: label + standard.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    standard: str

    @property
    def semantic_type(self) -> str:
        return "HarmonyLabel"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "standard": self.standard,
        }

    def to_dict(self) -> dict[str, object]:
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> HarmonyLabel | None:
        label = row.get("label")
        if label is None:
            return None
        return cls(label=str(label), standard=str(row.get("standard") or ""))

    def __repr__(self) -> str:
        return f"HarmonyLabel(label={self.label!r}, standard={self.standard!r})"

    def __str__(self) -> str:
        return self.label


class HarmonyLabelField(SemanticField[HarmonyLabel]):
    """Columnar wrapper for ``HarmonyLabel`` (paired Field)."""


class PitchBasedHarmony(ScalarVocabulary, BaseModel):
    """Harmony with root and bass (OHR model)."""

    model_config = ConfigDict(frozen=True)

    label: str
    standard: str
    root: int | None = None
    bass: int | None = None

    @property
    def semantic_type(self) -> str:
        return "PitchBasedHarmony"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "standard": self.standard,
        }

    def to_dict(self) -> dict[str, object]:
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> PitchBasedHarmony | None:
        label = row.get("label")
        if label is None:
            return None
        root_raw = row.get("root")
        bass_raw = row.get("bass", row.get("bass_note"))
        return cls(
            label=str(label),
            standard=str(row.get("standard") or ""),
            root=int(root_raw) if root_raw is not None else None,
            bass=int(bass_raw) if bass_raw is not None else None,
        )

    def __repr__(self) -> str:
        return f"PitchBasedHarmony(label={self.label!r}, root={self.root})"

    def __str__(self) -> str:
        return self.label


class PitchBasedHarmonyField(SemanticField[PitchBasedHarmony]):
    """Columnar wrapper for ``PitchBasedHarmony`` (paired Field)."""


class WesternTertianHarmony(ScalarVocabulary, BaseModel):
    """Western tertian chord."""

    model_config = ConfigDict(frozen=True)

    label: str
    standard: str
    root: int | None = None
    bass: int | None = None
    chord_type: str = ""
    inversion: int | None = None

    @property
    def semantic_type(self) -> str:
        return "WesternTertianHarmony"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "standard": self.standard,
        }

    def to_dict(self) -> dict[str, object]:
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> WesternTertianHarmony | None:
        label = row.get("label")
        if label is None:
            return None
        root_raw = row.get("root")
        bass_raw = row.get("bass", row.get("bass_note"))
        inversion_raw = row.get("inversion")
        if inversion_raw is None and "figbass" in row:
            inv = figbass_to_inversion(str(row.get("figbass") or ""))
            inversion_raw = int(inv) if inv is not None else None
        return cls(
            label=str(label),
            standard=str(row.get("standard") or ""),
            root=int(root_raw) if root_raw is not None else None,
            bass=int(bass_raw) if bass_raw is not None else None,
            chord_type=str(row.get("chord_type") or ""),
            inversion=int(inversion_raw) if inversion_raw is not None else None,
        )

    def __repr__(self) -> str:
        return (
            f"WesternTertianHarmony(label={self.label!r}, "
            f"chord_type={self.chord_type!r})"
        )

    def __str__(self) -> str:
        return self.label


class WesternTertianHarmonyField(SemanticField[WesternTertianHarmony]):
    """Columnar wrapper for ``WesternTertianHarmony`` (paired Field)."""


class RomanNumeralHarmony(ScalarVocabulary, BaseModel):
    """Roman-numeral analysis."""

    model_config = ConfigDict(frozen=True)

    label: str
    standard: str
    root: int | None = None
    bass: int | None = None
    chord_type: str = ""
    inversion: int | None = None
    numeral: str = ""
    localkey: str = ""
    globalkey: str = ""

    @property
    def semantic_type(self) -> str:
        return "RomanNumeralHarmony"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "standard": self.standard,
        }

    def to_dict(self) -> dict[str, object]:
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> RomanNumeralHarmony | None:
        label = row.get("label")
        if label is None:
            return None
        root_raw = row.get("root")
        bass_raw = row.get("bass", row.get("bass_note"))
        inversion_raw = row.get("inversion")
        if inversion_raw is None and "figbass" in row:
            inv = figbass_to_inversion(str(row.get("figbass") or ""))
            inversion_raw = int(inv) if inv is not None else None
        return cls(
            label=str(label),
            standard=str(row.get("standard") or ""),
            root=int(root_raw) if root_raw is not None else None,
            bass=int(bass_raw) if bass_raw is not None else None,
            chord_type=str(row.get("chord_type") or ""),
            inversion=int(inversion_raw) if inversion_raw is not None else None,
            numeral=str(row.get("numeral") or ""),
            localkey=str(row.get("localkey") or ""),
            globalkey=str(row.get("globalkey") or ""),
        )

    def __repr__(self) -> str:
        return (
            f"RomanNumeralHarmony(label={self.label!r}, "
            f"numeral={self.numeral!r}, key={self.globalkey}:{self.localkey})"
        )

    def __str__(self) -> str:
        return self.label


class RomanNumeralHarmonyField(SemanticField[RomanNumeralHarmony]):
    """Columnar wrapper for ``RomanNumeralHarmony`` (paired Field)."""


class DcmlHarmony(ScalarVocabulary, BaseModel):
    """DCML harmony annotation.  ``standard`` pinned to ``Literal["dcml"]``."""

    model_config = ConfigDict(frozen=True)

    label: str
    standard: Literal["dcml"] = "dcml"
    root: int | None = None
    bass: int | None = None
    chord_type: str = ""
    inversion: int | None = None
    numeral: str = ""
    localkey: str = ""
    globalkey: str = ""
    tonicized_key: str | None = None
    pedal: str | None = None

    @property
    def semantic_type(self) -> str:
        return "DcmlHarmony"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "standard": "dcml",
        }

    def to_dict(self) -> dict[str, object]:
        """Return a summary dict of all harmony properties.

        Root and bass are shown both as pitch class integers and as
        ``EnharmonicPitchClass`` objects for readability.
        """
        root_gpc = (
            EnharmonicPitchClass(pitch_class=self.root)
            if self.root is not None
            else None
        )
        bass_gpc = (
            EnharmonicPitchClass(pitch_class=self.bass)
            if self.bass is not None
            else None
        )
        return {
            "label": self.label,
            "numeral": self.numeral,
            "chord_type": self.chord_type,
            "inversion": self.inversion,
            "root": root_gpc,
            "bass": bass_gpc,
            "globalkey": self.globalkey,
            "localkey": self.localkey,
            "tonicized_key": self.tonicized_key,
        }

    @classmethod
    def from_label(
        cls,
        label: str,
        *,
        globalkey: str = "C",
        localkey: str = "I",
    ) -> DcmlHarmony:
        """Construct a fully populated ``DcmlHarmony`` from a DCML label string."""
        from ms3.expand_dcml import features2type
        from ms3.utils import fifths2pc, roman_numeral2fifths
        from ms3.utils.constants import DCML_REGEX

        m = DCML_REGEX.match(label)
        if m is None:
            raise ValueError(f"Cannot parse DCML label: {label!r}")

        parts = {k: v for k, v in m.groupdict().items() if v is not None}
        numeral = parts.get("numeral", "")
        form = parts.get("form")
        figbass = parts.get("figbass")
        relativeroot = parts.get("relativeroot")
        pedal = parts.get("pedal")

        chord_type = features2type(numeral, form, figbass) if numeral else ""
        inv = figbass_to_inversion(figbass or "")
        inversion = int(inv) if inv is not None else None

        root: int | None = None
        bass: int | None = None
        if numeral:
            root_tpc = roman_numeral2fifths(numeral)
            root = fifths2pc(root_tpc)
            try:
                from ms3 import chord2tpcs

                chord_str = parts.get("chord", label)
                tpcs = chord2tpcs(chord_str)
                if tpcs:
                    bass = fifths2pc(tpcs[0])
            except Exception:
                bass = root

        return cls(
            label=label,
            globalkey=globalkey,
            localkey=localkey,
            numeral=numeral,
            chord_type=chord_type,
            inversion=inversion,
            root=root,
            bass=bass,
            tonicized_key=relativeroot,
            pedal=pedal,
        )

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> DcmlHarmony | None:
        """Construct from a DCML storage row dict.

        Maps DCML storage field names to internal model names:
        - ``bass_note`` -> ``bass``
        - ``figbass`` -> ``inversion``
        - ``relativeroot`` -> ``tonicized_key``
        """
        label = row.get("label")
        if label is None:
            return None

        root_raw = row.get("root")
        root = int(root_raw) if root_raw is not None else None
        bass_raw = row.get("bass_note", row.get("bass"))
        bass = int(bass_raw) if bass_raw is not None else None
        figbass_raw = row.get("figbass", "")
        inversion_raw = row.get("inversion")
        if inversion_raw is not None:
            inversion = int(inversion_raw)
        else:
            inv = figbass_to_inversion(str(figbass_raw or ""))
            inversion = int(inv) if inv is not None else None

        globalkey = str(row.get("globalkey") or "")
        localkey = str(row.get("localkey") or "")

        return cls(
            label=str(label),
            globalkey=globalkey,
            localkey=localkey,
            numeral=str(row.get("numeral") or ""),
            chord_type=str(row.get("chord_type") or ""),
            inversion=inversion,
            root=root,
            bass=bass,
            tonicized_key=row.get("relativeroot") or row.get("tonicized_key"),
            pedal=row.get("pedal"),
        )

    def __repr__(self) -> str:
        return (
            f"DcmlHarmony(label={self.label!r}, key={self.globalkey}:{self.localkey})"
        )

    def __str__(self) -> str:
        return self.label


class DcmlHarmonyField(SemanticField[DcmlHarmony]):
    """Columnar wrapper for ``DcmlHarmony`` (paired Field)."""


# ═══════════════════════════════════════════════════════════════════════════
# 3. EVENT SCALARS — Note, Measure, MeasureNumber, Id
# ═══════════════════════════════════════════════════════════════════════════


class Note(ScalarVocabulary, BaseModel):
    """A single note or rest event.

    ``Note.pitch`` is annotated ``EnharmonicPitch | SpecificPitch | None``
    for materialised scalars — both are legitimate pitch representations
    on a note.  This union MUST NOT translate to Arrow ``dense_union``;
    instead, the field is dropped from the pa.Schema and pitch is
    represented exactly once on the EventData.  ``NoteEventData`` (a
    spelled score source) stores ``specific_pitch`` as the sole default
    semantic pitch field and keeps the source MIDI number as a
    non-default raw ``midi`` int that affords an ``EnharmonicPitch`` view
    on request.  Number-only sources store the bare number instead (as a
    raw ``pitch`` int or an ``EnharmonicPitch`` ``{midi_number}`` struct).
    See Contributing → §2.4 Architectural decision log
    (union-of-BaseModels rejected; columnar separation).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    start: Coordinate
    end: Coordinate | None = None
    duration: Duration | None = None
    pitch: EnharmonicPitch | SpecificPitch | None = None
    voice: int | None = None
    staff: int | None = None
    velocity: int | None = None
    instrument: str | None = None

    @field_validator("duration", mode="before")
    @classmethod
    def _coerce_duration_from_coordinate(cls, v: object) -> Duration | None:
        if v is None or isinstance(v, Duration):
            return v
        if isinstance(v, Coordinate):
            return Duration(v.value, v.unit)
        return v

    @property
    @data_shaped
    def is_rest(self) -> bool:
        return self.pitch is None

    @property
    def semantic_type(self) -> str:
        return "Note"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "has_pitch": str(self.pitch is not None).lower(),
        }

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dictionary.

        Pitch is intentionally absent — it is stored in a separate field
        on ``NoteEventData`` (columnar separation).
        """
        from .retrieval import coordinate_to_dict

        d: dict[str, object] = {
            "start": coordinate_to_dict(self.start),
            "end": coordinate_to_dict(self.end) if self.end is not None else None,
            "voice": self.voice,
            "staff": self.staff,
            "velocity": self.velocity,
            "instrument": self.instrument,
        }
        d["duration"] = (
            coordinate_to_dict(self.duration) if self.duration is not None else None
        )
        return d

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Note | None:
        """Construct from a ``NoteEventData`` row dict (trust-boundary regime)."""
        from .enums import TimeUnit
        from .retrieval import coordinate_from_dict

        start_raw = row.get("start")
        if start_raw is None:
            return None

        def _coerce_coord(raw: Any) -> Coordinate | None:
            if raw is None:
                return None
            if isinstance(raw, Coordinate):
                return raw
            if isinstance(raw, dict):
                if raw.get("value") is None:
                    return None
                if "number_type" in raw:
                    return coordinate_from_dict(raw)
                unit = raw.get("unit", TimeUnit.quarters)
                return Coordinate(struct_to_rational(raw), unit)
            return None

        def _coerce_duration(raw: Any) -> Duration | None:
            if raw is None:
                return None
            if isinstance(raw, Duration):
                return raw
            if isinstance(raw, Coordinate):
                return Duration(raw.value, raw.unit)
            if isinstance(raw, dict):
                if raw.get("value") is None:
                    return None
                if "number_type" in raw:
                    coordinate = coordinate_from_dict(raw)
                    return Duration(
                        coordinate.value,
                        coordinate.unit,
                        number_type=coordinate.number_type,
                    )
                unit = raw.get("unit", TimeUnit.quarters)
                return Duration(struct_to_rational(raw), unit)
            return None

        start = _coerce_coord(start_raw)
        if start is None:
            return None
        end = _coerce_coord(row.get("end"))
        duration = _coerce_duration(row.get("duration"))

        # Represent-once resolution.  A spelled source stores the full
        # ``specific_pitch`` (the most-expressive faithful type); prefer
        # it.  Otherwise fall back to the bare MIDI number a number-only
        # source carries — either the non-default raw ``midi`` int on a
        # spelled NoteEventData, or a ``pitch`` value that arrives as an
        # ``EnharmonicPitch`` ``{midi_number}`` struct dict (or a plain
        # int) from a number-only timeline.  Pitch is never stored twice,
        # so at most one of these is populated.
        pitch: EnharmonicPitch | SpecificPitch | None = None
        sp_raw = row.get("specific_pitch")
        if isinstance(sp_raw, dict):
            pitch = SpecificPitch.from_row(sp_raw)
        if pitch is None:
            midi_raw = row.get("midi")
            pitch_raw = row.get("pitch")
            if isinstance(midi_raw, (int, float)):
                pitch = EnharmonicPitch(midi_number=int(midi_raw))
            elif isinstance(pitch_raw, dict):
                pitch = EnharmonicPitch.from_row(pitch_raw)
            elif isinstance(pitch_raw, (int, float)):
                pitch = EnharmonicPitch(midi_number=int(pitch_raw))

        return cls(
            start=start,
            end=end,
            duration=duration,
            pitch=pitch,
            voice=row.get("voice"),
            staff=row.get("staff"),
            velocity=row.get("velocity"),
            instrument=row.get("instrument"),
        )

    def __repr__(self) -> str:
        pitch_str = repr(self.pitch) if self.pitch is not None else "rest"
        return f"Note(start={self.start}, duration={self.duration}, pitch={pitch_str})"

    def __str__(self) -> str:
        pitch_str = str(self.pitch) if self.pitch is not None else "rest"
        return f"{pitch_str} @{self.start}+{self.duration}"


def _drop_field(_model_cls: type[BaseModel], _name: str, _info: object) -> list[Any]:
    """Drop a polymorphic / out-of-band field from the derived pa.Schema."""
    return []


# Columnar-separation rule: drop ``Note.pitch`` from pa.Schema so that
# the represent-once pitch field (``specific_pitch`` for a spelled
# source, plus the non-default raw ``midi`` int) lives in its own
# column (see Contributing → §2.4 Architectural decision log).
register_value_projector(Note, "pitch", _drop_field)


class NoteField(SemanticField[Note]):
    """Field wrapper for ``Note`` (paired Field)."""

    @property
    def is_rest(self) -> pa.Array:
        """Vectorized rest predicate.

        A row is a rest when the EventData carries no pitch for it — i.e.
        the default ``specific_pitch`` semantic field and the raw ``midi``
        number it affords an ``EnharmonicPitch`` view from are both null.
        At Field level, only the ``Note`` struct columns are available;
        the columnar-separation rule means ``Note.pitch``
        does NOT exist on the Field's struct.

        ``Note.is_rest`` is data-shaped, but the pitch data lives outside the
        ``Note`` struct in the represent-once TTA layout — so this method must
        be invoked on a higher-level container that knows about the separated
        pitch columns. Until that container exists, the mirror raises with a
        clear message.
        """
        raise NotImplementedError(
            "NoteField.is_rest requires access to the represent-once pitch "
            "columns on NoteEventData (the default specific_pitch field and "
            "the raw midi number), not available on the bare Note struct "
            "field."
        )


# ---------------------------------------------------------------------------
# Measure + MeasureField
# ---------------------------------------------------------------------------


class Measure(ScalarVocabulary, BaseModel):
    """One measure-like unit in a work's immutable measure map.

    ``count`` and ``qstamp`` may be supplied by a source, but a containing
    measure map derives both from printed order and exact actual lengths.
    The map warns when a supplied value disagrees with that derivation.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    id: str | None = Field(  # noqa: A003 - Measure Map vocabulary
        default=None,
        validation_alias=AliasChoices("id", "ID"),
        serialization_alias="ID",
    )
    count: int | None = None
    qstamp: Fraction | None = None
    number: int | None = None
    name: str | None = None
    time_signature: str | None = None
    nominal_length: Fraction | None = None
    actual_length: Fraction | None = None
    start_repeat: bool = False
    end_repeat: bool = False
    next: tuple[str, ...] | None = None
    volta: int | None = None

    @classmethod
    def from_values(cls, **values: Any) -> Measure:
        """Build the measure of the concrete class its values describe.

        This is the one classification of a measure, shared by every lane
        that builds measures from source facts or document rows:

        - no ``nominal_length``: a plain :class:`Measure`;
        - a ``cadenza`` time signature: :class:`CadenzaMeasure`;
        - a non-zero ``offset_within_measure``: :class:`MeasureConstituent`;
        - ``actual_length`` equal to ``nominal_length``: :class:`RegularMeasure`;
        - otherwise: :class:`IrregularMeasure`.

        Neither ``number`` nor a neighbouring measure takes part: a shared
        number also marks alternative endings, so it never identifies a
        split measure.

        Args:
            **values: Measure fields, optionally with ``offset_within_measure``.

        Returns:
            The measure, of the class its values describe.

        Raises:
            ValueError: If a non-zero offset is stated for a measure without
                nominal length or for a cadenza, which cannot lie inside a
                nominal measure.
        """
        offset = values.pop("offset_within_measure", None) or Fraction(0)
        signature = values.get("time_signature")
        target: type[Measure]
        if values.get("nominal_length") is None:
            target = Measure
        elif signature is not None and str(signature).lower() == "cadenza":
            target = CadenzaMeasure
        elif offset:
            return MeasureConstituent(**values, offset_within_measure=offset)
        elif values.get("actual_length") == values["nominal_length"]:
            target = RegularMeasure
        else:
            target = IrregularMeasure
        if offset:
            raise ValueError(
                f"A {target.__name__} cannot lie at offset {offset} inside a "
                "nominal measure"
            )
        return target(**values)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Measure:
        """Restore the concrete measure class named by a dictionary payload.

        Args:
            data: Measure dictionary carrying a concrete ``class`` tag.

        Returns:
            A measure of the tagged concrete class.

        Raises:
            ValueError: If the tag is missing, unknown, or incompatible with
                the receiving subclass.
        """

        def descendants(parent: type[Measure]) -> list[type[Measure]]:
            children: list[type[Measure]] = []
            for child in parent.__subclasses__():
                children.append(child)
                children.extend(descendants(child))
            return children

        known_classes = [Measure, *descendants(Measure)]
        known = {
            measure_class.__name__: measure_class for measure_class in known_classes
        }
        class_tag = data.get("class")
        target = known.get(class_tag) if isinstance(class_tag, str) else None
        if target is None:
            raise ValueError(
                f"Unknown or missing Measure class {class_tag!r}; "
                f"known classes: {', '.join(sorted(known))}"
            )
        if cls is not Measure and target is not cls:
            raise ValueError(
                f"Measure class {class_tag!r} does not match receiving "
                f"subclass {cls.__name__!r}"
            )

        values = {key: value for key, value in data.items() if key != "class"}
        unknown_keys = sorted(set(values) - set(target.model_fields))
        if unknown_keys:
            raise ValueError(
                f"Measure class {target.__name__!r} does not declare the keys: "
                f"{', '.join(unknown_keys)}"
            )
        for key in ("qstamp", "nominal_length", "actual_length"):
            if values.get(key) is not None:
                values[key] = number_from_json(values[key])
        if values.get("next") is not None:
            values["next"] = tuple(values["next"])
        if (
            issubclass(target, MeasureConstituent)
            and values.get("offset_within_measure") is not None
        ):
            values["offset_within_measure"] = number_from_json(
                values["offset_within_measure"]
            )
        return target(**values)

    @field_validator("qstamp", "nominal_length", "actual_length", mode="before")
    @classmethod
    def _as_exact_quarters(cls, value: Any) -> Any:
        if value is None or isinstance(value, Fraction):
            return value
        return Fraction(value)

    @field_validator("next", mode="before")
    @classmethod
    def _as_measure_ids(cls, value: Any) -> tuple[str, ...] | None:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            text = value.strip().strip("()[]")
            return (
                tuple(part.strip() for part in text.split(",") if part.strip()) or None
            )
        return tuple(str(item) for item in value)

    @model_validator(mode="after")
    def _default_name(self) -> Measure:
        if self.name is None and self.number is not None:
            object.__setattr__(self, "name", str(self.number))
        return self

    @property
    def semantic_type(self) -> str:
        """Return the scalar's semantic name."""
        return "Measure"

    def __repr__(self) -> str:
        parts = [f"id={self.id!r}", f"count={self.count!r}", f"name={self.name!r}"]
        for member in ("number", "time_signature", "qstamp", "actual_length"):
            value = getattr(self, member)
            if value is not None:
                parts.append(f"{member}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __str__(self) -> str:
        return self.name or (self.id or "Measure")

    def to_dict(self) -> dict[str, Any]:
        """Return the concrete measure as a JSON-safe dictionary."""
        data: dict[str, Any] = {
            "class": type(self).__name__,
            "id": self.id,
            "count": self.count,
            "qstamp": (
                number_to_json(self.qstamp) if self.qstamp is not None else None
            ),
            "number": self.number,
            "name": self.name,
            "time_signature": self.time_signature,
            "nominal_length": (
                number_to_json(self.nominal_length)
                if self.nominal_length is not None
                else None
            ),
            "actual_length": (
                number_to_json(self.actual_length)
                if self.actual_length is not None
                else None
            ),
            "start_repeat": self.start_repeat,
            "end_repeat": self.end_repeat,
            "next": list(self.next) if self.next is not None else None,
            "volta": self.volta,
        }
        if isinstance(self, MeasureConstituent):
            data["offset_within_measure"] = number_to_json(self.offset_within_measure)
        return data


class MeasureField(SemanticField[Measure]):
    """Columnar wrapper for ``Measure`` (paired Field)."""


class RegularMeasure(Measure):
    """A measure whose actual and nominal lengths agree."""


class RegularMeasureField(SemanticField[RegularMeasure]):
    """Columnar wrapper for :class:`RegularMeasure`."""


class IrregularMeasure(Measure):
    """A measure whose sounding length differs from its nominal length."""


class IrregularMeasureField(SemanticField[IrregularMeasure]):
    """Columnar wrapper for :class:`IrregularMeasure`."""


class SplitRegularMeasure(RegularMeasure):
    """A regular measure represented by more than one source constituent."""


class SplitRegularMeasureField(SemanticField[SplitRegularMeasure]):
    """Columnar wrapper for :class:`SplitRegularMeasure`."""


class SplitIrregularMeasure(IrregularMeasure):
    """An irregular split measure."""


class SplitIrregularMeasureField(SemanticField[SplitIrregularMeasure]):
    """Columnar wrapper for :class:`SplitIrregularMeasure`."""


class MeasureConstituent(IrregularMeasure):
    """A split-measure constituent with an offset in its nominal measure."""

    offset_within_measure: Fraction = Fraction(0)

    @field_validator("offset_within_measure", mode="before")
    @classmethod
    def _as_exact_offset(cls, value: Any) -> Fraction:
        return value if isinstance(value, Fraction) else Fraction(value)


class MeasureConstituentField(SemanticField[MeasureConstituent]):
    """Columnar wrapper for :class:`MeasureConstituent`."""


class CadenzaMeasure(IrregularMeasure):
    """An unmetered or freely measured span."""


class CadenzaMeasureField(SemanticField[CadenzaMeasure]):
    """Columnar wrapper for :class:`CadenzaMeasure`."""


@dataclass(frozen=True, init=False)
class Gap:
    """A stretch of target time that carries no source material.

    A string passed positionally is a descriptive label. Numeric values are
    exact durations; omitting both makes the gap auto-sized by flow machinery.

    Two contexts give a duration-less gap different meanings. When a FlowMap
    assembles interval descriptors, the gap is auto-sized from the distance
    between its neighbouring spans. In a skeleton flow, inserted material has
    no counterpart on the shared structure, so a duration-less gap is a
    zero-extent marker — its extent is unknown and is never inferred.
    """

    duration: Fraction | None
    label: str | None

    def __init__(
        self,
        duration: Fraction | int | float | str | None = None,
        label: str | None = None,
    ) -> None:
        if isinstance(duration, str):
            if label is not None:
                raise ValueError(
                    "A positional Gap label cannot be combined with label="
                )
            label = duration
            duration = None
        exact = None if duration is None else Fraction(duration)
        if exact is not None and exact < 0:
            raise ValueError(f"A Gap duration cannot be negative, got {exact}")
        object.__setattr__(self, "duration", exact)
        object.__setattr__(self, "label", label)

    @property
    def is_auto(self) -> bool:
        """Whether neighbouring source spans determine this gap's duration."""
        return self.duration is None

    def __repr__(self) -> str:
        size = "auto" if self.is_auto else str(self.duration)
        label = f", label={self.label!r}" if self.label else ""
        return f"Gap({size}{label})"


class BeatPolicy(BaseModel):
    """How a bar is counted in beats: a grouping cycle over one division.

    A time signature says two things at once — which note value is
    counted, and how those values group into beats.  ``division`` is the
    counted value in quarters (``Fraction(1, 2)`` for an eighth), and
    ``grouping`` says how many of them each successive beat spans.  A
    compound ``6/8`` bar is ``grouping=(3, 3)`` over eighths; a simple
    ``4/4`` bar is ``grouping=(1, 1, 1, 1)`` over quarters; an uneven
    ``(3+2+3)/8`` bar is ``grouping=(3, 2, 3)`` over eighths.

    The beat lengths (:attr:`rods`) are derived from that pair, never
    stored: the source encoding is what a policy holds, and every rod,
    offset and index question is answered from it.

    Attributes:
        grouping: How many divisions each beat spans, in order.
        division: The counted note value, in quarters. Given either
            directly or through *beat_size*, which completes it.
        beat_size: The counted value as a typed duration in whole notes
            or quarters. Completed from *division* when omitted.
        name: Optional human-readable label, typically the signature the
            source spells.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    grouping: tuple[int, ...] = (1,)
    division: Fraction | None = None
    beat_size: Duration | None = None
    name: str | None = None

    @classmethod
    def from_time_signature(cls, signature: str) -> BeatPolicy:
        """Derive the default counting of a bar from its time signature.

        A signature ``"n/d"`` is read as compound when ``n`` is a
        multiple of three greater than three and the denominator is an
        eighth or shorter — ``6/8``, ``9/8``, ``12/16`` count in dotted
        beats of three divisions.  Everything else counts one division
        per beat.  A composite signature ``"a/d+b/d+c/d"`` spells its own
        grouping.  ``"C"`` is ``4/4`` and ``"C|"`` (also spelled
        ``"cut"``) is ``2/2``.

        Args:
            signature: The time signature as the source spells it.

        Returns:
            The policy that counts a bar of that signature.

        Raises:
            ValueError: If *signature* cannot be read.  There is no
                silent fallback to ``4/4``: choosing a default for an
                unreadable source is a loader's decision, not a
                policy's.
        """
        text = signature.strip()
        if not text:
            raise ValueError("Time signature must not be empty")
        if text in ("C", "c"):
            return cls(grouping=(1,) * 4, division=Fraction(1), name=text)
        if text in ("C|", "c|", "cut", "CUT"):
            return cls(grouping=(1,) * 2, division=Fraction(2), name=text)

        numerators: list[int] = []
        denominator: int | None = None
        shorthand = re.fullmatch(r"(\d+(?:\s*\+\s*\d+)*)\s*/\s*(\d+)", text)
        if shorthand is not None:
            numerators = [int(numerator) for numerator in shorthand.group(1).split("+")]
            denominator = int(shorthand.group(2))
            if any(numerator < 1 for numerator in numerators) or denominator < 1:
                raise ValueError(f"Cannot read time signature {signature!r}")
        else:
            terms = [term.strip() for term in text.split("+")]
            for term in terms:
                match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", term)
                if match is None:
                    raise ValueError(f"Cannot read time signature {signature!r}")
                numerator = int(match.group(1))
                term_denominator = int(match.group(2))
                if numerator < 1 or term_denominator < 1:
                    raise ValueError(f"Cannot read time signature {signature!r}")
                if denominator is None:
                    denominator = term_denominator
                elif denominator != term_denominator:
                    raise ValueError(
                        f"Composite time signature {signature!r} mixes denominators; "
                        "every term must share one denominator"
                    )
                numerators.append(numerator)
        assert denominator is not None
        division = Fraction(4, denominator)

        if len(numerators) > 1:
            return cls(grouping=tuple(numerators), division=division, name=text)

        count = numerators[0]
        if count % 3 == 0 and count > 3 and denominator >= 8:
            return cls(grouping=(3,) * (count // 3), division=division, name=text)
        return cls(grouping=(1,) * count, division=division, name=text)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BeatPolicy:
        """Restore a beat policy from its JSON dictionary representation."""
        return cls(
            grouping=tuple(data["grouping"]),
            beat_size=_duration_from_dict(data["beat_size"]),
            name=data.get("name"),
        )

    @classmethod
    def uniform(
        cls, division: Fraction, count: int, *, name: str | None = None
    ) -> BeatPolicy:
        """Count *count* beats of one *division* each.

        Args:
            division: The counted note value, in quarters.
            count: How many such beats fill the bar.
            name: Optional label, typically the signature the source
                spells.

        Returns:
            A policy with an all-ones grouping.
        """
        return cls(grouping=(1,) * count, division=Fraction(division), name=name)

    @field_validator("grouping")
    @classmethod
    def _validate_grouping(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("BeatPolicy grouping must not be empty")
        if any(entry < 1 for entry in value):
            raise ValueError("BeatPolicy grouping entries must be >= 1")
        return value

    @field_validator("division")
    @classmethod
    def _validate_division(cls, value: Fraction | None) -> Fraction | None:
        if value is None:
            return value
        if value <= 0:
            raise ValueError("BeatPolicy division must be positive")
        return value

    @model_validator(mode="after")
    def _complete_beat_size(self) -> BeatPolicy:
        """Keep the typed beat size and quarter-note division in agreement."""
        from .enums import TimeUnit

        division = self.division
        beat_size = self.beat_size
        if division is None and beat_size is None:
            raise ValueError("BeatPolicy requires beat_size or division")
        if division is None:
            assert beat_size is not None
            if beat_size.unit is TimeUnit.whole_notes:
                division = Fraction(beat_size.value) * 4
            elif beat_size.unit is TimeUnit.quarters:
                division = Fraction(beat_size.value)
            else:
                raise ValueError(
                    "BeatPolicy beat_size must use whole_notes or quarters"
                )
            object.__setattr__(self, "division", division)
        elif beat_size is not None:
            if beat_size.unit is TimeUnit.whole_notes:
                stated_division = Fraction(beat_size.value) * 4
            elif beat_size.unit is TimeUnit.quarters:
                stated_division = Fraction(beat_size.value)
            else:
                raise ValueError(
                    "BeatPolicy beat_size must use whole_notes or quarters"
                )
            if stated_division != division:
                raise ValueError(
                    "BeatPolicy division and beat_size disagree: "
                    f"{division} quarters != {stated_division} quarters"
                )
        if beat_size is None:
            object.__setattr__(
                self,
                "beat_size",
                Duration(Fraction(division, 4), TimeUnit.whole_notes),
            )
        return self

    @property
    def rods(self) -> tuple[Fraction, ...]:
        """Length of every beat in this bar, in quarters."""
        assert self.division is not None
        return tuple(entry * self.division for entry in self.grouping)

    @property
    def n_beats(self) -> int:
        """How many beats the bar is counted in."""
        return len(self.grouping)

    @property
    def span(self) -> Fraction:
        """Total length of one counted bar, in quarters."""
        return sum(self.rods, Fraction(0))

    def as_divisions(self) -> BeatPolicy:
        """Return the same bar counted once per source division.

        Returns:
            A policy whose grouping contains one entry for every counted
            division, with the original division and name preserved.
        """
        assert self.division is not None
        return type(self)(
            grouping=(1,) * sum(self.grouping),
            division=self.division,
            name=self.name,
        )

    def rod_for(self, index: int) -> Fraction:
        """Return the length of beat *index* (1-based), in quarters.

        Args:
            index: 1-based beat index; ``1`` is the downbeat.

        Returns:
            The beat's length in quarters.

        Raises:
            ValueError: If *index* is outside ``1..n_beats``.
        """
        self._check_index(index)
        return self.rods[index - 1]

    def offset_for(self, index: int) -> Fraction:
        """Return the distance from the downbeat to beat *index*.

        Args:
            index: 1-based beat index; ``1`` is the downbeat.

        Returns:
            Quarters from the bar's downbeat.

        Raises:
            ValueError: If *index* is outside ``1..n_beats``.
        """
        self._check_index(index)
        return sum(self.rods[: index - 1], Fraction(0))

    def index_at(self, offset: Fraction) -> int:
        """Return the 1-based index of the beat containing *offset*.

        Args:
            offset: Quarters from the bar's downbeat.

        Returns:
            The 1-based beat index.

        Raises:
            ValueError: If *offset* is negative or beyond the bar's span.
        """
        position = Fraction(offset)
        if position < 0 or position >= self.span:
            raise ValueError(
                f"Offset {position} lies outside a bar of {self.span} quarters"
            )
        running = Fraction(0)
        for index, rod in enumerate(self.rods, start=1):
            running += rod
            if position < running:
                return index
        raise ValueError(  # pragma: no cover - guarded by the span check above
            f"Offset {position} lies outside a bar of {self.span} quarters"
        )

    def _check_index(self, index: int) -> None:
        if not 1 <= index <= self.n_beats:
            raise ValueError(
                f"Beat index {index} is outside 1..{self.n_beats} for this policy"
            )

    def __repr__(self) -> str:
        grouping = "+".join(str(entry) for entry in self.grouping)
        name = f", {self.name!r}" if self.name is not None else ""
        return f"BeatPolicy({grouping} x {self.division}{name})"

    def __str__(self) -> str:
        return self.name or repr(self)

    def to_dict(self) -> dict[str, Any]:
        """Return this policy as a JSON-safe dictionary."""
        if self.beat_size is None:
            raise ValueError("BeatPolicy states no beat size")
        return {
            "grouping": list(self.grouping),
            "beat_size": _duration_to_dict(self.beat_size),
            "name": self.name,
        }


class BeatPolicyField(SemanticField[BeatPolicy]):
    """Paired Field for :class:`BeatPolicy`.

    Empty body — :class:`BeatPolicy` carries no ``@data_shaped``
    methods, so the parity check is trivially satisfied.  The class
    exists to give the paired-class shape (``Object`` + ``ObjectField``)
    a stable home and to integrate with
    ``EventData.get_field(BeatPolicy)`` dispatch.
    """


def _exact_optional(value: object, *, field_name: str) -> Fraction | None:
    """Read an optional exact scalar without guessing a nearby ratio."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Fraction)):
        raise TypeError(f"Tempo {field_name} must be a number or decimal string")
    return Fraction(value)


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _duration_to_dict(value: Duration | None) -> dict[str, Any] | None:
    """Encode an optional duration with its exact value and unit."""
    if value is None:
        return None
    return {
        "value": number_to_json(value.value),
        "unit": value.unit.value,
        "number_type": value.number_type.name,
    }


def _duration_from_dict(value: dict[str, Any] | None) -> Duration | None:
    """Restore an optional duration from its typed dictionary."""
    if value is None:
        return None
    from .enums import NumberType, TimeUnit

    return Duration(
        number_from_json(value["value"]),
        TimeUnit(value["unit"]),
        number_type=NumberType[value["number_type"]],
    )


class Tempo(BaseModel):
    """A stated tempo rate, independent of any metrical reading.

    Attributes:
        bpm: Beats per minute at the start of the statement.
        beat: The duration unit for which the rate is stated.
        to_bpm: Optional target rate of a ramp.
        mean_at: Optional share of the ramp span at which its mean occurs.
        text: Optional verbal tempo indication supplied by the source.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    bpm: Fraction | None
    beat: Duration | None = None
    to_bpm: Fraction | None = None
    mean_at: Fraction | None = None
    text: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tempo:
        """Restore a tempo from its JSON dictionary representation.

        Args:
            data: JSON-safe tempo dictionary.

        Returns:
            The restored exact tempo statement.
        """
        return cls(
            bpm=(
                None
                if data.get("bpm") is None
                else Fraction(number_from_json(data["bpm"]))
            ),
            beat=_duration_from_dict(data.get("beat")),
            to_bpm=(
                None
                if data.get("to_bpm") is None
                else Fraction(number_from_json(data["to_bpm"]))
            ),
            mean_at=(
                None
                if data.get("mean_at") is None
                else Fraction(number_from_json(data["mean_at"]))
            ),
            text=data.get("text"),
        )

    @field_validator("bpm", "to_bpm", "mean_at", mode="before")
    @classmethod
    def _coerce_exact_rate(cls, value: object, info: Any) -> Fraction | None:
        return _exact_optional(value, field_name=info.field_name)

    @field_validator("bpm", "to_bpm")
    @classmethod
    def _validate_positive_rate(cls, value: Fraction | None) -> Fraction | None:
        if value is not None and value <= 0:
            raise ValueError("Tempo rates must be positive")
        return value

    @field_validator("bpm", "to_bpm", "mean_at")
    @classmethod
    def _validate_exact_dict_range(
        cls, value: Fraction | None, info: Any
    ) -> Fraction | None:
        if value is None:
            return None
        if not (
            _INT64_MIN <= value.numerator <= _INT64_MAX
            and value.denominator <= _INT64_MAX
        ):
            raise ValueError(
                f"Tempo {info.field_name} value {value} exceeds the signed int64 "
                f"dictionary limit {_INT64_MAX}"
            )
        return value

    @model_validator(mode="after")
    def _validate_ramp_mean(self) -> Tempo:
        if self.mean_at is None:
            return self
        if not 0 < self.mean_at < 1:
            raise ValueError("Tempo mean_at must satisfy 0 < mean_at < 1")
        if self.to_bpm is None:
            raise ValueError("Tempo mean_at requires to_bpm")
        return self

    def __repr__(self) -> str:
        parts = [f"bpm={self.bpm!r}"]
        if self.beat is not None:
            parts.append(f"beat={self.beat!r}")
        if self.to_bpm is not None:
            parts.append(f"to_bpm={self.to_bpm!r}")
        if self.mean_at is not None:
            parts.append(f"mean_at={self.mean_at!r}")
        if self.text is not None:
            parts.append(f"text={self.text!r}")
        return f"Tempo({', '.join(parts)})"

    def to_dict(self) -> dict[str, Any]:
        """Return this tempo as a JSON-safe dictionary.

        Returns:
            A dictionary retaining every exact rational value.
        """
        return {
            "bpm": None if self.bpm is None else number_to_json(self.bpm),
            "beat": _duration_to_dict(self.beat),
            "to_bpm": (None if self.to_bpm is None else number_to_json(self.to_bpm)),
            "mean_at": (None if self.mean_at is None else number_to_json(self.mean_at)),
            "text": self.text,
        }


class TempoField(SemanticField[Tempo]):
    """Paired Field for :class:`Tempo`."""


class Address(BaseModel):
    """Abstract root of the typed positions that name measures and beats.

    An address identifies a place in a piece the way a musician does —
    "bar 12", "the second ending of bar 15", "beat 3" — rather than by a
    position on a coordinate axis.  Resolving an address into a
    coordinate is a :class:`~timetoalign.timelines.TimeSkeleton`
    operation; the address itself is inert data and carries no timeline.

    Subclasses are frozen and strict: a measure label is a string, a
    measure count is an integer, and neither is silently coerced into
    the other.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    rendition: int | None = None

    @classmethod
    def parse(
        cls,
        text: str,
        *,
        offset_denomination: Fraction = Fraction(1, 1),
    ) -> Address:
        """Read an address out of its printed form.

        Two forms are recognised:

        * ``"N+p/q"`` — a measure label and a within-measure offset,
          the spelling used by performance-alignment sources.  The
          offset fraction is read in *offset_denomination* whole notes
          and converted exactly to quarters, so the default reads
          ``"12+3/8"`` as three eighths of a whole note, i.e. 3/2
          quarters.
        * anything else — a bare measure label, suffix included.

        Args:
            text: The printed address.
            offset_denomination: The note value the offset fraction
                counts, as a fraction of a whole note.

        Returns:
            A :class:`MeasureNumberAddress` for the first form, a
            :class:`MeasureNumber` for the second.

        Raises:
            ValueError: If the offset part of the first form is not a
                readable fraction.
        """
        from .enums import TimeUnit

        label, separator, offset_text = text.partition("+")
        if not separator:
            return MeasureNumber(mn=text)
        try:
            offset = Fraction(offset_text)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(
                f"Cannot read the within-measure offset of address {text!r}"
            ) from error
        quarters = offset * Fraction(offset_denomination) * 4
        return MeasureNumberAddress(
            mn=label,
            at=Coordinate(quarters, TimeUnit.quarters),
        )

    @property
    def selects_span(self) -> bool:
        """Whether this address names a whole span rather than an instant."""
        raise NotImplementedError


class MeasureNumber(Address):
    """A measure addressed by its printed label.

    Storage struct (derived): ``{mc: int64, mn: string, volta: int64}``.
    The label is the number a musician reads in the margin, kept as a
    string because it may carry a suffix (``"237b"``) or be ``"0"`` for
    an anacrusis.  ``mc`` is a lossless carry-along of the source's own
    measure count where it has one; it is never a resolution input.

    Args:
        mn: The printed label.  An integer is accepted and stringified.
        mc: The source's measure count, when known.
        volta: The ending this label belongs to, when known.
    """

    mc: int | None = None
    mn: str
    volta: int | None = None
    section: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> MeasureNumber | None:
        """Rebuild a :class:`MeasureNumber` from a storage row."""
        if not isinstance(row, dict):
            return None
        mn = row.get("mn")
        if mn is None:
            return None
        return cls(
            mn=str(mn),
            mc=row.get("mc"),
            volta=row.get("volta"),
            section=row.get("section"),
            rendition=row.get("rendition"),
        )

    @field_validator("mn", mode="before")
    @classmethod
    def _coerce_label(cls, value: Any) -> Any:
        return str(value) if isinstance(value, int) else value

    @property
    def selects_span(self) -> bool:
        """A bare measure label names the whole bar."""
        return True

    def __repr__(self) -> str:
        parts = [f"mn={self.mn!r}"]
        if self.mc is not None:
            parts.append(f"mc={self.mc}")
        if self.volta is not None:
            parts.append(f"volta={self.volta}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __str__(self) -> str:
        return self.mn


class MeasureNumberField(SemanticField[MeasureNumber]):
    """Paired Field for :class:`MeasureNumber`.

    Empty body — :class:`MeasureNumber` carries no ``@data_shaped``
    methods, so the parity check is trivially satisfied.  The class
    exists to give the paired-class shape (``Object`` + ``ObjectField``)
    a stable home and to integrate with
    ``EventData.get_field(MeasureNumber)`` dispatch.
    """

    def from_array(
        self,
        source: pa.Array | pa.ChunkedArray,
        *,
        name: str | None = None,
    ) -> SemanticField[MeasureNumber]:
        """Read a column of printed labels into the ``{mc, mn, volta}`` struct.

        A source column carries the label and nothing else — the
        measure count and volta are facts a measure map holds, not facts
        a label column states — so both are left null.

        Args:
            source: The label column, of any Arrow type.
            name: Optional output field name.

        Returns:
            A live :class:`MeasureNumberField`.
        """
        if isinstance(source, pa.ChunkedArray):
            source = source.combine_chunks()
        schema = type(self).pa_schema
        assert schema is not None
        label_type = schema.field(schema.get_field_index("mn")).type
        labels = source if source.type == label_type else pc.cast(source, label_type)
        arrays = [
            labels if field.name == "mn" else pa.nulls(len(labels), type=field.type)
            for field in schema
        ]
        struct_arr = pa.StructArray.from_arrays(arrays, fields=list(schema))
        out_name = name if name is not None else self.name
        return type(self).from_field((struct_arr, pa.field(out_name, schema)))


class MeasureNumberAddress(MeasureNumber):
    """A measure label plus a position inside that measure.

    The within-measure position is one polymorphic field: a
    :class:`Beat` counts under a beat-size policy, a
    :class:`~timetoalign.core.Coordinate` measures a distance from the
    bar's notated downbeat.  Both resolve the same way — measure start
    plus within-measure position — which is why they share a field
    rather than splitting the class.

    Args:
        at: The position inside the measure.
    """

    at: Beat | Coordinate

    @property
    def selects_span(self) -> bool:
        """A measure-plus-offset address names an instant."""
        return False

    def __repr__(self) -> str:
        return f"MeasureNumberAddress(mn={self.mn!r}, at={self.at!r})"

    def __str__(self) -> str:
        return f"{self.mn}+{self.at}"


class MeasureId(Address):
    """A measure addressed by its identity.

    An integer value resolves positionally — the thirteenth measure-like
    unit in the piece — while a string value is looked up against the
    measures' own identifiers.  Sources that mint identifiers as
    ``str(count)`` make the two coincide; sources with opaque
    identifiers keep them apart.  Strictness is what keeps them apart:
    ``"13"`` is an identifier, never the count 13.

    Args:
        value: A positional count or an identifier.
    """

    value: int | str

    @classmethod
    def count(cls, value: int) -> MeasureId:
        """Address the *value*-th measure-like unit.

        Args:
            value: 1-based position.

        Returns:
            A positional address.

        Raises:
            ValueError: If *value* is not an integer.
        """
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"MeasureId.count requires an int position, got {value!r}")
        return cls(value=value)

    @classmethod
    def identifier(cls, value: str) -> MeasureId:
        """Address the measure whose identifier is *value*.

        Args:
            value: The measure's identifier.

        Returns:
            An identifier address.

        Raises:
            ValueError: If *value* is not a string.
        """
        if not isinstance(value, str):
            raise ValueError(f"MeasureId.identifier requires a str id, got {value!r}")
        return cls(value=value)

    def __init__(self, value: int | str | None = None, **data: Any) -> None:
        if value is not None:
            if "value" in data:
                raise TypeError("MeasureId value was supplied twice")
            data["value"] = value
        super().__init__(**data)

    @property
    def is_positional(self) -> bool:
        """Whether this address resolves by count rather than by identifier."""
        return isinstance(self.value, int)

    @property
    def selects_span(self) -> bool:
        """A bare measure identity names the whole bar."""
        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"

    def __str__(self) -> str:
        return str(self.value)


class MeasureIdAddress(MeasureId):
    """A measure identity plus a position inside that measure.

    Args:
        at: The position inside the measure.
    """

    at: Beat | Coordinate

    @property
    def selects_span(self) -> bool:
        """A measure-plus-offset address names an instant."""
        return False

    def __repr__(self) -> str:
        return f"MeasureIdAddress({self.value!r}, at={self.at!r})"

    def __str__(self) -> str:
        return f"{self.value}+{self.at}"


class Beat(Address):
    """A beat within an implied measure scope.

    Storage struct (derived): ``{index: int64, policy: struct<grouping,
    division, name>, level: int64}``.  The index is 1-based, so beat 1
    is the downbeat.  A beat carries a :class:`BeatPolicy` only when
    it means to override the bar's own counting: without one, the beat
    is counted the way the bar's time signature counts it, and only
    resolution — which knows the bar — can say how long it is.

    Args:
        index: 1-based beat index.
        policy: Counting override, or ``None`` for the bar's default.
        level: Metrical level; ``0`` is the pulse, negative levels are
            groove below the pulse, and positive levels above it carry no
            name of their own — a reading names the levels it states.
    """

    index: int
    policy: BeatPolicy | None = None
    level: int = 0

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Beat | None:
        """Rebuild a :class:`Beat` from a storage row."""
        if not isinstance(row, dict):
            return None
        index = row.get("index")
        if index is None:
            return None
        policy_row = row.get("policy")
        policy: BeatPolicy | None = None
        if isinstance(policy_row, dict) and policy_row.get("grouping") is not None:
            policy = BeatPolicy(
                grouping=tuple(int(entry) for entry in policy_row["grouping"]),
                division=Fraction(struct_to_rational(policy_row["division"])),
                name=policy_row.get("name"),
            )
        return cls(
            index=int(index),
            policy=policy,
            level=int(row.get("level") or 0),
            rendition=row.get("rendition"),
        )

    @field_validator("index")
    @classmethod
    def _validate_index(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Beat index is 1-based; beat 1 is the downbeat")
        return value

    @property
    def is_downbeat(self) -> bool:
        """Whether this beat is the bar's downbeat."""
        return self.index == 1 and self.level == 0

    @property
    def size(self) -> Fraction | None:
        """This beat's length in quarters, or ``None`` without a policy."""
        return None if self.policy is None else self.policy.rod_for(self.index)

    @property
    def selects_span(self) -> bool:
        """A beat names the span it occupies."""
        return True

    def offset(self, policy: BeatPolicy | None = None) -> Fraction:
        """Return the distance from the bar's downbeat to this beat.

        Args:
            policy: The bar's counting, used when the beat carries none.

        Returns:
            Quarters from the downbeat.

        Raises:
            ValueError: If neither the beat nor the caller supplies a
                policy, or the index is outside the policy's bar.
        """
        effective = self.policy if self.policy is not None else policy
        if effective is None:
            raise ValueError(
                f"Beat {self.index} needs a BeatPolicy to know where it sits"
            )
        return effective.offset_for(self.index)

    def __repr__(self) -> str:
        parts = [str(self.index)]
        if self.policy is not None:
            parts.append(f"policy={self.policy!r}")
        if self.level:
            parts.append(f"level={self.level}")
        return f"Beat({', '.join(parts)})"

    def __str__(self) -> str:
        return str(self.index)


class BeatField(SemanticField[Beat]):
    """Paired Field for :class:`Beat`.

    Empty body — :class:`Beat` carries no ``@data_shaped`` methods, so
    the parity check is trivially satisfied.  The class exists to give
    the paired-class shape (``Object`` + ``ObjectField``) a stable home
    and to integrate with ``EventData.get_field(Beat)`` dispatch.
    """


MeasureNumberAddress.model_rebuild()
MeasureIdAddress.model_rebuild()


class Id(BaseModel):
    """A string identity label for a tabular event.

    Storage struct (derived): ``{value: string}``.  Used to carry an
    event-level identity through loading and analysis — typically the
    note-id string emitted by a performance-analysis tool (``n1b8xktz``,
    ``nh9xux4``, …).  Equality / hashing are inherited from pydantic's
    frozen ``BaseModel``; two ``Id`` objects compare equal iff their
    ``value`` strings match.
    """

    model_config = ConfigDict(frozen=True)

    value: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Id | None:
        if not isinstance(row, dict):
            return None
        v = row.get("value")
        if v is None:
            return None
        return cls(value=str(v))

    def __repr__(self) -> str:
        return f"Id({self.value!r})"

    def __str__(self) -> str:
        return self.value


class IdField(SemanticField[Id]):
    """Paired Field for :class:`Id`.

    Empty body — :class:`Id` carries no ``@data_shaped`` methods, so the
    parity check is trivially satisfied.  The class exists to give the
    paired-class shape (``Object`` + ``ObjectField``) a stable home and
    to integrate with ``EventData.get_field(Id)`` dispatch.
    """


# ═══════════════════════════════════════════════════════════════════════════
# 4. MIDI EVENT (BASE + SCORE SUBCLASS)
# ═══════════════════════════════════════════════════════════════════════════


class MidiEvent(BaseModel):
    """Performance MIDI event scalar (mido-shaped). 7 cross-loader fields.

    Storage struct (derived): nested — ``pitch`` is a
    ``struct<midi_number: int64>`` (from :class:`EnharmonicPitch`); the
    other six sub-fields are nullable ``int64``.  Velocity, channel,
    track, control, value, program are all optional 0–127 (channel
    0–15, track for multi-track files).  Pitch is present for note
    events, ``None`` for Control Change / Program Change.  The wider
    score-side variant is :class:`ScoreMidiEvent`, which extends this
    base with three partitura-only fields.

    This scalar is the cross-loader intersection of the columns mido
    and partitura can produce, and is the storage shape emitted by
    ``PerformanceMidiLoader``.  Score-only fields (``voice`` /
    ``staff`` / ``part_id``) live on the subclass rather than as
    always-null columns on every performance MIDI table, so the
    storage stays minimal.
    """

    model_config = ConfigDict(frozen=True, slots=True)

    pitch: EnharmonicPitch | None = None
    velocity: int | None = None
    channel: int | None = None
    track: int | None = None
    control: int | None = None
    value: int | None = None
    program: int | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MidiEvent | None":
        """Reconstruct a :class:`MidiEvent` from a storage-row dict.

        Accepts the dict shape produced by ``pa.StructArray.to_pylist()``
        on the column-builder output:

        * ``pitch``: either ``None`` or a nested ``{"midi_number": int}``
          dict (translated into an :class:`EnharmonicPitch` via its own
          ``from_row``).
        * The six other fields are pass-through ``int | None``.

        Returns ``None`` for a ``None`` row (the null-struct slot).
        """
        if row is None:
            return None
        pitch_raw = row.get("pitch")
        if isinstance(pitch_raw, dict):
            pitch = EnharmonicPitch.from_row(pitch_raw)
        elif pitch_raw is None:
            pitch = None
        else:
            pitch = EnharmonicPitch(midi_number=int(pitch_raw))
        return cls(
            pitch=pitch,
            velocity=row.get("velocity"),
            channel=row.get("channel"),
            track=row.get("track"),
            control=row.get("control"),
            value=row.get("value"),
            program=row.get("program"),
        )

    def _repr_parts(self) -> list[str]:
        """Return ``"field=val"`` strings for every non-None field.

        Subclasses extend this (never the rendered string) so the
        inherited :meth:`__repr__` stays the single rendering site.
        """
        parts: list[str] = []
        if self.pitch is not None:
            parts.append(f"pitch={self.pitch!r}")
        if self.velocity is not None:
            parts.append(f"velocity={self.velocity}")
        if self.channel is not None:
            parts.append(f"channel={self.channel}")
        if self.track is not None:
            parts.append(f"track={self.track}")
        if self.control is not None:
            parts.append(f"control={self.control}")
        if self.value is not None:
            parts.append(f"value={self.value}")
        if self.program is not None:
            parts.append(f"program={self.program}")
        return parts

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(self._repr_parts())})"


class MidiEventField(SemanticField[MidiEvent]):
    """Paired Field for :class:`MidiEvent`.

    Empty body — :class:`MidiEvent` carries no ``@data_shaped``
    methods, so the parity check is trivially satisfied.  The class
    exists to give the paired-class shape (``Object`` + ``ObjectField``)
    a stable home and to integrate with
    ``EventData.get_field(MidiEvent)`` dispatch.
    """


class ScoreMidiEvent(MidiEvent):
    """Score MIDI event scalar (partitura-shaped).

    Extends :class:`MidiEvent` with three partitura-only fields:
    ``voice``, ``staff``, ``part_id``.  The derived ``pa.StructType``
    is a **separate (wider) struct** than :class:`MidiEvent`'s — the
    column-builder produces two distinct shapes; there is no shared
    layout and no struct subtyping.  This is the storage shape
    emitted by ``ScoreMidiLoader``.
    """

    voice: int | None = None
    staff: int | None = None
    part_id: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ScoreMidiEvent | None":
        """Reconstruct a :class:`ScoreMidiEvent` from a storage-row dict.

        Accepts the wider 10-field dict shape produced by
        :func:`build_struct_array` on this scalar; mirrors
        :meth:`MidiEvent.from_row` with the three score-only fields
        (``voice`` / ``staff`` / ``part_id``) appended.
        """
        if row is None:
            return None
        pitch_raw = row.get("pitch")
        if isinstance(pitch_raw, dict):
            pitch = EnharmonicPitch.from_row(pitch_raw)
        elif pitch_raw is None:
            pitch = None
        else:
            pitch = EnharmonicPitch(midi_number=int(pitch_raw))
        return cls(
            pitch=pitch,
            velocity=row.get("velocity"),
            channel=row.get("channel"),
            track=row.get("track"),
            control=row.get("control"),
            value=row.get("value"),
            program=row.get("program"),
            voice=row.get("voice"),
            staff=row.get("staff"),
            part_id=row.get("part_id"),
        )

    def _repr_parts(self) -> list[str]:
        parts = super()._repr_parts()
        if self.voice is not None:
            parts.append(f"voice={self.voice}")
        if self.staff is not None:
            parts.append(f"staff={self.staff}")
        if self.part_id is not None:
            parts.append(f"part_id={self.part_id!r}")
        return parts


class ScoreMidiEventField(SemanticField[ScoreMidiEvent]):
    """Paired Field for :class:`ScoreMidiEvent`.

    Empty body — :class:`ScoreMidiEvent` carries no ``@data_shaped``
    methods, so the parity check is trivially satisfied.  The class
    exists to give the paired-class shape (``Object`` + ``ObjectField``)
    a stable home and to integrate with
    ``EventData.get_field(ScoreMidiEvent)`` dispatch.
    """
