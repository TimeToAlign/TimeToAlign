"""Conversion maps for TimeToAlign!

This module provides coordinate conversion infrastructure for transforming
values between different units and timelines.

The key abstractions are:
- ConversionMap: Base protocol/ABC for all maps
- LinearMap: Affine transformation (y = ax + b)
- ScalarMap: Pure scaling (y = ax)
- ShiftMap: Offset transformation (y = x + b)
- TableMap: Lookup/interpolation-based mapping
- ChainMap: Composition of multiple maps
- PiecewiseMap: Region-based mapping with different maps for different intervals
- CombinationMap: Multi-output mapping (e.g., (x, y) or (measure, beat))
- MeasureRecordMap, MeasureNumberMap, FloatingMeasureMap, MeasureBeatMap:
  views reading a measure map
- RotationMap: Periodic/cyclic patterns via modular arithmetic
- FloorMap: Integer floor division (e.g., measure numbers)

Convenience classes for common conversions:
- TicksToQuarters, QuartersToTicks: MIDI tick <-> quarter note
- SamplesToSeconds, SecondsToSamples: Audio sample <-> seconds
"""

from __future__ import annotations

from .base import ConversionMap
from .combination import CombinationMap
from .composite import ChainMap, PiecewiseMap
from .constant import ConstantMap
from .convenience import (
    QuartersToTicks,
    SamplesToSeconds,
    SecondsToSamples,
    TicksToQuarters,
)
from .interpolation import InterpolationMap
from .interval import (
    IntervalToConstantMap,
    QuartersToFloatingMeasures,
    QuartersToMeasureNumber,
)
from .linear import LinearMap, ScalarMap, ShiftMap
from .measures import (
    FloatingMeasureMap,
    MeasureBeatMap,
    MeasureNumberMap,
    MeasureRecordMap,
)
from .meter import BeatInMeasureMap, MetricalPositionMap, MetricMap
from .periodic import FloorMap, RotationMap
from .table import TableMap

__all__ = [
    # Base
    "ConversionMap",
    # Constant map
    "ConstantMap",
    # Interpolation (internal engine)
    "InterpolationMap",
    # Interval-to-constant maps
    "IntervalToConstantMap",
    "QuartersToMeasureNumber",
    "QuartersToFloatingMeasures",
    # Linear maps
    "LinearMap",
    "ScalarMap",
    "ShiftMap",
    # Table maps
    "TableMap",
    # Composite maps
    "ChainMap",
    "PiecewiseMap",
    # Multi-output maps
    "CombinationMap",
    # Periodic/floor maps
    "RotationMap",
    "FloorMap",
    # Measure views
    "MeasureRecordMap",
    "MeasureNumberMap",
    "FloatingMeasureMap",
    "MeasureBeatMap",
    # Meter maps
    "MetricMap",
    "BeatInMeasureMap",
    "MetricalPositionMap",
    # Convenience classes
    "TicksToQuarters",
    "QuartersToTicks",
    "SamplesToSeconds",
    "SecondsToSamples",
]
