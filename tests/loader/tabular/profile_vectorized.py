"""Performance profiling for vectorized TabularLoader.

This script profiles the vectorized loading pipeline on real specimens
to validate performance claims and identify bottlenecks.

Usage:
    cd timetoalign
    python -m tests.loader.tabular.profile_vectorized

Output:
    Performance report with throughput metrics and zero iteration validation.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from timetoalign.loader.score import Ms3Loader
from timetoalign.loader.tabular import TabularLoader
from timetoalign.testdata import ensure_data

# region Profiling Infrastructure


@dataclass
class ProfileResult:
    """Results from profiling a single file."""

    specimen: str
    loader_class: str
    file_size_kb: float
    line_count: int
    event_count: int
    load_time_mean: float
    load_time_std: float
    load_time_min: float
    load_time_max: float
    throughput: float
    zero_iteration: bool


def profile_loader(
    loader_class: type[TabularLoader],
    file_path: Path,
    iterations: int = 10,
    **loader_kwargs: Any,
) -> ProfileResult:
    """Profile loader performance on a specimen file.

    Args:
        loader_class: The loader class to profile.
        file_path: Path to the specimen file.
        iterations: Number of iterations for timing.
        **loader_kwargs: Additional kwargs for loader initialization.

    Returns:
        ProfileResult with performance metrics.
    """
    # Get file metadata
    file_size_kb = file_path.stat().st_size / 1024
    with open(file_path) as f:
        line_count = sum(1 for _ in f)

    # Run timed iterations
    times = []
    event_count = 0

    for _ in range(iterations):
        loader = loader_class(**loader_kwargs)
        start = time.perf_counter()
        loader.load(file_path)
        end = time.perf_counter()
        times.append(end - start)
        event_count = len(loader.events)

    times_arr = np.array(times)

    return ProfileResult(
        specimen=file_path.name,
        loader_class=loader_class.__name__,
        file_size_kb=file_size_kb,
        line_count=line_count,
        event_count=event_count,
        load_time_mean=np.mean(times_arr),
        load_time_std=np.std(times_arr),
        load_time_min=np.min(times_arr),
        load_time_max=np.max(times_arr),
        throughput=event_count / np.mean(times_arr),
        zero_iteration=True,  # Will be validated separately
    )


def validate_zero_iteration(
    loader_class: type[TabularLoader],
    file_path: Path,
    **loader_kwargs: Any,
) -> bool:
    """Validate that loader never iterates over rows.

    Strategy:
    - Monkey-patch DataFrame.__iter__ to raise
    - Load file
    - If iteration occurred, test fails

    Args:
        loader_class: The loader class to test.
        file_path: Path to the specimen file.
        **loader_kwargs: Additional kwargs for loader initialization.

    Returns:
        True if no iteration occurred, False otherwise.
    """
    import pandas as pd

    original_df_iter = pd.DataFrame.__iter__

    iteration_detected = False

    def fail_on_iter(self):
        nonlocal iteration_detected
        iteration_detected = True
        # Still allow iteration (for pandas internals) but mark as detected
        return original_df_iter(self)

    pd.DataFrame.__iter__ = fail_on_iter

    try:
        loader = loader_class(**loader_kwargs)
        loader.load(file_path)
    finally:
        pd.DataFrame.__iter__ = original_df_iter

    return not iteration_detected


# endregion


# region Profiling Specimens


def get_specimens() -> list[tuple[Path, type[TabularLoader], dict[str, Any]]]:
    """Get list of specimens to profile with their loaders.

    Returns:
        List of (file_path, loader_class, loader_kwargs) tuples.
    """
    base = ensure_data("score")

    specimens = []

    # Beethoven WoO71 notes (ms3 format)
    beethoven_notes = base / "beethoven_woo71" / "WoO71.notes.tsv"
    if beethoven_notes.exists():
        specimens.append((beethoven_notes, Ms3Loader, {}))

    # Beethoven WoO71 measures (ms3 format)
    beethoven_measures = base / "beethoven_woo71" / "WoO71.measures.tsv"
    if beethoven_measures.exists():
        specimens.append((beethoven_measures, Ms3Loader, {}))

    # Rachmaninoff Concerto 2 notes (ms3 format)
    rach_notes = (
        base
        / "rachmaninoff_concerto2"
        / "score"
        / "Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.notes.tsv"
    )
    if rach_notes.exists():
        specimens.append((rach_notes, Ms3Loader, {}))

    # Original file before cleaning (larger)
    original_notes = (
        base / "beethoven_woo71" / "original_file_before_cleaning.notes.tsv"
    )
    if original_notes.exists():
        specimens.append((original_notes, Ms3Loader, {}))

    return specimens


# endregion


# region Report Generation


def print_report(results: list[ProfileResult]) -> None:
    """Print formatted profiling report.

    Args:
        results: List of profiling results.
    """
    print("\n" + "=" * 80)
    print("VECTORIZED TABULARLOADER PERFORMANCE PROFILING REPORT")
    print("=" * 80)

    print("\n## Test Environment")
    print(f"- Python: {sys.version.split()[0]}")
    print(f"- Platform: {sys.platform}")

    print("\n## Profiling Results\n")

    # Table header
    print(
        f"{'Specimen':<45} {'Events':>8} {'Time (s)':>10} {'Throughput':>12} {'Zero-Iter':>10}"
    )
    print("-" * 90)

    for r in results:
        status = "PASS" if r.zero_iteration else "FAIL"
        print(
            f"{r.specimen:<45} {r.event_count:>8} {r.load_time_mean:>10.4f} "
            f"{r.throughput:>10,.0f}/s {status:>10}"
        )

    print("\n## Detailed Results\n")

    for r in results:
        print(f"### {r.specimen}")
        print(f"- Loader: {r.loader_class}")
        print(f"- File size: {r.file_size_kb:.1f} KB ({r.line_count:,} lines)")
        print(f"- Events loaded: {r.event_count:,}")
        print(
            f"- Load time: {r.load_time_mean:.4f}s +/- {r.load_time_std:.4f}s "
            f"(min: {r.load_time_min:.4f}s, max: {r.load_time_max:.4f}s)"
        )
        print(f"- **Throughput: {r.throughput:,.0f} events/sec**")
        print(f"- Zero iteration: {'PASS' if r.zero_iteration else 'FAIL'}")
        print()

    # Performance summary
    print("\n## Performance Summary\n")

    all_pass = all(r.zero_iteration for r in results)
    min_throughput = min(r.throughput for r in results) if results else 0
    avg_throughput = np.mean([r.throughput for r in results]) if results else 0

    print(f"- Zero iteration validation: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"- Minimum throughput: {min_throughput:,.0f} events/sec")
    print(f"- Average throughput: {avg_throughput:,.0f} events/sec")

    target = 20000
    if min_throughput >= target:
        print(f"- **Target ({target:,} events/sec): ACHIEVED**")
    else:
        print(
            f"- **Target ({target:,} events/sec): NOT MET** (need {target - min_throughput:,.0f} more)"
        )


# endregion


# region Main


def main() -> int:
    """Run profiling and print report.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    specimens = get_specimens()

    if not specimens:
        print("ERROR: No specimens found. Check paths.")
        return 1

    print(f"Found {len(specimens)} specimens to profile...\n")

    results = []

    for file_path, loader_class, loader_kwargs in specimens:
        print(f"Profiling {file_path.name}...", end=" ", flush=True)

        # Profile performance
        result = profile_loader(loader_class, file_path, iterations=10, **loader_kwargs)

        # Validate zero iteration
        result.zero_iteration = validate_zero_iteration(
            loader_class, file_path, **loader_kwargs
        )

        results.append(result)
        print(f"{result.throughput:,.0f} events/sec")

    print_report(results)

    # Return failure if any tests failed or throughput below target
    all_pass = all(r.zero_iteration for r in results)
    min_throughput = min(r.throughput for r in results) if results else 0

    if not all_pass:
        print("\nFAILED: Row iteration detected!")
        return 1

    if min_throughput < 20000:
        print(f"\nWARNING: Throughput ({min_throughput:,.0f}) below target (20,000)")
        # Don't fail on throughput, just warn
        return 0

    print("\nSUCCESS: All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# endregion
