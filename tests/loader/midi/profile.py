"""Profiling script for MIDI loaders."""

import time
import warnings
from pathlib import Path

# Suppress pkg_resources deprecation warning from partitura
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="partitura.*")
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    import partitura as pt

from timetoalign.loader.midi import PerformanceMidiLoader, ScoreMidiLoader
from timetoalign.testdata import ensure_data

DATA_DIR = ensure_data("midi")


def profile_loader(name: str, loader_cls, path: Path, **kwargs):
    print(f"--- Profiling {name} ---")
    print(f"File: {path.name}")
    print(f"Size: {path.stat().st_size / 1024:.1f} KB")

    start_time = time.time()
    loader = loader_cls(**kwargs)
    loader.load(path)
    duration = time.time() - start_time

    print(f"Time: {duration:.4f} seconds")
    print(f"Events: {len(loader)}")
    print(f"Rate: {len(loader) / duration:.0f} events/sec")
    print("-" * 30)
    return duration


def profile_raw_partitura_perf(path: Path):
    print("--- Profiling Partitura Raw (load_performance_midi) ---")
    print(f"File: {path.name}")
    print(f"Size: {path.stat().st_size / 1024:.1f} KB")

    start_time = time.time()
    # load_performance_midi returns a Performance object
    perf = pt.load_performance_midi(path)
    # Access note array to ensure full loading
    _ = perf.note_array()
    duration = time.time() - start_time

    # Count notes + controls
    n_notes = len(perf.note_array())
    # Approximation of controls count not easily available without iterating tracks,
    # but note_array length is the main comparable metric

    print(f"Time: {duration:.4f} seconds")
    print(f"Events (Notes): {n_notes}")
    print(f"Rate: {n_notes / duration:.0f} notes/sec")
    print("-" * 30)
    return duration


def main():
    perf_file = DATA_DIR / "performance" / "supra_raw.mid"
    score_file = DATA_DIR / "score" / "beethoven_op18.mid"

    # 1. Profile PerformanceMidiLoader (mido) on raw performance data
    if perf_file.exists():
        t_mido = profile_loader(
            "PerformanceMidiLoader (mido)", PerformanceMidiLoader, perf_file
        )

    # 2. Profile Partitura Raw Performance loading
    if perf_file.exists():
        t_part_perf = profile_raw_partitura_perf(perf_file)

    # 3. Profile ScoreMidiLoader (partitura load_score_midi) on performance data
    # This checks the overhead of treating performance as score (quantization etc)
    if perf_file.exists():
        t_part_score = profile_loader(
            "ScoreMidiLoader (partitura load_score_midi)", ScoreMidiLoader, perf_file
        )

    # 4. Profile ScoreMidiLoader on actual score data (intended use)
    if score_file.exists():
        profile_loader("ScoreMidiLoader (partitura)", ScoreMidiLoader, score_file)

    if perf_file.exists():
        print("\nSummary on Performance Data:")
        print(f"Mido Speedup vs Partitura (Perf Mode): {t_part_perf/t_mido:.2f}x")
        print(f"Mido Speedup vs Partitura (Score Mode): {t_part_score/t_mido:.2f}x")


if __name__ == "__main__":
    main()
