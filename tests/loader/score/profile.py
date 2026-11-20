"""Profiling script for Symbolic Score Loaders."""

import timeit

from timetoalign.testdata import ensure_data

DATA_DIR = ensure_data("vienna_1x22")
MS3_DIR = DATA_DIR / "ms3"
CHOPIN_XML = DATA_DIR / "Chopin_op10_no3.musicxml"
CHOPIN_TSV = MS3_DIR / "chopin_op10_no3.notes.tsv"


def profile_loader(loader_cls, source, name, loops=5):
    """Run timing benchmark for a loader."""

    def _run():
        loader = loader_cls()
        loader.load(source)

    # Warmup
    try:
        _run()
    except Exception as e:
        print(f"Skipping {name}: {e}")
        return None

    # Timing
    t = timeit.timeit(_run, number=loops)
    avg_time = t / loops
    print(f"{name:<20}: {avg_time*1000:.2f} ms per run (avg of {loops})")
    return avg_time


def profile_all():
    """Profile all loaders and print ScoreBundle summary."""
    from timetoalign.loader.score.ms3 import Ms3Loader
    from timetoalign.loader.score.music21 import Music21Loader
    from timetoalign.loader.score.partitura import PartituraLoader

    print("=" * 60)
    print("Score Loader Profiling (Chopin Op. 10 No. 3)")
    print("=" * 60)
    print()

    # TSV
    print("--- Ms3Loader (notes.tsv) ---")
    profile_loader(Ms3Loader, CHOPIN_TSV, "Ms3Loader")
    tsv_bundle = Ms3Loader().load(CHOPIN_TSV)
    print(f"    {tsv_bundle}")
    print()

    # Partitura
    print("--- PartituraLoader (MusicXML) ---")
    profile_loader(PartituraLoader, CHOPIN_XML, "PartituraLoader")
    pt_bundle = PartituraLoader().load(CHOPIN_XML)
    print(f"    {pt_bundle}")
    print()

    # Music21
    print("--- Music21Loader (MusicXML) ---")
    profile_loader(Music21Loader, CHOPIN_XML, "Music21Loader")
    m21_bundle = Music21Loader().load(CHOPIN_XML)
    print(f"    {m21_bundle}")
    print()

    # Summary
    print("=" * 60)
    print("Note Counts Comparison")
    print("=" * 60)
    print(f"TSV:       {len(tsv_bundle.notes)} notes")
    print(
        f"Partitura: {len(pt_bundle.notes)} notes, {len(pt_bundle.measures)} measures"
    )
    print(
        f"Music21:   {len(m21_bundle.notes)} notes (has_rests={m21_bundle.notes.has_rests})"
    )


if __name__ == "__main__":
    profile_all()
