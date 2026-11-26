"""``create_timeline()`` works after a bare ``import timetoalign``.

Validation logic is documented in ``tests/storage/README.md`` (the
``test_timeline_factory_import.py`` section). Each check runs in a fresh
interpreter, since the pytest process has always imported the timeline layer
already.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

_PRELUDE = """
import sys

import timetoalign

assert "timetoalign.timelines" not in sys.modules, "timeline layer already imported"
data = timetoalign.EventData.from_dicts(
    [{"id": "n1", "event_type": "Note", "start": 0, "end": 1}],
    unit=timetoalign.TimeUnit.quarters,
)
"""


def _run(body: str) -> list[str]:
    """Run *body* after the prelude in a fresh interpreter; return its lines."""
    completed = subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.splitlines()


def test_event_data_creates_a_timeline_after_import_timetoalign() -> None:
    lines = _run("""
        timeline = data.create_timeline(uid="notes")
        print(type(timeline).__name__, timeline.id, len(timeline.get_events()))
        """)
    assert lines == ["ContinuousLogicalTimeline notes 1"]


def test_store_creates_a_timeline_after_import_timetoalign() -> None:
    lines = _run("""
        timeline = timetoalign.SingleStore(data).create_timeline(uid="score")
        ids = timeline.get_events().table.column("id").to_pylist()
        print(type(timeline).__name__, timeline.id, ids)
        """)
    assert lines == ["ContinuousLogicalTimeline score ['n1']"]
