"""Shared pytest fixtures for timetoalign tests."""

from __future__ import annotations

import contextlib
from pathlib import Path

# Import ms3 in the controller process first: its packaging machinery may
# (re)write a generated version module on import, and concurrent first
# imports across xdist workers can race on that write.
with contextlib.suppress(Exception):
    import ms3  # noqa: F401

import numpy as np
import pytest


def _c_int_bound_ldexp(real_ldexp):
    """Return ``numpy.ldexp`` as a 32-bit-``long`` build offers it.

    A handful of numpy ufuncs take an integer *argument* whose type is a C
    type rather than a numpy width — ``ldexp``'s exponent is one, offered as
    ``'di->d'`` and ``'dl->d'``. Where a C ``long`` is 64 bits (Linux, macOS)
    an ``int64`` exponent binds to the second loop; where it is 32 (Windows)
    neither loop fits and numpy raises rather than narrow the exponent for
    you. Reproducing that refusal is what lets a test hold the library to a
    platform its developers do not run on.
    """

    def ldexp(values, exponent, *args, **kwargs):
        if np.asarray(exponent).dtype.itemsize > 4:
            raise TypeError(
                "ufunc 'ldexp' not supported for the input types, and the "
                "inputs could not be safely coerced to any supported types "
                "according to the casting rule ''safe''"
            )
        return real_ldexp(values, exponent, *args, **kwargs)

    return ldexp


@pytest.fixture(params=["lp64", "llp64"])
def platform_contract(request, monkeypatch) -> str:
    """Run a test under both widths numpy binds its C-typed arguments to.

    A test taking this fixture runs twice: once as this machine offers numpy,
    and once with every C-typed ufunc argument narrowed to 32 bits, as an
    LLP64 build (Windows) offers it. Which functions those are is not open
    to drift — ``test_numpy_portability_hygiene.py`` fails if the library
    reaches for one outside this list.
    """
    if request.param == "llp64":
        monkeypatch.setattr(np, "ldexp", _c_int_bound_ldexp(np.ldexp))
    return request.param


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked slow (long-running corpus-scale integration tests)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test: pass --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def rekordbox_specimen() -> Path:
    """The Rekordbox collection specimen, skipping where it is absent.

    The specimen is not part of a fetched corpus; it lives in a local
    checkout of the test-data repository.
    """
    specimen = Path(
        "/home/laser/git/tta/tta_test_data/data/audio/moriero_dj_set/rekordbox.xml"
    )
    if not specimen.exists():
        pytest.skip(f"Test data file not found: {specimen}")
    return specimen
