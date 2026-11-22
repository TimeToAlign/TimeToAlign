"""Optional dependencies of the loaders.

Several loaders need a package that only one of the package's extras
installs. They all go through this module, so a missing package is handled
one way and reported with one message, naming the extra to install:
``Music21Loader needs music21: pip install "timetoalign[music21]"``.

- A sub-package exporting such loaders lists them as ``{name: (module,
  extra)}``, imports those it can with :func:`import_optional` and sets
  ``__getattr__ = resolve_optional(...)``, so asking for one whose package
  is missing raises :exc:`ImportError` with that message.
- A loader with a chain of backends asks :func:`optional_module`, which
  answers ``None`` for a missing package so the next backend can try.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from importlib import import_module
from types import ModuleType
from typing import Any

OptionalExports = Mapping[str, tuple[str, str]]
"""Export name -> (module of the sub-package defining it, extra installing its package)."""


def missing_dependency(subject: str, package: str | None, extra: str) -> str:
    """Return the message reporting that *subject* lacks *package*.

    Args:
        subject: What needs the package, such as a loader class name.
        package: The package that could not be imported.
        extra: The extra of ``timetoalign`` that installs it.

    Returns:
        ``'<subject> needs <package>: pip install "timetoalign[<extra>]"'``.
    """
    return f'{subject} needs {package}: pip install "timetoalign[{extra}]"'


def import_optional(
    package: str, exports: OptionalExports, namespace: MutableMapping[str, Any]
) -> list[str]:
    """Import every optional export whose module imports, into *namespace*.

    Args:
        package: The importing sub-package's ``__name__``.
        exports: The sub-package's optional exports.
        namespace: The sub-package's ``globals()``.

    Returns:
        The names imported, in the order of *exports*, for ``__all__``.
    """
    imported = []
    for name, (module, _extra) in exports.items():
        try:
            namespace[name] = getattr(import_module(f"{package}.{module}"), name)
        except ImportError:
            continue
        imported.append(name)
    return imported


def resolve_optional(package: str, exports: OptionalExports) -> Callable[[str], Any]:
    """Return a module ``__getattr__`` reporting an optional export's missing package.

    :func:`import_optional` has already bound every export that imports, so
    Python asks this function only for the others. It imports the export's
    module once more and turns a missing package into an :exc:`ImportError`
    naming the extra; any other import error propagates unchanged.

    Args:
        package: The sub-package's ``__name__``.
        exports: The sub-package's optional exports.

    Returns:
        The function to bind as the sub-package's ``__getattr__``.
    """

    def __getattr__(name: str) -> Any:
        target = exports.get(name)
        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        module, extra = target
        try:
            return getattr(import_module(f"{package}.{module}"), name)
        except ModuleNotFoundError as error:
            raise ImportError(missing_dependency(name, error.name, extra)) from error

    return __getattr__


def optional_module(name: str) -> ModuleType | None:
    """Import the package *name*, or answer ``None`` when it cannot be imported.

    Args:
        name: The package's import name, such as ``"soundfile"``.

    Returns:
        The module, or ``None``.
    """
    try:
        return import_module(name)
    except ImportError:
        return None
