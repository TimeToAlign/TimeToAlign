"""Building ZIP archives of test data, and the checks every archive test shares."""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def pack(archive: Path, members: Mapping[str, Path | bytes | str]) -> Path:
    """Write *members* (inner path → file, bytes or text) into the ZIP *archive*.

    The archive holds file entries only: every directory is implied by its
    members, as in archives made by many tools.
    """
    with zipfile.ZipFile(archive, "w") as handle:
        for name, content in members.items():
            if isinstance(content, Path):
                handle.write(content, name)
            else:
                handle.writestr(name, content)
    return archive


def lay_out(root: Path, members: Mapping[str, Path | bytes | str]) -> Path:
    """Write *members* below the folder *root*, as :func:`pack` packs them."""
    for name, content in members.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, Path):
            shutil.copyfile(content, target)
        elif isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    return root


def member(archive: Path, name: str) -> zipfile.Path:
    """The member *name* of *archive*."""
    return zipfile.Path(archive, at=name)


def file_members(
    directory: Path, prefix: str, *, only: Iterable[str] | None = None
) -> dict[str, Path]:
    """Every file below *directory*, named ``<prefix>/<path inside it>``.

    With *only*, just the files whose path inside *directory* starts with one
    of these prefixes.
    """
    kept = tuple(only) if only is not None else None
    files = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if not path.is_file() or path.name == ".tta_testdata_hash":
            continue
        if kept is not None and not relative.startswith(kept):
            continue
        files[f"{prefix}/{relative}"] = path
    return files


def same_rows(archived: Any, on_disk: Any) -> bool:
    """Whether two event tables are equal but for the extraction each row names.

    An extraction names its source file (a path, a member's ``jar:`` URI),
    so a member and the same bytes on disk are two extractions.
    """
    if "extraction_id" in archived.column_names:
        archived = archived.drop_columns(["extraction_id"])
    if "extraction_id" in on_disk.column_names:
        on_disk = on_disk.drop_columns(["extraction_id"])
    return archived.equals(on_disk)


def same_timeline(archived: Any, on_disk: Any) -> None:
    """Assert both loaders created the same timeline, exactly."""
    assert archived.create_timeline().to_dict() == on_disk.create_timeline().to_dict()


def strings_in(value: Any) -> list[str]:
    """Every string in *value*, keys and leaves, at any depth."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [
            text
            for key, item in value.items()
            for text in (*strings_in(key), *strings_in(item))
        ]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [text for item in value for text in strings_in(item)]
    return []


def copies(cache_dir: Path) -> list[Path]:
    """The files below the cache folder (no copy was made if it is empty)."""
    return sorted(path for path in cache_dir.rglob("*") if path.is_file())
