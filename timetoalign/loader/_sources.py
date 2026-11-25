"""Where a loader reads from: filesystem paths and members of ZIP archives.

A source is a filesystem path or a member of a ZIP archive, given as a
standard-library ``zipfile.Path``; the archive is read in place, never
extracted. A parser that takes a file object reads a member through
:func:`readable`; one that needs a file on disk reads the copy
:func:`materialize` makes (inside a loader: :func:`on_disk`, which also names
the member when that parser fails). Navigation inside an archive
(:func:`source_parent`, :func:`join_source`, :func:`sibling_source`,
:func:`find_sources`, :func:`sort_sources`) mirrors the filesystem.

Only :data:`SourceLike` and :func:`materialize` are public, exported from
:mod:`timetoalign.loader`; everything else serves the loaders.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import posixpath
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import IO, TypeAlias

#: What a loader reads: a filesystem path, as a ``Path`` or a ``str``, or a
#: member of a ZIP archive, as a ``zipfile.Path`` (the archive is read in
#: place, never extracted). A string always names a filesystem path:
#: ``"corpus.zip/notes/x.tsv"`` is a file of that name, not a member of
#: ``corpus.zip``.
SourceLike: TypeAlias = Path | str | zipfile.Path

#: A source as a loader holds it: a ``Path``, or a ``zipfile.Path`` member.
Source: TypeAlias = Path | zipfile.Path

#: Environment variable naming the folder that holds the copies
#: :func:`materialize` makes; unset, it is ``~/.cache/timetoalign``.
CACHE_DIR_VARIABLE = "TTA_CACHE_DIR"


# region Identity and display


def _archive_path(source: zipfile.Path) -> Path:
    """The absolute path of the ZIP archive *source* is a member of.

    Args:
        source: A member of a ZIP archive.

    Returns:
        The archive's absolute filesystem path.

    Raises:
        ValueError: If the archive was opened from a file object, so it has
            no path to record or to key a copy by.
    """
    filename = source.root.filename
    if filename is None:
        raise ValueError(
            f"The ZIP archive of member '{source.at}' was opened from a file object; "
            "a loader reads members of an archive opened from its path."
        )
    return Path(filename).absolute()


def _member_name(source: zipfile.Path) -> str:
    """The POSIX path of *source* inside its archive.

    Args:
        source: A member of a ZIP archive.

    Returns:
        The member's path, without a closing ``/`` (``""`` for the root).
    """
    return source.at.rstrip("/")


def as_source(source: SourceLike) -> Source:
    """Return *source* as a loader holds it.

    A ``str`` or ``Path`` becomes a ``Path`` (existence is the loader's
    concern, as it always was). A ``zipfile.Path`` is kept as it is; a member
    named without its closing ``/`` that is a directory is taken as that
    directory.

    Args:
        source: A filesystem path or a member of a ZIP archive.

    Returns:
        A ``Path`` or a ``zipfile.Path``.

    Raises:
        FileNotFoundError: If *source* is a member its archive does not hold,
            naming both the archive and the member.
        ValueError: If *source*'s archive was opened from a file object.
    """
    if not isinstance(source, zipfile.Path):
        return Path(source)
    archive = _archive_path(source)
    if source.exists():
        return source
    if source.at and not source.at.endswith("/"):
        directory = zipfile.Path(source.root, f"{source.at}/")
        if directory.exists():
            return directory
    raise FileNotFoundError(
        f"The ZIP archive '{archive}' holds no member '{_member_name(source)}'."
    )


def describe_source(source: object) -> str:
    """How messages, warnings and summaries name *source*.

    A filesystem path is named as ``str`` names it, as it always was. A
    member of a ZIP archive is named ``<archive> member <name>``: the
    archive's absolute path and the member's path inside it (``str`` of a
    ``zipfile.Path`` would join the two into a filesystem path that does not
    exist). Anything else is named as ``str`` names it.

    Args:
        source: A filesystem path or a member of a ZIP archive, as a loader
            is given it.

    Returns:
        The name to show.
    """
    if not isinstance(source, zipfile.Path):
        return str(source)
    filename = source.root.filename
    archive = Path(filename).absolute() if filename is not None else "<file object>"
    return f"{archive} member {_member_name(source)}"


def source_record(source: Source, key: str = "path") -> dict[str, str]:
    """The provenance of *source* as metadata records it.

    A filesystem source records ``{key: str(path)}``, the string it always
    recorded. A member of a ZIP archive records the archive's absolute path
    under *key* and the member's POSIX path inside it under ``member``, so
    no recorded value is a path that does not exist.

    Args:
        source: A filesystem path or a member of a ZIP archive.
        key: The name under which the path is recorded (``"path"`` in a
            loader's per-source metadata, ``"source"`` in a store's).

    Returns:
        The entries to record.

    Raises:
        ValueError: If *source*'s archive was opened from a file object.
    """
    if isinstance(source, Path):
        return {key: str(source)}
    return {key: str(_archive_path(source)), "member": _member_name(source)}


def member_uri(source: zipfile.Path) -> str:
    """The conventional URI of a member of a ZIP archive.

    ``jar:<archive file URI>!/<member>``, the archive's absolute path and
    the member's POSIX path both percent-encoded: for instance
    ``jar:file:///data/melod%20y.zip!/works/1.ttl``.

    Args:
        source: A member of a ZIP archive.

    Returns:
        The member's URI.

    Raises:
        ValueError: If *source*'s archive was opened from a file object.
    """
    from urllib.parse import quote

    return f"jar:{_archive_path(source).as_uri()}!/{quote(_member_name(source))}"


# endregion

# region Reading in place


@contextmanager
def readable(source: Source) -> Iterator[Path | IO[bytes]]:
    """Give a reader that takes a path or a binary file object what it reads.

    A filesystem source is given as its ``Path``, so the reader sees exactly
    what it always saw (pandas, for instance, infers compression from the
    name). An archive member is opened in binary mode and closed afterwards;
    nothing is copied.

    Args:
        source: A filesystem path or a member of a ZIP archive.

    Yields:
        The ``Path``, or the member's open binary file object.
    """
    if isinstance(source, Path):
        yield source
        return
    with source.open("rb") as handle:
        yield handle


# endregion

# region Navigation


def source_parent(source: Source) -> Source:
    """The directory holding *source*.

    A member's parent stays inside its archive: the archive's root directory
    is its own parent (``zipfile.Path.parent`` would leave the archive there,
    for the folder holding it).

    Args:
        source: A filesystem path or a member of a ZIP archive.

    Returns:
        The directory, of the same kind as *source*.
    """
    if isinstance(source, Path):
        return source.parent
    parent = posixpath.dirname(source.at.rstrip("/"))
    return zipfile.Path(source.root, f"{parent}/" if parent else "")


def join_source(directory: Source, relative: str) -> Source:
    """The file a path written in a source names, relative to *directory*.

    A filesystem directory joins *relative* as ``directory / relative``
    does. Inside an archive, ``.`` and ``..`` segments are resolved, as the
    filesystem resolves them (``./audio/take.png`` names ``audio/take.png``);
    a path leading above the archive's root names no member.

    Args:
        directory: A filesystem directory or a directory member.
        relative: A relative path, POSIX-separated.

    Returns:
        The named path or member (which need not exist).
    """
    if isinstance(directory, Path):
        return directory / relative
    joined = posixpath.normpath(posixpath.join(directory.at, relative))
    return zipfile.Path(directory.root).joinpath("" if joined == "." else joined)


def sibling_source(source: Source, relative: str) -> Source:
    """The file *relative* names beside *source*, in the directory holding it.

    Args:
        source: A filesystem path or a member of a ZIP archive.
        relative: A path relative to the directory holding *source*, as a
            source names a file next to it (``"take.wav"``,
            ``"./audio/take.png"``).

    Returns:
        ``join_source(source_parent(source), relative)``.
    """
    return join_source(source_parent(source), relative)


def _order_key(source: Source) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    """The key :func:`sort_sources` orders by.

    Args:
        source: A filesystem path or a member of a ZIP archive.

    Returns:
        Filesystem paths before members; then the path's parts, compared as
        strings (case-sensitive on every platform); a member's archive's
        parts, then its own parts inside the archive.
    """
    if isinstance(source, Path):
        return (0, source.parts, ())
    return (1, _archive_path(source).parts, PurePosixPath(source.at).parts)


def sort_sources(sources: Iterable[Source]) -> list[Source]:
    """*sources* in path order, one rule for paths and members.

    Paths are compared part by part, the parts as strings, on every
    platform (``sorted`` of ``Path`` objects ignores case on Windows); a
    member's parts inside its archive are compared the same way, so the
    files of a directory come in the same order from the filesystem and
    from an archive. Filesystem paths come before archive members.

    Args:
        sources: Filesystem paths and members of ZIP archives.

    Returns:
        The sources, ordered.
    """
    return sorted(sources, key=_order_key)


def find_sources(
    directory: Source, pattern: str, *, recursive: bool = False
) -> list[Source]:
    """The entries of *directory* whose name matches *pattern*, ordered.

    For a filesystem directory these are what ``directory.glob(pattern)``
    (``rglob`` when *recursive*) finds; an archive directory is searched the
    same way among its members, without extracting anything, directories
    included. A directory an archive holds only implicitly (it has members
    but no entry of its own) is found too: ``zipfile.Path`` lists such
    directories among the archive's names. Names in an archive are matched
    case-sensitively (``fnmatch.fnmatchcase``).

    Args:
        directory: A filesystem directory or a directory member of a ZIP
            archive.
        pattern: A shell-style pattern matched against entry names
            (``"*.wav"``).
        recursive: Whether to search every level below *directory*.

    Returns:
        The matching entries, in the order of :func:`sort_sources`.
    """
    if isinstance(directory, Path):
        return sort_sources(
            directory.rglob(pattern) if recursive else directory.glob(pattern)
        )
    prefix = directory.at
    matches: list[Source] = []
    # zipfile.Path's root lists implied directories among the names.
    for name in directory.root.namelist():
        if not name.startswith(prefix) or name == prefix:
            continue
        relative = name.removeprefix(prefix).rstrip("/")
        if not recursive and "/" in relative:
            continue
        if fnmatch.fnmatchcase(PurePosixPath(relative).name, pattern):
            matches.append(zipfile.Path(directory.root, name))
    return sort_sources(matches)


# endregion

# region Copies on disk


def cache_dir() -> Path:
    """The folder holding the copies :func:`materialize` makes.

    Returns:
        The value of the environment variable ``TTA_CACHE_DIR`` when it is
        set, else ``~/.cache/timetoalign``.
    """
    configured = os.environ.get(CACHE_DIR_VARIABLE)
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "timetoalign"


def _copy_folder(source: zipfile.Path, names: list[str]) -> Path:
    """The folder holding the copies of the members *names* of *source*'s archive.

    It is keyed by the archive's absolute path, size and modification time,
    and by each member's name, CRC-32 and size as the archive states them,
    so an archive whose members change gets fresh copies, even when it
    keeps its size and modification time.

    Args:
        source: A member of the archive.
        names: The archive names of the file members to be copied.

    Returns:
        The folder, below :func:`cache_dir`.
    """
    archive = _archive_path(source)
    status = archive.stat()
    key = hashlib.sha256(f"{archive}\0{status.st_size}\0{status.st_mtime_ns}".encode())
    for name in names:
        info = source.root.getinfo(name)
        key.update(f"\0{name}\0{info.CRC}\0{info.file_size}".encode())
    return cache_dir() / "archives" / f"{archive.stem}-{key.hexdigest()[:16]}"


def _copy_member(archive: zipfile.ZipFile, name: str, folder: Path) -> Path:
    """Copy the file member *name* of *archive* below *folder* once.

    The copy is written to a temporary file beside its place and renamed into
    it, so no reader ever sees half a copy; a copy already in place is
    returned as it is.

    Args:
        archive: The archive holding the member.
        name: The member's name in the archive.
        folder: The folder the member's path is laid out below.

    Returns:
        The copy's path.

    Raises:
        ValueError: If *name* points outside the archive (``..``, an
            absolute path).
    """
    parts = PurePosixPath(name).parts
    target = folder.joinpath(*parts)
    if (
        not parts
        or PurePosixPath(name).is_absolute()
        or ".." in parts
        or not target.resolve().is_relative_to(folder.resolve())
    ):
        raise ValueError(f"The ZIP member '{name}' names a path outside its archive.")
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".part"
    )
    try:
        with os.fdopen(descriptor, "wb") as copy, archive.open(name) as member:
            shutil.copyfileobj(member, copy)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def materialize(source: SourceLike) -> Path:
    """Return a filesystem path holding *source*, for a parser that needs one.

    A filesystem path is returned unchanged. A file member of a ZIP archive
    is copied once into the cache and the copy's path returned; a directory
    member is copied with everything below it. A later call for the same
    member of the same archive copies nothing; the copy is reused only while
    the archive keeps its absolute path, size and modification time and the
    member its CRC-32 and size, so a changed archive or member gets a fresh
    copy. Copies are written atomically, so parallel processes never see
    half of one.

    The cache lives in the folder the environment variable ``TTA_CACHE_DIR``
    names, else in ``~/.cache/timetoalign``, below
    ``archives/<archive stem>-<key>/<member path>``. It is never pruned:
    copies of archives that changed or are gone stay until the folder is
    deleted by hand.

    Args:
        source: A filesystem path or a member of a ZIP archive.

    Returns:
        The path itself, or the path of the member's copy.

    Raises:
        FileNotFoundError: If *source* is a member its archive does not hold.
        ValueError: If *source*'s archive was opened from a file object, or
            a member names a path outside its archive.

    Examples:
        >>> from zipfile import Path as ZipPath
        >>> score = materialize(ZipPath("corpus.zip", at="scores/piece.mei"))
    """
    held = as_source(source)
    if isinstance(held, Path):
        return held
    if not held.is_dir():
        return _copy_member(held.root, held.at, _copy_folder(held, [held.at]))
    names = [
        name
        for name in held.root.namelist()
        if name.startswith(held.at) and not name.endswith("/")
    ]
    folder = _copy_folder(held, names)
    for name in names:
        _copy_member(held.root, name, folder)
    directory = folder.joinpath(*PurePosixPath(_member_name(held)).parts)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@contextmanager
def on_disk(source: Source) -> Iterator[Path]:
    """Give a parser that needs a file on disk *source*'s path, or its copy.

    The path is :func:`materialize`'s. When the parser fails on the copy of
    a member, its exception is re-raised unchanged with a note naming the
    archive and the member (``<archive> member <name>``), so the error does
    not point at the cache alone.

    Args:
        source: A filesystem path or a member of a ZIP archive.

    Yields:
        The filesystem path, or the member's copy.

    Raises:
        FileNotFoundError: If *source* is a member its archive does not hold.
        ValueError: If *source*'s archive was opened from a file object.
    """
    path = materialize(source)
    try:
        yield path
    except Exception as error:
        if isinstance(source, zipfile.Path):
            error.add_note(f"{path} is the copy of {describe_source(source)}.")
        raise


# endregion
