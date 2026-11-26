"""Helpers for the IDs loaders mint: exact seeded IDs and minted shapes."""

from __future__ import annotations

from collections.abc import Iterable

from timetoalign.core.ids import IdMinter


def minted(seed: int, *prefixes: str) -> list[str]:
    """The IDs a fresh ``IdMinter(seed)`` mints for *prefixes*, in order."""
    minter = IdMinter(seed)
    return [minter.mint(prefix) for prefix in prefixes]


def shape(uid: str) -> str:
    """A minted ID without its random suffix: ``"score:clt"``.

    The suffix is checked to be the minter's 13 base-36 characters, so the
    shape stands for one minted ID.
    """
    head, suffix = uid.rsplit("_", 1)
    assert len(suffix) == 13, uid
    assert all(c in "0123456789abcdefghijklmnopqrstuvwxyz" for c in suffix), uid
    return head


def shapes(uids: Iterable[str]) -> list[str]:
    """The shapes of *uids*, in order."""
    return [shape(uid) for uid in uids]
