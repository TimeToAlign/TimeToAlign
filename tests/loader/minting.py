"""Helpers for the IDs loaders mint: exact seeded IDs and minted shapes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from fractions import Fraction
from typing import Any

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


def _place(destination: dict[str, Any]) -> tuple[int, Any, str]:
    """The canonical order of a Jump's destinations: no place, ``by``, ``to`` by ID."""
    if "to" in destination:
        return (2, 0, destination["to"])
    if "by" in destination:
        return (1, Fraction(*destination["by"]), "")
    return (0, 0, "")


def _renamed(value: Any, names: dict[str, str]) -> Any:
    """*value* with every string that is a key of *names* replaced, at any depth."""
    if isinstance(value, str):
        return names.get(value, value)
    if isinstance(value, dict):
        return {key: _renamed(member, names) for key, member in value.items()}
    if isinstance(value, list):
        return [_renamed(member, names) for member in value]
    return value


def named_flow_control(
    written: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Flow control as written, each element named by the order ``IdMinter(seed)`` minted it.

    Asserts the canonical order first: elements by coordinate, then by ID, and
    every Jump's destinations by place. Then each element is named
    ``<role>_<k>``, the k-th element of its role the minter minted (the
    position of an ID in the minter's sequence is read off ``IdMinter(seed)``),
    every reference is renamed with it, and the array is sorted by coordinate
    and name, destinations by place. Two arrays minted in one order compare
    equal whatever their seeds; a reference to another element does not.
    """

    def coordinate(element: dict[str, Any]) -> Fraction:
        return Fraction(*element["coordinate"])

    assert written == sorted(written, key=lambda e: (coordinate(e), e["id"]))
    for element in written:
        targets = element.get("targets", [])
        assert targets == sorted(targets, key=_place)

    minted = IdMinter(seed).mint_many("", 20_000).to_pylist()
    position = {suffix.removeprefix("_"): k for k, suffix in enumerate(minted)}
    by_role: dict[str, list[str]] = defaultdict(list)
    for element in written:
        by_role[element["role"]].append(element["id"])
    names = {
        element_id: f"{role}_{k}"
        for role, element_ids in by_role.items()
        for k, element_id in enumerate(
            sorted(element_ids, key=lambda i: position[i.removeprefix(f"{role}_")]),
            start=1,
        )
    }
    named = _renamed(written, names)
    for element in named:
        if "targets" in element:
            element["targets"] = sorted(element["targets"], key=_place)
    return sorted(named, key=lambda e: (coordinate(e), e["id"]))
