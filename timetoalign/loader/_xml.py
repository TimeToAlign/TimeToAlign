"""Names shared by the loaders that read XML."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
"""The ``xml:id`` attribute as :mod:`xml.etree.ElementTree` names it."""


def element_paths(root: ET.Element) -> dict[int, str]:
    """The path of every element of the tree *root*, keyed by ``id(element)``.

    An XPath from the root through local names (``/ieee1599/logic/los/part[2]``),
    a step's position among its parent's children of the same name added,
    from 1, where the parent holds more than one of them; namespaces are
    left out. One walk over the tree.
    """
    paths: dict[int, str] = {}

    def walk(element: ET.Element, path: str) -> None:
        paths[id(element)] = path
        children = list(element)
        counts = Counter(_local(child.tag) for child in children)
        seen: Counter[str] = Counter()
        for child in children:
            name = _local(child.tag)
            seen[name] += 1
            step = f"{name}[{seen[name]}]" if counts[name] > 1 else name
            walk(child, f"{path}/{step}")

    walk(root, f"/{_local(root.tag)}")
    return paths


def _local(tag: object) -> str:
    """An element's tag without its namespace (a comment's: its kind)."""
    return tag.rpartition("}")[2] if isinstance(tag, str) else "comment()"
