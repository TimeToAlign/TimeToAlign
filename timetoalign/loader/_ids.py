"""IDs a loader gives the elements it builds.

A loader mints every ID from its one minter (``Loader._id_minter``); the
helpers here hand that minter to rows that do not state their ID. An ID the
source states is kept as it is.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

from timetoalign.core.ids import IdMinter
from timetoalign.storage import mint_event_ids

#: The prefix of a conversion map's ID, every class alike.
CMAP_PREFIX = "cmap"

#: The prefix of a MatchClaim's ID.
CLAIM_PREFIX = "claim"

#: The prefix of a TimelineGroup's ID.
GROUP_PREFIX = "group"

#: The prefix of a measure map's ID.
MEASURE_MAP_PREFIX = "measure_map"


def minter_of(seed: int | None, minter: IdMinter | None) -> IdMinter | None:
    """The minter a constructor given *seed* or *minter* draws IDs from.

    A seed gives a fresh :class:`~timetoalign.core.ids.IdMinter`; neither
    gives None, for :func:`~timetoalign.storage.mint_event_ids`' content
    minter.

    Raises:
        TypeError: If both are given.
    """
    if seed is not None and minter is not None:
        raise TypeError("Give seed or minter, not both.")
    return IdMinter(seed) if seed is not None else minter


def give_event_ids(
    rows: Sequence[MutableMapping[str, Any]], minter: IdMinter | None
) -> None:
    """Give every row whose ``id`` is missing or None a minted event ID, in place.

    The IDs come from :func:`~timetoalign.storage.mint_event_ids`: the prefix
    is the row's ``event_type`` in lower case (``event`` without one), one
    batch per prefix in the order of its first row.

    Args:
        rows: Event rows; a row stating its ``id`` keeps it.
        minter: The minter the IDs are drawn from (a loader's), or None for
            the content minter of the event types.
    """
    missing = [row for row in rows if row.get("id") is None]
    if not missing:
        return
    ids = mint_event_ids([row.get("event_type") for row in missing], minter)
    for row, event_id in zip(missing, ids.to_pylist(), strict=True):
        row["id"] = event_id
