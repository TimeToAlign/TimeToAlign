"""What the metadata importers share: where pairs land, and the plan.

An import states many pairs on a bundle, its timelines and regions, and
adds a timeline per manifestation. :class:`Plan` makes it all or nothing:
every pair is first stated, proposed or passed over on a scratch copy of its
holder's statements, through the very methods a holder runs (TimeToShare's
functions and the library's prefix check); only then are the new timelines
added and the copies stored. A refusal leaves the bundle as it was.
"""

from __future__ import annotations

import copy
import datetime
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import timetoshare

from timetoalign.alignment.bundle import AlignmentBundle
from timetoalign.core.ids import mint_id
from timetoalign.timelines.base import Timeline
from timetoalign.timelines.statements import (
    MutableStatementsMixin,
    NodePath,
    StatementsMixin,
    _day,
    renamed_state,
)

__all__ = ["Plan", "Target", "day", "refuse_arguments", "refuse_value_entity"]

day = _day
"""A provenance entry's day: the date given, as the file writes it, else today."""


def refuse_arguments(
    into: AlignmentBundle | None,
    id: str | None,
    subjects: Mapping[str, str] | None,
) -> None:
    """Refuse an ``id`` beside ``into``, and ``subjects`` without ``into``.

    Raises:
        ValueError: ``id names a new bundle; ...`` or ``subjects needs
            into: ...``.
    """
    if into is not None and id is not None:
        raise ValueError(
            "id names a new bundle; the bundle given as into keeps its own"
        )
    if subjects and into is None:
        raise ValueError("subjects needs into: a new bundle holds no element to map to")


def refuse_value_entity(key: str, kind: str) -> None:
    """Refuse a key of ``subjects`` naming a work or an expression.

    Raises:
        ValueError: ``subjects maps 'w', a work; works and expressions are
            values, never elements``.
    """
    article = "an" if kind[0] in "aeiou" else "a"
    raise ValueError(
        f"subjects maps '{key}', {article} {kind}; works and expressions are values, "
        "never elements"
    )


@dataclass(frozen=True)
class Target:
    """Where an entity's pairs land.

    The bundle, a timeline (at any depth) or a region of it, or a timeline
    the import adds.

    Attributes:
        id: The ID a value names the element by (``{"id": "#<id>"}``); for a
            timeline the import adds, a provisional ID that :meth:`Plan.commit`
            replaces by the timeline's own.
        holder: The bundle, the timeline, or the timeline holding the region.
        region: The region's ID, or None.
        name: The holder as a message names it.
    """

    id: str
    holder: AlignmentBundle | Timeline
    region: str | None = None
    name: str = ""

    def view(self) -> StatementsMixin:
        """The holder's statements as they stand."""
        if self.region is None:
            return self.holder
        assert isinstance(self.holder, Timeline)
        return self.holder.get_region(self.region)

    def store(self, state: Mapping[str, Any]) -> None:
        """Store *state* as the holder's statements and provenance."""
        if self.region is None:
            self.holder._with_statement_state(state)
            return
        assert isinstance(self.holder, Timeline)
        self.holder.replace_region(self.view()._with_statement_state(state))


class _Scratch(MutableStatementsMixin):
    """A copy of a holder's statements that an import changes first."""

    def __init__(self, target: Target) -> None:
        view = target.view()
        self._statement_state = copy.deepcopy(dict(view._statement_state_view()))
        self._prefixes = dict(view._own_prefixes())
        self._name = target.name

    def _statement_holder_name(self) -> str:
        return self._name


class Plan:
    """The pairs of one import, stated on scratch copies of their holders.

    The importer finds its targets (:meth:`mapped`, :meth:`new_timeline`,
    :attr:`bundle_target`), then states and proposes pairs on them
    (:meth:`state`, :meth:`propose`); a pair a holder already states,
    proposes or rejects is passed over. Each change runs on the copy,
    through the methods the holder runs, so a refusal (a prefix no context
    declares, a pair TimeToShare refuses) is raised while the bundle is
    untouched. :meth:`commit` then changes the bundle.
    """

    def __init__(self, bundle: AlignmentBundle) -> None:
        self.bundle = bundle
        self.bundle_target = Target(
            bundle.id, bundle, name=bundle._statement_holder_name()
        )
        self._scratches: dict[tuple[int, str | None], tuple[Target, _Scratch]] = {}
        self._new: list[tuple[str, Timeline]] = []
        # A provisional ID is no ID of the bundle, so that no reference the
        # holders' statements hold already can name it.
        self._taken: set[str] = set(bundle._scope_ids())

    # region Targets

    def mapped(self, key: str, element_id: str) -> Target:
        """The timeline (at any depth) or region that ``subjects`` maps *key* to.

        Raises:
            ValueError: If the bundle holds no timeline or region *element_id*.
        """
        for root in self.bundle._roots():
            for timeline in (root, *(child for _, child in root._iter_descendants())):
                if timeline.id == element_id:
                    return Target(
                        element_id, timeline, name=timeline._statement_holder_name()
                    )
                if element_id in timeline.list_regions():
                    region = timeline.get_region(element_id)
                    return Target(
                        element_id,
                        timeline,
                        element_id,
                        name=region._statement_holder_name(),
                    )
        raise ValueError(
            f"subjects maps '{key}' to '{element_id}', which is no timeline or "
            f"region of {self.bundle._statement_holder_name()}"
        )

    def new_timeline(self, timeline: Timeline, entity: str) -> Target:
        """A timeline :meth:`commit` adds to the bundle, standing for *entity*."""
        provisional = mint_id("new", self._taken)
        self._taken.add(provisional)
        self._new.append((provisional, timeline))
        return Target(provisional, timeline, name=f"the new timeline of {entity}")

    # endregion

    # region Pairs

    def _scratch(self, target: Target) -> _Scratch:
        key = (id(target.holder), target.region)
        if key not in self._scratches:
            self._scratches[key] = (target, _Scratch(target))
        return self._scratches[key][1]

    def known(
        self, target: Target, property: str, value: Any, node: NodePath = ()
    ) -> bool:
        """Whether the target states, proposes or rejects the pair at *node*."""
        scratch = self._scratch(target)
        context = scratch._statement_context()
        wanted = (property, value)
        return any(
            timetoshare.same_pair(context, held, wanted)
            for held in scratch.get_pairs(node=node)
        ) or any(
            timetoshare.same_pair(context, (entry["property"], entry["value"]), wanted)
            for entry in scratch.get_provenance(node=node)
        )

    def held_index(
        self, target: Target, property: str, value: Any, node: NodePath = ()
    ) -> int | None:
        """The position of the value in its property's list at *node*, or None.

        None when the pair is not stated there: proposed, rejected or absent.
        """
        scratch = self._scratch(target)
        context = scratch._statement_context()
        values = [held for key, held in scratch.get_pairs(node=node) if key == property]
        return next(
            (
                index
                for index, held in enumerate(values)
                if timetoshare.same_pair(context, (property, held), (property, value))
            ),
            None,
        )

    def state(
        self,
        target: Target,
        property: str,
        value: Any,
        provenance: Mapping[str, Any],
        node: NodePath = (),
    ) -> None:
        """State the pair with its confirmed entry, unless the target knows it."""
        if not self.known(target, property, value, node):
            self._scratch(target).add_statement(
                property, value, node=node, provenance=provenance
            )

    def propose(
        self,
        target: Target,
        property: str,
        value: Any,
        *,
        stated_by: list[str],
        date: str | datetime.date | None,
        basis: list[str],
        note: str | None,
        node: NodePath = (),
    ) -> None:
        """Propose the pair (method ``imported``), unless the target knows it."""
        if not self.known(target, property, value, node):
            self._scratch(target).propose_statement(
                property,
                value,
                stated_by=stated_by,
                method="imported",
                date=date,
                basis=basis,
                note=note,
                node=node,
            )

    # endregion

    def commit(self) -> None:
        """Add the new timelines and store the copies.

        Every pair has been checked on its copy. The new timelines join the
        bundle, ungrouped, in the order they were planned, every provisional
        ID in the copies becomes the timeline's own, and each holder whose
        copy differs takes it.
        """
        renamed: dict[str, str | None] = {}
        for provisional, timeline in self._new:
            self.bundle.add_timeline(timeline)
            renamed[provisional] = timeline.id
        for target, scratch in self._scratches.values():
            state = renamed_state(scratch._statement_state, renamed)
            if state != dict(target.view()._statement_state_view()):
                target.store(state)
