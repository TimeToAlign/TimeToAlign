"""Timeline-side access to shared temporal structures."""

from __future__ import annotations

from typing import Any


class SkeletonsMixin:
    """Expose skeleton membership while keeping enrollment skeleton-owned."""

    @property
    def skeleton(self) -> Any:
        """Return the sole attached skeleton."""
        if len(self._skeleton_attachments) != 1:
            raise ValueError(
                f"Timeline {self.id!r} has {len(self._skeleton_attachments)} skeleton attachments; "
                "exactly one is required"
            )
        return self._skeleton_attachments[0]

    @property
    def skeletons(self) -> tuple[Any, ...]:
        """All skeletons this timeline participates in."""
        return tuple(self._skeleton_attachments)

    def attach(self, *args: Any, **kwargs: Any) -> None:
        """Refuse timeline-side enrollment."""
        raise NotImplementedError("Enroll timelines with TimeSkeleton.attach(...)")

    def create_skeleton(self, *args: Any, **kwargs: Any) -> Any:
        """Return structure already harvested for this timeline.

        Creating structure from arbitrary events is intentionally unsupported:
        measures are structural scalars, not timeline events.
        """
        if len(self._skeleton_attachments) == 1:
            return self._skeleton_attachments[0]
        raise ValueError(f"Timeline {self.id!r} has no harvestable temporal structure")

    def _add_skeleton_attachment(self, skeleton: Any) -> None:
        # Skeleton equality is structural, so two distinct shared structures
        # may compare equal; attachment bookkeeping distinguishes by identity.
        if not any(existing is skeleton for existing in self._skeleton_attachments):
            self._skeleton_attachments.append(skeleton)

    def _remove_skeleton_attachment(self, skeleton: Any) -> None:
        self._skeleton_attachments = [
            existing
            for existing in self._skeleton_attachments
            if existing is not skeleton
        ]
