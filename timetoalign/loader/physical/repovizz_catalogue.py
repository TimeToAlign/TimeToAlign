"""Catalogue models and XML parsing for RepoVizz manifests."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

from timetoalign.loader._sources import Source


@dataclass(frozen=True)
class CatalogueEntry:
    """Metadata for a loadable signal/audio/annotation from an XML manifest.

    This dataclass represents a single loadable item from a RepoVizz
    (or similar) XML manifest. It captures enough metadata to:
    1. Identify the item (xml_id, name)
    2. Categorize it (group, subgroup, file_type)
    3. Load the data (filename, sample_rate, n_samples)
    4. Handle special cases (related_ids for MoCap X/Y/Z grouping)

    The class is frozen (immutable) to enable use as dict keys and to
    ensure catalogue entries don't change after parsing.

    Attributes:
        xml_id: Unique identifier from the XML (ID attribute).
        name: Human-readable name from the XML (Name or name attribute).
        category: Category from the XML (e.g., "Ambient", "Pickup", "AuDesc").
        group: Top-level group: "audio", "score", "descriptors", "mocap".
        subgroup: Second-level grouping (e.g., "ambient", "vln1", "MoCap").
        filename: Data file referenced (None for container elements).
        file_type: File format: "BWF", "CSV", "NOTES", "txt", "wav", etc.
        sample_rate: Sampling rate in Hz (0.0 for non-signal entries).
        n_samples: Number of samples from XML (0 for non-signal entries).
        frame_size: For multi-dimensional signals (1 for scalar).
        metadata: Additional attributes from the XML element.
        related_ids: For MoCapMarker: tuple of (x_id, y_id, z_id) signal IDs.
    """

    xml_id: str
    name: str
    category: str
    group: str
    subgroup: str
    filename: str | None
    file_type: str
    sample_rate: float
    n_samples: int
    frame_size: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    related_ids: tuple[str, ...] = ()

    @property
    def is_signal(self) -> bool:
        """True if this entry represents a loadable signal (has samples)."""
        return self.n_samples > 0 and self.sample_rate > 0

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds (0 if not a signal)."""
        if not self.is_signal:
            return 0.0
        return self.n_samples / self.sample_rate


class _XmlCatalogueParser:
    """Build a RepoVizz catalogue from an XML manifest tree."""

    def __init__(
        self,
        xml_root_path: Source,
        metadata: dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        """Initialize parser state.

        Args:
            xml_root_path: Directory that contains the manifest's data files.
            metadata: Manifest metadata updated while parsing.
            logger: Logger associated with the owning loader.
        """
        self._xml_root_path = xml_root_path
        self._metadata = metadata
        self._logger = logger
        self._catalogue: dict[str, CatalogueEntry] = {}
        self._specs: list[dict[str, Any]] = []

    def parse_catalogue(
        self, root: ET.Element
    ) -> tuple[dict[str, CatalogueEntry], list[dict[str, Any]]]:
        """Parse a manifest root and return its catalogue and timeline specifications."""
        self._walk_xml(root, group="", subgroup="")

        for elem in root.iter():
            if elem.get("Category") == "METADATA":
                for child in elem:
                    if child.get("Category") == "TITLE":
                        self._metadata["title"] = child.get("Text", "")
                        break
                break

        return self._catalogue, self._specs

    def _walk_xml(self, elem: ET.Element, group: str, subgroup: str) -> None:
        """Recursively walk XML and build catalogue entries.

        Args:
            elem: Current XML element.
            group: Current top-level group name.
            subgroup: Current second-level group name.
        """
        tag = elem.tag
        category = elem.get("Category", elem.get("category", ""))
        xml_id = elem.get("ID", elem.get("id", ""))
        name = elem.get("Name", elem.get("name", ""))

        current_group = group
        current_subgroup = subgroup

        if tag == "Generic":
            if category in ("AudioGroup", "METADATA"):
                current_group = "audio" if category == "AudioGroup" else "metadata"
            elif category == "Score":
                current_group = "score"
            elif category == "DescriptorGroup":
                current_group = "descriptors"
            elif category == "MoCapGroup":
                current_group = "mocap"
            elif category == "MoCapMarker":
                current_subgroup = name or xml_id
                xyz_ids = self._collect_mocap_xyz(elem)
                if xyz_ids:
                    entry = CatalogueEntry(
                        xml_id=xml_id,
                        name=name,
                        category=category,
                        group="mocap",
                        subgroup=current_subgroup,
                        filename=None,
                        file_type="MoCapMarker",
                        sample_rate=240.0,
                        n_samples=0,
                        related_ids=tuple(xyz_ids),
                    )
                    self._catalogue[xml_id] = entry
                    self._specs.append(
                        {
                            "id": xml_id,
                            "name": name,
                            "group": "mocap",
                            "entry": entry,
                        }
                    )
            elif name:
                current_subgroup = name.split()[0].lower()

        elif tag == "Audio":
            entry = self._parse_audio_entry(elem, current_group, current_subgroup)
            self._catalogue[entry.xml_id] = entry
            self._specs.append(
                {
                    "id": entry.xml_id,
                    "name": entry.name,
                    "group": "audio",
                    "entry": entry,
                }
            )

        elif tag == "Signal":
            entry = self._parse_signal_entry(elem, current_group, current_subgroup)
            self._catalogue[entry.xml_id] = entry
            if current_group != "mocap" or category not in ("X", "Y", "Z"):
                self._specs.append(
                    {
                        "id": entry.xml_id,
                        "name": entry.name,
                        "group": entry.group,
                        "entry": entry,
                    }
                )

        elif tag == "Annotation":
            entry = self._parse_annotation_entry(elem, current_subgroup)
            self._catalogue[entry.xml_id] = entry
            self._specs.append(
                {
                    "id": entry.xml_id,
                    "name": entry.name,
                    "group": "score",
                    "entry": entry,
                }
            )

        for child in elem:
            self._walk_xml(child, current_group, current_subgroup)

    def _collect_mocap_xyz(self, marker_elem: ET.Element) -> list[str]:
        """Collect X, Y, Z signal IDs from a MoCapMarker element.

        Args:
            marker_elem: The Generic element with Category="MoCapMarker".

        Returns:
            List of signal IDs [x_id, y_id, z_id], or empty if not found.
        """
        xyz_ids: dict[str, str] = {}
        for child in marker_elem:
            if child.tag == "Signal":
                cat = child.get("Category", child.get("category", ""))
                xml_id = child.get("ID", child.get("id", ""))
                if cat in ("X", "Y", "Z") and xml_id:
                    xyz_ids[cat] = xml_id

        if "X" in xyz_ids and "Y" in xyz_ids and "Z" in xyz_ids:
            return [xyz_ids["X"], xyz_ids["Y"], xyz_ids["Z"]]
        return []

    def _parse_audio_entry(
        self,
        elem: ET.Element,
        group: str,
        subgroup: str,
    ) -> CatalogueEntry:
        """Parse an Audio element into a CatalogueEntry."""
        xml_id = elem.get("ID", elem.get("id", ""))
        name = elem.get("Name", elem.get("name", ""))
        filename = elem.get("Filename", elem.get("filename"))
        category = elem.get("Category", elem.get("category", ""))
        sample_rate = float(elem.get("SampleRate", elem.get("samplerate", "0")))
        n_samples = int(elem.get("NumSamples", elem.get("numsamples", "0")))
        file_type = elem.get("FileType", elem.get("filetype", "BWF"))

        return CatalogueEntry(
            xml_id=xml_id,
            name=name,
            category=category,
            group="audio",
            subgroup=subgroup or category.lower(),
            filename=filename,
            file_type=file_type,
            sample_rate=sample_rate,
            n_samples=n_samples,
        )

    def _parse_signal_entry(
        self,
        elem: ET.Element,
        group: str,
        subgroup: str,
    ) -> CatalogueEntry:
        """Parse a Signal element into a CatalogueEntry."""
        xml_id = elem.get("ID", elem.get("id", ""))
        name = elem.get("Name", elem.get("name", ""))
        filename = elem.get("Filename", elem.get("filename"))
        category = elem.get("Category", elem.get("category", ""))
        sample_rate_str = elem.get("SampleRate", elem.get("samplerate", "")) or "0"
        sample_rate = float(sample_rate_str) if sample_rate_str else 0.0
        n_samples_str = elem.get("NumSamples", elem.get("numsamples", "")) or "0"
        n_samples = int(n_samples_str) if n_samples_str else 0
        frame_size_str = elem.get("FrameSize", elem.get("framesize", "")) or "1"
        frame_size = int(frame_size_str) if frame_size_str else 1

        file_type = "wav"
        if filename:
            if filename.endswith(".csv"):
                file_type = "CSV"
            elif filename.endswith(".wav"):
                file_type = "wav"

        actual_group = group
        if category == "AuDesc":
            actual_group = "audio"
        elif category in ("X", "Y", "Z"):
            actual_group = "mocap"
        elif category == "Descriptor" or group == "descriptors":
            actual_group = "descriptors"

        return CatalogueEntry(
            xml_id=xml_id,
            name=name,
            category=category,
            group=actual_group,
            subgroup=subgroup,
            filename=filename,
            file_type=file_type,
            sample_rate=sample_rate,
            n_samples=n_samples,
            frame_size=frame_size,
        )

    def _parse_annotation_entry(
        self, elem: ET.Element, subgroup: str
    ) -> CatalogueEntry:
        """Parse an Annotation element into a CatalogueEntry."""
        xml_id = elem.get("ID", elem.get("id", ""))
        name = elem.get("Name", elem.get("name", ""))
        filename = elem.get("Filename", elem.get("filename"))
        category = elem.get("Category", elem.get("category", ""))
        file_type = elem.get("FileType", elem.get("filetype", "NOTES"))

        return CatalogueEntry(
            xml_id=xml_id,
            name=name,
            category=category,
            group="score",
            subgroup=subgroup,
            filename=filename,
            file_type=file_type,
            sample_rate=0.0,
            n_samples=0,
        )
