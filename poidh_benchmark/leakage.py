from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Literal

from poidh_detector.contracts import SampleRecord, require_safe_relative_path


_LOWERCASE_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class HoldoutRegistration:
    holdout_id: str
    manifest_path: Path
    manifest_sha256: str
    status: Literal["development_exposed", "frozen_unopened"]
    require_provenance: bool = False

    def __post_init__(self) -> None:
        if not self.holdout_id:
            raise ValueError("holdout_id must not be empty")
        if len(self.manifest_sha256) != 64 or any(
            character not in _LOWERCASE_HEX for character in self.manifest_sha256
        ):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.require_provenance, bool):
            raise ValueError("require_provenance must be a boolean")


@dataclass(frozen=True, slots=True)
class HoldoutIndex:
    registration: HoldoutRegistration
    sample_ids: frozenset[str]
    content_sha256: frozenset[str]
    provenance_groups: frozenset[str]


def load_registered_holdout(registration: HoldoutRegistration) -> HoldoutIndex:
    if not registration.manifest_path.is_file():
        raise ValueError(
            f"registered holdout manifest is missing: {registration.holdout_id}"
        )
    payload = registration.manifest_path.read_bytes()
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != registration.manifest_sha256:
        raise ValueError(
            f"registered holdout digest mismatch: {registration.holdout_id}"
        )
    document = json.loads(payload)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("unsupported registered holdout manifest")
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("registered holdout manifest must contain entries")

    sample_ids: set[str] = set()
    content_hashes: set[str] = set()
    provenance_groups: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("registered holdout entry must be an object")
        file_name = entry.get("file_name")
        content_hash = entry.get("content_sha256")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError("registered holdout entry requires file_name")
        require_safe_relative_path(file_name, "registered holdout file_name")
        if (
            not isinstance(content_hash, str)
            or len(content_hash) != 64
            or any(character not in _LOWERCASE_HEX for character in content_hash)
        ):
            raise ValueError(
                "registered holdout entry requires a lowercase SHA-256 digest"
            )
        sample_ids.add(file_name)
        content_hashes.add(content_hash)
        provenance_group = entry.get("provenance_group")
        if registration.require_provenance and (
            not isinstance(provenance_group, str)
            or not provenance_group
            or provenance_group != provenance_group.strip()
        ):
            raise ValueError("registered holdout entry requires provenance_group")
        if isinstance(provenance_group, str) and provenance_group:
            provenance_groups.add(provenance_group)
    return HoldoutIndex(
        registration=registration,
        sample_ids=frozenset(sample_ids),
        content_sha256=frozenset(content_hashes),
        provenance_groups=frozenset(provenance_groups),
    )


def reject_holdout_overlap(
    samples: Iterable[SampleRecord], holdouts: Iterable[HoldoutIndex]
) -> None:
    rows = list(samples)
    all_holdouts = list(holdouts)
    if not all_holdouts:
        raise ValueError("at least one verified holdout denylist is required")
    training_ids = {row.sample_id for row in rows} | {row.upstream_path for row in rows}
    training_hashes = {row.content_sha256 for row in rows} | {
        row.upstream_sha256 for row in rows if row.upstream_sha256 is not None
    }
    training_groups = {row.provenance_group for row in rows}
    for holdout in all_holdouts:
        if overlap := training_ids & holdout.sample_ids:
            raise ValueError(
                f"sample ID overlap with {holdout.registration.holdout_id}: {sorted(overlap)[0]}"
            )
        if overlap := training_hashes & holdout.content_sha256:
            raise ValueError(
                f"content overlap with {holdout.registration.holdout_id}: {sorted(overlap)[0]}"
            )
        if overlap := training_groups & holdout.provenance_groups:
            raise ValueError(
                f"provenance overlap with {holdout.registration.holdout_id}: {sorted(overlap)[0]}"
            )
