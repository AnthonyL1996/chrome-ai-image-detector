from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
import re
from typing import Literal


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
_ALLOWED_LICENSES = frozenset(
    {
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC-BY-4.0",
        "CC0-1.0",
        "MIT",
    }
)


def _require_text(value: str, field_name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty and trimmed")


def _require_sha256(value: str, field_name: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def require_safe_relative_path(value: str, field_name: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or value != path.as_posix()
    ):
        raise ValueError(f"{field_name} must be a safe relative POSIX path")


@dataclass(frozen=True, slots=True)
class LicenseAudit:
    spdx_expression: str
    declared_scope: Literal["repository", "dataset_card"]
    evidence_uri: str
    evidence_sha256: str
    dataset_card_sha256: str
    audited_at: str
    auditor: str
    decision: Literal["allow", "deny"]
    rationale: str
    image_level_verified: bool
    accepted_policy_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "spdx_expression",
            "evidence_uri",
            "audited_at",
            "auditor",
            "rationale",
            "accepted_policy_id",
        ):
            _require_text(getattr(self, field_name), field_name)
        _require_sha256(self.evidence_sha256, "evidence_sha256")
        _require_sha256(self.dataset_card_sha256, "dataset_card_sha256")

    @property
    def is_allowed(self) -> bool:
        return self.decision == "allow" and self.spdx_expression in _ALLOWED_LICENSES


@dataclass(frozen=True, slots=True)
class DatasetSource:
    source_id: str
    dataset_id: str
    revision: str
    upstream_uri: str
    metadata_sha256: str
    license_audit: LicenseAudit

    def __post_init__(self) -> None:
        for field_name in ("source_id", "dataset_id", "upstream_uri"):
            _require_text(getattr(self, field_name), field_name)
        if not _IMMUTABLE_REVISION.fullmatch(self.revision):
            raise ValueError("dataset source requires an immutable revision digest")
        _require_sha256(self.metadata_sha256, "metadata_sha256")
        if not self.license_audit.is_allowed:
            raise ValueError("dataset source requires an allowed license audit")


@dataclass(frozen=True, slots=True)
class SampleRecord:
    sample_id: str
    source_id: str
    upstream_path: str
    local_path: str
    label: Literal[0, 1]
    content_sha256: str
    provenance_group: str
    generator_family: str | None
    content_type: str
    upstream_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "sample_id",
            "source_id",
            "provenance_group",
            "content_type",
        ):
            _require_text(getattr(self, field_name), field_name)
        require_safe_relative_path(self.upstream_path, "upstream_path")
        require_safe_relative_path(self.local_path, "local_path")
        if isinstance(self.label, bool) or self.label not in (0, 1):
            raise ValueError("label must be 0 (real) or 1 (AI)")
        _require_sha256(self.content_sha256, "content_sha256")
        if self.label == 1 and not self.generator_family:
            raise ValueError("AI samples require generator_family")
        if self.generator_family is not None:
            _require_text(self.generator_family, "generator_family")
        if self.upstream_sha256 is not None:
            _require_sha256(self.upstream_sha256, "upstream_sha256")


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: int
    sources: tuple[DatasetSource, ...]
    samples: tuple[SampleRecord, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("unsupported dataset manifest schema version")
        if not self.sources:
            raise ValueError("dataset manifest requires at least one source")
        if not self.samples:
            raise ValueError("dataset manifest requires at least one sample")
        _reject_duplicates((source.source_id for source in self.sources), "source ID")
        source_ids = {source.source_id for source in self.sources}
        for sample in self.samples:
            if sample.source_id not in source_ids:
                raise ValueError(f"unknown source ID: {sample.source_id}")
        _reject_duplicates((sample.sample_id for sample in self.samples), "sample ID")
        _reject_duplicates((sample.local_path for sample in self.samples), "local path")
        _reject_duplicates(
            (sample.content_sha256 for sample in self.samples), "content hash"
        )

    def to_json_bytes(self) -> bytes:
        document = {
            "schema_version": self.schema_version,
            "sources": [
                asdict(source)
                for source in sorted(self.sources, key=lambda row: row.source_id)
            ],
            "samples": [
                asdict(sample)
                for sample in sorted(self.samples, key=lambda row: row.sample_id)
            ],
        }
        return (
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()

    def verify_materialized_files(self, root: Path) -> None:
        if not root.is_dir() or root.is_symlink():
            raise ValueError("materialized root must be a real directory")
        resolved_root = root.resolve()
        expected_paths = {sample.local_path for sample in self.samples}
        images_root = root / "images"
        materialized_entries = list(images_root.rglob("*"))
        if any(path.is_symlink() for path in materialized_entries):
            raise ValueError("materialized paths must not use symlinks")
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in materialized_entries
            if path.is_file()
        }
        if actual_paths != expected_paths:
            missing = sorted(expected_paths - actual_paths)[:3]
            extra = sorted(actual_paths - expected_paths)[:3]
            raise ValueError(
                f"materialized file set mismatch: missing={missing}, extra={extra}"
            )
        for sample in self.samples:
            path = root / PurePosixPath(sample.local_path)
            relative_parents = PurePosixPath(sample.local_path).parents
            materialized_parents = [
                root / parent
                for parent in relative_parents
                if parent != PurePosixPath(".")
            ]
            if path.is_symlink() or any(
                parent.is_symlink() for parent in materialized_parents
            ):
                raise ValueError(
                    f"materialized path must not use symlinks: {sample.local_path}"
                )
            if not path.is_file() or not path.resolve().is_relative_to(resolved_root):
                raise ValueError(
                    f"materialized file is missing or unsafe: {sample.local_path}"
                )
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != sample.content_sha256:
                raise ValueError(f"content hash mismatch: {sample.local_path}")


def _reject_duplicates(values: Iterable[str], description: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {description}: {value}")
        seen.add(value)
