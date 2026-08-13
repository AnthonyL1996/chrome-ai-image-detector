from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import quote

from poidh_benchmark.manifest import MirageRow


FetchBytes = Callable[[str], bytes]
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._ -]+")
_KNOWN_SUFFIXES = {".avif", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True, slots=True)
class PriorManifest:
    dataset_id: str
    dataset_revision: str
    entries: tuple[tuple[str, str], ...]
    file_names: frozenset[str]
    content_sha256: frozenset[str]
    manifest_sha256: str


def pinned_download_url(dataset_id: str, revision: str, file_name: str) -> str:
    if not dataset_id:
        raise ValueError("dataset_id must not be empty")
    if not revision:
        raise ValueError("revision must not be empty")
    if not file_name:
        raise ValueError("file_name must not be empty")
    return (
        "https://huggingface.co/datasets/"
        f"{quote(dataset_id, safe='/')}/resolve/{quote(revision, safe='')}/"
        f"{quote(file_name, safe='/')}"
    )


def materialize_entry(
    row: MirageRow,
    *,
    output_root: Path,
    dataset_id: str,
    revision: str,
    selection_seed: str,
    fetch: FetchBytes,
) -> dict[str, str | None]:
    if not selection_seed:
        raise ValueError("selection_seed must not be empty")
    download_url = pinned_download_url(dataset_id, revision, row.file_name)
    payload = fetch(download_url)
    if not payload:
        raise ValueError(f"downloaded empty image: {row.file_name}")

    selection_sha256 = hashlib.sha256(
        f"{selection_seed}\0{row.file_name}".encode("utf-8")
    ).hexdigest()
    content_sha256 = hashlib.sha256(payload).hexdigest()
    suffix = PurePosixPath(row.file_name).suffix.lower()
    if suffix not in _KNOWN_SUFFIXES:
        suffix = ".img"
    label = "ai" if row.label == "1_fake" else "real"
    relative_parts = [label, _safe_path_segment(row.content_type)]
    if row.generator_family is not None:
        relative_parts.append(_safe_path_segment(row.generator_family))
    relative_path = Path(*relative_parts, f"{selection_sha256}{suffix}")
    destination = output_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)

    return {
        "file_name": row.file_name,
        "label": label,
        "content_type": row.content_type,
        "generator_family": row.generator_family,
        "selection_sha256": selection_sha256,
        "content_sha256": content_sha256,
        "download_url": download_url,
        "local_path": relative_path.as_posix(),
    }


def validate_materialized_entries(
    images_root: Path, entries: Iterable[Mapping[str, object]]
) -> None:
    expected = {
        str(entry["local_path"])
        for entry in entries
        if isinstance(entry.get("local_path"), str)
    }
    actual = {
        path.relative_to(images_root).as_posix()
        for path in images_root.rglob("*")
        if path.is_file()
    }
    if expected != actual:
        missing = sorted(expected - actual)[:3]
        extra = sorted(actual - expected)[:3]
        raise RuntimeError(
            "materialized file set does not match manifest: "
            f"missing={missing}, extra={extra}"
        )


def commit_staging_directory(staging: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"output destination already exists: {destination}")
    staging.replace(destination)


def git_provenance(repository: Path, script_path: Path) -> dict[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("refusing to prepare holdout from a dirty worktree")
    return {
        "git_commit": commit,
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
    }


def read_prior_manifest(
    path: Path,
    *,
    expected_dataset_id: str,
    expected_revision: str,
) -> PriorManifest:
    payload = path.read_bytes()
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise ValueError("manifest root must be an object")
    if document.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema version")
    dataset_id = document.get("dataset_id")
    revision = document.get("dataset_revision")
    if dataset_id != expected_dataset_id:
        raise ValueError("prior manifest dataset ID does not match")
    if revision != expected_revision:
        raise ValueError("prior manifest dataset revision does not match")
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ValueError("manifest entries must be a list")
    names: list[str] = []
    hashes: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("every manifest entry must be an object")
        file_name = entry.get("file_name")
        content_hash = entry.get("content_sha256")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError("every manifest entry must contain a file_name")
        if (
            not isinstance(content_hash, str)
            or len(content_hash) != 64
            or any(character not in "0123456789abcdef" for character in content_hash)
        ):
            raise ValueError("every manifest entry must contain a content SHA-256")
        names.append(file_name)
        hashes.append(content_hash)
    if len(names) != len(set(names)):
        raise ValueError("duplicate file_name in prior manifest")
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate content SHA-256 in prior manifest")
    return PriorManifest(
        dataset_id=dataset_id,
        dataset_revision=revision,
        entries=tuple(sorted(zip(names, hashes, strict=True))),
        file_names=frozenset(names),
        content_sha256=frozenset(hashes),
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )


def reject_prior_content_overlap(
    content_hashes: Iterable[str], prior_manifests: Iterable[PriorManifest]
) -> None:
    prior_hashes = {
        digest for manifest in prior_manifests for digest in manifest.content_sha256
    }
    leaked_content = set(content_hashes) & prior_hashes
    if leaked_content:
        preview = ", ".join(sorted(leaked_content)[:3])
        raise RuntimeError(f"content overlap with prior holdout: {preview}")


def _safe_path_segment(value: str) -> str:
    sanitized = _SAFE_SEGMENT.sub("_", value).strip(" .")
    if not sanitized or sanitized in {".", ".."}:
        raise ValueError(f"unsafe path segment: {value!r}")
    return sanitized
