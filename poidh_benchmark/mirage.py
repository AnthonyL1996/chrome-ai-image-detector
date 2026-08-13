from __future__ import annotations

from collections.abc import Callable
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
from urllib.parse import quote

from poidh_benchmark.manifest import MirageRow


FetchBytes = Callable[[str], bytes]
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._ -]+")
_KNOWN_SUFFIXES = {".avif", ".jpeg", ".jpg", ".png", ".webp"}


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


def _safe_path_segment(value: str) -> str:
    sanitized = _SAFE_SEGMENT.sub("_", value).strip(" .")
    if not sanitized or sanitized in {".", ".."}:
        raise ValueError(f"unsafe path segment: {value!r}")
    return sanitized
