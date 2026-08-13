from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

from poidh_detector.contracts import SampleRecord


_SOURCE_CONTRACTS: dict[str, tuple[int, str, str | None]] = {
    "commoncatalog-cc-by": (0, "cc-by-4.0", None),
    "synthetic-flux-schnell": (1, "apache-2.0", "flux-1-schnell"),
    "synthetic-flux-klein": (1, "apache-2.0", "flux-2-klein-4b"),
    "synthetic-z-image": (1, "apache-2.0", "z-image"),
}


@dataclass(frozen=True, slots=True)
class MonetRow:
    key: str
    upstream_path: str
    source: str
    license: str
    upstream_sha256: str
    perceptual_hash: str
    sscd_cluster_id: str
    thumbnail: bytes

    def __post_init__(self) -> None:
        if self.source not in _SOURCE_CONTRACTS:
            raise ValueError(f"not an approved MONET source: {self.source}")
        expected_license = _SOURCE_CONTRACTS[self.source][1]
        if self.license.lower() != expected_license:
            raise ValueError(
                f"MONET license mismatch for {self.source}: "
                f"expected {expected_license}, found {self.license}"
            )
        for field_name in (
            "key",
            "upstream_path",
            "perceptual_hash",
            "sscd_cluster_id",
        ):
            value = getattr(self, field_name)
            if not value or value != value.strip():
                raise ValueError(f"{field_name} must be non-empty and trimmed")
        if len(self.upstream_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.upstream_sha256
        ):
            raise ValueError("upstream_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.thumbnail, bytes) or not self.thumbnail:
            raise ValueError("thumbnail must contain encoded image bytes")


def select_source_quotas(
    rows: Iterable[MonetRow], *, quotas: Mapping[str, int], seed: str
) -> list[MonetRow]:
    if not seed:
        raise ValueError("seed must not be empty")
    if not quotas:
        raise ValueError("quotas must not be empty")
    for source, quota in quotas.items():
        if source not in _SOURCE_CONTRACTS:
            raise ValueError(f"not an approved MONET source: {source}")
        if isinstance(quota, bool) or not isinstance(quota, int) or quota <= 0:
            raise ValueError("every source quota must be a positive integer")

    candidates = list(rows)
    _reject_duplicate_values((row.key for row in candidates), "MONET key")
    _reject_duplicate_values(
        (hashlib.sha256(row.thumbnail).hexdigest() for row in candidates),
        "thumbnail content",
    )
    unexpected = {row.source for row in candidates} - set(quotas)
    if unexpected:
        raise ValueError(f"rows contain source without quota: {sorted(unexpected)[0]}")

    selected: list[MonetRow] = []
    for source in sorted(quotas):
        source_rows = [row for row in candidates if row.source == source]
        quota = quotas[source]
        if len(source_rows) < quota:
            raise ValueError(
                f"insufficient rows for {source}: need {quota}, found {len(source_rows)}"
            )
        ranked = sorted(
            source_rows,
            key=lambda row: (
                hashlib.sha256(f"{seed}\0{source}\0{row.key}".encode()).digest(),
                row.key,
            ),
        )
        selected.extend(ranked[:quota])
    return selected


def materialize_selected_rows(
    rows: Iterable[MonetRow], *, output_root: Path
) -> list[SampleRecord]:
    sorted_rows = sorted(rows, key=lambda value: value.key)
    samples = sample_records_for_rows(sorted_rows)
    for row, sample in zip(sorted_rows, samples, strict=True):
        destination = output_root / sample.local_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".webp.part")
        temporary.write_bytes(row.thumbnail)
        temporary.replace(destination)
    return samples


def sample_records_for_rows(rows: Iterable[MonetRow]) -> list[SampleRecord]:
    samples: list[SampleRecord] = []
    for row in sorted(rows, key=lambda value: value.key):
        label, _, generator = _SOURCE_CONTRACTS[row.source]
        content_sha256 = hashlib.sha256(row.thumbnail).hexdigest()
        relative_path = Path(
            "images",
            "ai" if label == 1 else "real",
            row.source,
            f"{content_sha256}.webp",
        )
        samples.append(
            SampleRecord(
                sample_id=f"monet:{row.key}",
                source_id=f"monet-{row.source}",
                upstream_path=f"{row.upstream_path}/{row.key}.webp",
                local_path=relative_path.as_posix(),
                label=label,  # type: ignore[arg-type]
                content_sha256=content_sha256,
                provenance_group=f"monet:sscd:{row.sscd_cluster_id}",
                generator_family=generator,
                content_type="mixed",
                upstream_sha256=row.upstream_sha256,
            )
        )
    return samples


def _reject_duplicate_values(values: Iterable[str], description: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {description}: {value}")
        seen.add(value)
