from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Literal

from poidh_detector.contracts import SampleRecord


SplitName = Literal["train", "validation", "calibration"]


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float
    validation: float
    calibration: float

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.calibration)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("split ratios must be finite and positive")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError("split ratios must sum to one")


def assign_group_splits(
    samples: Iterable[SampleRecord], *, ratios: SplitRatios, seed: str
) -> dict[str, SplitName]:
    if not seed:
        raise ValueError("seed must not be empty")
    rows = list(samples)
    if not rows:
        raise ValueError("samples must not be empty")
    by_group: dict[str, list[SampleRecord]] = {}
    seen_ids: set[str] = set()
    for row in rows:
        if row.sample_id in seen_ids:
            raise ValueError(f"duplicate sample ID: {row.sample_id}")
        seen_ids.add(row.sample_id)
        by_group.setdefault(row.provenance_group, []).append(row)
    for group, grouped_rows in by_group.items():
        if len({row.label for row in grouped_rows}) != 1:
            raise ValueError(f"provenance group has conflicting labels: {group}")

    strata: dict[tuple[int, tuple[str, ...]], list[str]] = {}
    for group, grouped_rows in by_group.items():
        label = grouped_rows[0].label
        source_ids = tuple(sorted({row.source_id for row in grouped_rows}))
        strata.setdefault((label, source_ids), []).append(group)

    assignments: dict[str, SplitName] = {}
    split_names: tuple[SplitName, ...] = ("train", "validation", "calibration")
    for stratum, groups in sorted(strata.items()):
        if len(groups) < len(split_names):
            raise ValueError(
                f"source/class stratum needs at least three provenance groups: {stratum}"
            )
        ranked = sorted(
            groups,
            key=lambda group: (
                hashlib.sha256(f"{seed}\0{stratum}\0{group}".encode()).digest(),
                group,
            ),
        )
        counts = _split_group_counts(len(ranked), ratios)
        offset = 0
        for split, count in zip(split_names, counts, strict=True):
            for group in ranked[offset : offset + count]:
                for row in by_group[group]:
                    assignments[row.sample_id] = split
            offset += count
    return dict(sorted(assignments.items()))


def _split_group_counts(group_count: int, ratios: SplitRatios) -> tuple[int, int, int]:
    remaining = group_count - 3
    ratio_values = (ratios.train, ratios.validation, ratios.calibration)
    raw = [remaining * ratio for ratio in ratio_values]
    additional = [math.floor(value) for value in raw]
    unassigned = remaining - sum(additional)
    remainder_order = sorted(
        range(3), key=lambda index: (-(raw[index] - additional[index]), index)
    )
    for index in remainder_order[:unassigned]:
        additional[index] += 1
    return tuple(value + 1 for value in additional)  # type: ignore[return-value]
