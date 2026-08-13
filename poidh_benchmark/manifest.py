from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Iterable, Sequence


_REAL_LABEL = "0_real"
_AI_LABEL = "1_fake"
_LABELS = (_REAL_LABEL, _AI_LABEL)


@dataclass(frozen=True, slots=True)
class MirageRow:
    file_name: str
    label: str
    content_type: str

    def __post_init__(self) -> None:
        if not self.file_name:
            raise ValueError("file_name must not be empty")
        if self.label not in _LABELS:
            raise ValueError("label must be 0_real or 1_fake")
        if not self.content_type:
            raise ValueError("content_type must not be empty")

    @property
    def generator_family(self) -> str | None:
        if self.label == _REAL_LABEL:
            return None
        parts = self.file_name.split("/")
        if len(parts) < 4 or parts[1] != _AI_LABEL or not parts[2]:
            raise ValueError(f"cannot derive generator from {self.file_name!r}")
        return parts[2]


def select_balanced_strata(
    rows: Iterable[MirageRow],
    *,
    content_types: Sequence[str],
    per_class_per_content: int,
    seed: str,
) -> list[MirageRow]:
    if (
        isinstance(per_class_per_content, bool)
        or not isinstance(per_class_per_content, int)
        or per_class_per_content <= 0
    ):
        raise ValueError("per_class_per_content must be a positive integer")
    if not seed:
        raise ValueError("seed must not be empty")
    selected_content_types = tuple(content_types)
    if (
        not selected_content_types
        or any(not content_type for content_type in selected_content_types)
        or len(selected_content_types) != len(set(selected_content_types))
    ):
        raise ValueError("content_types must contain unique non-empty values")

    grouped: dict[tuple[str, str], list[MirageRow]] = {}
    seen_file_names: set[str] = set()
    for row in rows:
        if row.file_name in seen_file_names:
            raise ValueError(f"duplicate file_name: {row.file_name}")
        seen_file_names.add(row.file_name)
        grouped.setdefault((row.content_type, row.label), []).append(row)

    selected: list[MirageRow] = []
    for content_type in selected_content_types:
        for label in _LABELS:
            candidates = grouped.get((content_type, label), [])
            if len(candidates) < per_class_per_content:
                raise ValueError(
                    "incomplete stratum "
                    f"{content_type}/{label}: need {per_class_per_content}, "
                    f"found {len(candidates)}"
                )
            ranked = sorted(
                candidates,
                key=lambda row: (
                    hashlib.sha256(f"{seed}\0{row.file_name}".encode("utf-8")).digest(),
                    row.file_name,
                ),
            )
            selected.extend(ranked[:per_class_per_content])
    return selected


def select_balanced_generator_strata(
    rows: Iterable[MirageRow],
    *,
    content_types: Sequence[str],
    fake_generators: Sequence[str],
    per_class_per_content: int,
    seed: str,
) -> list[MirageRow]:
    _validate_selection_arguments(
        content_types=content_types,
        per_class_per_content=per_class_per_content,
        seed=seed,
    )
    generators = tuple(fake_generators)
    if (
        not generators
        or any(not generator for generator in generators)
        or len(generators) != len(set(generators))
    ):
        raise ValueError("fake_generators must contain unique non-empty values")

    all_rows = list(rows)
    _reject_duplicate_file_names(all_rows)
    grouped: dict[tuple[str, str], list[MirageRow]] = {}
    for row in all_rows:
        key = (row.content_type, row.label)
        grouped.setdefault(key, []).append(row)

    selected: list[MirageRow] = []
    base_quota, remainder = divmod(per_class_per_content, len(generators))
    for content_index, content_type in enumerate(content_types):
        real_rows = grouped.get((content_type, _REAL_LABEL), [])
        if len(real_rows) < per_class_per_content:
            raise ValueError(
                "incomplete stratum "
                f"{content_type}/{_REAL_LABEL}: need {per_class_per_content}, "
                f"found {len(real_rows)}"
            )
        selected.extend(_rank_rows(real_rows, seed=seed)[:per_class_per_content])

        fake_rows = grouped.get((content_type, _AI_LABEL), [])
        by_generator: dict[str, list[MirageRow]] = {}
        for row in fake_rows:
            generator = row.generator_family
            if generator is not None:
                by_generator.setdefault(generator, []).append(row)
        extra_start = (content_index * remainder) % len(generators)
        extra_indices = {
            (extra_start + offset) % len(generators) for offset in range(remainder)
        }
        for generator_index, generator in enumerate(generators):
            quota = base_quota + (generator_index in extra_indices)
            candidates = by_generator.get(generator, [])
            if len(candidates) < quota:
                raise ValueError(
                    "incomplete generator stratum "
                    f"{content_type}/{generator}: need {quota}, "
                    f"found {len(candidates)}"
                )
            selected.extend(_rank_rows(candidates, seed=seed)[:quota])
    return selected


def _validate_selection_arguments(
    *,
    content_types: Sequence[str],
    per_class_per_content: int,
    seed: str,
) -> None:
    if (
        isinstance(per_class_per_content, bool)
        or not isinstance(per_class_per_content, int)
        or per_class_per_content <= 0
    ):
        raise ValueError("per_class_per_content must be a positive integer")
    if not seed:
        raise ValueError("seed must not be empty")
    selected_content_types = tuple(content_types)
    if (
        not selected_content_types
        or any(not content_type for content_type in selected_content_types)
        or len(selected_content_types) != len(set(selected_content_types))
    ):
        raise ValueError("content_types must contain unique non-empty values")


def _reject_duplicate_file_names(rows: Iterable[MirageRow]) -> None:
    seen: set[str] = set()
    for row in rows:
        if row.file_name in seen:
            raise ValueError(f"duplicate file_name: {row.file_name}")
        seen.add(row.file_name)


def _rank_rows(rows: Iterable[MirageRow], *, seed: str) -> list[MirageRow]:
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{seed}\0{row.file_name}".encode("utf-8")).digest(),
            row.file_name,
        ),
    )
