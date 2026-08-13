from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Literal

from poidh_detector.contracts import (
    DatasetManifest,
    DatasetSource,
    LicenseAudit,
    SampleRecord,
)


SplitName = Literal["train", "validation", "calibration"]
ProfileName = Literal["overfit", "smoke", "pilot", "full"]

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = frozenset({"train", "validation", "calibration"})
_PROFILES = frozenset({"overfit", "smoke", "pilot", "full"})
_MAX_SEED = 2**32 - 1


@dataclass(frozen=True, slots=True)
class SplitManifest:
    dataset_manifest_sha256: str
    seed: int | str
    ratios: Mapping[str, float]
    assignments: Mapping[str, str]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported split manifest schema version")
        _require_sha256(self.dataset_manifest_sha256, "dataset_manifest_sha256")
        if isinstance(self.seed, str):
            if not self.seed or self.seed != self.seed.strip():
                raise ValueError("split seed must be non-empty and trimmed")
        else:
            _require_seed(self.seed)
        ratios = dict(self.ratios)
        if set(ratios) != _SPLITS:
            raise ValueError(
                "split ratios must define train, validation, and calibration"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
            for value in ratios.values()
        ):
            raise ValueError("split ratios must be finite positive numbers")
        if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
            raise ValueError("split ratios must sum to one")
        assignments = dict(self.assignments)
        if not assignments:
            raise ValueError("split manifest requires sample assignments")
        if any(
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id != sample_id.strip()
            for sample_id in assignments
        ):
            raise ValueError("split assignment IDs must be non-empty and trimmed")
        if any(split not in _SPLITS for split in assignments.values()):
            raise ValueError("unknown split assignment")
        object.__setattr__(self, "ratios", MappingProxyType(ratios))
        object.__setattr__(self, "assignments", MappingProxyType(assignments))

    def to_json_bytes(self) -> bytes:
        document = {
            "assignments": dict(sorted(self.assignments.items())),
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "ratios": dict(sorted(self.ratios.items())),
            "schema_version": self.schema_version,
            "seed": self.seed,
        }
        return _canonical_json(document)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()

    def validate_against(self, manifest: DatasetManifest) -> None:
        if self.dataset_manifest_sha256 != manifest.sha256:
            raise ValueError("split manifest dataset manifest digest does not match")
        expected = {sample.sample_id for sample in manifest.samples}
        actual = set(self.assignments)
        if actual != expected:
            missing = sorted(expected - actual)[:3]
            extra = sorted(actual - expected)[:3]
            raise ValueError(
                f"split assignment set mismatch: missing={missing}, extra={extra}"
            )
        group_splits: dict[str, str] = {}
        for sample in manifest.samples:
            split = self.assignments[sample.sample_id]
            previous = group_splits.setdefault(sample.provenance_group, split)
            if previous != split:
                raise ValueError(
                    "provenance group crosses splits: " + sample.provenance_group
                )


def load_dataset_manifest(path: Path) -> DatasetManifest:
    raw = path.read_bytes()
    document = _json_object(raw, "dataset manifest")
    _require_keys(
        document, {"schema_version", "sources", "samples"}, "dataset manifest"
    )
    sources = []
    for source_document in _object_list(document["sources"], "sources"):
        _require_keys(
            source_document,
            {
                "source_id",
                "dataset_id",
                "revision",
                "upstream_uri",
                "metadata_sha256",
                "license_audit",
            },
            "dataset source",
        )
        audit_document = source_document["license_audit"]
        if not isinstance(audit_document, dict):
            raise ValueError("license_audit must be an object")
        audit = LicenseAudit(**audit_document)
        sources.append(
            DatasetSource(
                **{
                    key: value
                    for key, value in source_document.items()
                    if key != "license_audit"
                },
                license_audit=audit,
            )
        )
    samples = [
        SampleRecord(**row) for row in _object_list(document["samples"], "samples")
    ]
    manifest = DatasetManifest(
        schema_version=document["schema_version"],
        sources=tuple(sources),
        samples=tuple(samples),
    )
    if raw != manifest.to_json_bytes():
        raise ValueError("dataset manifest must use canonical JSON encoding")
    return manifest


def load_split_manifest(
    path: Path,
    manifest: DatasetManifest,
    *,
    expected_sha256: str | None = None,
) -> SplitManifest:
    raw = path.read_bytes()
    document = _json_object(raw, "split manifest")
    _require_keys(
        document,
        {"schema_version", "dataset_manifest_sha256", "seed", "ratios", "assignments"},
        "split manifest",
    )
    split = SplitManifest(**document)
    if raw != split.to_json_bytes():
        raise ValueError("split manifest must use canonical JSON encoding")
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "expected_sha256")
        if split.sha256 != expected_sha256:
            raise ValueError("split manifest digest does not match expected digest")
    split.validate_against(manifest)
    return split


def samples_for_split(
    manifest: DatasetManifest, split_manifest: SplitManifest, split: SplitName
) -> tuple[SampleRecord, ...]:
    if split not in _SPLITS:
        raise ValueError(f"unknown split: {split}")
    split_manifest.validate_against(manifest)
    return tuple(
        sorted(
            (
                sample
                for sample in manifest.samples
                if split_manifest.assignments[sample.sample_id] == split
            ),
            key=lambda sample: sample.sample_id,
        )
    )


def select_profile_subset(
    samples: Sequence[SampleRecord],
    *,
    profile: ProfileName,
    split: SplitName,
    seed: int,
    per_class_cap: int | None,
) -> tuple[SampleRecord, ...]:
    """Select deterministic whole provenance groups without exceeding a class cap."""

    if profile not in _PROFILES:
        raise ValueError(f"unknown training profile: {profile}")
    if split not in _SPLITS:
        raise ValueError(f"unknown split: {split}")
    _require_seed(seed)
    if per_class_cap is not None and (
        isinstance(per_class_cap, bool)
        or not isinstance(per_class_cap, int)
        or per_class_cap <= 0
    ):
        raise ValueError("per_class_cap must be a positive integer or None")

    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for sample in samples:
        grouped[sample.provenance_group].append(sample)
    group_rows: list[tuple[int, str, tuple[SampleRecord, ...]]] = []
    for group, rows in grouped.items():
        labels = {row.label for row in rows}
        if len(labels) != 1:
            raise ValueError(f"provenance group mixes labels: {group}")
        ordered_rows = tuple(sorted(rows, key=lambda row: row.sample_id))
        group_rows.append((ordered_rows[0].label, group, ordered_rows))

    selected: list[SampleRecord] = []
    for label in (0, 1):
        used = 0
        ranked = sorted(
            (row for row in group_rows if row[0] == label),
            key=lambda row: _stable_rank(seed, profile, split, label, row[1]),
        )
        for _, _, rows in ranked:
            if per_class_cap is not None and used + len(rows) > per_class_cap:
                continue
            selected.extend(rows)
            used += len(rows)
    return tuple(sorted(selected, key=lambda row: row.sample_id))


def balanced_epoch_indices(
    samples: Sequence[SampleRecord], *, seed: int, epoch: int
) -> tuple[int, ...]:
    """Return deterministic, interleaved indices with minority-class oversampling."""

    _require_seed(seed)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    by_label = {
        label: [index for index, sample in enumerate(samples) if sample.label == label]
        for label in (0, 1)
    }
    if not by_label[0] or not by_label[1]:
        raise ValueError("balanced sampling requires both real and AI samples")
    target = max(len(by_label[0]), len(by_label[1]))
    expanded: dict[int, list[int]] = {}
    for label, indices in by_label.items():
        ranked = sorted(
            indices,
            key=lambda index: _stable_rank(
                seed, "balanced", epoch, label, samples[index].sample_id
            ),
        )
        expanded[label] = [ranked[offset % len(ranked)] for offset in range(target)]
    first_label = epoch % 2
    order: list[int] = []
    for offset in range(target):
        order.extend((expanded[first_label][offset], expanded[1 - first_label][offset]))
    return tuple(order)


def preprocess_rgb_image(
    image: Any,
    *,
    functional_module: Any | None = None,
    bicubic_value: Any | None = None,
) -> Any:
    """Apply the fixed 224px sRGB/ImageNet preprocessing contract."""

    functional = functional_module
    bicubic = bicubic_value
    if functional is None:
        try:
            functional = importlib.import_module("torchvision.transforms.functional")
            transforms = importlib.import_module("torchvision.transforms")
        except ModuleNotFoundError as error:
            if error.name and error.name.split(".")[0] in {"torch", "torchvision"}:
                raise RuntimeError(
                    "image preprocessing requires torch and torchvision; install training extras"
                ) from error
            raise
        bicubic = transforms.InterpolationMode.BICUBIC
    if bicubic is None:
        raise ValueError("bicubic_value is required with an injected functional module")
    rgb = image.convert("RGB")
    resized = functional.resize(
        rgb, [IMAGE_SIZE, IMAGE_SIZE], interpolation=bicubic, antialias=True
    )
    tensor = functional.to_tensor(resized)
    return functional.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class DatasetImageSamples:
    def __init__(
        self,
        samples: Sequence[SampleRecord],
        root: Path,
        *,
        transform: Callable[[Any], Any] = preprocess_rgb_image,
        image_module: Any | None = None,
    ) -> None:
        self._samples = tuple(samples)
        self._root = root
        self._transform = transform
        self._image_module = image_module

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Any, float, str]:
        sample = self._samples[index]
        images = self._image_module
        if images is None:
            try:
                images = importlib.import_module("PIL.Image")
            except ModuleNotFoundError as error:
                if error.name == "PIL":
                    raise RuntimeError(
                        "image loading requires Pillow; install training extras"
                    ) from error
                raise
        with images.open(self._root / sample.local_path) as opened:
            rgb = opened.convert("RGB")
            if hasattr(rgb, "copy"):
                rgb = rgb.copy()
        return self._transform(rgb), float(sample.label), sample.sample_id


def _json_object(raw: bytes, description: str) -> dict[str, Any]:
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} must be valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{description} must be a JSON object")
    return document


def _object_list(value: object, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{field_name} must be a list of objects")
    return value


def _require_keys(
    document: Mapping[str, object], keys: set[str], description: str
) -> None:
    if set(document) != keys:
        raise ValueError(f"{description} fields do not match the schema")


def _canonical_json(document: Mapping[str, object]) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _stable_rank(*values: object) -> str:
    encoded = "\0".join(str(value) for value in values).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_seed(seed: int) -> None:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= _MAX_SEED
    ):
        raise ValueError(f"seed must be an integer from 0 to {_MAX_SEED}")
