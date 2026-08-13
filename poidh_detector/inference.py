from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal

from poidh_benchmark.leakage import (
    HoldoutIndex,
    HoldoutRegistration,
    load_registered_holdout,
    reject_holdout_overlap,
)
from poidh_detector.calibration_fit import (
    CalibrationPrediction,
    CalibrationPredictions,
)
from poidh_detector.data import (
    DatasetImageSamples,
    SplitManifest,
    load_dataset_manifest,
    load_split_manifest,
    samples_for_split,
)
from poidh_detector.model import ConvNeXtV2NanoConfig, create_convnextv2_nano
from poidh_detector.reproducibility import (
    EnvironmentFingerprint,
    capture_environment,
    configure_determinism,
)
from poidh_detector.torch_training import (
    LoadedGeneration,
    OptimizationConfig,
    load_current_generation,
)
from poidh_detector.training import TrainingConfig


PartitionName = Literal["calibration", "validation"]
ModelFactory = Callable[[], Any]

FIXED_THRESHOLD = 0.65
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_PARTITIONS = frozenset({"calibration", "validation"})


@dataclass(frozen=True, slots=True)
class InferenceMetrics:
    auc: float
    balanced_accuracy: float
    threshold: float = FIXED_THRESHOLD

    def __post_init__(self) -> None:
        for field_name in ("auc", "balanced_accuracy"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{field_name} must be a finite number in [0, 1]")
        if self.threshold != FIXED_THRESHOLD:
            raise ValueError(f"threshold must remain fixed at {FIXED_THRESHOLD}")


@dataclass(frozen=True, slots=True)
class InferenceResult:
    partition: PartitionName
    checkpoint_sha256: str
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    partition_sha256: str
    training_config_sha256: str
    environment_sha256: str
    exposed_holdout_manifest_sha256: tuple[str, ...]
    predictions: tuple[CalibrationPrediction, ...]
    metrics: InferenceMetrics

    def predictions_json_bytes(self) -> bytes:
        if self.partition == "calibration":
            return CalibrationPredictions(
                schema_version=1,
                input_identifier="calibration",
                checkpoint_sha256=self.checkpoint_sha256,
                calibration_split_sha256=self.partition_sha256,
                training_config_sha256=self.training_config_sha256,
                predictions=self.predictions,
            ).to_json_bytes()
        return _canonical_json(self._report_document(include_predictions=True))

    def summary_json_bytes(self) -> bytes:
        return _canonical_json(self._report_document(include_predictions=False))

    def _report_document(self, *, include_predictions: bool) -> dict[str, Any]:
        document: dict[str, Any] = {
            "checkpoint_sha256": self.checkpoint_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "environment_sha256": self.environment_sha256,
            "exposed_holdout_manifest_sha256": list(
                self.exposed_holdout_manifest_sha256
            ),
            "metrics": asdict(self.metrics),
            "partition": self.partition,
            "partition_sha256": self.partition_sha256,
            "prediction_count": len(self.predictions),
            "schema_version": 1,
            "split_manifest_sha256": self.split_manifest_sha256,
            "training_config_sha256": self.training_config_sha256,
        }
        if include_predictions:
            document["predictions"] = [
                {
                    "label": row.label,
                    "raw_logit": row.raw_logit,
                    "sample_id": row.sample_id,
                }
                for row in self.predictions
            ]
        return document


@dataclass(frozen=True, slots=True)
class LoadedSelectedCheckpoint:
    model: Any
    training_config: TrainingConfig
    optimization_config: OptimizationConfig
    environment: EnvironmentFingerprint
    generation: LoadedGeneration
    checkpoint_sha256: str


def partition_sha256(assignments: Mapping[str, str], partition: str) -> str:
    if (
        not isinstance(partition, str)
        or not partition
        or partition != partition.strip()
    ):
        raise ValueError("partition must be non-empty and trimmed")
    sample_ids = sorted(
        sample_id
        for sample_id, assigned in assignments.items()
        if assigned == partition
    )
    if not sample_ids:
        raise ValueError(f"partition has no assigned samples: {partition}")
    return hashlib.sha256(
        _canonical_json(
            {"sample_ids": sample_ids, "schema_version": 1, "split": partition}
        )
    ).hexdigest()


def build_inference_result(
    *,
    partition: str,
    predictions: Sequence[CalibrationPrediction],
    training_config: TrainingConfig,
    split_manifest: SplitManifest,
    checkpoint_sha256: str,
    environment: EnvironmentFingerprint,
) -> InferenceResult:
    if partition not in _ALLOWED_PARTITIONS:
        raise ValueError("inference partition must be calibration or validation")
    if type(training_config) is not TrainingConfig:
        raise TypeError("training_config must be TrainingConfig")
    if type(split_manifest) is not SplitManifest:
        raise TypeError("split_manifest must be SplitManifest")
    if type(environment) is not EnvironmentFingerprint:
        raise TypeError("environment must be EnvironmentFingerprint")
    _require_sha256(checkpoint_sha256, "checkpoint_sha256")
    if (
        training_config.dataset_manifest_sha256
        != split_manifest.dataset_manifest_sha256
    ):
        raise ValueError("dataset manifest digest mismatch")
    if training_config.split_manifest_sha256 != split_manifest.sha256:
        raise ValueError("split manifest digest mismatch")

    selected_partition_sha256 = partition_sha256(split_manifest.assignments, partition)
    if (
        partition == "calibration"
        and selected_partition_sha256 != training_config.calibration_split_sha256
    ):
        raise ValueError("calibration split digest mismatch")
    rows = tuple(predictions)
    if any(type(row) is not CalibrationPrediction for row in rows):
        raise TypeError("predictions must contain CalibrationPrediction rows")
    rows = tuple(sorted(rows, key=lambda row: row.sample_id))
    identifiers = [row.sample_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate sample ID in inference predictions")
    expected_ids = {
        sample_id
        for sample_id, assigned in split_manifest.assignments.items()
        if assigned == partition
    }
    if set(identifiers) != expected_ids:
        raise ValueError("predictions must exactly match the named split partition")
    labels = [row.label for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError("inference metrics require both classes")

    metrics = InferenceMetrics(
        auc=_binary_auc(labels, [row.raw_logit for row in rows]),
        balanced_accuracy=_balanced_accuracy(rows),
    )
    return InferenceResult(
        partition=partition,
        checkpoint_sha256=checkpoint_sha256,
        dataset_manifest_sha256=training_config.dataset_manifest_sha256,
        split_manifest_sha256=split_manifest.sha256,
        partition_sha256=selected_partition_sha256,
        training_config_sha256=training_config.sha256,
        environment_sha256=_environment_sha256(environment),
        exposed_holdout_manifest_sha256=training_config.exposed_holdout_sha256,
        predictions=rows,
        metrics=metrics,
    )


def load_selected_checkpoint(
    run: Path,
    *,
    device: str,
    observed_environment: EnvironmentFingerprint | None = None,
    torch_module: Any | None = None,
    model_factory: ModelFactory | None = None,
) -> LoadedSelectedCheckpoint:
    if not run.is_dir() or run.is_symlink():
        raise ValueError("training run must be a real directory")
    training_config = _load_training_config(run / "training-config.json")
    optimization_config = _load_optimization_config(run / "optimization-config.json")
    recorded_environment = _load_environment(run / "environment.json")
    torch = torch_module or _required_torch()
    observed = observed_environment or capture_environment(torch_module=torch)
    if observed != recorded_environment:
        raise ValueError("environment provenance mismatch")
    generation = load_current_generation(
        run,
        config=training_config,
        optimization=optimization_config,
        environment=observed,
    )
    if generation is None:
        raise ValueError("training run has no selected immutable checkpoint")

    state = torch.load(
        generation.best_weights_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state, Mapping) or not state:
        raise ValueError("selected checkpoint must contain a model state mapping")
    factory = model_factory or (lambda: create_convnextv2_nano(ConvNeXtV2NanoConfig()))
    model = factory()
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return LoadedSelectedCheckpoint(
        model=model,
        training_config=training_config,
        optimization_config=optimization_config,
        environment=recorded_environment,
        generation=generation,
        checkpoint_sha256=generation.manifest.best_weights_sha256,
    )


def predict_logits(
    model: Any,
    dataset: Any,
    *,
    batch_size: int,
    workers: int,
    device: str,
    torch_module: Any | None = None,
) -> tuple[CalibrationPrediction, ...]:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("workers must be a non-negative integer")
    torch = torch_module or _required_torch()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
    )
    rows: list[CalibrationPrediction] = []
    model.eval()
    with torch.inference_mode():
        for images, labels, sample_ids in loader:
            logits = model(images.to(device)).detach().reshape(-1).to("cpu").tolist()
            raw_labels = labels.detach().reshape(-1).to("cpu").tolist()
            identifiers = list(sample_ids)
            if not len(logits) == len(raw_labels) == len(identifiers):
                raise ValueError("model output count does not match inference batch")
            for sample_id, raw_logit, raw_label in zip(
                identifiers, logits, raw_labels, strict=True
            ):
                if raw_label not in (0, 1, 0.0, 1.0) or isinstance(raw_label, bool):
                    raise ValueError("inference labels must be 0 (real) or 1 (AI)")
                rows.append(
                    CalibrationPrediction(
                        sample_id=sample_id,
                        raw_logit=raw_logit,
                        label=int(raw_label),
                    )
                )
    return tuple(rows)


def run_inference(
    dataset_root: Path,
    run: Path,
    *,
    partition: str,
    holdout_manifests: Sequence[Path],
    batch_size: int,
    workers: int,
    device: str,
    observed_environment: EnvironmentFingerprint | None = None,
    torch_module: Any | None = None,
    model_factory: ModelFactory | None = None,
) -> InferenceResult:
    if partition not in _ALLOWED_PARTITIONS:
        raise ValueError("inference partition must be calibration or validation")
    torch = torch_module or _required_torch()
    selected = load_selected_checkpoint(
        run,
        device=device,
        observed_environment=observed_environment,
        torch_module=torch,
        model_factory=model_factory,
    )
    manifest = load_dataset_manifest(dataset_root / "manifest.json")
    if manifest.sha256 != selected.training_config.dataset_manifest_sha256:
        raise ValueError("dataset manifest digest mismatch")
    _verify_preparation(
        dataset_root / "preparation.json",
        dataset_manifest_sha256=manifest.sha256,
        split_manifest_sha256=selected.training_config.split_manifest_sha256,
        exposed_holdout_sha256=selected.training_config.exposed_holdout_sha256,
    )
    holdouts = load_registered_exposed_holdouts(
        holdout_manifests,
        expected_sha256=selected.training_config.exposed_holdout_sha256,
    )
    split_manifest = load_split_manifest(
        dataset_root / "splits.json",
        manifest,
        expected_sha256=selected.training_config.split_manifest_sha256,
    )
    manifest.verify_materialized_files(dataset_root)
    samples = samples_for_split(manifest, split_manifest, partition)
    reject_holdout_overlap(samples, holdouts)
    configure_determinism(selected.training_config.seed, torch_module=torch)
    dataset = DatasetImageSamples(samples, dataset_root)
    predictions = predict_logits(
        selected.model,
        dataset,
        batch_size=batch_size,
        workers=workers,
        device=device,
        torch_module=torch,
    )
    return build_inference_result(
        partition=partition,
        predictions=predictions,
        training_config=selected.training_config,
        split_manifest=split_manifest,
        checkpoint_sha256=selected.checkpoint_sha256,
        environment=selected.environment,
    )


def load_registered_exposed_holdouts(
    paths: Sequence[Path], *, expected_sha256: Sequence[str]
) -> tuple[HoldoutIndex, ...]:
    expected = tuple(sorted(expected_sha256))
    if not expected:
        raise ValueError("at least one registered exposed holdout is required")
    for digest in expected:
        _require_sha256(digest, "expected exposed holdout digest")

    manifest_paths = tuple(paths)
    if len(manifest_paths) != len(expected):
        raise ValueError(
            "exactly the registered exposed holdout manifests are required"
        )
    loaded: list[HoldoutIndex] = []
    observed: list[str] = []
    for path in manifest_paths:
        if not isinstance(path, Path):
            raise TypeError("holdout manifest paths must be pathlib.Path values")
        if path.is_symlink() or not path.is_file():
            raise ValueError("registered exposed holdout must be a real manifest file")
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ValueError(
                "registered exposed holdout manifest is unreadable"
            ) from error
        digest = hashlib.sha256(payload).hexdigest()
        observed.append(digest)
        loaded.append(
            load_registered_holdout(
                HoldoutRegistration(
                    holdout_id=path.stem,
                    manifest_path=path,
                    manifest_sha256=digest,
                    status="development_exposed",
                )
            )
        )
    if tuple(sorted(observed)) != expected:
        raise ValueError(
            "exactly the registered exposed holdout manifests are required"
        )
    return tuple(loaded)


def _verify_preparation(
    path: Path,
    *,
    dataset_manifest_sha256: str,
    split_manifest_sha256: str,
    exposed_holdout_sha256: Sequence[str],
) -> None:
    document, _ = _canonical_object(path, "preparation provenance")
    required = {
        "manifest_sha256",
        "splits_sha256",
        "exposed_holdout_manifest_sha256",
    }
    if not required.issubset(document):
        raise ValueError("preparation provenance is missing required digest bindings")
    registered = document["exposed_holdout_manifest_sha256"]
    if not isinstance(registered, list) or not registered:
        raise ValueError("preparation provenance requires exposed holdout digests")
    if document["manifest_sha256"] != dataset_manifest_sha256:
        raise ValueError("preparation manifest digest mismatch")
    if document["splits_sha256"] != split_manifest_sha256:
        raise ValueError("preparation split digest mismatch")
    if tuple(sorted(registered)) != tuple(sorted(exposed_holdout_sha256)):
        raise ValueError("preparation exposed holdout digests mismatch")


def _load_training_config(path: Path) -> TrainingConfig:
    document, payload = _canonical_object(path, "training config")
    expected = {
        "architecture",
        "calibration_split_sha256",
        "dataset_manifest_sha256",
        "exposed_holdout_sha256",
        "pretrained",
        "schema_version",
        "seed",
        "selection_metric",
        "selection_minimize",
        "split_manifest_sha256",
        "weights_origin",
    }
    if set(document) != expected or not isinstance(
        document["exposed_holdout_sha256"], list
    ):
        raise ValueError("training config fields do not match the schema")
    config = TrainingConfig(
        dataset_manifest_sha256=document["dataset_manifest_sha256"],
        split_manifest_sha256=document["split_manifest_sha256"],
        calibration_split_sha256=document["calibration_split_sha256"],
        exposed_holdout_sha256=tuple(document["exposed_holdout_sha256"]),
        seed=document["seed"],
    )
    if config.to_json_bytes() != payload:
        raise ValueError("training config must be canonical and frozen")
    return config


def _load_optimization_config(path: Path) -> OptimizationConfig:
    document, payload = _canonical_object(path, "optimization config")
    declared_fields = fields(OptimizationConfig)
    if set(document) != {field.name for field in declared_fields}:
        raise ValueError("optimization config fields do not match the schema")
    config = OptimizationConfig(
        **{field.name: document[field.name] for field in declared_fields if field.init}
    )
    if config.to_json_bytes() != payload:
        raise ValueError("optimization config must be canonical and frozen")
    return config


def _load_environment(path: Path) -> EnvironmentFingerprint:
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid environment provenance JSON") from error
    if not isinstance(document, dict):
        raise ValueError("environment provenance must be a JSON object")
    environment = EnvironmentFingerprint(**document)
    expected = (
        json.dumps(environment.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode()
    if payload != expected:
        raise ValueError("environment provenance must be canonical and frozen")
    return environment


def _canonical_object(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description} JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{description} must be a JSON object")
    return document, payload


def _balanced_accuracy(rows: Sequence[CalibrationPrediction]) -> float:
    cutoff = math.log(FIXED_THRESHOLD / (1.0 - FIXED_THRESHOLD))
    real = [row for row in rows if row.label == 0]
    ai = [row for row in rows if row.label == 1]
    true_negative_rate = sum(row.raw_logit < cutoff for row in real) / len(real)
    true_positive_rate = sum(row.raw_logit >= cutoff for row in ai) / len(ai)
    return (true_negative_rate + true_positive_rate) / 2.0


def _binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = list(zip(labels, scores, strict=True))
    positive_count = sum(label == 1 for label, _ in pairs)
    negative_count = len(pairs) - positive_count
    ranked = sorted(pairs, key=lambda pair: pair[1])
    positive_rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][1] == ranked[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label == 1 for label, _ in ranked[start:end]
        )
        start = end
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _environment_sha256(environment: EnvironmentFingerprint) -> str:
    return hashlib.sha256(_canonical_json(environment.to_dict())).hexdigest()


def _canonical_json(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _required_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        raise RuntimeError("inference requires PyTorch") from error
