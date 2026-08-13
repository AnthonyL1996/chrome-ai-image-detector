from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Literal

from poidh_detector.reproducibility import EnvironmentFingerprint
from poidh_detector.training import (
    CheckpointCandidate,
    TrainingConfig,
    select_best_checkpoint,
)


ProfileName = Literal["overfit", "smoke", "pilot", "full"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GENERATION_ID = re.compile(r"^epoch-[0-9]{4,}-[0-9a-f]{16}$")
_PROFILE_VALUES: dict[str, dict[str, int | None]] = {
    "overfit": {
        "epochs": 30,
        "batch_size": 8,
        "train_per_class_cap": 16,
        "validation_per_class_cap": 16,
    },
    "smoke": {
        "epochs": 2,
        "batch_size": 32,
        "train_per_class_cap": 256,
        "validation_per_class_cap": 64,
    },
    "pilot": {
        "epochs": 10,
        "batch_size": 64,
        "train_per_class_cap": 10_000,
        "validation_per_class_cap": 2_000,
    },
    "full": {
        "epochs": 20,
        "batch_size": 64,
        "train_per_class_cap": None,
        "validation_per_class_cap": None,
    },
}


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    profile: ProfileName
    epochs: int
    batch_size: int
    train_per_class_cap: int | None
    validation_per_class_cap: int | None
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    ema_decay: float = 0.9999
    num_workers: int = 4
    optimizer: str = field(default="adamw", init=False)
    loss: str = field(default="bce_with_logits", init=False)
    schedule: str = field(default="linear_warmup_cosine", init=False)
    image_size: int = field(default=224, init=False)
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.profile not in _PROFILE_VALUES:
            raise ValueError(f"unknown training profile: {self.profile}")
        for field_name in ("epochs", "batch_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers must be a non-negative integer")
        for field_name in ("train_per_class_cap", "validation_per_class_cap"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{field_name} must be a positive integer or None")
        for field_name in ("learning_rate", "weight_decay"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{field_name} must be finite and positive")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")

    @classmethod
    def for_profile(cls, profile: ProfileName, **overrides: Any) -> OptimizationConfig:
        values = _PROFILE_VALUES.get(profile)
        if values is None:
            raise ValueError(f"unknown training profile: {profile}")
        return cls(profile=profile, **(values | overrides))

    def to_json_bytes(self) -> bytes:
        return _canonical_json(asdict(self))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    bce: float
    auc: float
    count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.bce, bool)
            or not isinstance(self.bce, (int, float))
            or not math.isfinite(self.bce)
            or self.bce < 0.0
        ):
            raise ValueError("bce must be a finite non-negative number")
        if (
            isinstance(self.auc, bool)
            or not isinstance(self.auc, (int, float))
            or not math.isfinite(self.auc)
            or not 0.0 <= self.auc <= 1.0
        ):
            raise ValueError("auc must be a finite number in [0, 1]")
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count <= 0
        ):
            raise ValueError("count must be a positive integer")


@dataclass(frozen=True, slots=True)
class ResumeContract:
    training_config_sha256: str
    optimization_config_sha256: str
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    environment: Mapping[str, str | bool | None]
    completed_epoch: int
    global_step: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported resume contract schema version")
        for field_name in (
            "training_config_sha256",
            "optimization_config_sha256",
            "dataset_manifest_sha256",
            "split_manifest_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if (
            isinstance(self.completed_epoch, bool)
            or not isinstance(self.completed_epoch, int)
            or self.completed_epoch < 0
        ):
            raise ValueError("completed_epoch must be a non-negative integer")
        if (
            isinstance(self.global_step, bool)
            or not isinstance(self.global_step, int)
            or self.global_step < 0
        ):
            raise ValueError("global_step must be a non-negative integer")
        environment = dict(self.environment)
        if not environment:
            raise ValueError("resume contract requires environment provenance")
        object.__setattr__(self, "environment", MappingProxyType(environment))

    @classmethod
    def create(
        cls,
        *,
        config: TrainingConfig,
        optimization: OptimizationConfig,
        environment: EnvironmentFingerprint,
        completed_epoch: int,
        global_step: int,
    ) -> ResumeContract:
        return cls(
            training_config_sha256=config.sha256,
            optimization_config_sha256=optimization.sha256,
            dataset_manifest_sha256=config.dataset_manifest_sha256,
            split_manifest_sha256=config.split_manifest_sha256,
            environment=environment.to_dict(),
            completed_epoch=completed_epoch,
            global_step=global_step,
        )

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> ResumeContract:
        expected = {
            "training_config_sha256",
            "optimization_config_sha256",
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "environment",
            "completed_epoch",
            "global_step",
            "schema_version",
        }
        if set(document) != expected:
            raise ValueError("resume contract fields do not match the schema")
        return cls(**document)

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_epoch": self.completed_epoch,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "environment": dict(self.environment),
            "global_step": self.global_step,
            "optimization_config_sha256": self.optimization_config_sha256,
            "schema_version": self.schema_version,
            "split_manifest_sha256": self.split_manifest_sha256,
            "training_config_sha256": self.training_config_sha256,
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class GenerationManifest:
    generation_id: str
    epoch: int
    global_step: int
    selected_checkpoint_id: str
    training_config_sha256: str
    optimization_config_sha256: str
    environment_sha256: str
    resume_sha256: str
    best_weights_sha256: str
    history_sha256: str
    history_count: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported generation manifest schema version")
        if not _GENERATION_ID.fullmatch(self.generation_id):
            raise ValueError("invalid generation ID")
        for field_name in ("epoch", "global_step", "history_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            not self.selected_checkpoint_id
            or self.selected_checkpoint_id != self.selected_checkpoint_id.strip()
        ):
            raise ValueError("selected_checkpoint_id must be non-empty and trimmed")
        for field_name in (
            "training_config_sha256",
            "optimization_config_sha256",
            "environment_sha256",
            "resume_sha256",
            "best_weights_sha256",
            "history_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> GenerationManifest:
        document = _json_object(payload, "generation manifest")
        expected = {
            "best_weights_sha256",
            "environment_sha256",
            "epoch",
            "generation_id",
            "global_step",
            "history_count",
            "history_sha256",
            "optimization_config_sha256",
            "resume_sha256",
            "schema_version",
            "selected_checkpoint_id",
            "training_config_sha256",
        }
        if set(document) != expected:
            raise ValueError("generation manifest fields do not match the schema")
        manifest = cls(**document)
        if manifest.to_json_bytes() != payload:
            raise ValueError("generation manifest must use canonical JSON encoding")
        return manifest

    def to_json_bytes(self) -> bytes:
        return _canonical_json(asdict(self))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class LoadedGeneration:
    path: Path
    manifest: GenerationManifest
    resume_path: Path
    best_weights_path: Path
    history_path: Path
    candidates: tuple[CheckpointCandidate, ...]


InterruptionHook = Callable[[str], None]
ArtifactWriter = Callable[[Path], object]


def reserve_run_directory(output: Path, *, os_name: str | None = None) -> None:
    """Atomically claim an absent output path for a new training run."""

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"training output already exists: {output}") from error
    (output / "generations").mkdir()
    _fsync_directory(output, os_name=os_name)
    _fsync_directory(output.parent, os_name=os_name)


def publish_generation(
    output: Path,
    *,
    contract: ResumeContract,
    candidates: Sequence[CheckpointCandidate],
    selected: CheckpointCandidate,
    write_resume: ArtifactWriter,
    write_best_weights: ArtifactWriter,
    interruption_hook: InterruptionHook | None = None,
    os_name: str | None = None,
) -> LoadedGeneration:
    """Publish all epoch state as one immutable generation, then advance CURRENT."""

    history = tuple(candidates)
    _validate_history(history, contract)
    if selected != select_best_checkpoint(history):
        raise ValueError("selected checkpoint is not the validation-BCE winner")
    generations = output / "generations"
    if (
        not output.is_dir()
        or output.is_symlink()
        or not generations.is_dir()
        or generations.is_symlink()
    ):
        raise ValueError("training output is not a reserved run directory")

    generation_id = f"epoch-{contract.completed_epoch:04d}-{secrets.token_hex(8)}"
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=generations))
    final = generations / generation_id
    current_temporary: Path | None = None
    try:
        resume_path = staging / "resume.pt"
        best_weights_path = staging / "best-model.pt"
        history_path = staging / "validation-history.json"
        write_resume(resume_path)
        write_best_weights(best_weights_path)
        for path, description in (
            (resume_path, "resume"),
            (best_weights_path, "best weights"),
        ):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"{description} writer did not create a regular file")
            _fsync_file(path)
        history_payload = _candidate_history_bytes(history)
        history_path.write_bytes(history_payload)
        _fsync_file(history_path)
        manifest = GenerationManifest(
            generation_id=generation_id,
            epoch=contract.completed_epoch,
            global_step=contract.global_step,
            selected_checkpoint_id=selected.checkpoint_id,
            training_config_sha256=contract.training_config_sha256,
            optimization_config_sha256=contract.optimization_config_sha256,
            environment_sha256=_environment_sha256(contract.environment),
            resume_sha256=_sha256_file(resume_path),
            best_weights_sha256=_sha256_file(best_weights_path),
            history_sha256=hashlib.sha256(history_payload).hexdigest(),
            history_count=len(history),
        )
        manifest_path = staging / "generation.json"
        manifest_path.write_bytes(manifest.to_json_bytes())
        _fsync_file(manifest_path)
        ready_path = staging / "READY"
        ready_path.write_text(manifest.sha256 + "\n", encoding="ascii")
        _fsync_file(ready_path)
        _fsync_directory(staging, os_name=os_name)
        _interrupt(interruption_hook, "after_artifacts")

        staging.replace(final)
        _fsync_directory(generations, os_name=os_name)
        _interrupt(interruption_hook, "after_generation_rename")

        pointer_payload = _canonical_json(
            {
                "generation": generation_id,
                "generation_manifest_sha256": manifest.sha256,
                "schema_version": 1,
            }
        )
        descriptor, temporary_name = tempfile.mkstemp(prefix=".CURRENT-", dir=output)
        current_temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(pointer_payload)
            stream.flush()
            os.fsync(stream.fileno())
        _interrupt(interruption_hook, "before_pointer_replace")
        os.replace(current_temporary, output / "CURRENT")
        current_temporary = None
        _fsync_directory(output, os_name=os_name)
        _interrupt(interruption_hook, "after_pointer_replace")
        return LoadedGeneration(
            path=final,
            manifest=manifest,
            resume_path=final / "resume.pt",
            best_weights_path=final / "best-model.pt",
            history_path=final / "validation-history.json",
            candidates=history,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if current_temporary is not None and current_temporary.exists():
            current_temporary.unlink()


def load_current_generation(
    output: Path,
    *,
    config: TrainingConfig,
    optimization: OptimizationConfig,
    environment: EnvironmentFingerprint,
) -> LoadedGeneration | None:
    """Load only the generation selected by an atomically published CURRENT."""

    if not output.is_dir() or output.is_symlink():
        raise ValueError("resume output must be a real run directory")
    pointer_path = output / "CURRENT"
    if not os.path.lexists(pointer_path):
        generations = output / "generations"
        if not generations.is_dir() or generations.is_symlink():
            raise ValueError("resume output has no safe generations directory")
        return None
    if not pointer_path.is_file() or pointer_path.is_symlink():
        raise ValueError("resume output has no published CURRENT generation")
    pointer_payload = pointer_path.read_bytes()
    pointer = _json_object(pointer_payload, "CURRENT pointer")
    if set(pointer) != {
        "generation",
        "generation_manifest_sha256",
        "schema_version",
    }:
        raise ValueError("CURRENT pointer fields do not match the schema")
    if type(pointer["schema_version"]) is not int or pointer["schema_version"] != 1:
        raise ValueError("unsupported CURRENT pointer schema version")
    if _canonical_json(pointer) != pointer_payload:
        raise ValueError("CURRENT pointer must use canonical JSON encoding")
    generation_id = pointer["generation"]
    if not isinstance(generation_id, str) or not _GENERATION_ID.fullmatch(
        generation_id
    ):
        raise ValueError("CURRENT pointer has an invalid generation ID")
    pointer_manifest_sha256 = pointer["generation_manifest_sha256"]
    _require_sha256(pointer_manifest_sha256, "generation_manifest_sha256")

    generation = output / "generations" / generation_id
    expected_names = {
        "READY",
        "best-model.pt",
        "generation.json",
        "resume.pt",
        "validation-history.json",
    }
    if (
        not generation.is_dir()
        or generation.is_symlink()
        or {path.name for path in generation.iterdir()} != expected_names
        or any(path.is_symlink() for path in generation.iterdir())
    ):
        raise ValueError("CURRENT generation is missing, unsafe, or incomplete")
    manifest_path = generation / "generation.json"
    manifest = GenerationManifest.from_json_bytes(manifest_path.read_bytes())
    if manifest.generation_id != generation_id:
        raise ValueError("generation manifest ID does not match CURRENT")
    if manifest.sha256 != pointer_manifest_sha256:
        raise ValueError("generation manifest digest does not match CURRENT")
    ready = (generation / "READY").read_bytes()
    if ready != (manifest.sha256 + "\n").encode("ascii"):
        raise ValueError("generation READY marker does not match manifest")
    if manifest.training_config_sha256 != config.sha256:
        raise ValueError("generation training configuration does not match")
    if manifest.optimization_config_sha256 != optimization.sha256:
        raise ValueError("generation optimization configuration does not match")
    if manifest.environment_sha256 != _environment_sha256(environment.to_dict()):
        raise ValueError("generation environment provenance does not match")

    resume_path = generation / "resume.pt"
    best_weights_path = generation / "best-model.pt"
    history_path = generation / "validation-history.json"
    if _sha256_file(resume_path) != manifest.resume_sha256:
        raise ValueError("resume artifact digest does not match generation manifest")
    if _sha256_file(best_weights_path) != manifest.best_weights_sha256:
        raise ValueError(
            "best weights artifact digest does not match generation manifest"
        )
    history_payload = history_path.read_bytes()
    if hashlib.sha256(history_payload).hexdigest() != manifest.history_sha256:
        raise ValueError("history artifact digest does not match generation manifest")
    candidates = _parse_candidate_history(history_payload)
    if _candidate_history_bytes(candidates) != history_payload:
        raise ValueError("validation history must use canonical JSON encoding")
    _validate_history_values(candidates, manifest.epoch, manifest.global_step)
    if len(candidates) != manifest.history_count:
        raise ValueError("validation history count does not match generation manifest")
    selected = select_best_checkpoint(candidates)
    if selected.checkpoint_id != manifest.selected_checkpoint_id:
        raise ValueError("selected checkpoint does not match validation history")
    return LoadedGeneration(
        path=generation,
        manifest=manifest,
        resume_path=resume_path,
        best_weights_path=best_weights_path,
        history_path=history_path,
        candidates=candidates,
    )


def validate_generation_resume_contract(
    generation: LoadedGeneration, contract: ResumeContract
) -> None:
    manifest = generation.manifest
    if contract.completed_epoch != manifest.epoch:
        raise ValueError("resume contract epoch does not match generation manifest")
    if contract.global_step != manifest.global_step:
        raise ValueError(
            "resume contract global step does not match generation manifest"
        )
    if contract.training_config_sha256 != manifest.training_config_sha256:
        raise ValueError("resume contract training config does not match generation")
    if contract.optimization_config_sha256 != manifest.optimization_config_sha256:
        raise ValueError(
            "resume contract optimization config does not match generation"
        )
    if _environment_sha256(contract.environment) != manifest.environment_sha256:
        raise ValueError("resume contract environment does not match generation")


def validate_resume_contract(
    contract: ResumeContract,
    config: TrainingConfig,
    optimization: OptimizationConfig,
    environment: EnvironmentFingerprint,
) -> None:
    if contract.training_config_sha256 != config.sha256:
        raise ValueError("resume training configuration does not match")
    if contract.optimization_config_sha256 != optimization.sha256:
        raise ValueError("resume optimization configuration does not match")
    if contract.dataset_manifest_sha256 != config.dataset_manifest_sha256:
        raise ValueError("resume dataset manifest does not match")
    if contract.split_manifest_sha256 != config.split_manifest_sha256:
        raise ValueError("resume split manifest does not match")
    if dict(contract.environment) != environment.to_dict():
        raise ValueError("resume environment provenance does not match")


def cosine_warmup_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    for field_name, value in (
        ("step", step),
        ("total_steps", total_steps),
        ("warmup_steps", warmup_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be less than total_steps")
    bounded_step = min(step, total_steps)
    if warmup_steps and bounded_step < warmup_steps:
        return bounded_step / warmup_steps
    progress = (bounded_step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def binary_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    pairs = list(zip(labels, scores, strict=True))
    if any(label not in (0, 1) for label, _ in pairs):
        raise ValueError("AUC labels must be 0 or 1")
    if any(not math.isfinite(score) for _, score in pairs):
        raise ValueError("AUC scores must be finite")
    positive_count = sum(label == 1 for label, _ in pairs)
    negative_count = len(pairs) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("AUC requires both classes")

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


def candidate_from_validation(
    *,
    epoch: int,
    global_step: int,
    training_bce: float,
    validation: ValidationMetrics,
) -> CheckpointCandidate:
    return CheckpointCandidate(
        checkpoint_id=f"epoch-{epoch:04d}",
        epoch=epoch,
        global_step=global_step,
        validation_bce=validation.bce,
        training_bce=training_bce,
    )


def create_optimizer_and_scheduler(
    model: Any,
    optimization: OptimizationConfig,
    *,
    total_steps: int,
    torch_module: Any | None = None,
) -> tuple[Any, Any]:
    torch = torch_module or _required_torch()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimization.learning_rate,
        weight_decay=optimization.weight_decay,
    )
    warmup_steps = int(total_steps * optimization.warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_warmup_multiplier(
            step, total_steps, warmup_steps
        ),
    )
    return optimizer, scheduler


class ExponentialMovingAverage:
    def __init__(self, model: Any, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if getattr(value, "is_floating_point", lambda: False)()
        }

    def update(self, model: Any) -> None:
        for name, value in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {name: value.clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError("EMA decay does not match resume state")
        incoming = dict(state["shadow"])
        if set(incoming) != set(self.shadow):
            raise ValueError("EMA parameter set does not match model")
        self.shadow = {name: value.clone() for name, value in incoming.items()}

    @contextmanager
    def average_parameters(self, model: Any) -> Iterable[None]:
        model_state = model.state_dict()
        backup = {name: model_state[name].detach().clone() for name in self.shadow}
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=False)


def train_one_epoch(
    model: Any,
    loader: Iterable[Any],
    optimizer: Any,
    scheduler: Any,
    ema: ExponentialMovingAverage,
    *,
    device: str,
    torch_module: Any | None = None,
) -> tuple[float, int]:
    torch = torch_module or _required_torch()
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    model.train()
    loss_sum = 0.0
    sample_count = 0
    steps = 0
    for images, labels, _sample_ids in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().reshape(-1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images).reshape(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)
        count = int(labels.numel())
        loss_sum += float(loss.detach().item()) * count
        sample_count += count
        steps += 1
    if sample_count == 0:
        raise ValueError("training loader must not be empty")
    return loss_sum / sample_count, steps


def evaluate_model(
    model: Any,
    loader: Iterable[Any],
    *,
    device: str,
    torch_module: Any | None = None,
) -> ValidationMetrics:
    torch = torch_module or _required_torch()
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    model.eval()
    loss_sum = 0.0
    all_labels: list[int] = []
    all_scores: list[float] = []
    with torch.no_grad():
        for images, labels, _sample_ids in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().reshape(-1)
            logits = model(images).reshape(-1)
            loss_sum += float(criterion(logits, labels).item())
            all_labels.extend(int(value) for value in labels.detach().cpu().tolist())
            all_scores.extend(
                float(value) for value in torch.sigmoid(logits).detach().cpu().tolist()
            )
    if not all_labels:
        raise ValueError("validation loader must not be empty")
    return ValidationMetrics(
        bce=loss_sum / len(all_labels),
        auc=binary_auc(all_labels, all_scores),
        count=len(all_labels),
    )


def save_resume_checkpoint(
    path: Path,
    *,
    contract: ResumeContract,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    ema: ExponentialMovingAverage,
    torch_module: Any | None = None,
) -> None:
    torch = torch_module or _required_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    payload = {
        "contract": contract.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_resume_checkpoint(
    path: Path,
    *,
    config: TrainingConfig,
    optimization: OptimizationConfig,
    environment: EnvironmentFingerprint,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    ema: ExponentialMovingAverage,
    torch_module: Any | None = None,
) -> ResumeContract:
    torch = torch_module or _required_torch()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "contract" not in payload:
        raise ValueError("resume checkpoint payload is invalid")
    contract = ResumeContract.from_dict(payload["contract"])
    validate_resume_contract(contract, config, optimization, environment)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    ema.load_state_dict(payload["ema"])
    torch.set_rng_state(payload["torch_rng_state"])
    if payload["cuda_rng_state"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("resume checkpoint requires CUDA RNG state")
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    return contract


def _validate_history(
    candidates: tuple[CheckpointCandidate, ...], contract: ResumeContract
) -> None:
    _validate_history_values(candidates, contract.completed_epoch, contract.global_step)


def _validate_history_values(
    candidates: Sequence[CheckpointCandidate], epoch: int, global_step: int
) -> None:
    if epoch <= 0 or global_step <= 0:
        raise ValueError("published generation progress must be positive")
    if len(candidates) != epoch:
        raise ValueError("validation history must contain one row per completed epoch")
    if [candidate.epoch for candidate in candidates] != list(range(1, epoch + 1)):
        raise ValueError("validation history epochs must be contiguous and ordered")
    steps = [candidate.global_step for candidate in candidates]
    if steps != sorted(set(steps)) or steps[-1] != global_step:
        raise ValueError("validation history global steps do not match progress")


def _candidate_history_bytes(
    candidates: Sequence[CheckpointCandidate],
) -> bytes:
    return _canonical_json(
        {
            "candidates": [asdict(candidate) for candidate in candidates],
            "schema_version": 1,
        }
    )


def _parse_candidate_history(payload: bytes) -> tuple[CheckpointCandidate, ...]:
    document = _json_object(payload, "validation history")
    if set(document) != {"candidates", "schema_version"}:
        raise ValueError("validation history fields do not match the schema")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("unsupported validation history schema version")
    rows = document["candidates"]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("validation history candidates must be a list of objects")
    expected = {
        "checkpoint_id",
        "epoch",
        "global_step",
        "training_bce",
        "validation_bce",
    }
    if any(set(row) != expected for row in rows):
        raise ValueError("validation history candidate fields do not match the schema")
    return tuple(CheckpointCandidate(**row) for row in rows)


def _environment_sha256(environment: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(environment))).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(payload: bytes, description: str) -> dict[str, Any]:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} must be valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{description} must be a JSON object")
    return document


def _interrupt(hook: InterruptionHook | None, stage: str) -> None:
    if hook is not None:
        hook(stage)


def _fsync_file(
    path: Path,
    *,
    open_file: Callable[[Path, str], Any] | None = None,
    fsync_descriptor: Callable[[int], object] | None = None,
) -> None:
    open_file = open_file or (lambda target, mode: target.open(mode))
    fsync_descriptor = fsync_descriptor or os.fsync
    with open_file(path, "r+b") as stream:
        stream.flush()
        fsync_descriptor(stream.fileno())


def _fsync_directory(
    path: Path,
    *,
    os_name: str | None = None,
    open_directory: Callable[[Path, int], int] | None = None,
    fsync_descriptor: Callable[[int], object] | None = None,
    close_descriptor: Callable[[int], object] | None = None,
) -> None:
    if (os_name or os.name) == "nt":
        return
    open_directory = open_directory or os.open
    fsync_descriptor = fsync_descriptor or os.fsync
    close_descriptor = close_descriptor or os.close
    descriptor = open_directory(path, os.O_RDONLY)
    try:
        fsync_descriptor(descriptor)
    finally:
        close_descriptor(descriptor)


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
        if error.name == "torch":
            raise RuntimeError(
                "training requires torch; install the training extras"
            ) from error
        raise
