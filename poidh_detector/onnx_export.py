from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import tempfile
from typing import Any

from poidh_detector.calibration import (
    CalibrationClassCounts,
    PlattCalibrationArtifact,
)
from poidh_detector.calibration_fit import CalibrationFitResult, CalibrationMetrics
from poidh_detector.export import (
    ONNX_OPSET_VERSION,
    ExportMetadata,
    OnnxArtifactSummary,
    OnnxTensor,
    validate_onnx_artifact,
)
from poidh_detector.model import ConvNeXtV2NanoConfig, create_convnextv2_nano
from poidh_detector.training import TrainingConfig


_MODEL_NAME = "detector.onnx"
_METADATA_NAME = "metadata.json"
_READY_NAME = "READY"
_INPUT_SHAPE = (1, 3, 224, 224)
_OUTPUT_SHAPE = (1, 1)


def export_detector_onnx(
    *,
    checkpoint: Path,
    calibrator: Path,
    training_config: Path,
    dataset_manifest: Path,
    code_provenance: Path,
    license_policy: Path,
    output: Path,
    torch_module: Any | None = None,
    timm_module: Any | None = None,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> ExportMetadata:
    """Export a frozen checkpoint and Platt calibration as one ONNX graph."""

    destination = output.absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"ONNX export output already exists: {destination}")

    checkpoint_payload = _read_regular_file(checkpoint, "checkpoint")
    config_payload = _read_regular_file(training_config, "training config")
    dataset_payload = _read_regular_file(dataset_manifest, "dataset manifest")
    calibrator_payload = _read_regular_file(calibrator, "calibrator")
    code_payload = _read_regular_file(code_provenance, "code provenance")
    license_payload = _read_regular_file(license_policy, "license policy")

    config = _load_training_config(config_payload)
    checkpoint_sha256 = _sha256(checkpoint_payload)
    dataset_manifest_sha256 = _sha256(dataset_payload)
    if config.dataset_manifest_sha256 != dataset_manifest_sha256:
        raise ValueError("dataset manifest digest mismatch")
    calibration = _load_calibrator(calibrator_payload, config)
    if calibration.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("calibrator checkpoint digest mismatch")

    onnx = _required_module("onnx", import_module)
    torch = torch_module or _required_module("torch", import_module)
    model = create_convnextv2_nano(
        ConvNeXtV2NanoConfig(),
        timm_module=timm_module,
        import_module=import_module,
    )
    state = torch.load(
        io.BytesIO(checkpoint_payload), map_location="cpu", weights_only=True
    )
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint must contain a non-empty model state mapping")
    if any(not isinstance(name, str) or not name for name in state):
        raise ValueError("checkpoint state keys must be non-empty strings")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)

    export_model = _calibrated_model(model, calibration, torch)
    dummy_input = torch.zeros(_INPUT_SHAPE, dtype=torch.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        model_path = staging / _MODEL_NAME
        torch.onnx.export(
            export_model,
            dummy_input,
            model_path,
            input_names=["image"],
            output_names=["probability_ai"],
            opset_version=ONNX_OPSET_VERSION,
            external_data=False,
            dynamic_axes=None,
            export_params=True,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            dynamo=False,
        )
        _require_single_model_file(staging, model_path)
        metadata = ExportMetadata(
            checkpoint_sha256=checkpoint_sha256,
            calibrator_sha256=_sha256(calibrator_payload),
            config_sha256=_sha256(config_payload),
            code_sha256=_sha256(code_payload),
            dataset_manifest_sha256=dataset_manifest_sha256,
            license_policy_sha256=_sha256(license_payload),
            model_sha256=_sha256(_read_regular_file(model_path, "ONNX model")),
        )
        _validate_onnx(model_path, metadata, onnx)
        metadata_path = staging / _METADATA_NAME
        metadata_path.write_bytes(metadata.to_json_bytes())
        _fsync_file(model_path)
        _fsync_file(metadata_path)
        _fsync_directory(staging)
        _publish_bundle(staging, destination, metadata)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
            _fsync_directory(destination.parent)
    return metadata


def validate_export_bundle(output: Path) -> ExportMetadata:
    """Verify canonical metadata and hashes for a published ONNX bundle."""

    bundle = output.absolute()
    if not bundle.is_dir() or bundle.is_symlink():
        raise ValueError(f"ONNX export bundle must be a real directory: {bundle}")
    expected = {_MODEL_NAME, _METADATA_NAME, _READY_NAME}
    if {entry.name for entry in bundle.iterdir()} != expected:
        raise ValueError("ONNX export bundle is incomplete or contains extra files")

    model_payload = _read_regular_file(bundle / _MODEL_NAME, "ONNX model")
    metadata_payload = _read_regular_file(bundle / _METADATA_NAME, "export metadata")
    ready_payload = _read_regular_file(bundle / _READY_NAME, "READY marker")
    expected_ready = (_sha256(metadata_payload) + "\n").encode("ascii")
    if ready_payload != expected_ready:
        raise ValueError("READY marker digest mismatch")
    metadata = _load_export_metadata(metadata_payload)
    if metadata.model_sha256 != _sha256(model_payload):
        raise ValueError("model digest mismatch")
    return metadata


def _calibrated_model(
    model: Any, calibration: PlattCalibrationArtifact, torch: Any
) -> Any:
    class CalibratedDetector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.detector = model
            self.register_buffer(
                "calibration_scale",
                torch.tensor(calibration.scale, dtype=torch.float32),
            )
            self.register_buffer(
                "calibration_bias",
                torch.tensor(calibration.bias, dtype=torch.float32),
            )

        def forward(self, image: Any) -> Any:
            raw_logit = self.detector(image).reshape(*_OUTPUT_SHAPE)
            calibrated_logit = self.calibration_scale * raw_logit
            calibrated_logit = calibrated_logit + self.calibration_bias
            return torch.sigmoid(calibrated_logit)

    return CalibratedDetector().eval().requires_grad_(False)


def _load_training_config(payload: bytes) -> TrainingConfig:
    document = _json_object(payload, "training config")
    expected_fields = {
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
    if set(document) != expected_fields:
        raise ValueError("training config fields do not match schema")
    holdouts = document["exposed_holdout_sha256"]
    if not isinstance(holdouts, list):
        raise ValueError("exposed_holdout_sha256 must be a JSON array")
    config = TrainingConfig(
        dataset_manifest_sha256=document["dataset_manifest_sha256"],
        split_manifest_sha256=document["split_manifest_sha256"],
        calibration_split_sha256=document["calibration_split_sha256"],
        exposed_holdout_sha256=tuple(holdouts),
        seed=document["seed"],
    )
    if config.to_json_bytes() != payload:
        raise ValueError("training config JSON must be canonical and frozen")
    return config


def _load_calibrator(
    payload: bytes, config: TrainingConfig
) -> PlattCalibrationArtifact:
    document = _json_object(payload, "calibrator")
    expected_fields = {
        "artifact",
        "calibrated",
        "ece_bins",
        "predictions_sha256",
        "schema_version",
        "threshold",
        "uncalibrated",
    }
    if set(document) != expected_fields:
        raise ValueError("calibrator fields do not match schema")
    artifact_document = document["artifact"]
    if not isinstance(artifact_document, dict):
        raise ValueError("calibrator artifact must be a JSON object")
    artifact_fields = {
        "bias",
        "calibration_split_sha256",
        "checkpoint_sha256",
        "class_counts",
        "input_identifier",
        "method",
        "sample_count",
        "scale",
        "schema_version",
        "threshold",
        "training_config_sha256",
    }
    if set(artifact_document) != artifact_fields:
        raise ValueError("calibrator artifact fields do not match schema")
    if artifact_document["training_config_sha256"] != config.sha256:
        raise ValueError("calibrator training config digest mismatch")
    counts = artifact_document["class_counts"]
    if not isinstance(counts, dict) or set(counts) != {"ai", "real"}:
        raise ValueError("calibrator class counts do not match schema")
    artifact = PlattCalibrationArtifact(
        scale=artifact_document["scale"],
        bias=artifact_document["bias"],
        checkpoint_sha256=artifact_document["checkpoint_sha256"],
        calibration_split_sha256=artifact_document["calibration_split_sha256"],
        training_config=config,
        input_identifier=artifact_document["input_identifier"],
        sample_count=artifact_document["sample_count"],
        class_counts=CalibrationClassCounts(
            real=counts["real"],
            ai=counts["ai"],
        ),
    )
    result = CalibrationFitResult(
        artifact=artifact,
        predictions_sha256=document["predictions_sha256"],
        uncalibrated=_load_calibration_metrics(document["uncalibrated"]),
        calibrated=_load_calibration_metrics(document["calibrated"]),
    )
    if result.to_json_bytes() != payload:
        raise ValueError("calibrator JSON must be canonical and frozen")
    return artifact


def _load_calibration_metrics(document: object) -> CalibrationMetrics:
    if not isinstance(document, dict) or set(document) != {
        "accuracy_at_threshold",
        "bce",
        "ece",
    }:
        raise ValueError("calibrator metrics do not match schema")
    return CalibrationMetrics(
        bce=document["bce"],
        ece=document["ece"],
        accuracy_at_threshold=document["accuracy_at_threshold"],
    )


def _validate_onnx(model_path: Path, metadata: ExportMetadata, onnx: Any) -> None:
    onnx.checker.check_model(str(model_path), full_check=True)
    model = onnx.load(str(model_path), load_external_data=False)
    graph = model.graph
    external = tuple(
        initializer.name
        for initializer in graph.initializer
        if initializer.data_location == onnx.TensorProto.EXTERNAL
        or bool(initializer.external_data)
    )
    operators = frozenset((node.domain, node.op_type) for node in graph.node)
    artifact = OnnxArtifactSummary(
        model_files=(model_path.name,),
        model_size_bytes=model_path.stat().st_size,
        inputs=tuple(_onnx_tensor(value, onnx) for value in graph.input),
        outputs=tuple(_onnx_tensor(value, onnx) for value in graph.output),
        opset_imports=tuple(
            (opset.domain, opset.version) for opset in model.opset_import
        ),
        operators=operators,
        calibration_embedded=("", "Sigmoid") in operators,
        external_data_files=external,
    )
    validate_onnx_artifact(artifact, metadata)


def _load_export_metadata(payload: bytes) -> ExportMetadata:
    document = _json_object(payload, "export metadata")
    metadata = ExportMetadata(
        checkpoint_sha256=document.get("checkpoint_sha256"),
        calibrator_sha256=document.get("calibrator_sha256"),
        config_sha256=document.get("config_sha256"),
        code_sha256=document.get("code_sha256"),
        dataset_manifest_sha256=document.get("dataset_manifest_sha256"),
        license_policy_sha256=document.get("license_policy_sha256"),
        model_sha256=document.get("model_sha256"),
    )
    if metadata.to_json_bytes() != payload:
        raise ValueError("export metadata JSON must be canonical and frozen")
    return metadata


def _publish_bundle(staging: Path, destination: Path, metadata: ExportMetadata) -> None:
    owns_destination = False
    reservation_token: str | None = None
    ready_temporary: Path | None = None
    try:
        try:
            destination.mkdir(mode=0o700)
        except FileExistsError as error:
            raise FileExistsError(
                f"ONNX export output already exists: {destination}"
            ) from error
        owns_destination = True

        reservation_token = secrets.token_hex(32)
        reservation = destination / ".reservation"
        reservation.write_text(reservation_token, encoding="ascii")
        _fsync_file(reservation)
        _fsync_directory(destination)
        _fsync_directory(destination.parent)

        (staging / _MODEL_NAME).replace(destination / _MODEL_NAME)
        (staging / _METADATA_NAME).replace(destination / _METADATA_NAME)
        _fsync_directory(destination)

        ready_payload = (_sha256(metadata.to_json_bytes()) + "\n").encode("ascii")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{_READY_NAME}.", suffix=".tmp", dir=destination
        )
        ready_temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(ready_payload)
            stream.flush()
            os.fsync(stream.fileno())

        reservation.unlink()
        owns_destination = False
        os.replace(ready_temporary, destination / _READY_NAME)
        ready_temporary = None
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
    finally:
        if ready_temporary is not None and os.path.lexists(ready_temporary):
            ready_temporary.unlink()
        if (
            owns_destination
            and reservation_token is not None
            and not os.path.lexists(destination / _READY_NAME)
        ):
            reservation = destination / ".reservation"
            try:
                owned = reservation.read_text(encoding="ascii") == reservation_token
            except OSError:
                owned = False
            if owned:
                shutil.rmtree(destination)
                _fsync_directory(destination.parent)


def _onnx_tensor(value: Any, onnx: Any) -> OnnxTensor:
    tensor_type = value.type.tensor_type
    if tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("ONNX graph tensors must use float32")
    dimensions: list[int] = []
    for dimension in tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise ValueError("ONNX graph tensor shapes must be fixed and positive")
        dimensions.append(dimension.dim_value)
    return OnnxTensor(value.name, tuple(dimensions), "float32")


def _require_single_model_file(staging: Path, model_path: Path) -> None:
    entries = set(staging.iterdir())
    if entries != {model_path} or not model_path.is_file() or model_path.is_symlink():
        raise ValueError("ONNX export produced forbidden external data files")


def _read_regular_file(path: Path, description: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{description} must be a real file")
    return path.read_bytes()


def _json_object(payload: bytes, description: str) -> dict[str, Any]:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description} JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{description} must be a JSON object")
    return document


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path, *, os_name: str | None = None) -> None:
    if (os_name or os.name) == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _required_module(name: str, import_module: Callable[[str], Any]) -> Any:
    try:
        return import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        raise RuntimeError(
            f"ONNX export requires {name}; install the training extras"
        ) from error
