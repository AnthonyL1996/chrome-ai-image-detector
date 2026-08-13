from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import json
import re


ONNX_OPSET_VERSION = 18
MAX_ONNX_BYTES = 100 * 1024 * 1024
SUPPORTED_ONNX_OPERATORS = frozenset(
    {
        "Add",
        "BatchNormalization",
        "Cast",
        "Clip",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Erf",
        "Expand",
        "Flatten",
        "Gather",
        "Gemm",
        "GlobalAveragePool",
        "Identity",
        "LayerNormalization",
        "Log",
        "MatMul",
        "Mul",
        "Pad",
        "Pow",
        "ReduceMean",
        "ReduceL2",
        "Reshape",
        "Resize",
        "Shape",
        "Sigmoid",
        "Slice",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESIZE = (224, 224)
_MEAN = (0.485, 0.456, 0.406)
_STANDARD_DEVIATION = (0.229, 0.224, 0.225)
_INPUT_SHAPE = (1, 3, *_RESIZE)
_OUTPUT_SHAPE = (1, 1)


@dataclass(frozen=True, slots=True)
class PreprocessingContract:
    """Preprocessing embedded in, or applied identically before, the ONNX graph."""

    resize: tuple[int, int] = _RESIZE
    interpolation: str = "bicubic"
    channel_order: str = "RGB"
    layout: str = "NCHW"
    input_dtype: str = "uint8"
    output_dtype: str = "float32"
    input_scale: float = 1.0 / 255.0
    mean: tuple[float, float, float] = _MEAN
    standard_deviation: tuple[float, float, float] = _STANDARD_DEVIATION

    def __post_init__(self) -> None:
        expected = {
            "resize": _RESIZE,
            "interpolation": "bicubic",
            "channel_order": "RGB",
            "layout": "NCHW",
            "input_dtype": "uint8",
            "output_dtype": "float32",
            "input_scale": 1.0 / 255.0,
            "mean": _MEAN,
            "standard_deviation": _STANDARD_DEVIATION,
        }
        for field_name, expected_value in expected.items():
            actual = getattr(self, field_name)
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise ValueError(f"preprocessing {field_name} is fixed")


@dataclass(frozen=True, slots=True)
class ExportMetadata:
    """Canonical metadata required beside a production detector export."""

    checkpoint_sha256: str
    calibrator_sha256: str
    config_sha256: str
    code_sha256: str
    dataset_manifest_sha256: str
    license_policy_sha256: str
    model_sha256: str
    preprocessing: PreprocessingContract = field(default_factory=PreprocessingContract)
    schema_version: int = 1
    model_format: str = "onnx"
    input_name: str = "image"
    input_shape: tuple[int, int, int, int] = _INPUT_SHAPE
    input_dtype: str = "float32"
    output_name: str = "probability_ai"
    output_shape: tuple[int, int] = _OUTPUT_SHAPE
    output_dtype: str = "float32"
    output_semantics: str = "calibrated_probability_ai"
    calibration: str = "platt_embedded"
    opset_version: int = ONNX_OPSET_VERSION
    single_file: bool = True
    uses_external_data: bool = False
    max_model_bytes: int = MAX_ONNX_BYTES
    allowed_operators: frozenset[str] = SUPPORTED_ONNX_OPERATORS

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("ExportMetadata cannot be subclassed")

    def __post_init__(self) -> None:
        if type(self) is not ExportMetadata:
            raise TypeError("export metadata must be exactly ExportMetadata")
        for field_name in (
            "checkpoint_sha256",
            "calibrator_sha256",
            "config_sha256",
            "code_sha256",
            "dataset_manifest_sha256",
            "license_policy_sha256",
            "model_sha256",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if type(self.preprocessing) is not PreprocessingContract:
            raise TypeError("preprocessing must be a PreprocessingContract")

        expected = {
            "schema_version": 1,
            "model_format": "onnx",
            "input_name": "image",
            "input_shape": _INPUT_SHAPE,
            "input_dtype": "float32",
            "output_name": "probability_ai",
            "output_shape": _OUTPUT_SHAPE,
            "output_dtype": "float32",
            "output_semantics": "calibrated_probability_ai",
            "calibration": "platt_embedded",
            "opset_version": ONNX_OPSET_VERSION,
            "single_file": True,
            "uses_external_data": False,
            "max_model_bytes": MAX_ONNX_BYTES,
            "allowed_operators": SUPPORTED_ONNX_OPERATORS,
        }
        for field_name, expected_value in expected.items():
            actual = getattr(self, field_name)
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise ValueError(f"export {field_name} is fixed")

    def to_json_bytes(self) -> bytes:
        if type(self) is not ExportMetadata:
            raise TypeError("export metadata must be exactly ExportMetadata")
        preprocessing = self.preprocessing
        document = {
            "allowed_operators": sorted(self.allowed_operators),
            "calibration": self.calibration,
            "calibrator_sha256": self.calibrator_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "input_dtype": self.input_dtype,
            "input_name": self.input_name,
            "input_shape": list(self.input_shape),
            "license_policy_sha256": self.license_policy_sha256,
            "max_model_bytes": self.max_model_bytes,
            "model_format": self.model_format,
            "model_sha256": self.model_sha256,
            "opset_version": self.opset_version,
            "output_dtype": self.output_dtype,
            "output_name": self.output_name,
            "output_semantics": self.output_semantics,
            "output_shape": list(self.output_shape),
            "preprocessing": {
                "channel_order": preprocessing.channel_order,
                "input_dtype": preprocessing.input_dtype,
                "input_scale": preprocessing.input_scale,
                "interpolation": preprocessing.interpolation,
                "layout": preprocessing.layout,
                "mean": list(preprocessing.mean),
                "output_dtype": preprocessing.output_dtype,
                "resize": list(preprocessing.resize),
                "standard_deviation": list(preprocessing.standard_deviation),
            },
            "schema_version": self.schema_version,
            "single_file": self.single_file,
            "uses_external_data": self.uses_external_data,
        }
        return (
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class OnnxTensor:
    """One graph input or output described without importing ONNX."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name != self.name.strip()
        ):
            raise ValueError("tensor name must be non-empty and trimmed")
        if type(self.shape) is not tuple or any(
            type(dimension) is not int or dimension <= 0 for dimension in self.shape
        ):
            raise ValueError("tensor shape must contain positive integer dimensions")
        if (
            not isinstance(self.dtype, str)
            or not self.dtype
            or self.dtype != self.dtype.strip()
        ):
            raise ValueError("tensor dtype must be non-empty and trimmed")


@dataclass(frozen=True, slots=True)
class OnnxArtifactSummary:
    """Complete graph facts extracted from an ONNX artifact by a future adapter."""

    model_files: tuple[str, ...]
    model_size_bytes: int
    inputs: tuple[OnnxTensor, ...]
    outputs: tuple[OnnxTensor, ...]
    opset_imports: tuple[tuple[str, int], ...]
    operators: frozenset[tuple[str, str]]
    calibration_embedded: bool
    external_data_files: tuple[str, ...]


def validate_supported_operators(operators: Iterable[tuple[str, str]]) -> None:
    """Reject graph operators outside the runtime-reviewed allowlist."""

    if isinstance(operators, str):
        raise ValueError("operators must be domain and operator pairs")
    observed: set[str] = set()
    for qualified_operator in operators:
        if (
            type(qualified_operator) is not tuple
            or len(qualified_operator) != 2
            or not all(isinstance(value, str) for value in qualified_operator)
        ):
            raise ValueError("operators must be domain and operator pairs")
        domain, operator = qualified_operator
        if domain:
            raise ValueError(f"unsupported ONNX operator domain: {domain}")
        if (
            not isinstance(operator, str)
            or not operator
            or operator != operator.strip()
        ):
            raise ValueError("operator names must be non-empty and trimmed")
        observed.add(operator)
    if not observed:
        raise ValueError("ONNX graph must contain at least one operator")
    unsupported = sorted(observed - SUPPORTED_ONNX_OPERATORS)
    if unsupported:
        raise ValueError(f"unsupported ONNX operators: {', '.join(unsupported)}")


def validate_onnx_artifact(
    artifact: OnnxArtifactSummary, metadata: ExportMetadata
) -> None:
    """Validate an extracted graph summary against the frozen export contract."""

    if not isinstance(artifact, OnnxArtifactSummary):
        raise TypeError("artifact must be an OnnxArtifactSummary")
    if type(metadata) is not ExportMetadata:
        raise TypeError("metadata must be ExportMetadata")
    if (
        type(artifact.model_files) is not tuple
        or any(type(path) is not str for path in artifact.model_files)
        or len(artifact.model_files) != 1
        or not artifact.model_files[0].endswith(".onnx")
        or "/" in artifact.model_files[0]
        or "\\" in artifact.model_files[0]
    ):
        raise ValueError("export must contain a single ONNX file")
    if type(artifact.external_data_files) is not tuple or artifact.external_data_files:
        raise ValueError("ONNX external data is forbidden")
    if (
        isinstance(artifact.model_size_bytes, bool)
        or not isinstance(artifact.model_size_bytes, int)
        or artifact.model_size_bytes <= 0
        or artifact.model_size_bytes > metadata.max_model_bytes
    ):
        raise ValueError("ONNX model exceeds or violates the size budget")
    expected_input = OnnxTensor(
        metadata.input_name, metadata.input_shape, metadata.input_dtype
    )
    if type(artifact.inputs) is not tuple or artifact.inputs != (expected_input,):
        raise ValueError("ONNX graph must have the single image input from metadata")
    expected_output = OnnxTensor(
        metadata.output_name, metadata.output_shape, metadata.output_dtype
    )
    if type(artifact.outputs) is not tuple or artifact.outputs != (expected_output,):
        raise ValueError("ONNX graph must have a single probability_ai output")
    expected_opsets = (("", metadata.opset_version),)
    if (
        type(artifact.opset_imports) is not tuple
        or artifact.opset_imports != expected_opsets
        or any(
            type(domain) is not str or type(version) is not int
            for domain, version in artifact.opset_imports
        )
    ):
        raise ValueError("ONNX opset imports do not match export metadata")
    if (
        type(artifact.calibration_embedded) is not bool
        or not artifact.calibration_embedded
    ):
        raise ValueError("Platt calibration must be embedded in the ONNX graph")
    if type(artifact.operators) is not frozenset:
        raise ValueError("operators must be a frozen set of domain and operator pairs")
    validate_supported_operators(artifact.operators)
