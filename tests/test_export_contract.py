from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
import json
import unittest

from poidh_detector.export import (
    MAX_ONNX_BYTES,
    ONNX_OPSET_VERSION,
    SUPPORTED_ONNX_OPERATORS,
    ExportMetadata,
    OnnxArtifactSummary,
    OnnxTensor,
    PreprocessingContract,
    validate_onnx_artifact,
    validate_supported_operators,
)


def _metadata(**overrides: object) -> ExportMetadata:
    values: dict[str, object] = {
        "checkpoint_sha256": "1" * 64,
        "calibrator_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "code_sha256": "4" * 64,
        "dataset_manifest_sha256": "5" * 64,
        "license_policy_sha256": "6" * 64,
        "model_sha256": "7" * 64,
    }
    values.update(overrides)
    return ExportMetadata(**values)  # type: ignore[arg-type]


def _artifact(**overrides: object) -> OnnxArtifactSummary:
    values: dict[str, object] = {
        "model_files": ("detector.onnx",),
        "model_size_bytes": 50 * 1024 * 1024,
        "inputs": (OnnxTensor("image", (1, 3, 224, 224), "float32"),),
        "outputs": (OnnxTensor("probability_ai", (1, 1), "float32"),),
        "opset_imports": (("", ONNX_OPSET_VERSION),),
        "operators": frozenset(
            {
                ("", "Conv"),
                ("", "LayerNormalization"),
                ("", "Sigmoid"),
            }
        ),
        "calibration_embedded": True,
        "external_data_files": (),
    }
    values.update(overrides)
    return OnnxArtifactSummary(**values)  # type: ignore[arg-type]


class OnnxExportContractTests(unittest.TestCase):
    def test_freezes_preprocessing_and_calibrated_single_output_contract(self) -> None:
        metadata = _metadata()

        self.assertEqual(metadata.preprocessing.resize, (224, 224))
        self.assertEqual(metadata.preprocessing.channel_order, "RGB")
        self.assertEqual(metadata.preprocessing.layout, "NCHW")
        self.assertEqual(metadata.preprocessing.input_scale, 1.0 / 255.0)
        self.assertEqual(metadata.input_name, "image")
        self.assertEqual(metadata.input_shape, (1, 3, 224, 224))
        self.assertEqual(metadata.output_name, "probability_ai")
        self.assertEqual(metadata.output_shape, (1, 1))
        self.assertEqual(metadata.output_semantics, "calibrated_probability_ai")
        self.assertEqual(metadata.calibration, "platt_embedded")
        self.assertEqual(metadata.opset_version, ONNX_OPSET_VERSION)

        with self.assertRaises(FrozenInstanceError):
            metadata.preprocessing.resize = (256, 256)  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "resize"):
            PreprocessingContract(resize=(256, 256))

    def test_rejects_preprocessing_subclasses_that_extend_the_schema(self) -> None:
        class ExtendedPreprocessing(PreprocessingContract):
            pass

        with self.assertRaisesRegex(TypeError, "PreprocessingContract"):
            _metadata(preprocessing=ExtendedPreprocessing())

    def test_rejects_export_metadata_subclasses_that_extend_the_schema(self) -> None:
        with self.assertRaisesRegex(TypeError, "cannot be subclassed"):

            @dataclass(frozen=True, slots=True)
            class ExtendedMetadata(ExportMetadata):
                extra: str = "unexpected"

        with self.assertRaisesRegex(TypeError, "cannot be subclassed"):

            class BypassMetadata(ExportMetadata):
                def __post_init__(self) -> None:
                    pass

    def test_requires_every_lowercase_sha256_provenance_digest(self) -> None:
        fields = (
            "checkpoint_sha256",
            "calibrator_sha256",
            "config_sha256",
            "code_sha256",
            "dataset_manifest_sha256",
            "license_policy_sha256",
            "model_sha256",
        )
        for field_name in fields:
            for invalid in ("f" * 63, "F" * 64, True):
                with self.subTest(field_name=field_name, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, field_name):
                        _metadata(**{field_name: invalid})

    def test_fixed_metadata_rejects_type_aliases(self) -> None:
        for field_name, invalid in (
            ("schema_version", True),
            ("opset_version", 18.0),
            ("single_file", 1),
            ("uses_external_data", 0),
            ("max_model_bytes", float(MAX_ONNX_BYTES)),
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    _metadata(**{field_name: invalid})

    def test_serializes_canonical_complete_export_metadata(self) -> None:
        document = json.loads(_metadata().to_json_bytes())

        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["model_format"], "onnx")
        self.assertEqual(document["output_name"], "probability_ai")
        self.assertEqual(document["max_model_bytes"], MAX_ONNX_BYTES)
        self.assertTrue(document["single_file"])
        self.assertFalse(document["uses_external_data"])
        self.assertEqual(
            document["allowed_operators"], sorted(SUPPORTED_ONNX_OPERATORS)
        )
        for field_name in (
            "checkpoint_sha256",
            "calibrator_sha256",
            "config_sha256",
            "code_sha256",
            "dataset_manifest_sha256",
            "license_policy_sha256",
            "model_sha256",
        ):
            self.assertIn(field_name, document)

    def test_accepts_artifact_that_matches_the_export_contract(self) -> None:
        validate_onnx_artifact(_artifact(), _metadata())

    def test_rejects_wrong_io_opset_or_missing_embedded_calibration(self) -> None:
        cases = (
            (
                replace(
                    _artifact(),
                    inputs=(OnnxTensor("pixels", (1, 3, 224, 224), "float32"),),
                ),
                "single image input",
            ),
            (
                replace(
                    _artifact(),
                    inputs=(OnnxTensor("image", (1, 3, 256, 256), "float32"),),
                ),
                "single image input",
            ),
            (
                replace(
                    _artifact(),
                    inputs=(
                        OnnxTensor("image", (1, 3, 224, 224), "float32"),
                        OnnxTensor("training", (), "bool"),
                    ),
                ),
                "single image input",
            ),
            (
                replace(
                    _artifact(),
                    outputs=(OnnxTensor("logit", (1, 1), "float32"),),
                ),
                "single probability_ai output",
            ),
            (
                replace(
                    _artifact(),
                    outputs=(OnnxTensor("probability_ai", (1, 2), "float32"),),
                ),
                "single probability_ai output",
            ),
            (
                replace(
                    _artifact(),
                    opset_imports=(("", ONNX_OPSET_VERSION - 1),),
                ),
                "opset imports",
            ),
            (
                replace(_artifact(), calibration_embedded=False),
                "calibration",
            ),
        )
        for artifact, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_onnx_artifact(artifact, _metadata())

    def test_rejects_non_integer_tensor_shape_aliases(self) -> None:
        for shape in ((1.0, 3, 224, 224), (True, 1)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "tensor shape"):
                    OnnxTensor("image", shape, "float32")  # type: ignore[arg-type]

    def test_enforces_single_file_no_external_data_and_size_budget(self) -> None:
        cases = (
            (
                replace(_artifact(), model_files=("model.onnx", "weights.bin")),
                "single ONNX file",
            ),
            (
                replace(_artifact(), external_data_files=("weights.bin",)),
                "external data",
            ),
            (
                replace(_artifact(), external_data_files=None),  # type: ignore[arg-type]
                "external data",
            ),
            (
                replace(_artifact(), model_size_bytes=MAX_ONNX_BYTES + 1),
                "size budget",
            ),
        )
        for artifact, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_onnx_artifact(artifact, _metadata())

    def test_operator_allowlist_reports_all_unsupported_operators(self) -> None:
        validate_supported_operators({("", "Conv"), ("", "Sigmoid")})

        with self.assertRaisesRegex(ValueError, "ExperimentalAttention, PythonOp"):
            validate_supported_operators(
                {
                    ("", "Conv"),
                    ("", "PythonOp"),
                    ("", "ExperimentalAttention"),
                }
            )

    def test_rejects_custom_operator_domains_and_imports(self) -> None:
        with self.assertRaisesRegex(ValueError, "operator domain"):
            validate_supported_operators({("com.example", "Conv")})

        with self.assertRaisesRegex(ValueError, "opset imports"):
            validate_onnx_artifact(
                replace(
                    _artifact(),
                    opset_imports=(
                        ("", ONNX_OPSET_VERSION),
                        ("com.example", 1),
                    ),
                ),
                _metadata(),
            )

        with self.assertRaisesRegex(ValueError, "operator domain"):
            validate_onnx_artifact(
                replace(
                    _artifact(),
                    operators=frozenset({("com.example", "Conv")}),
                ),
                _metadata(),
            )


if __name__ == "__main__":
    unittest.main()
