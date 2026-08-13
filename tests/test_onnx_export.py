from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from poidh_detector.calibration import (
    CalibrationClassCounts,
    PlattCalibrationArtifact,
)
from poidh_detector.calibration_fit import CalibrationFitResult, CalibrationMetrics
from poidh_detector.onnx_export import export_detector_onnx
from poidh_detector.training import TrainingConfig
from tools.export_detector_onnx import main


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _config(dataset_manifest: bytes) -> TrainingConfig:
    return TrainingConfig(
        dataset_manifest_sha256=_sha256(dataset_manifest),
        split_manifest_sha256="b" * 64,
        calibration_split_sha256="c" * 64,
        exposed_holdout_sha256=("d" * 64,),
        seed=323,
    )


class _Expression:
    def __init__(self, value: object) -> None:
        self.value = value

    def reshape(self, *shape: int) -> _Expression:
        return _Expression(("reshape", self.value, shape))

    def __mul__(self, other: object) -> _Expression:
        value = other.value if isinstance(other, _Expression) else other
        return _Expression(("mul", self.value, value))

    def __rmul__(self, other: object) -> _Expression:
        return self * other

    def __add__(self, other: object) -> _Expression:
        value = other.value if isinstance(other, _Expression) else other
        return _Expression(("add", self.value, value))

    def __radd__(self, other: object) -> _Expression:
        return self + other


class _FakeModule:
    def __init__(self) -> None:
        self.training = True
        self.requires_grad = True

    def __call__(self, *arguments: object) -> object:
        return self.forward(*arguments)

    def eval(self) -> _FakeModule:
        self.training = False
        return self

    def requires_grad_(self, value: bool) -> _FakeModule:
        self.requires_grad = value
        return self

    def register_buffer(self, name: str, value: object) -> None:
        setattr(self, name, value)


class _FakeDetector(_FakeModule):
    def __init__(self) -> None:
        super().__init__()
        self.loaded: tuple[object, bool] | None = None

    def load_state_dict(self, state: object, *, strict: bool) -> None:
        self.loaded = (state, strict)

    def forward(self, image: object) -> _Expression:
        return _Expression(("raw_logit", image))


class _FakeOnnxExporter:
    def __init__(self, *, create_external_data: bool = False) -> None:
        self.calls: list[tuple[object, object, Path, dict[str, object]]] = []
        self.create_external_data = create_external_data

    def export(
        self, model: object, arguments: object, destination: Path, **kwargs: object
    ) -> None:
        self.calls.append((model, arguments, destination, kwargs))
        destination.write_bytes(b"onnx-model")
        if self.create_external_data:
            destination.with_name(destination.name + ".data").write_bytes(b"weights")


class _FakeTorch:
    float32 = "float32"

    def __init__(self, *, create_external_data: bool = False) -> None:
        self.nn = SimpleNamespace(Module=_FakeModule)
        self.onnx = _FakeOnnxExporter(create_external_data=create_external_data)
        self.load_calls: list[tuple[Path, str, bool]] = []

    def load(self, path: Path, *, map_location: str, weights_only: bool) -> object:
        self.load_calls.append((path, map_location, weights_only))
        return {"head.weight": _Expression("weights")}

    def tensor(self, value: float, *, dtype: object) -> _Expression:
        self.assert_float32(dtype)
        return _Expression(value)

    def zeros(self, shape: tuple[int, ...], *, dtype: object) -> _Expression:
        self.assert_float32(dtype)
        return _Expression(("zeros", shape, dtype))

    def sigmoid(self, value: _Expression) -> _Expression:
        return _Expression(("sigmoid", value.value))

    @staticmethod
    def assert_float32(dtype: object) -> None:
        if dtype != "float32":
            raise AssertionError(f"unexpected dtype: {dtype}")


class _FakeTimm:
    def __init__(self, model: _FakeDetector) -> None:
        self.model = model
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create_model(self, name: str, **kwargs: object) -> _FakeDetector:
        self.calls.append((name, kwargs))
        return self.model


def _missing_onnx(name: str) -> object:
    error = ModuleNotFoundError(name)
    error.name = name
    raise error


class _Dimension:
    def __init__(self, value: int) -> None:
        self.dim_value = value

    def HasField(self, field: str) -> bool:
        return field == "dim_value"


def _value_info(name: str, shape: tuple[int, ...]) -> object:
    tensor_type = SimpleNamespace(
        elem_type=1,
        shape=SimpleNamespace(dim=[_Dimension(value) for value in shape]),
    )
    return SimpleNamespace(name=name, type=SimpleNamespace(tensor_type=tensor_type))


class _FakeOnnx:
    class TensorProto:
        FLOAT = 1
        EXTERNAL = 1

    def __init__(self) -> None:
        self.checked: list[tuple[Path, bool]] = []
        self.checker = SimpleNamespace(check_model=self._check_model)
        self.model = SimpleNamespace(
            graph=SimpleNamespace(
                input=[_value_info("image", (1, 3, 224, 224))],
                output=[_value_info("probability_ai", (1, 1))],
                node=[
                    SimpleNamespace(domain="", op_type="Conv"),
                    SimpleNamespace(domain="", op_type="Sigmoid"),
                ],
                initializer=[],
            ),
            opset_import=[SimpleNamespace(domain="", version=18)],
        )

    def load(self, path: Path, *, load_external_data: bool) -> object:
        if load_external_data:
            raise AssertionError("external ONNX data must not be loaded")
        return self.model

    def _check_model(self, path: Path, *, full_check: bool) -> None:
        self.checked.append((path, full_check))


class OnnxExporterTests(unittest.TestCase):
    def _inputs(
        self,
        root: Path,
        *,
        calibration_checkpoint_sha256: str | None = None,
        calibration_config: TrainingConfig | None = None,
    ) -> dict[str, Path]:
        root.mkdir(parents=True, exist_ok=True)
        dataset_manifest = b'[{"frozen":"dataset"}]\n'
        checkpoint = b"frozen-convnextv2-nano-state"
        config = _config(dataset_manifest)
        calibration = CalibrationFitResult(
            artifact=PlattCalibrationArtifact(
                scale=1.75,
                bias=-0.25,
                checkpoint_sha256=(
                    calibration_checkpoint_sha256 or _sha256(checkpoint)
                ),
                calibration_split_sha256=(
                    calibration_config or config
                ).calibration_split_sha256,
                training_config=calibration_config or config,
                input_identifier="calibration",
                sample_count=4,
                class_counts=CalibrationClassCounts(real=2, ai=2),
            ),
            predictions_sha256="e" * 64,
            uncalibrated=CalibrationMetrics(
                bce=0.7,
                ece=0.2,
                accuracy_at_threshold=0.5,
            ),
            calibrated=CalibrationMetrics(
                bce=0.6,
                ece=0.1,
                accuracy_at_threshold=0.75,
            ),
        )
        payloads = {
            "checkpoint": checkpoint,
            "calibrator": calibration.to_json_bytes(),
            "training_config": config.to_json_bytes(),
            "dataset_manifest": dataset_manifest,
            "code_provenance": b"git-tree:0123456789abcdef\n",
            "license_policy": b"approved-license-policy-v1\n",
        }
        paths: dict[str, Path] = {}
        for name, payload in payloads.items():
            path = root / f"{name}.bin"
            path.write_bytes(payload)
            paths[name] = path
        paths["output"] = root / "export"
        return paths

    def test_exports_fixed_calibrated_graph_and_canonical_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._inputs(root)
            torch = _FakeTorch()
            detector = _FakeDetector()
            timm = _FakeTimm(detector)

            metadata = export_detector_onnx(
                **paths,
                torch_module=torch,
                timm_module=timm,
                import_module=_missing_onnx,
            )

            self.assertEqual(
                set(paths["output"].iterdir()),
                {
                    paths["output"] / "detector.onnx",
                    paths["output"] / "metadata.json",
                },
            )
            self.assertEqual(
                (paths["output"] / "metadata.json").read_bytes(),
                metadata.to_json_bytes(),
            )
            document = json.loads(metadata.to_json_bytes())
            for field, path_name in (
                ("checkpoint_sha256", "checkpoint"),
                ("calibrator_sha256", "calibrator"),
                ("config_sha256", "training_config"),
                ("code_sha256", "code_provenance"),
                ("dataset_manifest_sha256", "dataset_manifest"),
                ("license_policy_sha256", "license_policy"),
            ):
                self.assertEqual(
                    document[field], _sha256(paths[path_name].read_bytes())
                )

            self.assertEqual(
                timm.calls,
                [
                    (
                        "convnextv2_nano",
                        {"pretrained": False, "num_classes": 1, "in_chans": 3},
                    )
                ],
            )
            self.assertEqual(torch.load_calls, [(paths["checkpoint"], "cpu", True)])
            self.assertEqual(
                detector.loaded, ({"head.weight": unittest.mock.ANY}, True)
            )
            self.assertFalse(detector.training)
            self.assertFalse(detector.requires_grad)

            exported_model, arguments, _, options = torch.onnx.calls[0]
            self.assertEqual(arguments.value, ("zeros", (1, 3, 224, 224), "float32"))
            self.assertEqual(options["input_names"], ["image"])
            self.assertEqual(options["output_names"], ["probability_ai"])
            self.assertEqual(options["opset_version"], 18)
            self.assertIs(options["external_data"], False)
            self.assertIsNone(options["dynamic_axes"])
            probability = exported_model(arguments)
            self.assertEqual(
                probability.value,
                (
                    "sigmoid",
                    (
                        "add",
                        (
                            "mul",
                            1.75,
                            ("reshape", ("raw_logit", arguments), (1, 1)),
                        ),
                        -0.25,
                    ),
                ),
            )

    def test_checks_export_with_onnx_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._inputs(Path(temporary))
            onnx = _FakeOnnx()

            export_detector_onnx(
                **paths,
                torch_module=_FakeTorch(),
                timm_module=_FakeTimm(_FakeDetector()),
                import_module=lambda name: onnx
                if name == "onnx"
                else _missing_onnx(name),
            )

            self.assertEqual(len(onnx.checked), 1)
            self.assertTrue(onnx.checked[0][1])

    def test_fails_closed_on_provenance_mismatch_before_loading_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                self._inputs(
                    root / "checkpoint", calibration_checkpoint_sha256="f" * 64
                ),
                self._inputs(
                    root / "config",
                    calibration_config=TrainingConfig(
                        dataset_manifest_sha256="1" * 64,
                        split_manifest_sha256="2" * 64,
                        calibration_split_sha256="3" * 64,
                        exposed_holdout_sha256=("4" * 64,),
                        seed=1,
                    ),
                ),
            )
            for paths in cases:
                with self.subTest(output=paths["output"]):
                    torch = _FakeTorch()
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        export_detector_onnx(
                            **paths,
                            torch_module=torch,
                            timm_module=_FakeTimm(_FakeDetector()),
                            import_module=_missing_onnx,
                        )
                    self.assertEqual(torch.load_calls, [])
                    self.assertFalse(paths["output"].exists())

    def test_refuses_existing_output_and_external_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._inputs(root / "existing")
            paths["output"].mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                export_detector_onnx(
                    **paths,
                    torch_module=_FakeTorch(),
                    timm_module=_FakeTimm(_FakeDetector()),
                    import_module=_missing_onnx,
                )

            external_paths = self._inputs(root / "external")
            with self.assertRaisesRegex(ValueError, "external data"):
                export_detector_onnx(
                    **external_paths,
                    torch_module=_FakeTorch(create_external_data=True),
                    timm_module=_FakeTimm(_FakeDetector()),
                    import_module=_missing_onnx,
                )
            self.assertFalse(external_paths["output"].exists())

    @patch("tools.export_detector_onnx.export_detector_onnx")
    def test_cli_forwards_all_frozen_inputs(self, export: MagicMock) -> None:
        arguments = [
            "--checkpoint",
            "best-model.pt",
            "--calibrator",
            "calibration.json",
            "--training-config",
            "training-config.json",
            "--dataset-manifest",
            "manifest.json",
            "--code-provenance",
            "code.txt",
            "--license-policy",
            "license-policy.json",
            "--output",
            "onnx-export",
        ]

        self.assertEqual(main(arguments), 0)

        export.assert_called_once_with(
            checkpoint=Path("best-model.pt"),
            calibrator=Path("calibration.json"),
            training_config=Path("training-config.json"),
            dataset_manifest=Path("manifest.json"),
            code_provenance=Path("code.txt"),
            license_policy=Path("license-policy.json"),
            output=Path("onnx-export"),
        )


if __name__ == "__main__":
    unittest.main()
