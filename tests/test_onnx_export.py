from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from poidh_detector.calibration import (
    CalibrationClassCounts,
    PlattCalibrationArtifact,
)
from poidh_detector.calibration_fit import CalibrationFitResult, CalibrationMetrics
from poidh_detector import onnx_export
from poidh_detector.export import ExportMetadata
from poidh_detector.model import ConvNeXtV2NanoConfig, create_convnextv2_nano
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
        self.load_calls: list[tuple[object, str, bool]] = []
        self.loaded_payloads: list[bytes | None] = []

    def load(self, source: object, *, map_location: str, weights_only: bool) -> object:
        self.load_calls.append((source, map_location, weights_only))
        read = getattr(source, "read", None)
        self.loaded_payloads.append(read() if callable(read) else None)
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
                import_module=lambda name: _FakeOnnx()
                if name == "onnx"
                else _missing_onnx(name),
            )

            self.assertEqual(
                set(paths["output"].iterdir()),
                {
                    paths["output"] / "detector.onnx",
                    paths["output"] / "metadata.json",
                    paths["output"] / "READY",
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
                document["model_sha256"],
                _sha256((paths["output"] / "detector.onnx").read_bytes()),
            )
            self.assertEqual(
                (paths["output"] / "READY").read_bytes(),
                (_sha256(metadata.to_json_bytes()) + "\n").encode("ascii"),
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
            self.assertEqual(len(torch.load_calls), 1)
            self.assertIsInstance(torch.load_calls[0][0], io.BytesIO)
            self.assertEqual(torch.loaded_payloads, [paths["checkpoint"].read_bytes()])
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

    def test_requires_onnx_before_loading_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._inputs(Path(temporary))
            torch = _FakeTorch()
            timm = _FakeTimm(_FakeDetector())

            with self.assertRaisesRegex(RuntimeError, "onnx.*training extras"):
                export_detector_onnx(
                    **paths,
                    torch_module=torch,
                    timm_module=timm,
                    import_module=_missing_onnx,
                )

            self.assertEqual(torch.load_calls, [])
            self.assertEqual(timm.calls, [])
            self.assertFalse(paths["output"].exists())

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
                    import_module=lambda name: _FakeOnnx()
                    if name == "onnx"
                    else _missing_onnx(name),
                )
            self.assertFalse(external_paths["output"].exists())

    def test_atomic_reservation_does_not_clobber_a_concurrent_destination(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._inputs(Path(temporary))
            destination = paths["output"].absolute()
            real_rename = onnx_export._rename_directory_no_replace
            injected = False

            def rename_with_collision(
                parent_descriptor: int,
                source_name: str,
                destination_name: str,
            ) -> None:
                nonlocal injected
                injected = True
                destination.mkdir()
                real_rename(parent_descriptor, source_name, destination_name)

            with patch.object(
                onnx_export,
                "_rename_directory_no_replace",
                rename_with_collision,
            ):
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    export_detector_onnx(
                        **paths,
                        torch_module=_FakeTorch(),
                        timm_module=_FakeTimm(_FakeDetector()),
                        import_module=lambda name: _FakeOnnx()
                        if name == "onnx"
                        else _missing_onnx(name),
                    )

            self.assertTrue(injected)
            self.assertEqual(list(destination.iterdir()), [])

    def test_fsyncs_bundle_and_detects_model_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._inputs(Path(temporary))
            with patch.object(os, "fsync", wraps=os.fsync) as fsync:
                metadata = export_detector_onnx(
                    **paths,
                    torch_module=_FakeTorch(),
                    timm_module=_FakeTimm(_FakeDetector()),
                    import_module=lambda name: _FakeOnnx()
                    if name == "onnx"
                    else _missing_onnx(name),
                )

            self.assertGreaterEqual(fsync.call_count, 4)
            self.assertEqual(
                onnx_export.validate_export_bundle(paths["output"]), metadata
            )
            with (paths["output"] / "detector.onnx").open("ab") as stream:
                stream.write(b"mutated")
            with self.assertRaisesRegex(ValueError, "model digest mismatch"):
                onnx_export.validate_export_bundle(paths["output"])

    def test_cleanup_never_deletes_replaced_destination(self) -> None:
        for replacement_stage in ("before-rename", "after-rename"):
            with self.subTest(replacement_stage=replacement_stage):
                with tempfile.TemporaryDirectory() as temporary:
                    paths = self._inputs(Path(temporary))
                    destination = paths["output"].absolute()
                    displaced = destination.with_name("displaced-export")
                    foreign_file = destination / "foreign.txt"
                    real_rename = onnx_export._rename_directory_no_replace

                    def replace_around_rename(
                        parent_descriptor: int,
                        source_name: str,
                        destination_name: str,
                    ) -> None:
                        if replacement_stage == "before-rename":
                            destination.mkdir()
                            foreign_file.write_text("foreign", encoding="ascii")
                            real_rename(
                                parent_descriptor, source_name, destination_name
                            )
                            return
                        real_rename(parent_descriptor, source_name, destination_name)
                        destination.rename(displaced)
                        destination.mkdir()
                        foreign_file.write_text("foreign", encoding="ascii")
                        raise RuntimeError("fault after rename")

                    with patch.object(
                        onnx_export,
                        "_rename_directory_no_replace",
                        replace_around_rename,
                    ):
                        expected = (
                            FileExistsError
                            if replacement_stage == "before-rename"
                            else RuntimeError
                        )
                        with self.assertRaises(expected):
                            export_detector_onnx(
                                **paths,
                                torch_module=_FakeTorch(),
                                timm_module=_FakeTimm(_FakeDetector()),
                                import_module=lambda name: _FakeOnnx()
                                if name == "onnx"
                                else _missing_onnx(name),
                            )

                    self.assertTrue(destination.is_dir())
                    self.assertEqual(
                        foreign_file.read_text(encoding="ascii"), "foreign"
                    )
                    self.assertFalse((destination / "READY").exists())

    def test_publication_does_not_write_into_destination_replaced_during_token(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._inputs(Path(temporary))
            destination = paths["output"].absolute()
            displaced = destination.with_name("displaced-export")
            foreign_model = destination / "detector.onnx"

            def replace_destination(_: int) -> str:
                destination.mkdir()
                foreign_model.write_bytes(b"foreign-model")
                return "a" * 64

            with patch.object(onnx_export.secrets, "token_hex", replace_destination):
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    export_detector_onnx(
                        **paths,
                        torch_module=_FakeTorch(),
                        timm_module=_FakeTimm(_FakeDetector()),
                        import_module=lambda name: _FakeOnnx()
                        if name == "onnx"
                        else _missing_onnx(name),
                    )

            self.assertEqual(foreign_model.read_bytes(), b"foreign-model")
            self.assertFalse((destination / "READY").exists())
            self.assertFalse(displaced.exists())

    def test_publication_does_not_acquire_a_destination_replaced_after_mkdir(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._inputs(Path(temporary))
            destination = paths["output"].absolute()
            displaced = destination.with_name("displaced-export")
            foreign_model = destination / "detector.onnx"
            real_mkdir = Path.mkdir
            destination_mkdir_called = False

            def replace_after_destination_mkdir(
                path: Path,
                mode: int = 0o777,
                parents: bool = False,
                exist_ok: bool = False,
            ) -> None:
                nonlocal destination_mkdir_called
                real_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)
                if path.absolute() == destination:
                    destination_mkdir_called = True
                    destination.rename(displaced)
                    real_mkdir(destination)
                    foreign_model.write_bytes(b"foreign-model")

            with patch.object(Path, "mkdir", replace_after_destination_mkdir):
                metadata = export_detector_onnx(
                    **paths,
                    torch_module=_FakeTorch(),
                    timm_module=_FakeTimm(_FakeDetector()),
                    import_module=lambda name: _FakeOnnx()
                    if name == "onnx"
                    else _missing_onnx(name),
                )

            self.assertFalse(destination_mkdir_called)
            self.assertFalse(displaced.exists())
            self.assertEqual(onnx_export.validate_export_bundle(destination), metadata)

    def test_cleanup_does_not_delete_replacement_after_ownership_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._inputs(Path(temporary))
            destination = paths["output"].absolute()
            displaced = destination.with_name("displaced-export")
            foreign_file = destination / "foreign.txt"
            real_require_identity = onnx_export._require_directory_identity
            replaced = False

            def replace_after_identity(path: Path, descriptor: int) -> None:
                nonlocal replaced
                real_require_identity(path, descriptor)
                if path == destination and not replaced:
                    replaced = True
                    destination.rename(displaced)
                    destination.mkdir()
                    foreign_file.write_text("foreign", encoding="ascii")
                    raise RuntimeError("publish fault")

            with patch.object(
                onnx_export, "_require_directory_identity", replace_after_identity
            ):
                with self.assertRaisesRegex(RuntimeError, "publish fault"):
                    export_detector_onnx(
                        **paths,
                        torch_module=_FakeTorch(),
                        timm_module=_FakeTimm(_FakeDetector()),
                        import_module=lambda name: _FakeOnnx()
                        if name == "onnx"
                        else _missing_onnx(name),
                    )

            self.assertEqual(foreign_file.read_text(encoding="ascii"), "foreign")
            self.assertTrue(displaced.is_dir())

    def test_publication_fails_if_destination_is_replaced_during_final_parent_fsync(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._inputs(Path(temporary))
            destination = paths["output"].absolute()
            displaced = destination.with_name("displaced-export")
            foreign_file = destination / "foreign.txt"
            real_fsync = os.fsync
            replaced = False

            def replace_during_final_parent_fsync(descriptor: int) -> None:
                nonlocal replaced
                real_fsync(descriptor)
                if destination.exists() and not replaced:
                    parent_stat = os.stat(destination.parent)
                    descriptor_stat = os.fstat(descriptor)
                    if (parent_stat.st_dev, parent_stat.st_ino) == (
                        descriptor_stat.st_dev,
                        descriptor_stat.st_ino,
                    ):
                        replaced = True
                        destination.rename(displaced)
                        destination.mkdir()
                        foreign_file.write_text("foreign", encoding="ascii")

            with patch.object(os, "fsync", replace_during_final_parent_fsync):
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    export_detector_onnx(
                        **paths,
                        torch_module=_FakeTorch(),
                        timm_module=_FakeTimm(_FakeDetector()),
                        import_module=lambda name: _FakeOnnx()
                        if name == "onnx"
                        else _missing_onnx(name),
                    )

            self.assertEqual(foreign_file.read_text(encoding="ascii"), "foreign")
            self.assertFalse((displaced / "READY").exists())

    def test_publication_fails_if_destination_is_replaced_during_staging_cleanup(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._inputs(Path(temporary))
            destination = paths["output"].absolute()
            displaced = destination.with_name("displaced-export")
            foreign_file = destination / "foreign.txt"
            real_fsync = os.fsync
            published_parent_syncs = 0

            def replace_during_staging_cleanup(descriptor: int) -> None:
                nonlocal published_parent_syncs
                real_fsync(descriptor)
                if destination.exists():
                    parent_stat = os.stat(destination.parent)
                    descriptor_stat = os.fstat(descriptor)
                    if (parent_stat.st_dev, parent_stat.st_ino) == (
                        descriptor_stat.st_dev,
                        descriptor_stat.st_ino,
                    ):
                        published_parent_syncs += 1
                    if published_parent_syncs == 2:
                        destination.rename(displaced)
                        destination.mkdir()
                        foreign_file.write_text("foreign", encoding="ascii")

            with patch.object(os, "fsync", replace_during_staging_cleanup):
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    export_detector_onnx(
                        **paths,
                        torch_module=_FakeTorch(),
                        timm_module=_FakeTimm(_FakeDetector()),
                        import_module=lambda name: _FakeOnnx()
                        if name == "onnx"
                        else _missing_onnx(name),
                    )

            self.assertEqual(foreign_file.read_text(encoding="ascii"), "foreign")
            self.assertTrue((displaced / "READY").is_file())

    def test_staging_cleanup_does_not_delete_a_replacement_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._inputs(Path(temporary))
            staging_path: Path | None = None
            displaced_staging: Path | None = None
            foreign_file: Path | None = None
            real_publish_bundle = onnx_export._publish_bundle

            def replace_staging_before_publication(
                staging: Path, *arguments: object
            ) -> int:
                nonlocal staging_path, displaced_staging, foreign_file
                staging_path = staging
                displaced_staging = staging.with_name(f"{staging.name}.displaced")
                staging.rename(displaced_staging)
                staging.mkdir()
                foreign_file = staging / "foreign.txt"
                foreign_file.write_text("foreign", encoding="ascii")
                return real_publish_bundle(staging, *arguments)

            with patch.object(
                onnx_export,
                "_publish_bundle",
                replace_staging_before_publication,
            ):
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    export_detector_onnx(
                        **paths,
                        torch_module=_FakeTorch(),
                        timm_module=_FakeTimm(_FakeDetector()),
                        import_module=lambda name: _FakeOnnx()
                        if name == "onnx"
                        else _missing_onnx(name),
                    )

            assert staging_path is not None
            assert displaced_staging is not None
            assert foreign_file is not None
            self.assertEqual(foreign_file.read_text(encoding="ascii"), "foreign")
            self.assertTrue(displaced_staging.is_dir())

    def test_publication_does_not_move_files_from_a_replaced_staging_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._inputs(Path(temporary))
            destination = paths["output"].absolute()
            staging_replacement: Path | None = None
            displaced_staging: Path | None = None
            real_publish_bundle = onnx_export._publish_bundle

            def replace_staging_before_publication(
                staging: Path, *arguments: object
            ) -> int:
                nonlocal staging_replacement, displaced_staging
                metadata = next(
                    argument
                    for argument in arguments
                    if isinstance(argument, ExportMetadata)
                )
                displaced_staging = staging.with_name(f"{staging.name}.displaced")
                staging.rename(displaced_staging)
                staging.mkdir()
                staging_replacement = staging
                (staging / "detector.onnx").write_bytes(b"foreign-model")
                (staging / "metadata.json").write_bytes(metadata.to_json_bytes())
                return real_publish_bundle(staging, *arguments)

            with patch.object(
                onnx_export,
                "_publish_bundle",
                replace_staging_before_publication,
            ):
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    export_detector_onnx(
                        **paths,
                        torch_module=_FakeTorch(),
                        timm_module=_FakeTimm(_FakeDetector()),
                        import_module=lambda name: _FakeOnnx()
                        if name == "onnx"
                        else _missing_onnx(name),
                    )

            assert staging_replacement is not None
            assert displaced_staging is not None
            self.assertEqual(
                (staging_replacement / "detector.onnx").read_bytes(),
                b"foreign-model",
            )
            self.assertTrue(displaced_staging.is_dir())
            self.assertFalse((destination / "READY").exists())

    def test_publication_strips_ready_from_source_replaced_at_atomic_rename(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._inputs(Path(temporary))
            destination = paths["output"].absolute()
            foreign_model = destination / "detector.onnx"
            displaced_staging: Path | None = None
            real_rename = onnx_export._rename_directory_no_replace

            def replace_source_at_rename(
                parent_descriptor: int,
                source_name: str,
                destination_name: str,
            ) -> None:
                nonlocal displaced_staging
                parent = destination.parent
                source = parent / source_name
                displaced_staging = source.with_name(f"{source.name}.displaced")
                source.rename(displaced_staging)
                source.mkdir()
                (source / "detector.onnx").write_bytes(b"foreign-model")
                (source / "metadata.json").write_bytes(b"foreign-metadata")
                (source / "READY").write_bytes(b"foreign-ready")
                real_rename(parent_descriptor, source_name, destination_name)

            with patch.object(
                onnx_export,
                "_rename_directory_no_replace",
                replace_source_at_rename,
            ):
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    export_detector_onnx(
                        **paths,
                        torch_module=_FakeTorch(),
                        timm_module=_FakeTimm(_FakeDetector()),
                        import_module=lambda name: _FakeOnnx()
                        if name == "onnx"
                        else _missing_onnx(name),
                    )

            assert displaced_staging is not None
            self.assertEqual(foreign_model.read_bytes(), b"foreign-model")
            self.assertFalse((destination / "READY").exists())
            self.assertTrue(displaced_staging.is_dir())

    def test_staging_cleanup_does_not_remove_an_empty_replacement_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._inputs(Path(temporary))
            staging_path: Path | None = None
            displaced_staging: Path | None = None
            real_publish_bundle = onnx_export._publish_bundle

            def replace_with_empty_staging(staging: Path, *arguments: object) -> int:
                nonlocal staging_path, displaced_staging
                staging_path = staging
                displaced_staging = staging.with_name(f"{staging.name}.displaced")
                staging.rename(displaced_staging)
                staging.mkdir()
                return real_publish_bundle(staging, *arguments)

            with patch.object(
                onnx_export,
                "_publish_bundle",
                replace_with_empty_staging,
            ):
                with self.assertRaisesRegex(RuntimeError, "identity"):
                    export_detector_onnx(
                        **paths,
                        torch_module=_FakeTorch(),
                        timm_module=_FakeTimm(_FakeDetector()),
                        import_module=lambda name: _FakeOnnx()
                        if name == "onnx"
                        else _missing_onnx(name),
                    )

            assert staging_path is not None
            assert displaced_staging is not None
            self.assertTrue(staging_path.is_dir())
            self.assertTrue(displaced_staging.is_dir())

    def test_windows_directory_fsync_is_a_supported_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(os, "fsync") as fsync:
                onnx_export._fsync_directory(Path(temporary), os_name="nt")
            fsync.assert_not_called()

    def test_training_extra_declares_real_export_dependencies(self) -> None:
        document = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        requirements = document["project"]["optional-dependencies"]["training"]
        normalized = {
            requirement.split("<", 1)[0].split(">", 1)[0]
            for requirement in requirements
        }

        self.assertIn("onnx", normalized)
        self.assertIn("onnxruntime", normalized)

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


_REAL_EXPORT_DEPENDENCIES = ("numpy", "onnx", "onnxruntime", "timm", "torch")


@unittest.skipUnless(
    all(
        importlib.util.find_spec(name) is not None for name in _REAL_EXPORT_DEPENDENCIES
    ),
    "requires the optional training/export dependencies",
)
class RealOnnxIntegrationTests(unittest.TestCase):
    def test_real_export_matches_pytorch_and_detects_model_mutation(self) -> None:
        import numpy as np
        import onnx
        import onnxruntime
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = OnnxExporterTests()._inputs(root)
            torch.manual_seed(323)
            source = create_convnextv2_nano(ConvNeXtV2NanoConfig()).eval()
            torch.save(source.state_dict(), paths["checkpoint"])
            checkpoint_sha256 = _sha256(paths["checkpoint"].read_bytes())
            config = _config(paths["dataset_manifest"].read_bytes())
            calibration = CalibrationFitResult(
                artifact=PlattCalibrationArtifact(
                    scale=1.75,
                    bias=-0.25,
                    checkpoint_sha256=checkpoint_sha256,
                    calibration_split_sha256=config.calibration_split_sha256,
                    training_config=config,
                    input_identifier="calibration",
                    sample_count=4,
                    class_counts=CalibrationClassCounts(real=2, ai=2),
                ),
                predictions_sha256="e" * 64,
                uncalibrated=CalibrationMetrics(0.7, 0.2, 0.5),
                calibrated=CalibrationMetrics(0.6, 0.1, 0.75),
            )
            paths["calibrator"].write_bytes(calibration.to_json_bytes())

            metadata = export_detector_onnx(**paths)
            model_path = paths["output"] / "detector.onnx"
            graph = onnx.load(str(model_path), load_external_data=False)
            onnx.checker.check_model(graph, full_check=True)
            self.assertFalse(
                any(
                    initializer.data_location == onnx.TensorProto.EXTERNAL
                    or initializer.external_data
                    for initializer in graph.graph.initializer
                )
            )
            self.assertEqual(metadata.model_sha256, _sha256(model_path.read_bytes()))

            image = (
                np.random.default_rng(323)
                .normal(size=(1, 3, 224, 224))
                .astype(np.float32)
            )
            with torch.no_grad():
                raw_logit = source(torch.from_numpy(image)).reshape(1, 1)
                expected = torch.sigmoid(1.75 * raw_logit - 0.25).numpy()
            session = onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            actual = session.run(["probability_ai"], {"image": image})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            self.assertEqual(
                onnx_export.validate_export_bundle(paths["output"]), metadata
            )

            with model_path.open("ab") as stream:
                stream.write(b"mutation")
            with self.assertRaisesRegex(ValueError, "model digest mismatch"):
                onnx_export.validate_export_bundle(paths["output"])


if __name__ == "__main__":
    unittest.main()
