import os
import random
import unittest
from unittest.mock import patch

from poidh_detector.reproducibility import (
    capture_environment,
    configure_determinism,
)


class _FakeCuda:
    def __init__(self) -> None:
        self.seed: int | None = None

    def is_available(self) -> bool:
        return True

    def manual_seed_all(self, seed: int) -> None:
        self.seed = seed

    def get_device_name(self, index: int) -> str:
        if index != 0:
            raise AssertionError("only device zero is expected")
        return "Test GPU"


class _FakeCudnn:
    version = staticmethod(lambda: 9100)
    benchmark = True
    deterministic = False


class _FakeTorch:
    __version__ = "2.test"
    version = type("Version", (), {"cuda": "13.test"})()

    def __init__(self) -> None:
        self.cuda = _FakeCuda()
        self.backends = type("Backends", (), {"cudnn": _FakeCudnn()})()
        self.cpu_seed: int | None = None
        self.deterministic_call: tuple[bool, bool] | None = None

    def manual_seed(self, seed: int) -> None:
        self.cpu_seed = seed

    def use_deterministic_algorithms(self, enabled: bool, *, warn_only: bool) -> None:
        self.deterministic_call = (enabled, warn_only)


class _FakeNumpyRandom:
    def __init__(self) -> None:
        self.seed_value: int | None = None

    def seed(self, seed: int) -> None:
        self.seed_value = seed


class _FakeNumpy:
    __version__ = "2.test"

    def __init__(self) -> None:
        self.random = _FakeNumpyRandom()


class ReproducibilityTests(unittest.TestCase):
    def test_configures_all_rngs_and_cuda_determinism(self) -> None:
        torch = _FakeTorch()
        numpy = _FakeNumpy()

        with patch.dict(os.environ, {}, clear=True):
            configure_determinism(323, torch_module=torch, numpy_module=numpy)

            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

        self.assertEqual(torch.cpu_seed, 323)
        self.assertEqual(torch.cuda.seed, 323)
        self.assertEqual(numpy.random.seed_value, 323)
        self.assertEqual(torch.deterministic_call, (True, False))
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)

        with patch.dict(os.environ, {}, clear=True):
            random.seed(0)
            configure_determinism(323, torch_module=torch, numpy_module=numpy)
            first = random.random()
            configure_determinism(323, torch_module=torch, numpy_module=numpy)
            self.assertEqual(first, random.random())

    def test_rejects_invalid_seed_and_preserves_existing_cublas_setting(self) -> None:
        torch = _FakeTorch()
        numpy = _FakeNumpy()

        for seed in (-1, True, 1.5):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "seed"):
                    configure_determinism(
                        seed,
                        torch_module=torch,
                        numpy_module=numpy,
                    )

        with patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": ":16:8"},
            clear=True,
        ):
            configure_determinism(1, torch_module=torch, numpy_module=numpy)
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":16:8")

    def test_rejects_invalid_cublas_setting(self) -> None:
        torch = _FakeTorch()
        numpy = _FakeNumpy()

        with patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": "invalid"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "CUBLAS_WORKSPACE_CONFIG"):
                configure_determinism(
                    323,
                    torch_module=torch,
                    numpy_module=numpy,
                )

    def test_captures_serializable_environment_without_importing_dependencies(
        self,
    ) -> None:
        torch = _FakeTorch()
        numpy = _FakeNumpy()
        timm = type("Timm", (), {"__version__": "1.test"})()

        fingerprint = capture_environment(
            torch_module=torch,
            numpy_module=numpy,
            timm_module=timm,
        )

        self.assertEqual(fingerprint.torch_version, "2.test")
        self.assertEqual(fingerprint.numpy_version, "2.test")
        self.assertEqual(fingerprint.timm_version, "1.test")
        self.assertEqual(fingerprint.cuda_version, "13.test")
        self.assertEqual(fingerprint.cudnn_version, "9100")
        self.assertEqual(fingerprint.cuda_device, "Test GPU")
        self.assertIsInstance(fingerprint.to_dict()["python_version"], str)

    def test_environment_capture_marks_unavailable_optional_dependencies(self) -> None:
        fingerprint = capture_environment(
            torch_module=None,
            numpy_module=None,
            timm_module=None,
            import_optional=False,
        )

        self.assertIsNone(fingerprint.torch_version)
        self.assertIsNone(fingerprint.numpy_version)
        self.assertIsNone(fingerprint.timm_version)
        self.assertFalse(fingerprint.cuda_available)


if __name__ == "__main__":
    unittest.main()
