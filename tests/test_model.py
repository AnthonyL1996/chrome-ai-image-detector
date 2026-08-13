import unittest

from poidh_detector.model import ConvNeXtV2NanoConfig, create_convnextv2_nano


class _FakeTimm:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create_model(self, name: str, **kwargs: object) -> object:
        self.calls.append((name, kwargs))
        return {"name": name, **kwargs}


class ConvNeXtV2NanoFactoryTests(unittest.TestCase):
    def test_builds_randomly_initialized_binary_classifier(self) -> None:
        timm = _FakeTimm()
        config = ConvNeXtV2NanoConfig(num_classes=1, input_channels=3)

        model = create_convnextv2_nano(config, timm_module=timm)

        self.assertEqual(
            timm.calls,
            [
                (
                    "convnextv2_nano",
                    {"pretrained": False, "num_classes": 1, "in_chans": 3},
                )
            ],
        )
        self.assertFalse(model["pretrained"])

    def test_pretrained_weights_cannot_be_requested(self) -> None:
        timm = _FakeTimm()

        with self.assertRaisesRegex(ValueError, "pretrained weights are forbidden"):
            create_convnextv2_nano(
                ConvNeXtV2NanoConfig(),
                pretrained=True,
                timm_module=timm,
            )

        self.assertEqual(timm.calls, [])

    def test_rejects_invalid_classifier_shape(self) -> None:
        for field, value in (("num_classes", 0), ("input_channels", True)):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    ConvNeXtV2NanoConfig(**{field: value})

    def test_missing_timm_fails_with_actionable_message(self) -> None:
        def missing_import(name: str) -> object:
            error = ModuleNotFoundError(name)
            error.name = name
            raise error

        with self.assertRaisesRegex(RuntimeError, "install.*timm"):
            create_convnextv2_nano(
                ConvNeXtV2NanoConfig(),
                import_module=missing_import,
            )

    def test_broken_timm_dependency_is_not_misreported_as_missing_timm(self) -> None:
        def broken_import(name: str) -> object:
            self.assertEqual(name, "timm")
            error = ModuleNotFoundError("optional image backend")
            error.name = "image_backend"
            raise error

        with self.assertRaisesRegex(ModuleNotFoundError, "image backend"):
            create_convnextv2_nano(
                ConvNeXtV2NanoConfig(),
                import_module=broken_import,
            )


if __name__ == "__main__":
    unittest.main()
