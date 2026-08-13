from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from poidh_detector.contracts import (
    DatasetManifest,
    DatasetSource,
    LicenseAudit,
    SampleRecord,
)
from poidh_detector.data import (
    DatasetImageSamples,
    SplitManifest,
    balanced_epoch_indices,
    load_dataset_manifest,
    load_split_manifest,
    preprocess_rgb_image,
    select_profile_subset,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _manifest() -> DatasetManifest:
    audit = LicenseAudit(
        spdx_expression="MIT",
        declared_scope="repository",
        evidence_uri="https://example.invalid/license",
        evidence_sha256=_digest("license"),
        dataset_card_sha256=_digest("card"),
        audited_at="2026-08-13T00:00:00Z",
        auditor="test",
        decision="allow",
        rationale="test fixture",
        image_level_verified=True,
        accepted_policy_id="test-v1",
    )
    source = DatasetSource(
        source_id="source",
        dataset_id="owner/dataset",
        revision="a" * 40,
        upstream_uri="https://example.invalid/dataset",
        metadata_sha256=_digest("metadata"),
        license_audit=audit,
    )
    rows = []
    for index, (label, group) in enumerate(
        ((0, "real-a"), (0, "real-a"), (0, "real-b"), (1, "ai-a"), (1, "ai-b"))
    ):
        rows.append(
            SampleRecord(
                sample_id=f"sample-{index}",
                source_id="source",
                upstream_path=f"upstream/{index}.png",
                local_path=f"images/{index}.png",
                label=label,
                content_sha256=_digest(f"content-{index}"),
                provenance_group=group,
                generator_family="generator" if label else None,
                content_type="image/png",
            )
        )
    return DatasetManifest(schema_version=1, sources=(source,), samples=tuple(rows))


class ManifestLoadingTests(unittest.TestCase):
    def test_round_trips_dataset_and_validates_split_groups(self) -> None:
        manifest = _manifest()
        assignments = {
            "sample-0": "train",
            "sample-1": "train",
            "sample-2": "validation",
            "sample-3": "train",
            "sample-4": "calibration",
        }
        split = SplitManifest(
            dataset_manifest_sha256=manifest.sha256,
            seed="poidh323:monet:splits:v1",
            ratios={"train": 0.6, "validation": 0.2, "calibration": 0.2},
            assignments=assignments,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset.json"
            split_path = root / "splits.json"
            dataset_path.write_bytes(manifest.to_json_bytes())
            split_path.write_bytes(split.to_json_bytes())

            loaded = load_dataset_manifest(dataset_path)
            loaded_split = load_split_manifest(split_path, loaded)

        self.assertEqual(loaded, manifest)
        self.assertEqual(loaded_split.sha256, split.sha256)

        crossing = dict(assignments)
        crossing["sample-1"] = "validation"
        with self.assertRaisesRegex(ValueError, "provenance group crosses splits"):
            SplitManifest(
                dataset_manifest_sha256=manifest.sha256,
                seed=323,
                ratios=split.ratios,
                assignments=crossing,
            ).validate_against(manifest)

    def test_rejects_missing_assignments_and_wrong_dataset(self) -> None:
        manifest = _manifest()
        split = SplitManifest(
            dataset_manifest_sha256=_digest("wrong"),
            seed=323,
            ratios={"train": 0.8, "validation": 0.1, "calibration": 0.1},
            assignments={sample.sample_id: "train" for sample in manifest.samples[:-1]},
        )
        with self.assertRaisesRegex(ValueError, "dataset manifest digest"):
            split.validate_against(manifest)


class SubsetAndSamplingTests(unittest.TestCase):
    def test_subset_is_deterministic_capped_and_never_splits_groups(self) -> None:
        samples = _manifest().samples

        selected = select_profile_subset(
            samples, profile="overfit", split="train", seed=323, per_class_cap=2
        )
        reordered = select_profile_subset(
            tuple(reversed(samples)),
            profile="overfit",
            split="train",
            seed=323,
            per_class_cap=2,
        )

        self.assertEqual(
            [sample.sample_id for sample in selected],
            [sample.sample_id for sample in reordered],
        )
        for label in (0, 1):
            self.assertLessEqual(sum(sample.label == label for sample in selected), 2)
        selected_ids = {sample.sample_id for sample in selected}
        for group in {sample.provenance_group for sample in samples}:
            group_ids = {
                sample.sample_id
                for sample in samples
                if sample.provenance_group == group
            }
            self.assertIn(len(selected_ids & group_ids), (0, len(group_ids)))

    def test_balanced_epoch_indices_are_repeatable_and_equalize_classes(self) -> None:
        samples = _manifest().samples
        first = balanced_epoch_indices(samples, seed=323, epoch=1)
        repeat = balanced_epoch_indices(samples, seed=323, epoch=1)
        next_epoch = balanced_epoch_indices(samples, seed=323, epoch=2)

        self.assertEqual(first, repeat)
        self.assertNotEqual(first, next_epoch)
        labels = [samples[index].label for index in first]
        self.assertEqual(labels.count(0), labels.count(1))


class ImageDatasetTests(unittest.TestCase):
    def test_preprocess_is_rgb_224_bicubic_tensor_normalization(self) -> None:
        calls: list[object] = []

        class Image:
            def convert(self, mode: str) -> "Image":
                calls.append(("convert", mode))
                return self

        class Functional:
            @staticmethod
            def resize(image: object, dimensions: list[int], **kwargs: object) -> str:
                calls.append(("resize", dimensions, kwargs))
                return "resized"

            @staticmethod
            def to_tensor(image: object) -> str:
                calls.append(("to_tensor", image))
                return "tensor"

            @staticmethod
            def normalize(
                tensor: object, mean: tuple[float, ...], std: tuple[float, ...]
            ) -> str:
                calls.append(("normalize", tensor, mean, std))
                return "normalized"

        result = preprocess_rgb_image(
            Image(), functional_module=Functional, bicubic_value="bicubic"
        )

        self.assertEqual(result, "normalized")
        self.assertEqual(calls[0], ("convert", "RGB"))
        self.assertEqual(calls[1][0:2], ("resize", [224, 224]))
        self.assertEqual(calls[1][2]["interpolation"], "bicubic")
        self.assertEqual(calls[-1][0], "normalize")

    def test_dataset_returns_float_label_and_sample_identity(self) -> None:
        sample = _manifest().samples[0]

        class Opened:
            def __enter__(self) -> "Opened":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def convert(self, mode: str) -> str:
                return f"{mode}-image"

        class Images:
            @staticmethod
            def open(path: Path) -> Opened:
                return Opened()

        dataset = DatasetImageSamples(
            (sample,),
            Path("/dataset"),
            transform=lambda image: f"x:{image}",
            image_module=Images,
        )

        self.assertEqual(dataset[0], ("x:RGB-image", 0.0, sample.sample_id))


if __name__ == "__main__":
    unittest.main()
