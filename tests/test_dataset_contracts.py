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


def _audit() -> LicenseAudit:
    return LicenseAudit(
        spdx_expression="MIT",
        declared_scope="dataset_card",
        evidence_uri="https://huggingface.co/datasets/example/data",
        evidence_sha256="1" * 64,
        dataset_card_sha256="2" * 64,
        audited_at="2026-08-13T00:00:00Z",
        auditor="poidh-team",
        decision="allow",
        rationale="Repository license accepted under policy.",
        image_level_verified=False,
        accepted_policy_id="repo-license-plus-audit-v1",
    )


def _source() -> DatasetSource:
    return DatasetSource(
        source_id="example",
        dataset_id="example/data",
        revision="a" * 40,
        upstream_uri="https://huggingface.co/datasets/example/data",
        metadata_sha256="3" * 64,
        license_audit=_audit(),
    )


def _sample(**overrides: object) -> SampleRecord:
    values: dict[str, object] = {
        "sample_id": "example:one",
        "source_id": "example",
        "upstream_path": "train/one.png",
        "local_path": "images/ai/one.png",
        "label": 1,
        "content_sha256": hashlib.sha256(b"one").hexdigest(),
        "provenance_group": "prompt:one",
        "generator_family": "flux-1-dev",
        "content_type": "scene",
        "upstream_sha256": "4" * 64,
    }
    values.update(overrides)
    return SampleRecord(**values)  # type: ignore[arg-type]


class DatasetContractTests(unittest.TestCase):
    def test_source_requires_immutable_revision(self) -> None:
        for revision in ("", "main", "latest", "abc123"):
            with self.subTest(revision=revision):
                with self.assertRaisesRegex(ValueError, "immutable revision"):
                    DatasetSource(
                        source_id="example",
                        dataset_id="example/data",
                        revision=revision,
                        upstream_uri="https://example.test/data",
                        metadata_sha256="3" * 64,
                        license_audit=_audit(),
                    )

    def test_source_requires_allowed_audit_with_evidence(self) -> None:
        denied = LicenseAudit(
            spdx_expression="CC-BY-NC-4.0",
            declared_scope="dataset_card",
            evidence_uri="https://example.test/card",
            evidence_sha256="1" * 64,
            dataset_card_sha256="2" * 64,
            audited_at="2026-08-13T00:00:00Z",
            auditor="poidh-team",
            decision="deny",
            rationale="Noncommercial restriction.",
            image_level_verified=False,
            accepted_policy_id="repo-license-plus-audit-v1",
        )

        with self.assertRaisesRegex(ValueError, "allowed license audit"):
            DatasetSource(
                source_id="example",
                dataset_id="example/data",
                revision="a" * 40,
                upstream_uri="https://example.test/data",
                metadata_sha256="3" * 64,
                license_audit=denied,
            )

    def test_sample_rejects_unsafe_paths_and_missing_ai_generator(self) -> None:
        for path in (
            "/absolute.png",
            "../escape.png",
            "images/../escape.png",
            "images/./alias.png",
            "images//alias.png",
            "./images/alias.png",
            "images/alias.png/",
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "safe relative"):
                    _sample(local_path=path)
        with self.assertRaisesRegex(ValueError, "generator_family"):
            _sample(generator_family=None)

    def test_manifest_rejects_duplicate_content(self) -> None:
        first = _sample()
        duplicate = _sample(
            sample_id="example:two",
            local_path="images/ai/two.png",
            upstream_path="train/two.png",
            provenance_group="prompt:two",
        )

        with self.assertRaisesRegex(ValueError, "duplicate content"):
            DatasetManifest(
                schema_version=1, sources=(_source(),), samples=(first, duplicate)
            )

    def test_rejects_boolean_schema_and_label_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "label"):
            _sample(label=True)
        with self.assertRaisesRegex(ValueError, "schema version"):
            DatasetManifest(
                schema_version=True, sources=(_source(),), samples=(_sample(),)
            )

    def test_manifest_digest_is_order_stable(self) -> None:
        first = _sample()
        second = _sample(
            sample_id="example:two",
            local_path="images/real/two.png",
            upstream_path="train/two.png",
            label=0,
            content_sha256=hashlib.sha256(b"two").hexdigest(),
            provenance_group="capture:two",
            generator_family=None,
            upstream_sha256="5" * 64,
        )
        forward = DatasetManifest(1, (_source(),), (first, second))
        reverse = DatasetManifest(1, (_source(),), (second, first))

        self.assertEqual(forward.sha256, reverse.sha256)
        self.assertEqual(forward.to_json_bytes(), reverse.to_json_bytes())

    def test_verify_materialized_files_checks_bytes_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "images" / "ai" / "one.png"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"one")
            manifest = DatasetManifest(1, (_source(),), (_sample(),))
            manifest.verify_materialized_files(root)

            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                manifest.verify_materialized_files(root)

            path.write_bytes(b"one")
            (root / "images" / "extra.png").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "file set"):
                manifest.verify_materialized_files(root)

    def test_verify_materialized_files_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            outside = base / "outside"
            outside.mkdir()
            (outside / "one.png").write_bytes(b"one")
            manifest = DatasetManifest(1, (_source(),), (_sample(),))

            root = base / "root"
            (root / "images").mkdir(parents=True)
            (root / "images" / "ai").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                manifest.verify_materialized_files(root)

            real_root = base / "real-root"
            (real_root / "images" / "ai").mkdir(parents=True)
            (real_root / "images" / "ai" / "one.png").write_bytes(b"one")
            root_link = base / "root-link"
            root_link.symlink_to(real_root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "real directory"):
                manifest.verify_materialized_files(root_link)


if __name__ == "__main__":
    unittest.main()
