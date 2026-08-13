from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from poidh_benchmark.leakage import (
    HoldoutRegistration,
    load_registered_holdout,
    reject_holdout_overlap,
)
from poidh_detector.contracts import SampleRecord


class LeakageTests(unittest.TestCase):
    def _write_manifest(self, root: Path) -> tuple[Path, str]:
        document = {
            "schema_version": 1,
            "dataset_id": "Yunncheng/Mirage-Test",
            "dataset_revision": "a" * 40,
            "entries": [
                {
                    "file_name": "Human/1_fake/Flux/image.png",
                    "content_sha256": hashlib.sha256(b"holdout").hexdigest(),
                    "provenance_group": "prompt:secret",
                }
            ],
        }
        payload = json.dumps(document, sort_keys=True).encode()
        path = root / "manifest.json"
        path.write_bytes(payload)
        return path, hashlib.sha256(payload).hexdigest()

    def test_registered_manifest_digest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._write_manifest(Path(temporary))
            registration = HoldoutRegistration(
                "mirage-v1", path, digest, "development_exposed"
            )
            loaded = load_registered_holdout(registration)
            self.assertEqual(loaded.registration, registration)

            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_registered_holdout(registration)

    def test_rejects_noncanonical_holdout_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self._write_manifest(Path(temporary))
            document = json.loads(path.read_text(encoding="utf-8"))
            document["entries"][0]["content_sha256"] = document["entries"][0][
                "content_sha256"
            ].upper()
            payload = json.dumps(document, sort_keys=True).encode()
            path.write_bytes(payload)
            registration = HoldoutRegistration(
                "mirage-v1",
                path,
                hashlib.sha256(payload).hexdigest(),
                "development_exposed",
            )

            with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                load_registered_holdout(registration)

    def test_rejects_renamed_duplicate_content_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._write_manifest(Path(temporary))
            holdout = load_registered_holdout(
                HoldoutRegistration("mirage-v1", path, digest, "development_exposed")
            )
            renamed = SampleRecord(
                sample_id="training:renamed",
                source_id="training",
                upstream_path="renamed.png",
                local_path="images/renamed.png",
                label=1,
                content_sha256=hashlib.sha256(b"holdout").hexdigest(),
                provenance_group="different",
                generator_family="flux",
                content_type="human",
            )
            with self.assertRaisesRegex(ValueError, "content overlap"):
                reject_holdout_overlap([renamed], [holdout])

            same_provenance = SampleRecord(
                sample_id="training:other",
                source_id="training",
                upstream_path="other.png",
                local_path="images/other.png",
                label=1,
                content_sha256=hashlib.sha256(b"other").hexdigest(),
                provenance_group="prompt:secret",
                generator_family="flux",
                content_type="human",
            )
            with self.assertRaisesRegex(ValueError, "provenance overlap"):
                reject_holdout_overlap([same_provenance], [holdout])

    def test_rejects_upstream_hash_overlap_after_reencoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._write_manifest(Path(temporary))
            holdout = load_registered_holdout(
                HoldoutRegistration("mirage-v1", path, digest, "development_exposed")
            )
            transformed = SampleRecord(
                sample_id="training:transformed",
                source_id="training",
                upstream_path="transformed.png",
                local_path="images/transformed.webp",
                label=1,
                content_sha256=hashlib.sha256(b"re-encoded").hexdigest(),
                provenance_group="different",
                generator_family="flux",
                content_type="human",
                upstream_sha256=hashlib.sha256(b"holdout").hexdigest(),
            )

            with self.assertRaisesRegex(ValueError, "content overlap"):
                reject_holdout_overlap([transformed], [holdout])

    def test_can_require_complete_provenance_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self._write_manifest(Path(temporary))
            document = json.loads(path.read_text(encoding="utf-8"))
            del document["entries"][0]["provenance_group"]
            payload = json.dumps(document, sort_keys=True).encode()
            path.write_bytes(payload)
            registration = HoldoutRegistration(
                "future-v3",
                path,
                hashlib.sha256(payload).hexdigest(),
                "frozen_unopened",
                require_provenance=True,
            )

            with self.assertRaisesRegex(ValueError, "provenance_group"):
                load_registered_holdout(registration)


if __name__ == "__main__":
    unittest.main()
