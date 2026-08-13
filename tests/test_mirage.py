import hashlib
import tempfile
import unittest
from pathlib import Path

from poidh_benchmark.manifest import MirageRow
from poidh_benchmark.mirage import materialize_entry, pinned_download_url


class MirageMaterializationTests(unittest.TestCase):
    def test_uses_revision_pinned_url_and_records_content_hash(self) -> None:
        row = MirageRow(
            file_name="Human/1_fake/Flux xhs/img 1.jpg",
            label="1_fake",
            content_type="Human",
        )
        payload = b"fake-jpeg-payload"
        requested_urls: list[str] = []

        def fetch(url: str) -> bytes:
            requested_urls.append(url)
            return payload

        with tempfile.TemporaryDirectory() as directory:
            entry = materialize_entry(
                row,
                output_root=Path(directory),
                dataset_id="Yunncheng/Mirage-Test",
                revision="abc123",
                selection_seed="poidh323",
                fetch=fetch,
            )
            local_path = Path(directory, entry["local_path"])

            self.assertEqual(local_path.read_bytes(), payload)
            self.assertEqual(
                entry["content_sha256"], hashlib.sha256(payload).hexdigest()
            )
            self.assertEqual(entry["generator_family"], "Flux xhs")
            self.assertEqual(entry["label"], "ai")
            self.assertEqual(requested_urls, [entry["download_url"]])
            self.assertIn("/resolve/abc123/", entry["download_url"])
            self.assertTrue(entry["download_url"].endswith("Flux%20xhs/img%201.jpg"))

    def test_places_real_and_ai_rows_in_separate_benchmark_folders(self) -> None:
        rows = [
            MirageRow("Scene/0_real/real.png", "0_real", "Scene"),
            MirageRow("Scene/1_fake/sd3.5/fake.webp", "1_fake", "Scene"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            entries = [
                materialize_entry(
                    row,
                    output_root=Path(directory),
                    dataset_id="Yunncheng/Mirage-Test",
                    revision="abc123",
                    selection_seed="poidh323",
                    fetch=lambda _: b"image",
                )
                for row in rows
            ]

            self.assertTrue(entries[0]["local_path"].startswith("real/Scene/"))
            self.assertTrue(entries[1]["local_path"].startswith("ai/Scene/sd3.5/"))

    def test_rejects_empty_download(self) -> None:
        row = MirageRow("Scene/0_real/real.jpg", "0_real", "Scene")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "empty image"):
                materialize_entry(
                    row,
                    output_root=Path(directory),
                    dataset_id="Yunncheng/Mirage-Test",
                    revision="abc123",
                    selection_seed="poidh323",
                    fetch=lambda _: b"",
                )

    def test_url_rejects_missing_revision(self) -> None:
        with self.assertRaisesRegex(ValueError, "revision"):
            pinned_download_url("dataset", "", "image.jpg")


if __name__ == "__main__":
    unittest.main()
