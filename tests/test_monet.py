from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from poidh_detector.monet import (
    MonetRow,
    materialize_selected_rows,
    sample_records_for_rows,
    select_source_quotas,
)


def _row(source: str, index: int, *, thumbnail: bytes | None = None) -> MonetRow:
    label = 0 if source == "commoncatalog-cc-by" else 1
    return MonetRow(
        key=f"{source}:{index}",
        upstream_path=f"v1.2.0/{source}/shard-{index}.parquet",
        source=source,
        license="cc-by-4.0" if label == 0 else "apache-2.0",
        upstream_sha256=hashlib.sha256(
            f"upstream:{source}:{index}".encode()
        ).hexdigest(),
        perceptual_hash=f"{index:016x}",
        sscd_cluster_id=f"cluster:{source}:{index // 2}",
        thumbnail=thumbnail or f"thumbnail:{source}:{index}".encode(),
    )


class MonetTests(unittest.TestCase):
    def test_rejects_unapproved_source_or_mismatched_license(self) -> None:
        with self.assertRaisesRegex(ValueError, "approved MONET source"):
            _row("laion", 1)
        with self.assertRaisesRegex(ValueError, "license mismatch"):
            MonetRow(
                key="one",
                upstream_path="v1.2.0/synthetic/one.parquet",
                source="synthetic-flux-schnell",
                license="cc-by-4.0",
                upstream_sha256="1" * 64,
                perceptual_hash="a" * 16,
                sscd_cluster_id="cluster:one",
                thumbnail=b"image",
            )

    def test_selects_exact_source_quotas_independent_of_input_order(self) -> None:
        rows = [
            *(_row("synthetic-flux-schnell", index) for index in range(10)),
            *(_row("synthetic-flux-klein", index) for index in range(10)),
            *(_row("synthetic-z-image", index) for index in range(10)),
            *(_row("commoncatalog-cc-by", index) for index in range(20)),
        ]
        quotas = {
            "synthetic-flux-schnell": 3,
            "synthetic-flux-klein": 3,
            "synthetic-z-image": 3,
            "commoncatalog-cc-by": 6,
        }
        forward = select_source_quotas(rows, quotas=quotas, seed="323")
        reverse = select_source_quotas(reversed(rows), quotas=quotas, seed="323")

        self.assertEqual(forward, reverse)
        self.assertEqual(
            {source: sum(row.source == source for row in forward) for source in quotas},
            quotas,
        )

    def test_selection_rejects_duplicate_keys_and_thumbnail_content(self) -> None:
        duplicate_key = _row("synthetic-flux-schnell", 1)
        with self.assertRaisesRegex(ValueError, "duplicate MONET key"):
            select_source_quotas(
                [duplicate_key, duplicate_key],
                quotas={"synthetic-flux-schnell": 1},
                seed="323",
            )
        with self.assertRaisesRegex(ValueError, "duplicate thumbnail content"):
            select_source_quotas(
                [
                    _row("synthetic-flux-schnell", 1, thumbnail=b"same"),
                    _row("synthetic-flux-schnell", 2, thumbnail=b"same"),
                ],
                quotas={"synthetic-flux-schnell": 1},
                seed="323",
            )

    def test_materialization_records_local_and_upstream_hashes(self) -> None:
        rows = [_row("synthetic-flux-schnell", 1), _row("commoncatalog-cc-by", 2)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = materialize_selected_rows(rows, output_root=root)

            self.assertEqual(len(samples), 2)
            for row, sample in zip(
                sorted(rows, key=lambda value: value.key), samples, strict=True
            ):
                path = root / sample.local_path
                self.assertEqual(path.read_bytes(), row.thumbnail)
                self.assertEqual(
                    sample.content_sha256, hashlib.sha256(row.thumbnail).hexdigest()
                )
                self.assertEqual(sample.upstream_sha256, row.upstream_sha256)
            self.assertEqual({sample.label for sample in samples}, {0, 1})

    def test_can_build_records_without_writing_images(self) -> None:
        rows = [_row("synthetic-flux-schnell", 1), _row("commoncatalog-cc-by", 2)]
        samples = sample_records_for_rows(rows)

        self.assertEqual(len(samples), 2)
        self.assertEqual({sample.label for sample in samples}, {0, 1})


if __name__ == "__main__":
    unittest.main()
