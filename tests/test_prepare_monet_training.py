from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.prepare_monet_training import (
    _load_exposed_holdouts,
    _parse_rows,
    _retain_best_candidates,
)


def _raw_row(index: int, source: str = "synthetic-flux-schnell") -> dict[str, object]:
    return {
        "__key__": f"row-{index}",
        "hash_sha256": hashlib.sha256(f"upstream-{index}".encode()).hexdigest(),
        "hash_perceptual": f"{index:016x}",
        "sscd_cluster_id": f"cluster-{index}",
        "source": source,
        "license": "apache-2.0",
        "thumbnail": f"image-{index}".encode(),
    }


class PrepareMonetTrainingTests(unittest.TestCase):
    def test_parse_rows_matches_pinned_binary_thumbnail_schema(self) -> None:
        parsed = _parse_rows(
            [_raw_row(1)],
            shard_path="v1.2.0/synthetic/flux-schnell/0-0/000000.parquet",
            expected_source="synthetic-flux-schnell",
        )

        self.assertEqual(parsed[0].thumbnail, b"image-1")
        self.assertEqual(parsed[0].source, "synthetic-flux-schnell")

    def test_parse_rows_rejects_missing_or_wrong_typed_metadata(self) -> None:
        for field, value in (
            ("thumbnail", None),
            ("sscd_cluster_id", None),
            ("source", 7),
        ):
            row = _raw_row(1)
            row[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    _parse_rows(
                        [row],
                        shard_path="v1.2.0/synthetic/flux-schnell/0-0/000000.parquet",
                        expected_source="synthetic-flux-schnell",
                    )

    def test_parse_rows_rejects_source_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            _parse_rows(
                [_raw_row(1, source="synthetic-z-image")],
                shard_path="v1.2.0/synthetic/flux-schnell/0-0/000000.parquet",
                expected_source="synthetic-flux-schnell",
            )

    def test_bounded_candidates_are_order_independent(self) -> None:
        rows = _parse_rows(
            [_raw_row(index) for index in range(20)],
            shard_path="v1.2.0/synthetic/flux-schnell/0-0/000000.parquet",
            expected_source="synthetic-flux-schnell",
        )
        forward = _retain_best_candidates(
            rows, source="synthetic-flux-schnell", limit=5
        )
        reverse = _retain_best_candidates(
            reversed(rows), source="synthetic-flux-schnell", limit=5
        )

        self.assertEqual(forward, reverse)
        self.assertEqual(len(forward), 5)

    def test_holdout_loader_requires_exact_registered_digest_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths: list[Path] = []
            digests: set[str] = set()
            for index in range(2):
                path = root / f"holdout-{index}.json"
                payload = json.dumps(
                    {
                        "schema_version": 1,
                        "entries": [
                            {
                                "file_name": f"Human/1_fake/Flux/{index}.png",
                                "content_sha256": hashlib.sha256(
                                    f"holdout-{index}".encode()
                                ).hexdigest(),
                            }
                        ],
                    },
                    sort_keys=True,
                ).encode()
                path.write_bytes(payload)
                paths.append(path)
                digests.add(hashlib.sha256(payload).hexdigest())

            self.assertEqual(
                len(_load_exposed_holdouts(paths, expected_digests=frozenset(digests))),
                2,
            )
            with self.assertRaisesRegex(ValueError, "exactly the registered"):
                _load_exposed_holdouts(paths[:1], expected_digests=frozenset(digests))


if __name__ == "__main__":
    unittest.main()
