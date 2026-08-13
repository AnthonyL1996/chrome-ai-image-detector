from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from tools.prepare_monet_training import (
    SOURCE_PATHS,
    _load_candidates,
    _load_exposed_holdouts,
    _parse_rows,
    _retain_best_candidates,
    main,
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

    def test_bounded_candidates_reject_duplicate_keys_at_cutoff(self) -> None:
        rows = _parse_rows(
            [_raw_row(index) for index in range(20)],
            shard_path="v1.2.0/synthetic/flux-schnell/0-0/000000.parquet",
            expected_source="synthetic-flux-schnell",
        )
        ranked = _retain_best_candidates(
            rows, source="synthetic-flux-schnell", limit=len(rows)
        )
        duplicate = ranked[4]
        conflicting = type(duplicate)(
            key=duplicate.key,
            upstream_path="another-shard.parquet",
            source=duplicate.source,
            license=duplicate.license,
            upstream_sha256=hashlib.sha256(b"different-upstream").hexdigest(),
            perceptual_hash="ffffffffffffffff",
            sscd_cluster_id="different-cluster",
            thumbnail=b"different-image",
        )

        for candidates in ([*ranked[:5], conflicting], [conflicting, *ranked[:5]]):
            with self.subTest(conflict_first=candidates[0] is conflicting):
                with self.assertRaisesRegex(ValueError, "duplicate MONET key"):
                    _retain_best_candidates(
                        candidates,
                        source="synthetic-flux-schnell",
                        limit=5,
                    )

    def test_main_validates_holdouts_before_loading_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls: list[str] = []
            arguments = SimpleNamespace(
                output=root / "output",
                cache_dir=root / "cache",
                synthetic_shards=6,
                real_shards=16,
                holdout_manifest=[root / "v1.json", root / "v2.json"],
            )

            def load_holdouts(paths: list[Path]) -> list[object]:
                self.assertEqual(paths, arguments.holdout_manifest)
                calls.append("holdouts")
                return []

            def load_candidates(**kwargs: object) -> None:
                self.assertEqual(calls, ["holdouts"])
                raise RuntimeError("stop after ordering assertion")

            with (
                patch(
                    "tools.prepare_monet_training._parse_arguments",
                    return_value=arguments,
                ),
                patch(
                    "tools.prepare_monet_training._capture_provenance",
                    return_value=("commit", "script"),
                ),
                patch(
                    "tools.prepare_monet_training._load_exposed_holdouts",
                    side_effect=load_holdouts,
                ),
                patch(
                    "tools.prepare_monet_training._load_candidates",
                    side_effect=load_candidates,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "stop after ordering assertion"
                ):
                    main()

            self.assertEqual(calls, ["holdouts"])

    def test_all_source_shard_counts_are_checked_before_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            card = root / "README.md"
            card.write_text("card", encoding="utf-8")
            downloads: list[str] = []
            discoveries: list[str] = []

            hub = ModuleType("huggingface_hub")

            def list_repo_tree(
                dataset_id: str, *, path_in_repo: str, **kwargs: object
            ) -> list[SimpleNamespace]:
                discoveries.append(path_in_repo)
                if path_in_repo == SOURCE_PATHS["synthetic-z-image"]:
                    return []
                return [SimpleNamespace(path=f"{path_in_repo}/000000.parquet")]

            def hf_hub_download(
                dataset_id: str, filename: str, **kwargs: object
            ) -> str:
                downloads.append(filename)
                return str(card if filename == "README.md" else root / filename)

            hub.list_repo_tree = list_repo_tree  # type: ignore[attr-defined]
            hub.hf_hub_download = hf_hub_download  # type: ignore[attr-defined]

            parquet = ModuleType("pyarrow.parquet")

            class Table:
                def __init__(self, path: str) -> None:
                    self.path = path

                def to_pylist(self) -> list[dict[str, object]]:
                    source = next(
                        source
                        for source, source_path in SOURCE_PATHS.items()
                        if source_path in self.path
                    )
                    row = _raw_row(len(downloads), source=source)
                    if source == "commoncatalog-cc-by":
                        row["license"] = "cc-by-4.0"
                    return [row]

            def read_table(path: str, *, columns: object) -> Table:
                self.assertIsInstance(columns, list)
                return Table(str(path))

            parquet.read_table = read_table  # type: ignore[attr-defined]
            pyarrow = ModuleType("pyarrow")
            pyarrow.parquet = parquet  # type: ignore[attr-defined]

            with patch.dict(
                sys.modules,
                {
                    "huggingface_hub": hub,
                    "pyarrow": pyarrow,
                    "pyarrow.parquet": parquet,
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "not enough parquet shards"):
                    _load_candidates(
                        cache_dir=root / "cache", synthetic_shards=1, real_shards=1
                    )

            self.assertEqual(discoveries, list(SOURCE_PATHS.values()))
            self.assertEqual(downloads, [])

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
