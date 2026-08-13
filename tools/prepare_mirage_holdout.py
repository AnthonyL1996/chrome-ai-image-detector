#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poidh_benchmark.manifest import (  # noqa: E402
    MirageRow,
    select_balanced_generator_strata,
)
from poidh_benchmark.mirage import materialize_entry  # noqa: E402


DATASET_ID = "Yunncheng/Mirage-Test"
DATASET_REVISION = "820a191bb0844ae74a1bd9eb57b28268257a4053"
CONTENT_TYPES = ("Animal", "Human", "Object", "Scene")
FAKE_GENERATORS = (
    "Digicam",
    "Flux_xhs_v2",
    "amateurphoto-v6-forcu",
    "realistic_photography_v1",
    "sd3.5",
    "ultrarealisticFinetune_v4",
)
SELECTION_SEED = "poidh323:mirage:holdout:v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen, generator-balanced Mirage-Test holdout."
    )
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--per-class-per-content", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--exclude-shuffle-seed",
        type=int,
        default=323,
        help="Reconstruct the already-viewed streamed development slice.",
    )
    parser.add_argument("--exclude-count", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.exclude_count < 0:
        raise ValueError("exclude-count must not be negative")

    from datasets import Image, load_dataset
    from huggingface_hub import hf_hub_download

    metadata_path = Path(
        hf_hub_download(
            DATASET_ID,
            "test.parquet",
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
    )
    metadata_sha256 = _sha256_file(metadata_path)
    metadata = load_dataset("parquet", data_files=str(metadata_path), split="train")
    rows = [
        MirageRow(
            file_name=item["file_name"],
            label=item["is_real"],
            content_type=item["content_type"],
        )
        for item in metadata
    ]

    excluded_file_names: set[str] = set()
    if args.exclude_count:
        streamed = load_dataset(
            DATASET_ID,
            split="test",
            streaming=True,
            revision=DATASET_REVISION,
        ).cast_column("image", Image(decode=False))
        shuffled = streamed.shuffle(seed=args.exclude_shuffle_seed)
        excluded_file_names = {
            item["file_name"] for item in itertools.islice(shuffled, args.exclude_count)
        }
        if len(excluded_file_names) != args.exclude_count:
            raise RuntimeError("could not reconstruct the full excluded slice")

    eligible_rows = [row for row in rows if row.file_name not in excluded_file_names]
    selected = select_balanced_generator_strata(
        eligible_rows,
        content_types=CONTENT_TYPES,
        fake_generators=FAKE_GENERATORS,
        per_class_per_content=args.per_class_per_content,
        seed=SELECTION_SEED,
    )

    images_root = args.output_root / "images"
    entries: list[dict[str, str | None]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(
                materialize_entry,
                row,
                output_root=images_root,
                dataset_id=DATASET_ID,
                revision=DATASET_REVISION,
                selection_seed=SELECTION_SEED,
                fetch=_download_with_retries,
            ): row
            for row in selected
        }
        for completed, future in enumerate(as_completed(pending), start=1):
            entries.append(future.result())
            if completed % 25 == 0 or completed == len(pending):
                print(f"Downloaded {completed}/{len(pending)}", flush=True)

    content_hashes = [entry["content_sha256"] for entry in entries]
    if len(content_hashes) != len(set(content_hashes)):
        raise RuntimeError("duplicate image content detected in frozen holdout")

    entries.sort(key=lambda entry: str(entry["file_name"]))
    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "metadata_file": "test.parquet",
        "metadata_sha256": metadata_sha256,
        "selection": {
            "seed": SELECTION_SEED,
            "content_types": list(CONTENT_TYPES),
            "fake_generators": list(FAKE_GENERATORS),
            "per_class_per_content": args.per_class_per_content,
            "excluded_development_shuffle_seed": args.exclude_shuffle_seed,
            "excluded_development_count": len(excluded_file_names),
            "excluded_file_names": sorted(excluded_file_names),
        },
        "counts": {
            "total": len(entries),
            "real": sum(entry["label"] == "real" for entry in entries),
            "ai": sum(entry["label"] == "ai" for entry in entries),
        },
        "entries": entries,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(f"Wrote {manifest_path}", flush=True)


def _download_with_retries(url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            request = Request(url, headers={"User-Agent": "poidh-ai-detector/0.1"})
            with urlopen(request, timeout=120) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - final error is re-raised with URL
            last_error = error
            if attempt < 4:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}") from last_error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
