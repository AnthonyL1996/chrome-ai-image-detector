#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poidh_benchmark.manifest import (  # noqa: E402
    MirageRow,
    select_balanced_generator_strata,
)
from poidh_benchmark.mirage import (  # noqa: E402
    commit_staging_directory,
    git_provenance,
    materialize_entry,
    read_prior_manifest,
    reject_prior_content_overlap,
    validate_materialized_entries,
)


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
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Exclude every file_name from a previously frozen manifest.",
    )
    parser.add_argument("--selection-seed", default=SELECTION_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.exclude_count < 0:
        raise ValueError("exclude-count must not be negative")
    if args.output_root.exists():
        raise FileExistsError(f"output destination already exists: {args.output_root}")

    repository = Path(__file__).resolve().parents[1]
    preparation = git_provenance(repository, Path(__file__).resolve())

    import datasets
    import huggingface_hub
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

    ordered_excluded_file_names: list[str] = []
    excluded_file_names: set[str] = set()
    if args.exclude_count:
        streamed = load_dataset(
            DATASET_ID,
            split="test",
            streaming=True,
            revision=DATASET_REVISION,
        ).cast_column("image", Image(decode=False))
        shuffled = streamed.shuffle(
            seed=args.exclude_shuffle_seed,
            buffer_size=1_000,
        )
        ordered_excluded_file_names = [
            item["file_name"] for item in itertools.islice(shuffled, args.exclude_count)
        ]
        excluded_file_names = set(ordered_excluded_file_names)
        if len(excluded_file_names) != args.exclude_count:
            raise RuntimeError("could not reconstruct the full excluded slice")

    prior_exclusions = [
        read_prior_manifest(
            path,
            expected_dataset_id=DATASET_ID,
            expected_revision=DATASET_REVISION,
        )
        for path in args.exclude_manifest
    ]
    for prior in prior_exclusions:
        excluded_file_names.update(prior.file_names)
    known_file_names = {row.file_name for row in rows}
    unknown_exclusions = excluded_file_names - known_file_names
    if unknown_exclusions:
        preview = ", ".join(sorted(unknown_exclusions)[:3])
        raise ValueError(f"excluded names missing from pinned metadata: {preview}")
    eligible_rows = [row for row in rows if row.file_name not in excluded_file_names]
    selected = select_balanced_generator_strata(
        eligible_rows,
        content_types=CONTENT_TYPES,
        fake_generators=FAKE_GENERATORS,
        per_class_per_content=args.per_class_per_content,
        seed=args.selection_seed,
    )

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_root.name}.staging-",
            dir=args.output_root.parent,
        )
    )
    published = False
    try:
        images_root = staging_root / "images"
        entries: list[dict[str, str | None]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {
                pool.submit(
                    materialize_entry,
                    row,
                    output_root=images_root,
                    dataset_id=DATASET_ID,
                    revision=DATASET_REVISION,
                    selection_seed=args.selection_seed,
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
        reject_prior_content_overlap(content_hashes, prior_exclusions)

        entries.sort(key=lambda entry: str(entry["file_name"]))
        validate_materialized_entries(images_root, entries)
        dataset_card_path = Path(
            hf_hub_download(
                DATASET_ID,
                "README.md",
                repo_type="dataset",
                revision=DATASET_REVISION,
            )
        )
        excluded_manifest_sha256 = hashlib.sha256(
            "\n".join(ordered_excluded_file_names).encode("utf-8")
        ).hexdigest()
        manifest = {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "metadata_file": "test.parquet",
            "metadata_sha256": metadata_sha256,
            "provenance": {
                "declared_repository_license": "MIT",
                "dataset_card_sha256": _sha256_file(dataset_card_path),
                "evaluation_only": True,
                "redistribute_images": False,
                "preparation_git_commit": preparation["git_commit"],
                "preparation_script_sha256": preparation["script_sha256"],
                "python_version": platform.python_version(),
                "datasets_version": datasets.__version__,
                "huggingface_hub_version": huggingface_hub.__version__,
            },
            "selection": {
                "seed": args.selection_seed,
                "content_types": list(CONTENT_TYPES),
                "fake_generators": list(FAKE_GENERATORS),
                "per_class_per_content": args.per_class_per_content,
                "excluded_development_shuffle_seed": args.exclude_shuffle_seed,
                "excluded_development_shuffle_buffer_size": 1_000,
                "excluded_development_count": len(ordered_excluded_file_names),
                "excluded_file_names": ordered_excluded_file_names,
                "excluded_manifest_sha256": excluded_manifest_sha256,
                "prior_exclusion_manifests": [
                    {
                        "manifest_sha256": prior.manifest_sha256,
                        "dataset_id": prior.dataset_id,
                        "dataset_revision": prior.dataset_revision,
                        "excluded_count": len(prior.file_names),
                        "entries": [
                            {"file_name": file_name, "content_sha256": content_hash}
                            for file_name, content_hash in prior.entries
                        ],
                    }
                    for prior in prior_exclusions
                ],
                "total_unique_excluded_count": len(excluded_file_names),
            },
            "counts": {
                "total": len(entries),
                "real": sum(entry["label"] == "real" for entry in entries),
                "ai": sum(entry["label"] == "ai" for entry in entries),
            },
            "entries": entries,
        }
        manifest_path = staging_root / "manifest.json"
        temporary = manifest_path.with_suffix(".json.part")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
        commit_staging_directory(staging_root, args.output_root)
        published = True
        print(f"Wrote {args.output_root / 'manifest.json'}", flush=True)
    finally:
        if not published and staging_root.exists():
            shutil.rmtree(staging_root)


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
