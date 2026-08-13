from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poidh_detector.contracts import DatasetManifest, DatasetSource, LicenseAudit
from poidh_detector.monet import (
    MonetRow,
    materialize_selected_rows,
    sample_records_for_rows,
    select_source_quotas,
)
from poidh_detector.splitting import SplitRatios, assign_group_splits
from poidh_benchmark.leakage import (
    HoldoutIndex,
    HoldoutRegistration,
    load_registered_holdout,
    reject_holdout_overlap,
)


DATASET_ID = "jasperai/monet"
DATASET_REVISION = "baae102c4c96c6571f248b86784c67c5af4fd57a"
SELECTION_SEED = "poidh323:monet:training:v1"
SPLIT_SEED = "poidh323:monet:splits:v1"
QUOTAS = {
    "commoncatalog-cc-by": 150_000,
    "synthetic-flux-klein": 50_000,
    "synthetic-flux-schnell": 50_000,
    "synthetic-z-image": 50_000,
}
SOURCE_PATHS = {
    "commoncatalog-cc-by": "v1.2.0/commoncatalog-cc-by",
    "synthetic-flux-klein": "v1.2.0/synthetic/flux-klein",
    "synthetic-flux-schnell": "v1.2.0/synthetic/flux-schnell",
    "synthetic-z-image": "v1.2.0/synthetic/z-image",
}
SOURCE_LICENSES = {
    "commoncatalog-cc-by": "CC-BY-4.0",
    "synthetic-flux-klein": "Apache-2.0",
    "synthetic-flux-schnell": "Apache-2.0",
    "synthetic-z-image": "Apache-2.0",
}
PARQUET_COLUMNS = (
    "__key__",
    "hash_sha256",
    "hash_perceptual",
    "sscd_cluster_id",
    "source",
    "license",
    "thumbnail",
)
EXPECTED_EXPOSED_HOLDOUT_DIGESTS = frozenset(
    {
        "f92cf3e694b14cb96c64aae0b36fb44b47bd81f0189dc18c23b652e242e4f94a",
        "857b7410280011a5b98a55d22c1155d49212519331b4a63bf344a43b5836eadc",
    }
)
_CANDIDATE_RESERVE = 2_000


class PreparationProvenance(NamedTuple):
    git_commit: str
    script_sha256: str


def main() -> None:
    arguments = _parse_arguments()
    repository = Path(__file__).resolve().parents[1]
    provenance = _capture_provenance(repository)
    holdouts = _load_exposed_holdouts(arguments.holdout_manifest)
    destination = arguments.output.resolve()
    if os.path.lexists(destination):
        raise FileExistsError(f"output destination already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    published = False
    try:
        rows, card_payload, selected_shards = _load_candidates(
            cache_dir=arguments.cache_dir,
            synthetic_shards=arguments.synthetic_shards,
            real_shards=arguments.real_shards,
        )
        selected = select_source_quotas(rows, quotas=QUOTAS, seed=SELECTION_SEED)
        samples = sample_records_for_rows(selected)
        reject_holdout_overlap(samples, holdouts)
        materialized = materialize_selected_rows(selected, output_root=staging)
        if materialized != samples:
            raise RuntimeError("materialized sample records changed unexpectedly")
        sources = _build_sources(card_payload)
        manifest = DatasetManifest(1, tuple(sources), tuple(samples))
        manifest.verify_materialized_files(staging)
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(manifest.to_json_bytes())

        assignments = assign_group_splits(
            samples,
            ratios=SplitRatios(train=0.8, validation=0.1, calibration=0.1),
            seed=SPLIT_SEED,
        )
        split_document = {
            "schema_version": 1,
            "dataset_manifest_sha256": manifest.sha256,
            "seed": SPLIT_SEED,
            "ratios": {"train": 0.8, "validation": 0.1, "calibration": 0.1},
            "assignments": assignments,
        }
        split_payload = (
            json.dumps(split_document, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        (staging / "splits.json").write_bytes(split_payload)
        preparation = {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "selection_seed": SELECTION_SEED,
            "source_quotas": QUOTAS,
            "selected_shards": selected_shards,
            "manifest_sha256": manifest.sha256,
            "splits_sha256": hashlib.sha256(split_payload).hexdigest(),
            "script_sha256": provenance.script_sha256,
            "git_commit": provenance.git_commit,
            "exposed_holdout_manifest_sha256": sorted(EXPECTED_EXPOSED_HOLDOUT_DIGESTS),
        }
        (staging / "preparation.json").write_text(
            json.dumps(preparation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _revalidate_provenance(repository, provenance)
        staging.replace(destination)
        published = True
        print(json.dumps(preparation, indent=2, sort_keys=True))
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _load_candidates(
    *, cache_dir: Path, synthetic_shards: int, real_shards: int
) -> tuple[list[MonetRow], bytes, dict[str, list[str]]]:
    try:
        import pyarrow.parquet as parquet
        from huggingface_hub import hf_hub_download, list_repo_tree
    except ImportError as error:
        raise RuntimeError("huggingface_hub and pyarrow are required") from error

    selected_shards: dict[str, list[str]] = {}
    for source, source_path in SOURCE_PATHS.items():
        wanted = real_shards if source == "commoncatalog-cc-by" else synthetic_shards
        tree = list_repo_tree(
            DATASET_ID,
            path_in_repo=source_path,
            recursive=True,
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
        candidates = [item.path for item in tree if item.path.endswith(".parquet")]
        ranked = sorted(
            candidates,
            key=lambda path: (
                hashlib.sha256(f"{SELECTION_SEED}\0shard\0{path}".encode()).digest(),
                path,
            ),
        )
        if len(ranked) < wanted:
            raise RuntimeError(f"not enough parquet shards for {source}")
        selected_shards[source] = ranked[:wanted]

    card_path = hf_hub_download(
        DATASET_ID,
        "README.md",
        repo_type="dataset",
        revision=DATASET_REVISION,
        cache_dir=cache_dir,
    )
    all_rows: list[MonetRow] = []
    for source in SOURCE_PATHS:
        source_rows: list[MonetRow] = []
        for shard_path in selected_shards[source]:
            local_path = hf_hub_download(
                DATASET_ID,
                shard_path,
                repo_type="dataset",
                revision=DATASET_REVISION,
                cache_dir=cache_dir,
            )
            table = parquet.read_table(local_path, columns=PARQUET_COLUMNS)
            source_rows.extend(
                _parse_rows(
                    table.to_pylist(),
                    shard_path=shard_path,
                    expected_source=source,
                )
            )
            source_rows = _retain_best_candidates(
                source_rows, source=source, limit=QUOTAS[source] + _CANDIDATE_RESERVE
            )
            del table
        all_rows.extend(source_rows)
    return all_rows, Path(card_path).read_bytes(), selected_shards


def _parse_rows(
    rows: Iterable[dict[str, object]], *, shard_path: str, expected_source: str
) -> list[MonetRow]:
    parsed: list[MonetRow] = []
    for row in rows:
        for field_name in (
            "__key__",
            "hash_sha256",
            "hash_perceptual",
            "sscd_cluster_id",
            "source",
            "license",
        ):
            if not isinstance(row.get(field_name), str) or not row[field_name]:
                raise ValueError(f"MONET row requires string {field_name}")
        thumbnail = row.get("thumbnail")
        if not isinstance(thumbnail, bytes) or not thumbnail:
            raise ValueError("MONET row requires binary thumbnail")
        if row["source"] != expected_source:
            raise ValueError(
                f"MONET source mismatch: expected {expected_source}, found {row['source']}"
            )
        parsed.append(
            MonetRow(
                key=row["__key__"],  # type: ignore[arg-type]
                upstream_path=shard_path,
                source=row["source"],  # type: ignore[arg-type]
                license=row["license"],  # type: ignore[arg-type]
                upstream_sha256=row["hash_sha256"],  # type: ignore[arg-type]
                perceptual_hash=row["hash_perceptual"],  # type: ignore[arg-type]
                sscd_cluster_id=row["sscd_cluster_id"],  # type: ignore[arg-type]
                thumbnail=thumbnail,
            )
        )
    return parsed


def _retain_best_candidates(
    rows: Iterable[MonetRow], *, source: str, limit: int
) -> list[MonetRow]:
    candidates = list(rows)
    seen_keys: set[str] = set()
    for row in candidates:
        if row.key in seen_keys:
            raise ValueError(f"duplicate MONET key: {row.key}")
        seen_keys.add(row.key)
    return sorted(
        candidates,
        key=lambda row: (
            hashlib.sha256(f"{SELECTION_SEED}\0{source}\0{row.key}".encode()).digest(),
            row.key,
        ),
    )[:limit]


def _load_exposed_holdouts(
    paths: list[Path],
    *,
    expected_digests: frozenset[str] = EXPECTED_EXPOSED_HOLDOUT_DIGESTS,
) -> list[HoldoutIndex]:
    registrations = []
    observed_digests = set()
    for path in paths:
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        observed_digests.add(digest)
        registrations.append(
            HoldoutRegistration(path.stem, path, digest, "development_exposed")
        )
    if observed_digests != expected_digests:
        raise ValueError(
            "exactly the registered Mirage v1/v2 holdout manifests are required"
        )
    return [load_registered_holdout(registration) for registration in registrations]


def _build_sources(card_payload: bytes) -> list[DatasetSource]:
    card_hash = hashlib.sha256(card_payload).hexdigest()
    sources: list[DatasetSource] = []
    for source, license_expression in sorted(SOURCE_LICENSES.items()):
        audit = LicenseAudit(
            spdx_expression=license_expression,
            declared_scope="dataset_card",
            evidence_uri=f"https://huggingface.co/datasets/{DATASET_ID}/blob/{DATASET_REVISION}/README.md",
            evidence_sha256=card_hash,
            dataset_card_sha256=card_hash,
            audited_at="2026-08-13T00:00:00Z",
            auditor="poidh-ai-detector",
            decision="allow",
            rationale=f"Pinned MONET card declares {source} as {license_expression}.",
            image_level_verified=False,
            accepted_policy_id="repo-license-plus-audit-v1",
        )
        sources.append(
            DatasetSource(
                source_id=f"monet-{source}",
                dataset_id=DATASET_ID,
                revision=DATASET_REVISION,
                upstream_uri=f"https://huggingface.co/datasets/{DATASET_ID}",
                metadata_sha256=card_hash,
                license_audit=audit,
            )
        )
    return sources


def _capture_provenance(repository: Path) -> PreparationProvenance:
    import subprocess

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("refusing to prepare training data from a dirty worktree")
    return PreparationProvenance(
        git_commit=_git_commit(repository),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def _revalidate_provenance(repository: Path, expected: PreparationProvenance) -> None:
    actual = _capture_provenance(repository)
    if actual != expected:
        raise RuntimeError("repository provenance changed during data preparation")


def _git_commit(repository: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a pinned, audited MONET training set"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--synthetic-shards", type=int, default=6)
    parser.add_argument("--real-shards", type=int, default=16)
    parser.add_argument(
        "--holdout-manifest",
        action="append",
        type=Path,
        required=True,
        help="Path to an exposed holdout manifest; provide Mirage v1 and v2",
    )
    arguments = parser.parse_args()
    if arguments.synthetic_shards < 6 or arguments.real_shards < 16:
        parser.error(
            "at least 6 synthetic and 16 real shards are required for the fixed quotas"
        )
    return arguments


if __name__ == "__main__":
    main()
