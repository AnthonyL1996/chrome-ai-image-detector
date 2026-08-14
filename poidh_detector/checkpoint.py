from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import tempfile

from poidh_detector.training import (
    CheckpointCandidate,
    TrainingConfig,
    checkpoint_selection_value,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class GitProvenance:
    commit: str

    def __post_init__(self) -> None:
        if not _GIT_COMMIT.fullmatch(self.commit):
            raise ValueError("git commit must be a lowercase 40-character digest")


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    schema_version: int
    checkpoint_id: str
    epoch: int
    global_step: int
    selection_metric: str
    selection_value: float
    weights_file: str
    weights_format: str
    weights_sha256: str
    training_config_sha256: str
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    exposed_holdout_sha256: tuple[str, ...]
    git_commit: str

    def to_json_bytes(self) -> bytes:
        document = asdict(self)
        document["exposed_holdout_sha256"] = list(self.exposed_holdout_sha256)
        return (
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")


ProvenanceReader = Callable[[Path], GitProvenance]


def publish_checkpoint(
    destination: Path,
    *,
    weights: bytes,
    config: TrainingConfig,
    candidate: CheckpointCandidate,
    dataset_manifest_sha256: str,
    split_manifest_sha256: str,
    exposed_holdout_sha256: Iterable[str],
    repository: Path,
    provenance_reader: ProvenanceReader | None = None,
) -> CheckpointManifest:
    if not isinstance(weights, bytes) or not weights:
        raise ValueError("weights must contain serialized checkpoint bytes")
    _require_matching_digest(
        dataset_manifest_sha256,
        config.dataset_manifest_sha256,
        "dataset manifest",
    )
    _require_matching_digest(
        split_manifest_sha256,
        config.split_manifest_sha256,
        "split manifest",
    )
    observed_holdouts = tuple(exposed_holdout_sha256)
    for digest in observed_holdouts:
        _require_sha256(digest, "exposed holdout")
    if (
        len(observed_holdouts) != len(set(observed_holdouts))
        or tuple(sorted(observed_holdouts)) != config.exposed_holdout_sha256
    ):
        raise ValueError("exposed holdout digest mismatch")

    # Keep the final path lexical so a broken destination symlink cannot be
    # resolved away and redirect publication to its missing target.
    resolved_destination = destination.absolute()
    if os.path.lexists(resolved_destination):
        raise FileExistsError(
            f"checkpoint destination already exists: {resolved_destination}"
        )
    read_provenance = provenance_reader or capture_git_provenance
    initial_provenance = read_provenance(repository)
    manifest = CheckpointManifest(
        schema_version=1,
        checkpoint_id=candidate.checkpoint_id,
        epoch=candidate.epoch,
        global_step=candidate.global_step,
        selection_metric=config.selection_metric,
        selection_value=checkpoint_selection_value(
            candidate, config.selection_metric
        ),
        weights_file="model.bin",
        weights_format="opaque_binary",
        weights_sha256=hashlib.sha256(weights).hexdigest(),
        training_config_sha256=config.sha256,
        dataset_manifest_sha256=config.dataset_manifest_sha256,
        split_manifest_sha256=config.split_manifest_sha256,
        exposed_holdout_sha256=config.exposed_holdout_sha256,
        git_commit=initial_provenance.commit,
    )

    resolved_destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{resolved_destination.name}.staging-",
            dir=resolved_destination.parent,
        )
    )
    published = False
    reservation_token: str | None = None
    ready_temporary: Path | None = None
    try:
        (staging / manifest.weights_file).write_bytes(weights)
        manifest_payload = manifest.to_json_bytes()
        (staging / "checkpoint.json").write_bytes(manifest_payload)
        if read_provenance(repository) != initial_provenance:
            raise RuntimeError(
                "repository provenance changed during checkpoint publication"
            )

        # mkdir is the portable atomic no-replace primitive for directories.
        # The final READY marker is the publication boundary: consumers must
        # ignore a reserved directory until it exists and matches the manifest.
        try:
            resolved_destination.mkdir()
        except FileExistsError as error:
            raise FileExistsError(
                f"checkpoint destination already exists: {resolved_destination}"
            ) from error
        reservation_token = secrets.token_hex(32)
        reservation = resolved_destination / ".reservation"
        reservation.write_text(reservation_token, encoding="ascii")
        (staging / manifest.weights_file).replace(
            resolved_destination / manifest.weights_file
        )
        (staging / "checkpoint.json").replace(resolved_destination / "checkpoint.json")
        ready_payload = hashlib.sha256(manifest_payload).hexdigest() + "\n"

        # Once ownership is released, failures leave an inert directory without
        # READY. The only operation that makes it consumable is the final atomic
        # rename of a fully written marker inside the same directory.
        reservation.unlink()
        reservation_token = None
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".READY.",
            suffix=".tmp",
            dir=resolved_destination,
        )
        ready_temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="ascii", newline="") as stream:
            stream.write(ready_payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(ready_temporary, resolved_destination / "READY")
        ready_temporary = None
        published = True
        return manifest
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
        if ready_temporary is not None and os.path.lexists(ready_temporary):
            ready_temporary.unlink()
        if reservation_token is not None:
            reservation = resolved_destination / ".reservation"
            try:
                owned = reservation.read_text(encoding="ascii") == reservation_token
            except OSError:
                owned = False
            if owned and not os.path.lexists(resolved_destination / "READY"):
                shutil.rmtree(resolved_destination)


def capture_git_provenance(repository: Path) -> GitProvenance:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("refusing checkpoint publication from a dirty worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return GitProvenance(commit)


def _require_matching_digest(observed: str, expected: str, description: str) -> None:
    _require_sha256(observed, description)
    if observed != expected:
        raise ValueError(f"{description} digest mismatch")


def _require_sha256(value: str, description: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
