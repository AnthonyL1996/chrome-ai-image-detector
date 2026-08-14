import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from poidh_detector.checkpoint import (
    GitProvenance,
    capture_git_provenance,
    publish_checkpoint,
)
from poidh_detector.training import CheckpointCandidate, TrainingConfig


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _config() -> TrainingConfig:
    return TrainingConfig(
        dataset_manifest_sha256=_digest("dataset"),
        split_manifest_sha256=_digest("splits"),
        calibration_split_sha256=_digest("calibration"),
        exposed_holdout_sha256=(_digest("v1"), _digest("v2")),
        seed=323,
    )


class CheckpointPublicationTests(unittest.TestCase):
    def test_publishes_weights_and_canonical_manifest_atomically(self) -> None:
        config = _config()
        candidate = CheckpointCandidate("epoch-2", 2, 200, 0.2, training_bce=0.1)
        calls = 0

        def clean(_: Path) -> GitProvenance:
            nonlocal calls
            calls += 1
            return GitProvenance("a" * 40)

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary, "checkpoint")
            manifest = publish_checkpoint(
                destination,
                weights=b"safetensors",
                config=config,
                candidate=candidate,
                dataset_manifest_sha256=config.dataset_manifest_sha256,
                split_manifest_sha256=config.split_manifest_sha256,
                exposed_holdout_sha256=reversed(config.exposed_holdout_sha256),
                repository=Path(temporary),
                provenance_reader=clean,
            )

            self.assertEqual(calls, 2)
            self.assertEqual((destination / "model.bin").read_bytes(), b"safetensors")
            manifest_payload = (destination / "checkpoint.json").read_bytes()
            self.assertEqual(manifest_payload, manifest.to_json_bytes())
            document = json.loads(manifest_payload)
            self.assertEqual(document["weights_sha256"], _digest("safetensors"))
            self.assertEqual(document["weights_format"], "opaque_binary")
            self.assertEqual(document["weights_file"], "model.bin")
            self.assertEqual(document["training_config_sha256"], config.sha256)
            self.assertEqual(document["selection_metric"], "validation_bce")
            self.assertEqual(document["selection_value"], 0.2)
            self.assertEqual(
                (destination / "READY").read_text(encoding="ascii"),
                hashlib.sha256(manifest_payload).hexdigest() + "\n",
            )

    def test_manifest_records_balanced_accuracy_selection_value(self) -> None:
        config = TrainingConfig(
            dataset_manifest_sha256=_digest("dataset"),
            split_manifest_sha256=_digest("splits"),
            calibration_split_sha256=_digest("calibration"),
            exposed_holdout_sha256=(_digest("v1"), _digest("v2")),
            seed=323,
            selection_metric="validation_balanced_accuracy",
        )
        candidate = CheckpointCandidate(
            "epoch-2",
            2,
            200,
            0.2,
            training_bce=0.1,
            validation_balanced_accuracy=0.81,
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest = publish_checkpoint(
                Path(temporary, "checkpoint"),
                weights=b"weights",
                config=config,
                candidate=candidate,
                dataset_manifest_sha256=config.dataset_manifest_sha256,
                split_manifest_sha256=config.split_manifest_sha256,
                exposed_holdout_sha256=config.exposed_holdout_sha256,
                repository=Path(temporary),
                provenance_reader=lambda _: GitProvenance("a" * 40),
            )
        self.assertEqual(manifest.selection_metric, "validation_balanced_accuracy")
        self.assertEqual(manifest.selection_value, 0.81)

    def test_refuses_mismatched_input_digests_before_publication(self) -> None:
        config = _config()
        candidate = CheckpointCandidate("epoch-1", 1, 10, 0.3)
        cases = (
            {"dataset_manifest_sha256": _digest("other")},
            {"split_manifest_sha256": _digest("other")},
            {"exposed_holdout_sha256": (_digest("other"),)},
        )

        with tempfile.TemporaryDirectory() as temporary:
            for index, changes in enumerate(cases):
                with self.subTest(changes=changes):
                    destination = Path(temporary, f"checkpoint-{index}")
                    arguments = {
                        "dataset_manifest_sha256": config.dataset_manifest_sha256,
                        "split_manifest_sha256": config.split_manifest_sha256,
                        "exposed_holdout_sha256": config.exposed_holdout_sha256,
                    }
                    arguments.update(changes)
                    with self.assertRaisesRegex(ValueError, "digest mismatch"):
                        publish_checkpoint(
                            destination,
                            weights=b"weights",
                            config=config,
                            candidate=candidate,
                            repository=Path(temporary),
                            provenance_reader=lambda _: GitProvenance("a" * 40),
                            **arguments,
                        )
                    self.assertFalse(destination.exists())

    def test_refuses_existing_destination_and_changed_git_provenance(self) -> None:
        config = _config()
        candidate = CheckpointCandidate("epoch-1", 1, 10, 0.3)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                publish_checkpoint(
                    existing,
                    weights=b"weights",
                    config=config,
                    candidate=candidate,
                    dataset_manifest_sha256=config.dataset_manifest_sha256,
                    split_manifest_sha256=config.split_manifest_sha256,
                    exposed_holdout_sha256=config.exposed_holdout_sha256,
                    repository=root,
                    provenance_reader=lambda _: GitProvenance("a" * 40),
                )

            provenance = iter((GitProvenance("a" * 40), GitProvenance("b" * 40)))
            changed = root / "changed"
            with self.assertRaisesRegex(RuntimeError, "provenance changed"):
                publish_checkpoint(
                    changed,
                    weights=b"weights",
                    config=config,
                    candidate=candidate,
                    dataset_manifest_sha256=config.dataset_manifest_sha256,
                    split_manifest_sha256=config.split_manifest_sha256,
                    exposed_holdout_sha256=config.exposed_holdout_sha256,
                    repository=root,
                    provenance_reader=lambda _: next(provenance),
                )
            self.assertFalse(changed.exists())

    def test_concurrent_destination_creation_is_reserved_without_clobber(self) -> None:
        config = _config()
        candidate = CheckpointCandidate("epoch-1", 1, 10, 0.3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "checkpoint"
            calls = 0

            def concurrent_creator(_: Path) -> GitProvenance:
                nonlocal calls
                calls += 1
                if calls == 2:
                    destination.mkdir()
                    (destination / "owner.txt").write_text(
                        "concurrent owner", encoding="utf-8"
                    )
                return GitProvenance("a" * 40)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                publish_checkpoint(
                    destination,
                    weights=b"weights",
                    config=config,
                    candidate=candidate,
                    dataset_manifest_sha256=config.dataset_manifest_sha256,
                    split_manifest_sha256=config.split_manifest_sha256,
                    exposed_holdout_sha256=config.exposed_holdout_sha256,
                    repository=root,
                    provenance_reader=concurrent_creator,
                )

            self.assertEqual(
                (destination / "owner.txt").read_text(encoding="utf-8"),
                "concurrent owner",
            )
            self.assertEqual(list(destination.iterdir()), [destination / "owner.txt"])

    def test_ready_rename_is_the_atomic_final_publication_operation(self) -> None:
        config = _config()
        candidate = CheckpointCandidate("epoch-1", 1, 10, 0.3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "checkpoint"
            original_replace = os.replace
            observations: list[str] = []

            def observe_final_replace(source: str, target: str) -> None:
                source_path = Path(source)
                target_path = Path(target)
                if target_path != destination / "READY":
                    original_replace(source, target)
                    return
                self.assertEqual(target_path, destination / "READY")
                self.assertEqual(source_path.parent, destination)
                self.assertTrue(source_path.name.startswith(".READY."))
                self.assertFalse((destination / ".reservation").exists())
                self.assertFalse(target_path.exists())
                self.assertTrue((destination / "model.bin").is_file())
                self.assertTrue((destination / "checkpoint.json").is_file())
                observations.append("final-ready")
                original_replace(source, target)

            with patch(
                "poidh_detector.checkpoint.os.replace",
                side_effect=observe_final_replace,
            ):
                publish_checkpoint(
                    destination,
                    weights=b"weights",
                    config=config,
                    candidate=candidate,
                    dataset_manifest_sha256=config.dataset_manifest_sha256,
                    split_manifest_sha256=config.split_manifest_sha256,
                    exposed_holdout_sha256=config.exposed_holdout_sha256,
                    repository=root,
                    provenance_reader=lambda _: GitProvenance("a" * 40),
                )

            self.assertEqual(observations, ["final-ready"])
            self.assertTrue((destination / "READY").is_file())

    def test_ready_rename_failure_leaves_unpublished_directory_without_ready(
        self,
    ) -> None:
        config = _config()
        candidate = CheckpointCandidate("epoch-1", 1, 10, 0.3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "checkpoint"
            original_replace = os.replace

            def fail_final_replace(source: str, target: str) -> None:
                if Path(target) == destination / "READY":
                    raise OSError("injected final publication failure")
                original_replace(source, target)

            with (
                patch(
                    "poidh_detector.checkpoint.os.replace",
                    side_effect=fail_final_replace,
                ),
                self.assertRaisesRegex(OSError, "final publication failure"),
            ):
                publish_checkpoint(
                    destination,
                    weights=b"weights",
                    config=config,
                    candidate=candidate,
                    dataset_manifest_sha256=config.dataset_manifest_sha256,
                    split_manifest_sha256=config.split_manifest_sha256,
                    exposed_holdout_sha256=config.exposed_holdout_sha256,
                    repository=root,
                    provenance_reader=lambda _: GitProvenance("a" * 40),
                )

            self.assertTrue(destination.is_dir())
            self.assertFalse((destination / "READY").exists())
            self.assertFalse((destination / ".reservation").exists())
            self.assertEqual(list(destination.glob(".READY.*")), [])
            self.assertTrue((destination / "model.bin").is_file())
            self.assertTrue((destination / "checkpoint.json").is_file())

    def test_refuses_broken_symlink_destination(self) -> None:
        config = _config()
        candidate = CheckpointCandidate("epoch-1", 1, 10, 0.3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "checkpoint"
            try:
                destination.symlink_to(
                    root / "missing-target", target_is_directory=True
                )
            except OSError as error:
                self.skipTest(f"symlinks are unavailable: {error}")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                publish_checkpoint(
                    destination,
                    weights=b"weights",
                    config=config,
                    candidate=candidate,
                    dataset_manifest_sha256=config.dataset_manifest_sha256,
                    split_manifest_sha256=config.split_manifest_sha256,
                    exposed_holdout_sha256=config.exposed_holdout_sha256,
                    repository=root,
                    provenance_reader=lambda _: GitProvenance("a" * 40),
                )

    @patch("poidh_detector.checkpoint.subprocess.run")
    def test_git_provenance_refuses_dirty_worktree(self, run) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout=" M model.py\n"),
        ]

        with self.assertRaisesRegex(RuntimeError, "dirty worktree"):
            capture_git_provenance(Path("/repo"))

    @patch("poidh_detector.checkpoint.subprocess.run")
    def test_git_provenance_captures_clean_commit(self, run) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0, stdout=""),
            subprocess.CompletedProcess([], 0, stdout="a" * 40 + "\n"),
        ]

        self.assertEqual(capture_git_provenance(Path("/repo")), GitProvenance("a" * 40))
        self.assertEqual(
            run.call_args_list[0].args[0],
            ["git", "status", "--porcelain", "--untracked-files=all"],
        )


if __name__ == "__main__":
    unittest.main()
