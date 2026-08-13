#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poidh_detector.data import (  # noqa: E402
    DatasetImageSamples,
    balanced_epoch_indices,
    load_dataset_manifest,
    load_split_manifest,
    samples_for_split,
    select_profile_subset,
)
from poidh_detector.model import (  # noqa: E402
    ConvNeXtV2NanoConfig,
    create_convnextv2_nano,
)
from poidh_detector.reproducibility import (  # noqa: E402
    capture_environment,
    configure_determinism,
)
from poidh_detector.torch_training import (  # noqa: E402
    ExponentialMovingAverage,
    OptimizationConfig,
    ResumeContract,
    candidate_from_validation,
    create_optimizer_and_scheduler,
    evaluate_model,
    load_current_generation,
    load_resume_checkpoint,
    publish_generation,
    reserve_run_directory,
    save_resume_checkpoint,
    train_one_epoch,
    validate_generation_resume_contract,
)
from poidh_detector.training import (  # noqa: E402
    CheckpointCandidate,
    TrainingConfig,
    select_best_checkpoint,
)


def main() -> None:
    arguments = _parse_arguments()
    dataset_root = arguments.dataset_root.resolve()
    manifest = load_dataset_manifest(dataset_root / "manifest.json")
    preparation = _load_preparation(dataset_root / "preparation.json")
    if preparation["manifest_sha256"] != manifest.sha256:
        raise ValueError("preparation manifest digest does not match manifest.json")
    split_manifest = load_split_manifest(
        dataset_root / "splits.json",
        manifest,
        expected_sha256=preparation["splits_sha256"],
    )
    manifest.verify_materialized_files(dataset_root)

    holdout_hashes = tuple(preparation["exposed_holdout_manifest_sha256"])
    if arguments.holdout_sha256 and set(arguments.holdout_sha256) != set(
        holdout_hashes
    ):
        raise ValueError("explicit holdout digests do not match preparation provenance")
    calibration_hash = _partition_sha256(split_manifest.assignments, "calibration")
    config = TrainingConfig(
        dataset_manifest_sha256=manifest.sha256,
        split_manifest_sha256=split_manifest.sha256,
        calibration_split_sha256=calibration_hash,
        exposed_holdout_sha256=holdout_hashes,
        seed=arguments.seed,
    )
    overrides: dict[str, Any] = {"num_workers": arguments.workers}
    if arguments.epochs is not None:
        overrides["epochs"] = arguments.epochs
    if arguments.batch_size is not None:
        overrides["batch_size"] = arguments.batch_size
    optimization = OptimizationConfig.for_profile(arguments.profile, **overrides)

    train_samples = select_profile_subset(
        samples_for_split(manifest, split_manifest, "train"),
        profile=optimization.profile,
        split="train",
        seed=config.seed,
        per_class_cap=optimization.train_per_class_cap,
    )
    validation_samples = select_profile_subset(
        samples_for_split(manifest, split_manifest, "validation"),
        profile=optimization.profile,
        split="validation",
        seed=config.seed,
        per_class_cap=optimization.validation_per_class_cap,
    )
    _require_two_classes(train_samples, "training")
    _require_two_classes(validation_samples, "validation")

    plan = {
        "training_config_sha256": config.sha256,
        "optimization_config_sha256": optimization.sha256,
        "dataset_manifest_sha256": manifest.sha256,
        "split_manifest_sha256": split_manifest.sha256,
        "profile": optimization.profile,
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "train_class_counts": _class_counts(train_samples),
        "validation_class_counts": _class_counts(validation_samples),
    }
    if arguments.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "training requires torch, torchvision, Pillow, NumPy, and timm"
        ) from error

    configure_determinism(config.seed, torch_module=torch)
    environment = capture_environment(torch_module=torch)
    device = _resolve_device(arguments.device, torch)
    output = arguments.output.resolve()
    _prepare_run_output(
        output,
        resume=arguments.resume,
        config=config,
        optimization=optimization,
        environment=environment,
        plan=plan,
    )

    model = create_convnextv2_nano(ConvNeXtV2NanoConfig()).to(device)
    train_dataset = DatasetImageSamples(train_samples, dataset_root)
    validation_dataset = DatasetImageSamples(validation_samples, dataset_root)

    epoch_indices = balanced_epoch_indices(train_samples, seed=config.seed, epoch=1)
    steps_per_epoch = math.ceil(len(epoch_indices) / optimization.batch_size)
    total_steps = steps_per_epoch * optimization.epochs
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, optimization, total_steps=total_steps, torch_module=torch
    )
    ema = ExponentialMovingAverage(model, optimization.ema_decay)

    completed_epoch = 0
    global_step = 0
    candidates: list[CheckpointCandidate] = []
    current_generation = None
    if arguments.resume:
        current_generation = load_current_generation(
            output,
            config=config,
            optimization=optimization,
            environment=environment,
        )
        if current_generation is not None:
            resumed = load_resume_checkpoint(
                current_generation.resume_path,
                config=config,
                optimization=optimization,
                environment=environment,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                torch_module=torch,
            )
            validate_generation_resume_contract(current_generation, resumed)
            completed_epoch = resumed.completed_epoch
            global_step = resumed.global_step
            candidates = list(current_generation.candidates)

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=optimization.batch_size,
        shuffle=False,
        num_workers=optimization.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    for epoch in range(completed_epoch + 1, optimization.epochs + 1):
        indices = balanced_epoch_indices(train_samples, seed=config.seed, epoch=epoch)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=optimization.batch_size,
            sampler=indices,
            num_workers=optimization.num_workers,
            pin_memory=device.startswith("cuda"),
        )
        training_bce, epoch_steps = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            ema,
            device=device,
            torch_module=torch,
        )
        global_step += epoch_steps
        with ema.average_parameters(model):
            validation = evaluate_model(
                model,
                validation_loader,
                device=device,
                torch_module=torch,
            )
            candidate = candidate_from_validation(
                epoch=epoch,
                global_step=global_step,
                training_bce=training_bce,
                validation=validation,
            )
            candidates.append(candidate)
            selected = select_best_checkpoint(candidates)

        contract = ResumeContract.create(
            config=config,
            optimization=optimization,
            environment=environment,
            completed_epoch=epoch,
            global_step=global_step,
        )

        def write_resume(path: Path) -> None:
            save_resume_checkpoint(
                path,
                contract=contract,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                torch_module=torch,
            )

        if selected == candidate:

            def write_best_weights(path: Path) -> None:
                with ema.average_parameters(model):
                    torch.save(model.state_dict(), path)

        else:
            if current_generation is None:
                raise RuntimeError("selected checkpoint weights are unavailable")

            def write_best_weights(path: Path) -> None:
                shutil.copyfile(current_generation.best_weights_path, path)

        current_generation = publish_generation(
            output,
            contract=contract,
            candidates=candidates,
            selected=selected,
            write_resume=write_resume,
            write_best_weights=write_best_weights,
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "training_bce": training_bce,
                    "validation_bce": validation.bce,
                    "validation_auc": validation.auc,
                    "selected_checkpoint": selected.checkpoint_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if not candidates:
        raise ValueError("resume checkpoint already exceeds configured epoch count")
    selected = select_best_checkpoint(candidates)
    if current_generation is None:
        raise RuntimeError("training completed without a published generation")
    print(
        json.dumps(
            {
                "best_checkpoint": selected.checkpoint_id,
                "best_validation_bce": selected.validation_bce,
                "weights": str(current_generation.best_weights_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the random-init ConvNeXtV2-Nano detector in staged profiles."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--profile", choices=("overfit", "smoke", "pilot", "full"), default="smoke"
    )
    parser.add_argument("--seed", type=int, default=323)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the transactionally published CURRENT generation in output.",
    )
    parser.add_argument("--holdout-sha256", action="append", default=[])
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Verify all data and print the exact subset plan without importing Torch.",
    )
    return parser.parse_args()


def _load_preparation(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict):
        raise ValueError("preparation.json must be a JSON object")
    for field_name in (
        "manifest_sha256",
        "splits_sha256",
        "exposed_holdout_manifest_sha256",
    ):
        if field_name not in document:
            raise ValueError(f"preparation.json is missing {field_name}")
    holdouts = document["exposed_holdout_manifest_sha256"]
    if not isinstance(holdouts, list) or not holdouts:
        raise ValueError("preparation.json requires exposed holdout digests")
    return document


def _partition_sha256(assignments: Any, split: str) -> str:
    document = {
        "sample_ids": sorted(
            sample_id
            for sample_id, assigned in assignments.items()
            if assigned == split
        ),
        "schema_version": 1,
        "split": split,
    }
    payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_two_classes(samples: Any, description: str) -> None:
    if {sample.label for sample in samples} != {0, 1}:
        raise ValueError(f"{description} subset must contain both real and AI samples")


def _class_counts(samples: Any) -> dict[str, int]:
    return {
        "real": sum(sample.label == 0 for sample in samples),
        "ai": sum(sample.label == 1 for sample in samples),
    }


def _resolve_device(requested: str, torch: Any) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")
    return requested


def _write_provenance(
    output: Path,
    config: TrainingConfig,
    optimization: OptimizationConfig,
    environment: Any,
    plan: dict[str, Any],
) -> None:
    for path, payload in _provenance_payloads(
        output, config, optimization, environment, plan
    ):
        with path.open("xb") as stream:
            stream.write(payload)


def _prepare_run_output(
    output: Path,
    *,
    resume: bool,
    config: TrainingConfig,
    optimization: OptimizationConfig,
    environment: Any,
    plan: dict[str, Any],
) -> None:
    if resume:
        for path, payload in _provenance_payloads(
            output, config, optimization, environment, plan
        ):
            if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                raise ValueError(f"existing run provenance does not match: {path.name}")
        return
    reserve_run_directory(output)
    _write_provenance(output, config, optimization, environment, plan)


def _provenance_payloads(
    output: Path,
    config: TrainingConfig,
    optimization: OptimizationConfig,
    environment: Any,
    plan: dict[str, Any],
) -> tuple[tuple[Path, bytes], ...]:
    return (
        (output / "training-config.json", config.to_json_bytes()),
        (output / "optimization-config.json", optimization.to_json_bytes()),
        (
            output / "environment.json",
            (
                json.dumps(environment.to_dict(), indent=2, sort_keys=True) + "\n"
            ).encode(),
        ),
        (
            output / "plan.json",
            (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode(),
        ),
    )


if __name__ == "__main__":
    main()
