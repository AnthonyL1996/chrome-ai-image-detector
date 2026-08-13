from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HPC_ROOT = REPOSITORY_ROOT / "hpc"
SMOKE_PATH = HPC_ROOT / "wice_h100_smoke.slurm"
TRAIN_PATH = HPC_ROOT / "wice_h100_train.slurm"
README_PATH = HPC_ROOT / "README.md"
MODULE_PYTHONPATH = "/module/python:/module/torch"

REQUIRED_SBATCH = {
    "account": "lp_edu_maibi_anndl",
    "clusters": "wice",
    "partition": "gpu_h100",
    "gpus-per-node": "1",
}
REQUIRED_MODULE_COMMANDS = (
    "module --force purge",
    "module load cluster/wice/gpu_h100",
    "module load PyTorch-bundle/2.9.1-foss-2025a-CUDA-12.8.0-whl",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sbatch_directives(script: str) -> dict[str, str]:
    directives: dict[str, str] = {}
    for line in script.splitlines():
        match = re.fullmatch(r"#SBATCH --([a-z-]+)=(\S+)", line)
        if match:
            directives[match.group(1)] = match.group(2)
    return directives


def _duration_seconds(duration: str) -> int:
    hours, minutes, seconds = (int(part) for part in duration.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _run_training_script(
    temporary: str,
    *,
    run_name: str,
    resume: bool,
    dependency_mode: str = "valid",
    worker_count: str | None = None,
    script_path: Path = TRAIN_PATH,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    root = Path(temporary)
    scratch = root / "scratch"
    data_root = scratch / "data"
    run_root = scratch / "runs"
    fake_bin = root / "bin"
    invocation_log = root / "python-arguments.txt"
    pythonpath_log = root / "pythonpath.txt"
    for directory in (data_root, run_root, fake_bin):
        directory.mkdir(parents=True, exist_ok=True)
    _write_executable(fake_bin / "module", "#!/bin/sh\nexit 0\n")
    _write_executable(fake_bin / "nvidia-smi", "#!/bin/sh\nexit 0\n")

    if dependency_mode == "outside":
        dependency_root = root / "outside-dependencies"
    else:
        dependency_root = scratch / "python-dependencies"
    if dependency_mode != "missing":
        dependency_root.mkdir(parents=True)
    if dependency_mode in {"valid", "outside"}:
        (dependency_root / "timm").mkdir()
        (dependency_root / "timm" / "__init__.py").write_text(
            '__version__ = "1.0.28"\n',
            encoding="utf-8",
        )
        (dependency_root / "timm-1.0.28.dist-info").mkdir()

    _write_executable(
        fake_bin / "python",
        '#!/bin/sh\nprintf "%s\\n" "$PYTHONPATH" > "$PYTHONPATH_LOG"\nprintf "%s\\n" "$@" > "$INVOCATION_LOG"\n',
    )
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "INVOCATION_LOG": str(invocation_log),
        "PYTHONPATH": MODULE_PYTHONPATH,
        "PYTHONPATH_LOG": str(pythonpath_log),
        "SLURM_CPUS_PER_TASK": "3",
        "SLURM_JOB_ID": "12345",
        "SLURM_SUBMIT_DIR": str(REPOSITORY_ROOT),
    }
    for variable in (
        "POIDH_DATA_ROOT",
        "POIDH_PYTHON_DEPS",
        "POIDH_RUN_NAME",
        "POIDH_RUN_ROOT",
        "TRAIN_PROFILE",
        "TRAIN_RESUME",
        "VSC_SCRATCH",
        "POIDH_WORKERS",
    ):
        environment.pop(variable, None)
    if worker_count is not None:
        environment["POIDH_WORKERS"] = worker_count
    if script_path == SMOKE_PATH:
        command = [
            "bash",
            str(script_path),
            str(scratch),
            str(data_root),
            str(run_root),
            str(dependency_root),
        ]
    else:
        command = [
            "bash",
            str(script_path),
            str(scratch),
            str(data_root),
            str(run_root),
            str(dependency_root),
            "pilot",
            "1" if resume else "0",
            run_name,
        ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, invocation_log, run_root, pythonpath_log


class HpcScriptContracts(unittest.TestCase):
    def test_both_jobs_request_the_pinned_wice_h100_allocation(self) -> None:
        for path in (SMOKE_PATH, TRAIN_PATH):
            with self.subTest(path=path.name):
                directives = _sbatch_directives(_read(path))
                for option, expected in REQUIRED_SBATCH.items():
                    self.assertEqual(directives.get(option), expected)
                self.assertNotIn("gres", directives)

    def test_smoke_job_is_capped_at_ten_minutes(self) -> None:
        duration = _sbatch_directives(_read(SMOKE_PATH))["time"]
        self.assertLessEqual(_duration_seconds(duration), 10 * 60)

    def test_jobs_load_only_the_pinned_cluster_environment(self) -> None:
        for path in (SMOKE_PATH, TRAIN_PATH):
            with self.subTest(path=path.name):
                script = _read(path)
                positions = [
                    script.index(command) for command in REQUIRED_MODULE_COMMANDS
                ]
                self.assertEqual(positions, sorted(positions))
                self.assertNotIn("pip install", script)
                self.assertNotRegex(script, r"\b(?:curl|wget|git clone)\b")

    def test_jobs_pin_deterministic_environment_and_require_scratch_paths(self) -> None:
        for path in (SMOKE_PATH, TRAIN_PATH):
            with self.subTest(path=path.name):
                script = _read(path)
                self.assertIn("export PYTHONHASHSEED=323", script)
                self.assertIn("export CUBLAS_WORKSPACE_CONFIG=:4096:8", script)
                self.assertIn('${VSC_SCRATCH:?"VSC_SCRATCH must be set"}', script)
                self.assertIn(
                    '${POIDH_DATA_ROOT:?"POIDH_DATA_ROOT must be set"}', script
                )
                self.assertIn('${POIDH_RUN_ROOT:?"POIDH_RUN_ROOT must be set"}', script)
                self.assertIn(
                    '${POIDH_PYTHON_DEPS:?"POIDH_PYTHON_DEPS must be set"}', script
                )
                self.assertIn('case "${DATA_ROOT}" in', script)
                self.assertIn('case "${RUN_ROOT}" in', script)
                self.assertIn('case "${DEPENDENCY_ROOT}" in', script)

    def test_smoke_job_checks_gpu_and_python_imports(self) -> None:
        script = _read(SMOKE_PATH)
        self.assertIn("nvidia-smi", script)
        self.assertIn("import torch", script)
        self.assertIn("import timm", script)
        self.assertIn("torch.cuda.is_available()", script)
        self.assertIn('torch.__version__.split("+", 1)[0] != "2.9.1"', script)
        self.assertIn('torchvision.__version__.split("+", 1)[0] != "0.24.1"', script)
        self.assertIn('timm.__version__ != "1.0.28"', script)
        self.assertLess(script.index("import torch"), script.index("import timm"))
        self.assertLess(
            script.index('"torch": torch.__version__'), script.index("import timm")
        )

    def test_training_job_uses_environment_profile_and_training_entrypoint(
        self,
    ) -> None:
        script = _read(TRAIN_PATH)
        self.assertIn('TRAIN_PROFILE="${TRAIN_PROFILE:-pilot}"', script)
        self.assertIn("python tools/train_detector.py \\\n", script)
        self.assertIn('"${DATA_ROOT}" \\\n', script)
        self.assertIn('"${RUN_DIRECTORY}" \\\n', script)
        self.assertIn('--profile "${TRAIN_PROFILE}"', script)
        self.assertIn("--seed 323", script)
        self.assertIn('TRAIN_WORKERS="${POIDH_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"', script)
        self.assertIn('--workers "${TRAIN_WORKERS}"', script)

    def test_training_worker_count_can_be_pinned_independently_of_slurm_rounding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed, invocation_log, _, _ = _run_training_script(
                temporary,
                run_name="pinned-workers",
                resume=False,
                worker_count="4",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            arguments = invocation_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(arguments[arguments.index("--workers") + 1], "4")

        with tempfile.TemporaryDirectory() as temporary:
            completed, invocation_log, _, _ = _run_training_script(
                temporary,
                run_name="invalid-workers",
                resume=False,
                worker_count="0",
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(invocation_log.exists())

    def test_fresh_training_accepts_only_an_absent_run_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed, invocation_log, run_root, pythonpath_log = _run_training_script(
                temporary, run_name="fresh", resume=False
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                pythonpath_log.read_text(encoding="utf-8").strip(),
                f"{(Path(temporary) / 'scratch' / 'python-dependencies').resolve()}:{MODULE_PYTHONPATH}",
            )
            arguments = invocation_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(arguments[0], "tools/train_detector.py")
            self.assertEqual(arguments[2], str(run_root.resolve() / "fresh"))
            self.assertNotIn("--resume", arguments)

        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "scratch" / "runs" / "existing"
            existing.mkdir(parents=True)
            completed, invocation_log, _, _ = _run_training_script(
                temporary, run_name="existing", resume=False
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(invocation_log.exists())

    def test_resume_requires_a_real_contained_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "scratch" / "runs" / "existing"
            existing.mkdir(parents=True)
            completed, invocation_log, run_root, pythonpath_log = _run_training_script(
                temporary, run_name="existing", resume=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(pythonpath_log.is_file())
            arguments = invocation_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(arguments[2], str(run_root.resolve() / "existing"))
            self.assertIn("--resume", arguments)

        with tempfile.TemporaryDirectory() as temporary:
            outside = Path(temporary) / "outside"
            link = Path(temporary) / "scratch" / "runs" / "escaped"
            outside.mkdir()
            link.parent.mkdir(parents=True)
            link.symlink_to(outside, target_is_directory=True)
            completed, invocation_log, _, _ = _run_training_script(
                temporary, run_name="escaped", resume=True
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(invocation_log.exists())

    def test_training_rejects_missing_invalid_or_external_dependency_root(self) -> None:
        for mode in ("missing", "bad", "outside"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                completed, invocation_log, _, pythonpath_log = _run_training_script(
                    temporary,
                    run_name="fresh",
                    resume=False,
                    dependency_mode=mode,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse(invocation_log.exists())
                self.assertFalse(pythonpath_log.exists())

    def test_smoke_uses_only_a_valid_contained_dependency_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed, _, _, pythonpath_log = _run_training_script(
                temporary,
                run_name="unused",
                resume=False,
                script_path=SMOKE_PATH,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                pythonpath_log.read_text(encoding="utf-8").strip(),
                f"{(Path(temporary) / 'scratch' / 'python-dependencies').resolve()}:{MODULE_PYTHONPATH}",
            )

        for mode in ("missing", "bad", "outside"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                completed, invocation_log, _, pythonpath_log = _run_training_script(
                    temporary,
                    run_name="unused",
                    resume=False,
                    dependency_mode=mode,
                    script_path=SMOKE_PATH,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse(invocation_log.exists())
                self.assertFalse(pythonpath_log.exists())

    def test_readme_gives_exact_transfer_submit_and_monitor_commands(self) -> None:
        readme = _read(README_PATH)
        required_commands = (
            "rsync -az",
            "hpc/wice_h100_smoke.slurm",
            '--time="${TRAIN_WALLTIME}"',
            'TRAIN_PROFILE="pilot"',
            "POIDH_WORKERS=4",
            "squeue --me",
            "sacct -j",
            "tail -f",
            "--export=POIDH_WORKERS=",
            "--no-deps --target",
            "timm==1.0.28",
            "PyTorch-bundle/2.9.1-foss-2025a-CUDA-12.8.0-whl",
        )
        for command in required_commands:
            with self.subTest(command=command):
                self.assertIn(command, readme)
        self.assertNotIn("--export=ALL", readme)
        self.assertIn('--export=POIDH_WORKERS="${POIDH_WORKERS}"', readme)
        self.assertIn('"${POIDH_PYTHON_DEPS}"', readme)

    def test_compute_jobs_never_install_or_download_packages(self) -> None:
        for path in (SMOKE_PATH, TRAIN_PATH):
            with self.subTest(path=path.name):
                script = _read(path)
                self.assertNotRegex(script, r"(?i)\b(?:pip|uv|conda)\s+install\b")
                self.assertNotRegex(script, r"(?i)\b(?:curl|wget)\b")

    def test_readme_states_cost_and_bart_first_policy(self) -> None:
        readme = _read(README_PATH)
        self.assertRegex(readme, r"597 credits(?:/| per )hour")
        self.assertIn("Bart", readme)
        self.assertRegex(readme, r"(?i)Bart.*smoke")
        self.assertRegex(readme, r"(?i)H100.*pilot")


if __name__ == "__main__":
    unittest.main()
