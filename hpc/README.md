# wICE H100 jobs

These scripts run the detector on one wICE H100 GPU under account
`lp_edu_maibi_anndl`. They use only the pinned wICE PyTorch module; the jobs do
not download code, data, or packages and must not receive credentials or
secrets through exported environment variables.

## Cost and promotion policy

An H100 allocation costs **597 credits/hour**. Run the import and tiny-data
smoke checks on **Bart** first; a Bart smoke must pass before any H100 job is
submitted. Use `wice_h100_smoke.slurm` only to confirm the wICE GPU/module
environment, then promote to a short **H100 pilot**. Review pilot logs and
validation output before considering a longer `full` profile. A ten-minute
H100 smoke can consume up to 99.5 credits.

## Transfer code and prepared data

Run these commands on the local workstation from the repository root. Replace
the login placeholder and the local dataset path. The dataset must already be
prepared and contain `manifest.json`, `splits.json`, `preparation.json`, and the
materialized images.

```bash
export VSC_LOGIN="vscXXXXX@login.hpc.kuleuven.be"
export LOCAL_DATASET_ROOT="/absolute/path/to/prepared-monet"
export REMOTE_SCRATCH="$(ssh "${VSC_LOGIN}" 'printf "%s" "$VSC_SCRATCH"')"

ssh "${VSC_LOGIN}" "mkdir -p '${REMOTE_SCRATCH}/poidh-ai-detector' '${REMOTE_SCRATCH}/poidh-data/monet-v1' '${REMOTE_SCRATCH}/poidh-runs/logs'"
rsync -az --exclude='.git/' --exclude='.venv/' --exclude='.env' --exclude='.env.*' ./ "${VSC_LOGIN}:${REMOTE_SCRATCH}/poidh-ai-detector/"
rsync -az "${LOCAL_DATASET_ROOT}/" "${VSC_LOGIN}:${REMOTE_SCRATCH}/poidh-data/monet-v1/"
```

Do not transfer SSH keys, tokens, `.env` files, third-party pretrained weights,
or unreviewed data. Re-run both `rsync` commands whenever code or prepared data
changes; neither command deletes remote files.

## Submit the environment smoke

Log in, define the two required scratch paths, and submit from the repository
root. Both scripts reject data or run paths that resolve outside
`VSC_SCRATCH`.

```bash
ssh "${VSC_LOGIN}"
cd "${VSC_SCRATCH}/poidh-ai-detector"
export POIDH_DATA_ROOT="${VSC_SCRATCH}/poidh-data/monet-v1"
export POIDH_RUN_ROOT="${VSC_SCRATCH}/poidh-runs"
mkdir -p "${POIDH_RUN_ROOT}/logs"

SMOKE_JOB_ID="$(sbatch --parsable --output="${POIDH_RUN_ROOT}/logs/h100-smoke-%j.out" --export=NONE hpc/wice_h100_smoke.slurm "${VSC_SCRATCH}" "${POIDH_DATA_ROOT}" "${POIDH_RUN_ROOT}")"
printf 'smoke job: %s\n' "${SMOKE_JOB_ID}"
```

The smoke job is hard-capped at `00:10:00` and runs `nvidia-smi` plus imports
for PyTorch, torchvision, and timm. It does not train or modify the dataset.
`--export=NONE` prevents unrelated login-shell variables, including accidental
credentials, from entering the job. Scratch paths are non-secret positional
arguments; Slurm still supplies its `SLURM_*` variables, and the scripts use a
login Bash shell to initialize the module command before loading the pinned
environment.

## Submit staged training

Start with `pilot`. `TRAIN_WALLTIME` is passed as an `sbatch` option because
Slurm does not expand shell variables in `#SBATCH` directives. The script's
`02:00:00` value is only a default; the submission command below is the exact
walltime override. `TRAIN_PROFILE` accepts `overfit`, `smoke`, `pilot`, or
`full` and defaults to `pilot`.

```bash
TRAIN_PROFILE="pilot"
TRAIN_WALLTIME="02:00:00"
TRAIN_JOB_ID="$(sbatch --parsable --time="${TRAIN_WALLTIME}" --output="${POIDH_RUN_ROOT}/logs/h100-train-%j.out" --export=NONE hpc/wice_h100_train.slurm "${VSC_SCRATCH}" "${POIDH_DATA_ROOT}" "${POIDH_RUN_ROOT}" "${TRAIN_PROFILE}" 0)"
printf 'training job: %s\n' "${TRAIN_JOB_ID}"
```

The positional profile value initializes the script's `TRAIN_PROFILE` setting
without exporting the rest of the submission environment. Direct execution
also accepts `TRAIN_PROFILE` from the environment.
Each fresh job writes to
`${POIDH_RUN_ROOT}/${TRAIN_PROFILE}-${SLURM_JOB_ID}`. To resume a
transactionally published run after a walltime stop, reuse its one-component
name and pass the explicit resume flag:

```bash
TRAIN_PROFILE="pilot"
TRAIN_WALLTIME="02:00:00"
POIDH_RUN_NAME="pilot-12345678"
TRAIN_JOB_ID="$(sbatch --parsable --time="${TRAIN_WALLTIME}" --output="${POIDH_RUN_ROOT}/logs/h100-train-%j.out" --export=NONE hpc/wice_h100_train.slurm "${VSC_SCRATCH}" "${POIDH_DATA_ROOT}" "${POIDH_RUN_ROOT}" "${TRAIN_PROFILE}" 1 "${POIDH_RUN_NAME}")"
```

The script rejects a pre-existing fresh-run path. Resume requires an existing,
non-symlink directory whose canonical path remains below the canonical run
root, plus a matching training provenance contract.

## Monitor and inspect

Use the job ID printed by `sbatch`:

```bash
JOB_ID="${TRAIN_JOB_ID}"
squeue --me
scontrol show job "${JOB_ID}"
sacct -j "${JOB_ID}" --format=JobID,JobName,Cluster,Partition,State,Elapsed,Timelimit,AllocTRES,ExitCode
tail -f "${POIDH_RUN_ROOT}/logs/h100-train-${JOB_ID}.out"
```

For the smoke job, set `JOB_ID="${SMOKE_JOB_ID}"` and tail
`${POIDH_RUN_ROOT}/logs/h100-smoke-${JOB_ID}.out`. A zero Slurm exit code is
necessary but not sufficient: inspect the reported GPU, CUDA/PyTorch versions,
training epoch JSON, validation metrics, and the run's provenance files before
promoting to a larger profile.
