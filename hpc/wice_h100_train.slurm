#!/bin/bash -l
#SBATCH --job-name=poidh-h100-train
#SBATCH --account=lp_edu_maibi_anndl
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

if (( $# != 0 && $# != 6 && $# != 7 )); then
    echo "usage: $0 [VSC_SCRATCH POIDH_DATA_ROOT POIDH_RUN_ROOT POIDH_PYTHON_DEPS PROFILE RESUME [RUN_NAME]]" >&2
    exit 2
fi
if (( $# >= 6 )); then
    VSC_SCRATCH=$1
    POIDH_DATA_ROOT=$2
    POIDH_RUN_ROOT=$3
    POIDH_PYTHON_DEPS=$4
    TRAIN_PROFILE=$5
    TRAIN_RESUME=$6
    if (( $# == 7 )); then
        POIDH_RUN_NAME=$7
    fi
fi

VSC_SCRATCH=${VSC_SCRATCH:?"VSC_SCRATCH must be set"}
POIDH_DATA_ROOT=${POIDH_DATA_ROOT:?"POIDH_DATA_ROOT must be set"}
POIDH_RUN_ROOT=${POIDH_RUN_ROOT:?"POIDH_RUN_ROOT must be set"}
POIDH_PYTHON_DEPS=${POIDH_PYTHON_DEPS:?"POIDH_PYTHON_DEPS must be set"}
TRAIN_PROFILE="${TRAIN_PROFILE:-pilot}"
TRAIN_RESUME="${TRAIN_RESUME:-0}"
TRAIN_WORKERS="${POIDH_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

canonical_directory() {
    local requested=$1
    local description=$2
    if [[ ! -d "${requested}" ]]; then
        echo "${description} must be an existing directory" >&2
        return 2
    fi
    (cd -P -- "${requested}" && pwd -P)
}

SCRATCH_ROOT="$(canonical_directory "${VSC_SCRATCH}" VSC_SCRATCH)"
DATA_ROOT="$(canonical_directory "${POIDH_DATA_ROOT}" POIDH_DATA_ROOT)"
RUN_ROOT="$(canonical_directory "${POIDH_RUN_ROOT}" POIDH_RUN_ROOT)"
DEPENDENCY_ROOT="$(canonical_directory "${POIDH_PYTHON_DEPS}" POIDH_PYTHON_DEPS)"

case "${DATA_ROOT}" in
    "${SCRATCH_ROOT}"/*) ;;
    *) echo "POIDH_DATA_ROOT must be below VSC_SCRATCH" >&2; exit 2 ;;
esac
case "${RUN_ROOT}" in
    "${SCRATCH_ROOT}"/*) ;;
    *) echo "POIDH_RUN_ROOT must be below VSC_SCRATCH" >&2; exit 2 ;;
esac
case "${DEPENDENCY_ROOT}" in
    "${SCRATCH_ROOT}"/*) ;;
    *) echo "POIDH_PYTHON_DEPS must be below VSC_SCRATCH" >&2; exit 2 ;;
esac
if [[ "${DEPENDENCY_ROOT}" == *:* ]]; then
    echo "POIDH_PYTHON_DEPS must not contain a colon" >&2
    exit 2
fi
if [[ -L "${DEPENDENCY_ROOT}/timm" || -L "${DEPENDENCY_ROOT}/timm/__init__.py" || ! -f "${DEPENDENCY_ROOT}/timm/__init__.py" || -L "${DEPENDENCY_ROOT}/timm-1.0.28.dist-info" || ! -d "${DEPENDENCY_ROOT}/timm-1.0.28.dist-info" ]]; then
    echo "POIDH_PYTHON_DEPS must contain the staged timm 1.0.28 target" >&2
    exit 2
fi
case "${TRAIN_PROFILE}" in
    overfit|smoke|pilot|full) ;;
    *) echo "TRAIN_PROFILE must be overfit, smoke, pilot, or full" >&2; exit 2 ;;
esac
case "${TRAIN_RESUME}" in
    0|1) ;;
    *) echo "TRAIN_RESUME must be 0 or 1" >&2; exit 2 ;;
esac
if ! [[ "${TRAIN_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "POIDH_WORKERS must be a positive integer" >&2
    exit 2
fi

SUBMIT_DIRECTORY=${SLURM_SUBMIT_DIR:?"SLURM_SUBMIT_DIR must be set"}
if [[ ! -f "${SUBMIT_DIRECTORY}/tools/train_detector.py" ]]; then
    echo "submit this job from the repository root" >&2
    exit 2
fi
cd "${SUBMIT_DIRECTORY}"

RUN_NAME="${POIDH_RUN_NAME:-${TRAIN_PROFILE}-${SLURM_JOB_ID:?}}"
case "${RUN_NAME}" in
    ""|.|..|*/*) echo "POIDH_RUN_NAME must be one path component" >&2; exit 2 ;;
esac
CANDIDATE_RUN_DIRECTORY="${RUN_ROOT}/${RUN_NAME}"

RESUME_ARGUMENTS=()
if [[ "${TRAIN_RESUME}" == "1" ]]; then
    if [[ -L "${CANDIDATE_RUN_DIRECTORY}" || ! -d "${CANDIDATE_RUN_DIRECTORY}" ]]; then
        echo "resume requires an existing, non-symlink run directory" >&2
        exit 2
    fi
    RUN_DIRECTORY="$(canonical_directory "${CANDIDATE_RUN_DIRECTORY}" run_directory)"
    case "${RUN_DIRECTORY}" in
        "${RUN_ROOT}"/*) ;;
        *) echo "resume run directory escaped POIDH_RUN_ROOT" >&2; exit 2 ;;
    esac
    RESUME_ARGUMENTS+=(--resume)
else
    if [[ -e "${CANDIDATE_RUN_DIRECTORY}" || -L "${CANDIDATE_RUN_DIRECTORY}" ]]; then
        echo "fresh run directory must not already exist" >&2
        exit 2
    fi
    RUN_DIRECTORY="${CANDIDATE_RUN_DIRECTORY}"
fi

module --force purge
module load cluster/wice/gpu_h100
module load PyTorch-bundle/2.9.1-foss-2025a-CUDA-12.8.0-whl

export PYTHONHASHSEED=323
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${DEPENDENCY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

nvidia-smi
python tools/train_detector.py \
    "${DATA_ROOT}" \
    "${RUN_DIRECTORY}" \
    --profile "${TRAIN_PROFILE}" \
    --seed 323 \
    --workers "${TRAIN_WORKERS}" \
    "${RESUME_ARGUMENTS[@]}"
