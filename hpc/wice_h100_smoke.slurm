#!/bin/bash -l
#SBATCH --job-name=poidh-h100-smoke
#SBATCH --account=lp_edu_maibi_anndl
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail

if (( $# != 0 && $# != 4 )); then
    echo "usage: $0 [VSC_SCRATCH POIDH_DATA_ROOT POIDH_RUN_ROOT POIDH_PYTHON_DEPS]" >&2
    exit 2
fi
if (( $# == 4 )); then
    VSC_SCRATCH=$1
    POIDH_DATA_ROOT=$2
    POIDH_RUN_ROOT=$3
    POIDH_PYTHON_DEPS=$4
fi

VSC_SCRATCH=${VSC_SCRATCH:?"VSC_SCRATCH must be set"}
POIDH_DATA_ROOT=${POIDH_DATA_ROOT:?"POIDH_DATA_ROOT must be set"}
POIDH_RUN_ROOT=${POIDH_RUN_ROOT:?"POIDH_RUN_ROOT must be set"}
POIDH_PYTHON_DEPS=${POIDH_PYTHON_DEPS:?"POIDH_PYTHON_DEPS must be set"}

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

module --force purge
module load cluster/wice/gpu_h100
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0-whl

export PYTHONHASHSEED=323
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${DEPENDENCY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

nvidia-smi
python - <<'PY'
import torch
import torchvision

if not torch.cuda.is_available():
    raise RuntimeError("the requested H100 is not visible to PyTorch")

print(
    {
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
)
PY
python - <<'PY'
import timm

if timm.__version__ != "1.0.28":
    raise RuntimeError(f"expected timm 1.0.28, found {timm.__version__}")

print({"timm": timm.__version__})
PY
