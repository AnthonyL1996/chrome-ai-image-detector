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

if (( $# != 0 && $# != 3 )); then
    echo "usage: $0 [VSC_SCRATCH POIDH_DATA_ROOT POIDH_RUN_ROOT]" >&2
    exit 2
fi
if (( $# == 3 )); then
    VSC_SCRATCH=$1
    POIDH_DATA_ROOT=$2
    POIDH_RUN_ROOT=$3
fi

VSC_SCRATCH=${VSC_SCRATCH:?"VSC_SCRATCH must be set"}
POIDH_DATA_ROOT=${POIDH_DATA_ROOT:?"POIDH_DATA_ROOT must be set"}
POIDH_RUN_ROOT=${POIDH_RUN_ROOT:?"POIDH_RUN_ROOT must be set"}

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

case "${DATA_ROOT}" in
    "${SCRATCH_ROOT}"/*) ;;
    *) echo "POIDH_DATA_ROOT must be below VSC_SCRATCH" >&2; exit 2 ;;
esac
case "${RUN_ROOT}" in
    "${SCRATCH_ROOT}"/*) ;;
    *) echo "POIDH_RUN_ROOT must be below VSC_SCRATCH" >&2; exit 2 ;;
esac

module --force purge
module load cluster/wice/gpu_h100
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0-whl

export PYTHONHASHSEED=323
export CUBLAS_WORKSPACE_CONFIG=:4096:8

nvidia-smi
python - <<'PY'
import timm
import torch
import torchvision

if not torch.cuda.is_available():
    raise RuntimeError("the requested H100 is not visible to PyTorch")

print(
    {
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "timm": timm.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
)
PY
