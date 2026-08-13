from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import os
import platform
import random
from typing import Any


_MAX_NUMPY_SEED = 2**32 - 1
_DEFAULT_CUBLAS_WORKSPACE = ":4096:8"
_VALID_CUBLAS_WORKSPACES = frozenset({_DEFAULT_CUBLAS_WORKSPACE, ":16:8"})


@dataclass(frozen=True, slots=True)
class EnvironmentFingerprint:
    python_version: str
    platform: str
    machine: str
    torch_version: str | None
    numpy_version: str | None
    timm_version: str | None
    cuda_available: bool
    cuda_version: str | None
    cudnn_version: str | None
    cuda_device: str | None

    def to_dict(self) -> dict[str, str | bool | None]:
        return asdict(self)


def configure_determinism(
    seed: int,
    *,
    torch_module: Any | None = None,
    numpy_module: Any | None = None,
) -> None:
    """Seed the training stack and enable deterministic Torch operations.

    Optional dependencies are imported only when the caller does not inject
    modules. Training cannot be reproducible without both Torch and NumPy, so a
    missing dependency is reported before any partial configuration is applied.

    Python's active hash secret cannot be inspected or changed after interpreter
    startup. Launchers must set ``PYTHONHASHSEED`` before starting Python, and
    training code must use canonical ordering rather than hash iteration order.
    """

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= _MAX_NUMPY_SEED
    ):
        raise ValueError(f"seed must be an integer from 0 to {_MAX_NUMPY_SEED}")

    torch = torch_module or _required_module("torch")
    numpy = numpy_module or _required_module("numpy")

    cublas_workspace = os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG", _DEFAULT_CUBLAS_WORKSPACE
    )
    if cublas_workspace not in _VALID_CUBLAS_WORKSPACES:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic CUDA"
        )
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_environment(
    *,
    torch_module: Any | None = None,
    numpy_module: Any | None = None,
    timm_module: Any | None = None,
    import_optional: bool = True,
) -> EnvironmentFingerprint:
    """Capture a JSON-serializable training environment fingerprint."""

    torch = _optional_module("torch", torch_module, import_optional)
    numpy = _optional_module("numpy", numpy_module, import_optional)
    timm = _optional_module("timm", timm_module, import_optional)

    cuda_available = bool(torch is not None and torch.cuda.is_available())
    cuda_version = None
    cudnn_version = None
    cuda_device = None
    if torch is not None:
        cuda_version = _string_or_none(getattr(torch.version, "cuda", None))
        cudnn_value = torch.backends.cudnn.version()
        cudnn_version = _string_or_none(cudnn_value)
        if cuda_available:
            cuda_device = str(torch.cuda.get_device_name(0))

    return EnvironmentFingerprint(
        python_version=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        torch_version=_module_version(torch),
        numpy_version=_module_version(numpy),
        timm_version=_module_version(timm),
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        cuda_device=cuda_device,
    )


def _required_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        raise RuntimeError(
            f"deterministic training requires {name}; install the training extras"
        ) from error


def _optional_module(
    name: str, supplied: Any | None, should_import: bool
) -> Any | None:
    if supplied is not None:
        return supplied
    if not should_import:
        return None
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        return None


def _module_version(module: Any | None) -> str | None:
    if module is None:
        return None
    return _string_or_none(getattr(module, "__version__", None))


def _string_or_none(value: object | None) -> str | None:
    return str(value) if value is not None else None
