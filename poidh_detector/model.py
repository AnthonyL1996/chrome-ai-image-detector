from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib
from typing import Any


_ARCHITECTURE = "convnextv2_nano"


@dataclass(frozen=True, slots=True)
class ConvNeXtV2NanoConfig:
    num_classes: int = 1
    input_channels: int = 3

    def __post_init__(self) -> None:
        for field_name, value in (
            ("num_classes", self.num_classes),
            ("input_channels", self.input_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


def create_convnextv2_nano(
    config: ConvNeXtV2NanoConfig,
    *,
    pretrained: bool = False,
    timm_module: Any | None = None,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> Any:
    """Create the detector architecture with random initialization only."""

    if pretrained is not False:
        raise ValueError("pretrained weights are forbidden for this model")
    if not isinstance(config, ConvNeXtV2NanoConfig):
        raise TypeError("config must be a ConvNeXtV2NanoConfig")

    timm = timm_module
    if timm is None:
        try:
            timm = import_module("timm")
        except ModuleNotFoundError as error:
            if error.name != "timm":
                raise
            raise RuntimeError(
                "model creation requires timm; install timm via the training extras"
            ) from error

    return timm.create_model(
        _ARCHITECTURE,
        pretrained=False,
        num_classes=config.num_classes,
        in_chans=config.input_channels,
    )
