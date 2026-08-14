"""Wrap the public Community Forensics ONNX export for the MV3 runtime.

The source model is kept outside this repository because its upstream artifact
is large. This deterministic transform renames the I/O tensors and bakes the
fixed +2.29 calibration offset plus sigmoid into the graph, so the extension
can expose a single probability output without JavaScript-side calibration.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


CALIBRATION_OFFSET = 2.29
SOURCE_MODEL_SHA256 = "0fb7bf7c74cf2808b9c0b6a068126739cb5b2dae72be33fa971babe912ec466e"


def wrap_model(source: Path, destination: Path) -> str:
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_digest != SOURCE_MODEL_SHA256:
        raise ValueError(
            "source ONNX digest does not match the audited public export: "
            f"{source_digest}"
        )
    model = onnx.load(source, load_external_data=False)
    graph = model.graph
    if [value.name for value in graph.input] != ["pixel_values"]:
        raise ValueError("source ONNX input contract is not pixel_values-only")
    if [value.name for value in graph.output] != ["logit"]:
        raise ValueError("source ONNX output contract is not logit-only")
    for value in graph.input:
        if value.name == "pixel_values":
            value.name = "image"
    for node in graph.node:
        node.input[:] = ["image" if name == "pixel_values" else name for name in node.input]
    for value in (*graph.input, *graph.output):
        dimensions = value.type.tensor_type.shape.dim
        if dimensions and dimensions[0].WhichOneof("value") == "dim_param":
            dimensions[0].ClearField("dim_param")
            dimensions[0].dim_value = 1

    original_output = "logit"
    offset_name = "poidh_calibration_offset"
    calibrated_logit = "poidh_calibrated_logit"
    output_name = "probability_ai"
    graph.initializer.append(
        numpy_helper.from_array(np.asarray(CALIBRATION_OFFSET, dtype=np.float32), name=offset_name)
    )
    graph.node.extend(
        [
            helper.make_node(
                "Add",
                [original_output, offset_name],
                [calibrated_logit],
                name="poidh_calibration_add",
            ),
            helper.make_node(
                "Sigmoid",
                [calibrated_logit],
                [output_name],
                name="poidh_calibration_sigmoid",
            ),
        ]
    )
    output = graph.output[0]
    output.name = output_name
    output.type.tensor_type.elem_type = TensorProto.FLOAT
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="public Community Forensics-derived ONNX export")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("extension/model/detector.onnx"),
    )
    args = parser.parse_args()
    digest = wrap_model(args.source, args.destination)
    print(f"{args.destination} {args.destination.stat().st_size} bytes sha256={digest}")


if __name__ == "__main__":
    main()
