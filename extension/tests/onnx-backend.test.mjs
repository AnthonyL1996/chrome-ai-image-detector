import assert from "node:assert/strict";
import test from "node:test";

import { loadOnnxBackend } from "../runtime/onnx-backend.mjs";

const MODEL_DIGEST = "0".repeat(64);

function metadata(overrides = {}) {
  return {
    schema_version: 1,
    model_format: "onnx",
    model_sha256: MODEL_DIGEST,
    input_name: "image",
    input_shape: [1, 3, 224, 224],
    input_dtype: "float32",
    output_name: "probability_ai",
    output_shape: [1, 1],
    output_dtype: "float32",
    output_semantics: "calibrated_probability_ai",
    calibration: "platt_embedded",
    preprocessing: {
      resize: [224, 224],
      interpolation: "bicubic",
      channel_order: "RGB",
      layout: "NCHW",
      input_dtype: "uint8",
      output_dtype: "float32",
      input_scale: 1 / 255,
      mean: [0.485, 0.456, 0.406],
      standard_deviation: [0.229, 0.224, 0.225],
    },
    single_file: true,
    uses_external_data: false,
    max_model_bytes: 100 * 1024 * 1024,
    ...overrides,
  };
}

function response(payload, { ok = true } = {}) {
  return {
    ok,
    async arrayBuffer() {
      return payload instanceof ArrayBuffer
        ? payload
        : Uint8Array.from(payload).buffer;
    },
    headers: { get: () => null },
  };
}

function dependencies({ session, imageTensor } = {}) {
  const seen = [];
  const actualSession = session || {
    inputNames: ["image"],
    outputNames: ["probability_ai"],
    async run(feeds) {
      seen.push(feeds);
      return { probability_ai: { data: Float32Array.of(0.75), dims: [1, 1] } };
    },
  };
  const ort = {
    InferenceSession: { create: async () => actualSession },
    Tensor: class Tensor {
      constructor(type, data, dims) {
        this.type = type;
        this.data = data;
        this.dims = dims;
      }
    },
  };
  const fetchImpl = async (url) => {
    if (url === "metadata.json") {
      return response(new TextEncoder().encode(JSON.stringify(metadata())));
    }
    if (url === "detector.onnx") {
      return response([1, 2, 3]);
    }
    throw new Error(`unexpected resource: ${url}`);
  };
  const cryptoImpl = {
    subtle: {
      digest: async () => new Uint8Array(32).buffer,
    },
  };
  return {
    ort,
    fetchImpl,
    cryptoImpl,
    imageTensor: imageTensor || (() => Float32Array.of(0.1)),
    seen,
  };
}

test("loads a hash-checked local ONNX backend and scores images", async () => {
  const deps = dependencies();
  const backend = await loadOnnxBackend({
    ort: deps.ort,
    modelUrl: "detector.onnx",
    metadataUrl: "metadata.json",
    fetchImpl: deps.fetchImpl,
    cryptoImpl: deps.cryptoImpl,
    imageTensor: deps.imageTensor,
  });

  assert.deepEqual(backend.status(), {
    engine: "onnxruntime-web-wasm",
    localOnly: true,
    ready: true,
  });
  assert.deepEqual(
    await backend.scoreImages([
      { id: "one", source: "https://example.test/one.png", alt: "One" },
    ]),
    [{ id: "one", status: "ok", confidence: 0.75 }],
  );
  assert.equal(deps.seen.length, 1);
  assert.deepEqual(deps.seen[0].image.dims, [1, 3, 224, 224]);
});

test("returns a terminal per-image error when image tensor preparation fails", async () => {
  const deps = dependencies({ imageTensor: async () => { throw new Error("decode"); } });
  const backend = await loadOnnxBackend({
    ort: deps.ort,
    modelUrl: "detector.onnx",
    metadataUrl: "metadata.json",
    fetchImpl: deps.fetchImpl,
    cryptoImpl: deps.cryptoImpl,
    imageTensor: deps.imageTensor,
  });

  assert.deepEqual(
    await backend.scoreImages([{ id: "bad", source: "https://example.test/bad.png", alt: "" }]),
    [{ id: "bad", status: "error", code: "IMAGE_DECODE_FAILED", message: "decode" }],
  );
});

test("rejects a model whose digest or session I/O contract is wrong", async () => {
  const deps = dependencies({
    session: { inputNames: ["pixels"], outputNames: ["probability_ai"], run: async () => ({}) },
  });
  await assert.rejects(
    loadOnnxBackend({
      ort: deps.ort,
      modelUrl: "detector.onnx",
      metadataUrl: "metadata.json",
      fetchImpl: deps.fetchImpl,
      cryptoImpl: { subtle: { digest: async () => Uint8Array.of(1).buffer } },
      imageTensor: deps.imageTensor,
    }),
    /digest/,
  );

  const ioDeps = dependencies({
    session: { inputNames: ["pixels"], outputNames: ["probability_ai"], run: async () => ({}) },
  });
  await assert.rejects(
    loadOnnxBackend({
      ort: ioDeps.ort,
      modelUrl: "detector.onnx",
      metadataUrl: "metadata.json",
      fetchImpl: ioDeps.fetchImpl,
      cryptoImpl: ioDeps.cryptoImpl,
      imageTensor: ioDeps.imageTensor,
    }),
    /input|output|contract/,
  );
});
