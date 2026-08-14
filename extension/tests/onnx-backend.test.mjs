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
    input_shape: [1, 3, 384, 384],
    input_dtype: "float32",
    output_name: "probability_ai",
    output_shape: [1, 1],
    output_dtype: "float32",
    output_semantics: "calibrated_probability_ai",
    calibration: "sigmoid_embedded_offset_2.29",
    preprocessing: {
      resize_shorter_side: 440,
      center_crop: [384, 384],
      interpolation: "canvas_2d_high",
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
      return {
        probability_ai: {
          type: "float32",
          data: Float32Array.of(0.75),
          dims: [1, 1],
        },
      };
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
    imageTensor: imageTensor || (() => new Float32Array(3 * 384 * 384)),
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
  assert.deepEqual(deps.seen[0].image.dims, [1, 3, 384, 384]);
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

test("bounds stalled image loads and aborts their request", async () => {
  const deps = dependencies();
  const modelFetch = deps.fetchImpl;
  let aborted = false;
  deps.fetchImpl = async (url, options) => {
    if (url === "https://example.test/stalled.png") {
      options.signal.addEventListener("abort", () => { aborted = true; }, { once: true });
      return new Promise(() => {});
    }
    return modelFetch(url, options);
  };
  const backend = await loadOnnxBackend({
    ort: deps.ort,
    modelUrl: "detector.onnx",
    metadataUrl: "metadata.json",
    fetchImpl: deps.fetchImpl,
    cryptoImpl: deps.cryptoImpl,
    imageLoadTimeoutMs: 5,
    imageBitmapFactory: async () => ({ width: 1, height: 1, close() {} }),
    BlobConstructor: class Blob {},
  });

  assert.deepEqual(
    await backend.scoreImages([{ id: "stalled", source: "https://example.test/stalled.png" }]),
    [{
      id: "stalled",
      status: "error",
      code: "IMAGE_DECODE_FAILED",
      message: "image decode/load timed out",
    }],
  );
  assert.equal(aborted, true);
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

test("rejects remote model and metadata URLs", async () => {
  const deps = dependencies();
  await assert.rejects(
    loadOnnxBackend({
      ort: deps.ort,
      modelUrl: "https://models.example/detector.onnx",
      metadataUrl: "metadata.json",
      fetchImpl: deps.fetchImpl,
      cryptoImpl: deps.cryptoImpl,
      imageTensor: deps.imageTensor,
    }),
    /extension-local/i,
  );
});

test("rejects a model output with the wrong type or shape", async () => {
  for (const output of [
    { type: "float64", data: Float32Array.of(0.75), dims: [1, 1] },
    { type: "float32", data: Float32Array.of(0.75), dims: [1] },
  ]) {
    const deps = dependencies({
      session: {
        inputNames: ["image"],
        outputNames: ["probability_ai"],
        run: async () => ({ probability_ai: output }),
      },
    });
    const backend = await loadOnnxBackend({
      ort: deps.ort,
      modelUrl: "detector.onnx",
      metadataUrl: "metadata.json",
      fetchImpl: deps.fetchImpl,
      cryptoImpl: deps.cryptoImpl,
      imageTensor: deps.imageTensor,
    });
    await assert.deepEqual(
      await backend.scoreImages([{ id: "bad-output", source: "https://example.test/x" }]),
      [{
        id: "bad-output",
        status: "error",
        code: "MODEL_INFERENCE_FAILED",
        message: "model output has an invalid shape",
      }],
    );
  }
});

test("matches the 440px shorter-side and 384px center-crop preprocessing", async () => {
  const width = 447;
  const height = 320;
  const outputWidth = 384;
  const outputHeight = 384;
  const rgba = new Uint8ClampedArray(outputWidth * outputHeight * 4);
  for (let y = 0; y < outputHeight; y += 1) {
    for (let x = 0; x < outputWidth; x += 1) {
      const offset = (y * outputWidth + x) * 4;
      rgba[offset] = (x * 13 + y * 7) % 256;
      rgba[offset + 1] = (x * 5 + y * 17 + 3) % 256;
      rgba[offset + 2] = (x * 19 + y * 11 + 9) % 256;
      rgba[offset + 3] = 255;
    }
  }

  let captured;
  let drawArgs;
  let smoothing;
  const deps = dependencies({
    session: {
      inputNames: ["image"],
      outputNames: ["probability_ai"],
      async run(feeds) {
        captured = feeds.image.data;
        return {
          probability_ai: {
            type: "float32",
            data: Float32Array.of(0.5),
            dims: [1, 1],
          },
        };
      },
    },
  });
  const modelFetch = deps.fetchImpl;
  deps.fetchImpl = async (url, options) => {
    if (url === "https://example.test/source.png") {
      return response(rgba.buffer);
    }
    return modelFetch(url, options);
  };
  const backend = await loadOnnxBackend({
    ort: deps.ort,
    modelUrl: "detector.onnx",
    metadataUrl: "metadata.json",
    fetchImpl: deps.fetchImpl,
    cryptoImpl: deps.cryptoImpl,
    imageBitmapFactory: async () => ({ width, height, close() {} }),
    canvasFactory: (canvasWidth, canvasHeight) => ({
      getContext: () => ({
        set imageSmoothingEnabled(value) {
          smoothing = { ...(smoothing || {}), enabled: value };
        },
        set imageSmoothingQuality(value) {
          smoothing = { ...(smoothing || {}), quality: value };
        },
        drawImage(...args) {
          drawArgs = args;
        },
        getImageData: () => ({ data: rgba }),
      }),
      width: canvasWidth,
      height: canvasHeight,
    }),
    BlobConstructor: class Blob {},
  });

  await backend.scoreImages([{ id: "fixture", source: "https://example.test/source.png" }]);
  assert.ok(captured instanceof Float32Array);
  assert.equal(captured.length, 3 * outputWidth * outputHeight);
  assert.equal(drawArgs[0].width, width);
  assert.equal(drawArgs[0].height, height);
  assert.deepEqual(drawArgs.slice(1), [-115.5, -28, 615, 440]);
  assert.deepEqual(smoothing, { enabled: true, quality: "high" });
  const plane = outputWidth * outputHeight;
  for (const [x, y] of [[0, 0], [11, 7], [383, 383], [203, 121]]) {
    const pixel = y * outputWidth + x;
    const rgbaOffset = pixel * 4;
    const expected = [0, 1, 2].map((channel) => (
      rgba[rgbaOffset + channel] / 255 - [0.485, 0.456, 0.406][channel]
    ) / [0.229, 0.224, 0.225][channel]);
    assert.ok(Math.abs(captured[pixel] - expected[0]) < 1e-6);
    assert.ok(Math.abs(captured[plane + pixel] - expected[1]) < 1e-6);
    assert.ok(Math.abs(captured[2 * plane + pixel] - expected[2]) < 1e-6);
  }
});

test("rejects extreme source dimensions before canvas rasterization", async () => {
  const deps = dependencies();
  const modelFetch = deps.fetchImpl;
  deps.fetchImpl = async (url, options) => {
    if (url === "https://example.test/wide.png") {
      return response([1]);
    }
    return modelFetch(url, options);
  };
  const backend = await loadOnnxBackend({
    ort: deps.ort,
    modelUrl: "detector.onnx",
    metadataUrl: "metadata.json",
    fetchImpl: deps.fetchImpl,
    cryptoImpl: deps.cryptoImpl,
    imageBitmapFactory: async () => ({ width: 16_384, height: 1, close() {} }),
    canvasFactory: () => { throw new Error("canvas must not be allocated"); },
    BlobConstructor: class Blob {},
  });

  assert.deepEqual(
    await backend.scoreImages([{ id: "wide", source: "https://example.test/wide.png" }]),
    [{
      id: "wide",
      status: "error",
      code: "IMAGE_DECODE_FAILED",
      message: "decoded image dimensions exceed the runtime limits",
    }],
  );
});
