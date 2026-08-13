const IMAGE_SIZE = 224;
const CHANNELS = 3;
const TENSOR_LENGTH = CHANNELS * IMAGE_SIZE * IMAGE_SIZE;
const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
const MAX_MODEL_BYTES = 100 * 1024 * 1024;
const MODEL_INPUT = "image";
const MODEL_OUTPUT = "probability_ai";
const SUPPORTED_SOURCE = /^(?:https?:|blob:|data:image\/)/i;
const MEAN = [0.485, 0.456, 0.406];
const STANDARD_DEVIATION = [0.229, 0.224, 0.225];

export async function loadOnnxBackend({
  ort,
  modelUrl,
  metadataUrl,
  fetchImpl = globalThis.fetch,
  cryptoImpl = globalThis.crypto,
  imageTensor,
  imageBitmapFactory = globalThis.createImageBitmap,
  canvasFactory = (width, height) => new OffscreenCanvas(width, height),
  BlobConstructor = globalThis.Blob,
} = {}) {
  if (!ort?.InferenceSession || typeof ort.InferenceSession.create !== "function") {
    throw new TypeError("onnxruntime-web must provide InferenceSession.create");
  }
  if (typeof ort.Tensor !== "function") {
    throw new TypeError("onnxruntime-web must provide Tensor");
  }
  if (typeof fetchImpl !== "function") {
    throw new TypeError("a fetch implementation is required");
  }
  if (typeof modelUrl !== "string" || !modelUrl.trim()) {
    throw new TypeError("modelUrl must be a non-empty string");
  }
  if (typeof metadataUrl !== "string" || !metadataUrl.trim()) {
    throw new TypeError("metadataUrl must be a non-empty string");
  }
  if (!cryptoImpl?.subtle || typeof cryptoImpl.subtle.digest !== "function") {
    throw new TypeError("Web Crypto SHA-256 is required");
  }

  const [modelBytes, metadata] = await Promise.all([
    loadResourceBytes(fetchImpl, modelUrl, "ONNX model"),
    loadMetadata(fetchImpl, metadataUrl),
  ]);
  validateMetadata(metadata);
  if (modelBytes.byteLength > MAX_MODEL_BYTES) {
    throw new Error("ONNX model exceeds the 100 MiB runtime limit");
  }
  if (metadata.max_model_bytes !== MAX_MODEL_BYTES) {
    throw new Error("ONNX metadata model size policy is not fixed");
  }
  const digest = await sha256Hex(cryptoImpl, modelBytes);
  if (digest !== metadata.model_sha256) {
    throw new Error("ONNX model digest does not match metadata");
  }

  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  validateSession(session);
  const tensorLoader = imageTensor || createImageTensorLoader({
    fetchImpl,
    imageBitmapFactory,
    canvasFactory,
    BlobConstructor,
  });

  return Object.freeze({
    status() {
      return { engine: "onnxruntime-web-wasm", localOnly: true, ready: true };
    },

    async scoreImages(images) {
      if (!Array.isArray(images)) {
        throw new TypeError("images must be an array");
      }
      const results = [];
      for (const image of images) {
        results.push(await scoreImage(image, session, ort, tensorLoader));
      }
      return results;
    },
  });
}

async function scoreImage(image, session, ort, tensorLoader) {
  const id = typeof image?.id === "string" ? image.id : "";
  if (!id) {
    throw new TypeError("each image requires a non-empty id");
  }
  try {
    const data = await tensorLoader(image.source);
    if (!(data instanceof Float32Array) || data.length !== TENSOR_LENGTH) {
      throw new Error("image tensor has an invalid shape");
    }
    const input = new ort.Tensor("float32", data, [1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]);
    const outputs = await session.run({ [MODEL_INPUT]: input });
    const confidence = readProbability(outputs);
    return { id, status: "ok", confidence };
  } catch (error) {
    const message = safeMessage(error);
    const code = message.includes("tensor") || message.includes("decode")
      ? "IMAGE_DECODE_FAILED"
      : "MODEL_INFERENCE_FAILED";
    return { id, status: "error", code, message };
  }
}

function readProbability(outputs) {
  const output = outputs?.[MODEL_OUTPUT];
  const values = output?.data;
  if (!values || values.length !== 1) {
    throw new Error("model output has an invalid shape");
  }
  const confidence = Number(values[0]);
  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    throw new Error("model output probability is invalid");
  }
  return confidence;
}

function validateSession(session) {
  if (
    !session ||
    !Array.isArray(session.inputNames) ||
    !Array.isArray(session.outputNames) ||
    session.inputNames.length !== 1 ||
    session.outputNames.length !== 1 ||
    session.inputNames[0] !== MODEL_INPUT ||
    session.outputNames[0] !== MODEL_OUTPUT ||
    typeof session.run !== "function"
  ) {
    throw new Error("ONNX session I/O contract is invalid");
  }
}

function validateMetadata(metadata) {
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    throw new Error("ONNX metadata must be an object");
  }
  if (
    metadata.schema_version !== 1 ||
    metadata.model_format !== "onnx" ||
    metadata.model_sha256 !== undefined && !/^[0-9a-f]{64}$/.test(metadata.model_sha256) ||
    metadata.input_name !== MODEL_INPUT ||
    !sameArray(metadata.input_shape, [1, 3, IMAGE_SIZE, IMAGE_SIZE]) ||
    metadata.input_dtype !== "float32" ||
    metadata.output_name !== MODEL_OUTPUT ||
    !sameArray(metadata.output_shape, [1, 1]) ||
    metadata.output_dtype !== "float32" ||
    metadata.output_semantics !== "calibrated_probability_ai" ||
    metadata.calibration !== "platt_embedded" ||
    metadata.single_file !== true ||
    metadata.uses_external_data !== false
  ) {
    throw new Error("ONNX metadata I/O contract is invalid");
  }
  const preprocessing = metadata.preprocessing;
  if (
    !preprocessing ||
    !sameArray(preprocessing.resize, [IMAGE_SIZE, IMAGE_SIZE]) ||
    preprocessing.interpolation !== "bicubic" ||
    preprocessing.channel_order !== "RGB" ||
    preprocessing.layout !== "NCHW" ||
    preprocessing.input_dtype !== "uint8" ||
    preprocessing.output_dtype !== "float32" ||
    preprocessing.input_scale !== 1 / 255 ||
    !sameArray(preprocessing.mean, MEAN) ||
    !sameArray(preprocessing.standard_deviation, STANDARD_DEVIATION)
  ) {
    throw new Error("ONNX preprocessing contract is invalid");
  }
  if (metadata.model_sha256 === undefined) {
    throw new Error("ONNX metadata requires model_sha256");
  }
}

function sameArray(actual, expected) {
  return Array.isArray(actual) &&
    actual.length === expected.length &&
    actual.every((value, index) => value === expected[index]);
}

async function loadMetadata(fetchImpl, url) {
  const payload = await loadResourceBytes(fetchImpl, url, "ONNX metadata");
  let metadata;
  try {
    metadata = JSON.parse(new TextDecoder().decode(payload));
  } catch (error) {
    throw new Error(`ONNX metadata is not valid JSON: ${safeMessage(error)}`);
  }
  return metadata;
}

async function loadResourceBytes(fetchImpl, url, description) {
  const response = await fetchImpl(url, {
    cache: "no-store",
    credentials: "omit",
    redirect: "error",
  });
  if (!response?.ok || typeof response.arrayBuffer !== "function") {
    throw new Error(`${description} could not be loaded`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength > MAX_MODEL_BYTES) {
    throw new Error(`${description} exceeds the runtime size limit`);
  }
  return bytes;
}

async function sha256Hex(cryptoImpl, bytes) {
  const digest = new Uint8Array(await cryptoImpl.subtle.digest("SHA-256", bytes));
  return [...digest].map((value) => value.toString(16).padStart(2, "0")).join("");
}

function createImageTensorLoader({
  fetchImpl,
  imageBitmapFactory,
  canvasFactory,
  BlobConstructor,
}) {
  if (typeof imageBitmapFactory !== "function") {
    throw new TypeError("createImageBitmap is required for browser image decoding");
  }
  if (typeof canvasFactory !== "function") {
    throw new TypeError("an OffscreenCanvas factory is required");
  }
  if (typeof BlobConstructor !== "function") {
    throw new TypeError("Blob is required for browser image decoding");
  }

  return async (source) => {
    if (typeof source !== "string" || !SUPPORTED_SOURCE.test(source.trim())) {
      throw new Error("image source is unsupported");
    }
    const response = await fetchImpl(source, {
      cache: "no-store",
      credentials: "omit",
      redirect: "error",
    });
    if (!response?.ok || typeof response.arrayBuffer !== "function") {
      throw new Error("image could not be loaded");
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength === 0 || bytes.byteLength > MAX_IMAGE_BYTES) {
      throw new Error("image exceeds the 20 MiB decode limit");
    }
    const mime = response.headers?.get?.("content-type") || "application/octet-stream";
    const bitmap = await imageBitmapFactory(new BlobConstructor([bytes], { type: mime }));
    const canvas = canvasFactory(IMAGE_SIZE, IMAGE_SIZE);
    const context = canvas?.getContext?.("2d", { willReadFrequently: true });
    if (!context || typeof context.drawImage !== "function" || typeof context.getImageData !== "function") {
      bitmap?.close?.();
      throw new Error("browser image canvas is unavailable");
    }
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    try {
      context.drawImage(bitmap, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
      return tensorFromRgba(context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data);
    } finally {
      bitmap?.close?.();
    }
  };
}

function tensorFromRgba(rgba) {
  if (!rgba || rgba.length !== IMAGE_SIZE * IMAGE_SIZE * 4) {
    throw new Error("decoded image has an invalid size");
  }
  const tensor = new Float32Array(TENSOR_LENGTH);
  const plane = IMAGE_SIZE * IMAGE_SIZE;
  for (let pixel = 0; pixel < plane; pixel += 1) {
    const source = pixel * 4;
    for (let channel = 0; channel < CHANNELS; channel += 1) {
      const value = rgba[source + channel] / 255;
      tensor[channel * plane + pixel] =
        (value - MEAN[channel]) / STANDARD_DEVIATION[channel];
    }
  }
  return tensor;
}

function safeMessage(error) {
  const message = error instanceof Error ? error.message : String(error);
  const trimmed = message.trim();
  return trimmed ? trimmed.slice(0, 240) : "Local model operation failed.";
}

