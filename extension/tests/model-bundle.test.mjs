import { createHash } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import test from "node:test";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const MODEL_PATH = resolve(ROOT, "model/detector.onnx");
const METADATA_PATH = resolve(ROOT, "model/metadata.json");
const MAX_MODEL_BYTES = 100 * 1024 * 1024;

test("bundled ONNX model matches its audited metadata contract", () => {
  const model = readFileSync(MODEL_PATH);
  const metadata = JSON.parse(readFileSync(METADATA_PATH, "utf8"));
  const digest = createHash("sha256").update(model).digest("hex");

  assert.equal(statSync(MODEL_PATH).size, model.byteLength);
  assert.ok(model.byteLength <= MAX_MODEL_BYTES);
  assert.equal(digest, metadata.model_sha256);
  assert.deepEqual(metadata.input_shape, [1, 3, 384, 384]);
  assert.deepEqual(metadata.output_shape, [1, 1]);
  assert.equal(metadata.calibration, "sigmoid_embedded_offset_2.29");
  assert.equal(metadata.preprocessing.resize_shorter_side, 440);
  assert.deepEqual(metadata.preprocessing.center_crop, [384, 384]);
  assert.equal(metadata.uses_external_data, false);
});
