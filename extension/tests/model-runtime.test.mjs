import assert from "node:assert/strict";
import test from "node:test";

import {
  createModelRuntime,
  validateImageRequests,
} from "../runtime/model-runtime.mjs";

test("unconfigured runtime reports local-only unavailable results", async () => {
  const runtime = createModelRuntime();
  assert.deepEqual(runtime.status(), {
    engine: "unconfigured",
    localOnly: true,
    ready: false,
  });

  assert.deepEqual(
    await runtime.scoreImages([
      { id: "image-1", source: "https://example.test/image.png" },
    ]),
    [
      {
        code: "MODEL_RUNTIME_UNAVAILABLE",
        id: "image-1",
        message: "Local model runtime is not bundled yet.",
        status: "error",
      },
    ],
  );
});

test("adapter delegates validated requests to a future local backend", async () => {
  const seen = [];
  const backend = {
    status: () => ({ engine: "test-local", localOnly: true, ready: true }),
    scoreImages: async (images) => {
      seen.push(images);
      return [{ id: images[0].id, status: "ok", confidence: 0.75 }];
    },
  };
  const runtime = createModelRuntime({ backend });
  const requests = [{ id: "image-1", source: "data:image/png;base64,AA==" }];

  assert.deepEqual(runtime.status(), backend.status());
  assert.deepEqual(await runtime.scoreImages(requests), [
    { id: "image-1", status: "ok", confidence: 0.75 },
  ]);
  assert.deepEqual(seen, [
    [{ id: "image-1", source: "data:image/png;base64,AA==", alt: "" }],
  ]);
});

test("request validation rejects malformed or duplicated image requests", () => {
  assert.throws(() => validateImageRequests("not-an-array"), /must be an array/);
  assert.throws(() => validateImageRequests([null]), /must be an object/);
  assert.throws(
    () => validateImageRequests([{ id: "", source: "https://example.test/a" }]),
    /non-empty id/,
  );
  assert.throws(
    () => validateImageRequests([{ id: "a", source: "javascript:alert(1)" }]),
    /supported source/,
  );
  assert.throws(
    () =>
      validateImageRequests([
        { id: "a", source: "https://example.test/a" },
        { id: "a", source: "https://example.test/b" },
      ]),
    /duplicate image id/,
  );
  assert.deepEqual(
    validateImageRequests([
      { id: "a", source: "blob:https://example.test/id", alt: " Blob image " },
    ]),
    [{ id: "a", source: "blob:https://example.test/id", alt: "Blob image" }],
  );
});

test("adapter rejects a backend that is missing the local runtime boundary", () => {
  assert.throws(() => createModelRuntime({ backend: {} }), /backend/);
  assert.throws(
    () =>
      createModelRuntime({
        backend: {
          scoreImages: async () => [],
          status: () => ({ ready: true, localOnly: false, engine: "remote" }),
        },
      }).status(),
    /local-only/,
  );
});
