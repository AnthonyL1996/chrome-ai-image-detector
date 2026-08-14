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
        message: "Local model runtime is unavailable.",
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

test("configured backends must be local-only and ready at construction", () => {
  for (const status of [
    { engine: "remote", localOnly: false, ready: true },
    { engine: "loading", localOnly: true, ready: false },
    { engine: "invalid", localOnly: true, ready: "yes" },
  ]) {
    assert.throws(
      () =>
        createModelRuntime({
          backend: {
            status: () => status,
            scoreImages: async () => [],
          },
        }),
      /local-only|ready/,
    );
  }
});

test("scoreImages rechecks the local-only ready boundary before delegation", async () => {
  let status = { engine: "test-local", localOnly: true, ready: true };
  let calls = 0;
  const runtime = createModelRuntime({
    backend: {
      status: () => status,
      scoreImages: async () => {
        calls += 1;
        return [{ id: "image-1", status: "ok", confidence: 0.5 }];
      },
    },
  });

  status = { engine: "remote", localOnly: false, ready: true };
  await assert.rejects(
    runtime.scoreImages([
      { id: "image-1", source: "https://example.test/image.png" },
    ]),
    /local-only/,
  );
  assert.equal(calls, 0);

  status = { engine: "loading", localOnly: true, ready: false };
  await assert.rejects(
    runtime.scoreImages([
      { id: "image-1", source: "https://example.test/image.png" },
    ]),
    /ready/,
  );
  assert.equal(calls, 0);
});

test("backend results are exact terminal one-to-one records", async (context) => {
  const requests = [
    { id: "image-1", source: "https://example.test/one.png" },
    { id: "image-2", source: "https://example.test/two.png" },
  ];
  const invalidResults = [
    null,
    [],
    [{ id: "image-1", status: "ok", confidence: 0.5 }],
    [
      { id: "image-1", status: "ok", confidence: 0.5 },
      { id: "image-1", status: "ok", confidence: 0.6 },
    ],
    [
      { id: "image-1", status: "ok", confidence: 0.5 },
      { id: "unknown", status: "ok", confidence: 0.6 },
    ],
    [
      { id: "image-1", status: "pending" },
      { id: "image-2", status: "ok", confidence: 0.6 },
    ],
    [
      { id: "image-1", status: "ok", confidence: Number.NaN },
      { id: "image-2", status: "ok", confidence: 0.6 },
    ],
    [
      { id: "image-1", status: "ok", confidence: 1.1 },
      { id: "image-2", status: "ok", confidence: 0.6 },
    ],
    [
      { id: "image-1", status: "ok", confidence: 0.5, extra: true },
      { id: "image-2", status: "ok", confidence: 0.6 },
    ],
    [
      { id: "image-1", status: "error", message: "failed" },
      { id: "image-2", status: "ok", confidence: 0.6 },
    ],
    [
      { id: "image-1", status: "error", code: "FAILED", message: "" },
      { id: "image-2", status: "ok", confidence: 0.6 },
    ],
  ];

  for (const [index, results] of invalidResults.entries()) {
    await context.test(`rejects malformed result set ${index + 1}`, async () => {
      const runtime = createModelRuntime({
        backend: {
          status: () => ({ engine: "test", localOnly: true, ready: true }),
          scoreImages: async () => results,
        },
      });
      await assert.rejects(runtime.scoreImages(requests), /result|confidence|error/i);
    });
  }
});

test("validated backend results are returned in request order", async () => {
  const runtime = createModelRuntime({
    backend: {
      status: () => ({ engine: "test", localOnly: true, ready: true }),
      scoreImages: async () => [
        {
          id: "image-2",
          status: "error",
          code: "DECODE_FAILED",
          message: "Could not decode image.",
        },
        { id: "image-1", status: "ok", confidence: 0.25 },
      ],
    },
  });

  assert.deepEqual(
    await runtime.scoreImages([
      { id: "image-1", source: "https://example.test/one.png" },
      { id: "image-2", source: "https://example.test/two.png" },
    ]),
    [
      { id: "image-1", status: "ok", confidence: 0.25 },
      {
        id: "image-2",
        status: "error",
        code: "DECODE_FAILED",
        message: "Could not decode image.",
      },
    ],
  );
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
