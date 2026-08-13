"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

require("../content.js");

const { collectImageCandidates, resultLabel, scanPage } = globalThis.POIDHContent;

function fakePage(images) {
  const annotations = [];
  const oldAnnotation = { removed: false, remove() { this.removed = true; } };
  for (const image of images) {
    image.after = (annotation) => annotations.push(annotation);
  }
  return {
    annotations,
    oldAnnotation,
    root: {
      createElement: () => ({
        className: "",
        dataset: {},
        setAttribute(name, value) { this[name] = value; },
        tabIndex: -1,
        textContent: "",
      }),
      querySelectorAll: (selector) =>
        selector === "img" ? images : selector === ".poidh-ai-result" ? [oldAnnotation] : [],
    },
  };
}

test("collectImageCandidates returns every ordinary webpage image element", () => {
  const images = [
    { currentSrc: "https://example.test/a.png", src: "ignored", alt: "first" },
    { currentSrc: "", src: "https://example.test/b.jpg", alt: "" },
    { currentSrc: "https://example.test/a.png", src: "", alt: "duplicate" },
    { currentSrc: "", src: "", alt: "missing" },
  ];
  const root = { querySelectorAll: () => images };

  assert.deepEqual(collectImageCandidates(root), [
    { id: "image-1", source: "https://example.test/a.png", alt: "first" },
    { id: "image-2", source: "https://example.test/b.jpg", alt: "" },
    { id: "image-3", source: "https://example.test/a.png", alt: "duplicate" },
  ]);
});

test("resultLabel never presents unavailable inference as a confidence", () => {
  assert.equal(resultLabel({ status: "pending" }), "Local AI confidence: pending");
  assert.equal(
    resultLabel({ status: "error", message: "Local model runtime is not bundled yet." }),
    "Local AI confidence unavailable: Local model runtime is not bundled yet.",
  );
  assert.equal(resultLabel({ status: "ok", confidence: 0.428 }), "Local AI confidence: 42.8%");
  assert.equal(
    resultLabel({ status: "ok", confidence: Number.NaN }),
    "Local AI confidence unavailable: Invalid local runtime result.",
  );
});

test("scanPage creates accessible per-image status and applies local errors", async () => {
  const page = fakePage([
    { currentSrc: "https://example.test/a.png", src: "", alt: "A diagram" },
    { currentSrc: "", src: "", alt: "not loaded" },
  ]);
  const messages = [];

  const summary = await scanPage({
    root: page.root,
    sendMessage: async (message) => {
      messages.push(message);
      assert.equal(page.annotations[0].textContent, "Local AI confidence: pending");
      return {
        ok: true,
        results: [
          {
            id: "image-1",
            status: "error",
            message: "Local model runtime is not bundled yet.",
          },
        ],
      };
    },
  });

  assert.equal(page.oldAnnotation.removed, true);
  assert.deepEqual(messages, [
    {
      type: "SCORE_IMAGES",
      images: [
        { id: "image-1", source: "https://example.test/a.png", alt: "A diagram" },
      ],
    },
  ]);
  assert.equal(page.annotations[0].role, "status");
  assert.equal(page.annotations[0].tabIndex, 0);
  assert.equal(
    page.annotations[0].textContent,
    "Local AI confidence unavailable: Local model runtime is not bundled yet.",
  );
  assert.deepEqual(summary, { count: 1, errors: 1 });
});

test("scanPage turns messaging failures into visible per-image errors", async () => {
  const page = fakePage([
    { currentSrc: "data:image/png;base64,AA==", src: "", alt: "" },
  ]);

  const summary = await scanPage({
    root: page.root,
    sendMessage: async () => {
      throw new Error("service worker unavailable");
    },
  });

  assert.match(page.annotations[0].textContent, /Scan failed locally/);
  assert.deepEqual(summary, { count: 1, errors: 1 });
});
