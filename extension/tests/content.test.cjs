"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

require("../content.js");

const { collectImageCandidates, resultLabel, scanPage } = globalThis.POIDHContent;

function fakePage(images) {
  const annotations = [];
  const liveRegions = [];
  const oldAnnotation = { removed: false, remove() { this.removed = true; } };
  for (const image of images) {
    const attributes = new Map();
    image.after = (annotation) => annotations.push(annotation);
    image.getAttribute = (name) => attributes.get(name) ?? null;
    image.setAttribute = (name, value) => attributes.set(name, String(value));
    image.removeAttribute = (name) => attributes.delete(name);
  }
  const createElement = (tagName) => ({
    tagName,
    attributes: new Map(),
    children: [],
    className: "",
    dataset: {},
    id: "",
    removed: false,
    tabIndex: -1,
    textContent: "",
    append(...children) { this.children.push(...children); },
    appendChild(child) { this.children.push(child); return child; },
    attachShadow() {
      this.shadowRootCreated = true;
      return { children: [], append: (...children) => this.shadowChildren = children };
    },
    getAttribute(name) { return this.attributes.get(name) ?? null; },
    setAttribute(name, value) {
      this.attributes.set(name, String(value));
      this[name] = String(value);
    },
    remove() { this.removed = true; },
  });
  return {
    annotations,
    liveRegions,
    oldAnnotation,
    root: {
      baseURI: "https://example.test/page",
      body: { append: (node) => liveRegions.push(node) },
      createElement,
      querySelectorAll: (selector) =>
        selector === "img" ? images : selector === ".poidh-ai-result" ? [oldAnnotation] : [],
    },
  };
}

test("collectImageCandidates deduplicates canonical sources", () => {
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
  ]);
});

test("resultLabel never presents unavailable inference as a confidence", () => {
  assert.equal(resultLabel({ status: "pending" }), "Local AI confidence: pending");
  assert.equal(
    resultLabel({ status: "error", message: "Local model runtime is unavailable." }),
    "Local AI confidence unavailable: Local model runtime is unavailable.",
  );
  assert.equal(resultLabel({ status: "ok", confidence: 0.428 }), "Local AI confidence: 42.8%");
  assert.equal(
    resultLabel({ status: "ok", confidence: Number.NaN }),
    "Local AI confidence unavailable: Invalid local runtime result.",
  );
});

test("scanPage fans out deduplicated results with one non-focusable live summary", async () => {
  const page = fakePage([
    { currentSrc: "https://example.test/a.png", src: "", alt: "A diagram" },
    { currentSrc: "https://example.test/a.png#fragment", src: "", alt: "Duplicate" },
    { currentSrc: "", src: "", alt: "unresolved" },
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
            message: "Local model runtime is unavailable.",
          },
        ],
      };
    },
  });

  assert.equal(page.oldAnnotation.removed, false);
  assert.deepEqual(messages, [
    {
      type: "SCORE_IMAGES",
      images: [
        { id: "image-1", source: "https://example.test/a.png", alt: "A diagram" },
      ],
    },
  ]);
  assert.equal(page.annotations.length, 2);
  for (const annotation of page.annotations) {
    assert.notEqual(annotation.role, "status");
    assert.equal(annotation.tabIndex, -1);
    assert.equal(annotation.shadowRootCreated, true);
    assert.equal(
      annotation.textContent,
      "Local AI confidence unavailable: Local model runtime is unavailable.",
    );
  }
  assert.equal(page.liveRegions.length, 1);
  assert.equal(page.liveRegions[0].role, "status");
  assert.equal(page.liveRegions[0]["aria-live"], "polite");
  assert.match(page.liveRegions[0].textContent, /2 images.*1 skipped/i);
  assert.match(page.root.querySelectorAll("img")[0].getAttribute("aria-describedby"), /poidh/);
  assert.match(page.root.querySelectorAll("img")[1].getAttribute("aria-describedby"), /poidh/);
  assert.deepEqual(summary, { count: 2, errors: 2, skipped: 1 });
});

test("scanPage materializes same-origin blob images in the page context", async () => {
  const page = fakePage([
    { currentSrc: "blob:https://example.test/image", src: "", alt: "Blob" },
  ]);
  let sent;
  const summary = await scanPage({
    root: page.root,
    fetchImpl: async () => ({
      ok: true,
      arrayBuffer: async () => Uint8Array.of(0, 1, 2).buffer,
      headers: { get: () => "image/png" },
    }),
    sendMessage: async (message) => {
      sent = message;
      return {
        ok: true,
        results: [{ id: "image-1", status: "ok", confidence: 0.75 }],
      };
    },
  });

  assert.equal(summary.errors, 0);
  assert.match(sent.images[0].source, /^data:image\/png;base64,/);
});

test("scanPage bounds stalled blob materialization and keeps the source unavailable", async () => {
  const page = fakePage([
    { currentSrc: "blob:https://example.test/stalled", src: "", alt: "Blob" },
  ]);
  let aborted = false;
  let sent;
  const summary = await scanPage({
    root: page.root,
    blobFetchTimeoutMs: 5,
    fetchImpl: async (_url, options) => {
      options.signal.addEventListener("abort", () => { aborted = true; }, { once: true });
      return new Promise(() => {});
    },
    sendMessage: async (message) => {
      sent = message;
      return {
        ok: true,
        results: [{ id: "image-1", status: "error", message: "blob unavailable" }],
      };
    },
  });

  assert.equal(aborted, true);
  assert.equal(sent.images[0].source, "blob:https://example.test/stalled");
  assert.deepEqual(summary, { count: 1, errors: 1, skipped: 0 });
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
  assert.deepEqual(summary, { count: 1, errors: 1, skipped: 0 });
});

test("scanPage reports an empty or unsupported page without messaging", async () => {
  const page = fakePage([
    { currentSrc: "", src: "file:///tmp/private.png", alt: "unsupported" },
  ]);

  const summary = await scanPage({
    root: page.root,
    sendMessage: async () => assert.fail("empty scans must not request scoring"),
  });

  assert.deepEqual(summary, { count: 0, errors: 0, skipped: 1 });
  assert.equal(page.annotations.length, 0);
  assert.equal(page.liveRegions[0].textContent, "0 images scanned; 0 unavailable; 1 skipped.");
});

test("scanPage rejects non-terminal, mismatched, and duplicated result records", async (t) => {
  const responses = [
    { ok: false, error: "runtime unavailable" },
    { ok: true, results: [] },
    { ok: true, results: [{ id: "unexpected", status: "error", message: "no" }] },
    { ok: true, results: [{ id: "image-1", status: "pending" }] },
  ];

  for (const [index, response] of responses.entries()) {
    await t.test(`invalid response ${index + 1}`, async () => {
      const page = fakePage([
        { currentSrc: "blob:https://example.test/image", src: "", alt: "" },
      ]);
      const summary = await scanPage({
        root: page.root,
        sendMessage: async () => response,
      });
      assert.deepEqual(summary, { count: 1, errors: 1, skipped: 0 });
      assert.match(page.annotations[0].textContent, /Scan failed locally/);
    });
  }
});

test("scanPage removes only nodes it owns and restores image descriptions", async () => {
  const image = { currentSrc: "https://example.test/a.png", src: "", alt: "A" };
  const page = fakePage([image]);
  image.setAttribute("aria-describedby", "page-description");

  await scanPage({
    root: page.root,
    sendMessage: async () => ({
      ok: true,
      results: [{ id: "image-1", status: "ok", confidence: 0.5 }],
    }),
  });
  const firstAnnotation = page.annotations[0];
  const firstLiveRegion = page.liveRegions[0];

  await scanPage({
    root: page.root,
    sendMessage: async () => ({
      ok: true,
      results: [{ id: "image-1", status: "ok", confidence: 0.6 }],
    }),
  });

  assert.equal(firstAnnotation.removed, true);
  assert.equal(firstLiveRegion.removed, true);
  assert.equal(page.oldAnnotation.removed, false);
  assert.match(image.getAttribute("aria-describedby"), /^page-description poidh/);
});

test("cleanup preserves page-owned descriptions added during a scan", async () => {
  const image = { currentSrc: "https://example.test/a.png", src: "", alt: "A" };
  const page = fakePage([image]);
  image.setAttribute("aria-describedby", "page-description");

  await scanPage({
    root: page.root,
    sendMessage: async () => {
      image.setAttribute(
        "aria-describedby",
        `${image.getAttribute("aria-describedby")} page-dynamic`,
      );
      return {
        ok: true,
        results: [{ id: "image-1", status: "ok", confidence: 0.5 }],
      };
    },
  });
  const firstOwnedId = page.annotations[0].id;

  await scanPage({
    root: page.root,
    sendMessage: async () => ({
      ok: true,
      results: [{ id: "image-1", status: "ok", confidence: 0.6 }],
    }),
  });

  const tokens = image.getAttribute("aria-describedby").split(/\s+/);
  assert.equal(tokens.includes(firstOwnedId), false);
  assert.equal(tokens.includes("page-description"), true);
  assert.equal(tokens.includes("page-dynamic"), true);
  assert.equal(tokens.filter((token) => token.startsWith("poidh-ai-result-")).length, 1);
});
