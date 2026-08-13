import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const extensionRoot = new URL("../", import.meta.url);

async function loadServiceWorker(overrides = {}) {
  let listener;
  const calls = [];
  const chrome = {
    runtime: {
      id: "extension-id",
      getURL: (path) => `chrome-extension://extension-id/${path}`,
      onMessage: { addListener: (registered) => { listener = registered; } },
    },
    scripting: {
      insertCSS: async (options) => calls.push(["insertCSS", options]),
      executeScript: async (options) => calls.push(["executeScript", options]),
    },
    tabs: {
      query: async () => [{ id: 17 }],
      sendMessage: async () => ({ ok: true, count: 0, errors: 0, skipped: 0 }),
    },
    permissions: {
      contains: async () => true,
      request: async () => true,
    },
    ...overrides,
  };
  const context = vm.createContext({ chrome, console, URL });
  const modules = new Map();
  async function moduleFor(url) {
    const href = url.href;
    if (modules.has(href)) return modules.get(href);
    const source = await readFile(url, "utf8");
    const module = new vm.SourceTextModule(source, {
      context,
      identifier: href,
    });
    modules.set(href, module);
    await module.link((specifier, referencing) =>
      moduleFor(new URL(specifier, referencing.identifier))
    );
    return module;
  }
  const module = await moduleFor(new URL("service-worker.js", extensionRoot));
  await module.evaluate();

  async function dispatch(message, sender) {
    return new Promise((resolve) => {
      assert.equal(listener(message, sender, resolve), true);
    });
  }
  return { calls, chrome, dispatch };
}

test("only the extension popup may request an active-tab scan", async () => {
  const worker = await loadServiceWorker();
  const response = await worker.dispatch(
    { type: "SCAN_ACTIVE_TAB" },
    { id: "extension-id", tab: { id: 99 }, url: "https://example.test" },
  );

  assert.equal(response.ok, false);
  assert.match(response.error, /popup|sender/i);
  assert.deepEqual(worker.calls, []);
});

test("image scoring is bound to the top frame of the tab being scanned", async () => {
  let dispatch;
  const calls = [];
  const worker = await loadServiceWorker({
    scripting: {
      insertCSS: async (options) => calls.push(["insertCSS", options]),
      executeScript: async (options) => calls.push(["executeScript", options]),
    },
    tabs: {
      query: async () => [{ id: 17 }],
      sendMessage: async () => {
        const scoring = await dispatch(
          {
            type: "SCORE_IMAGES",
            images: [{ id: "image-1", source: "https://example.test/a.png" }],
          },
          { id: "extension-id", tab: { id: 17 }, frameId: 0 },
        );
        assert.equal(scoring.ok, true);
        return { ok: true, count: 1, errors: 1, skipped: 0 };
      },
    },
  });
  dispatch = worker.dispatch;

  const rejected = await dispatch(
    {
      type: "SCORE_IMAGES",
      images: [{ id: "image-1", source: "https://example.test/a.png" }],
    },
    { id: "extension-id", tab: { id: 99 }, frameId: 0 },
  );
  assert.equal(rejected.ok, false);
  assert.match(rejected.error, /active scan|tab/i);

  const scan = await dispatch(
    { type: "SCAN_ACTIVE_TAB" },
    {
      id: "extension-id",
      url: "chrome-extension://extension-id/popup.html",
    },
  );
  assert.deepEqual(scan, { ok: true, count: 1, errors: 1, skipped: 0 });
  assert.deepEqual(JSON.parse(JSON.stringify(calls)), [
    ["insertCSS", { target: { tabId: 17 }, files: ["content.css"] }],
    ["executeScript", { target: { tabId: 17 }, files: ["content.js"] }],
  ]);

  const afterScan = await dispatch(
    { type: "SCORE_IMAGES", images: [] },
    { id: "extension-id", tab: { id: 17 }, frameId: 0 },
  );
  assert.equal(afterScan.ok, false);
  assert.match(afterScan.error, /active scan|tab/i);
});

test("scoring rejects extension-ID mismatches and non-top frames", async () => {
  const worker = await loadServiceWorker();
  for (const sender of [
    { id: "other-extension", tab: { id: 17 }, frameId: 0 },
    { id: "extension-id", tab: { id: 17 }, frameId: 3 },
  ]) {
    const response = await worker.dispatch(
      { type: "SCORE_IMAGES", images: [] },
      sender,
    );
    assert.equal(response.ok, false);
    assert.match(response.error, /sender|frame|active scan/i);
  }
});

test("scan requests reject missing active tabs and unknown messages", async () => {
  const worker = await loadServiceWorker({
    tabs: {
      query: async () => [],
      sendMessage: async () => assert.fail("no tab should be messaged"),
    },
  });
  const popup = {
    id: "extension-id",
    url: "chrome-extension://extension-id/popup.html",
  };

  const noTab = await worker.dispatch({ type: "SCAN_ACTIVE_TAB" }, popup);
  assert.equal(noTab.ok, false);
  assert.match(noTab.error, /no active webpage tab/i);

  const unknown = await worker.dispatch({ type: "NOT_A_REAL_MESSAGE" }, popup);
  assert.equal(unknown.ok, false);
  assert.match(unknown.error, /unknown extension message/i);
});

test("a tab is reserved before asynchronous scan setup begins", async () => {
  let releaseSetup;
  let setupStarted;
  const setupGate = new Promise((resolve) => { releaseSetup = resolve; });
  const setupSignal = new Promise((resolve) => { setupStarted = resolve; });
  const worker = await loadServiceWorker({
    scripting: {
      insertCSS: async () => {
        setupStarted();
        await setupGate;
      },
      executeScript: async () => {},
    },
  });
  const popup = {
    id: "extension-id",
    url: "chrome-extension://extension-id/popup.html",
  };

  const firstScan = worker.dispatch({ type: "SCAN_ACTIVE_TAB" }, popup);
  await setupSignal;
  const overlapping = await worker.dispatch({ type: "SCAN_ACTIVE_TAB" }, popup);
  assert.equal(overlapping.ok, false);
  assert.match(overlapping.error, /already has an active image scan/i);

  releaseSetup();
  assert.equal((await firstScan).ok, true);
});

test("scan requests optional access only for the active page origin", async () => {
  const requested = [];
  const worker = await loadServiceWorker({
    permissions: {
      contains: async () => false,
      request: async (details) => {
        requested.push(details);
        return true;
      },
    },
    tabs: {
      query: async () => [{ id: 17, url: "https://example.test:8443/article" }],
      sendMessage: async () => ({ ok: true, count: 0, errors: 0, skipped: 0 }),
    },
  });

  const response = await worker.dispatch(
    { type: "SCAN_ACTIVE_TAB" },
    { id: "extension-id", url: "chrome-extension://extension-id/popup.html" },
  );

  assert.equal(response.ok, true);
  assert.deepEqual(requested, [{ origins: ["https://example.test:8443/*"] }]);
});

test("scan stops before script injection when optional origin access is denied", async () => {
  const worker = await loadServiceWorker({
    permissions: {
      contains: async () => false,
      request: async () => false,
    },
    tabs: {
      query: async () => [{ id: 17, url: "https://example.test/article" }],
      sendMessage: async () => assert.fail("denied scans must not message the page"),
    },
  });

  const response = await worker.dispatch(
    { type: "SCAN_ACTIVE_TAB" },
    { id: "extension-id", url: "chrome-extension://extension-id/popup.html" },
  );

  assert.equal(response.ok, false);
  assert.match(response.error, /access|permission|denied/i);
  assert.deepEqual(worker.calls, []);
});
