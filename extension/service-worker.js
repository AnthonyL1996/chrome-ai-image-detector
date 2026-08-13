import { createModelRuntime } from "./runtime/model-runtime.mjs";
import { loadOnnxBackend } from "./runtime/onnx-backend.mjs";

const activeScanTabs = new Set();
let runtimePromise;

function requirePopupSender(sender) {
  const expectedUrl = chrome.runtime.getURL("popup.html");
  if (
    sender?.id !== chrome.runtime.id ||
    sender.tab !== undefined ||
    sender.url !== expectedUrl
  ) {
    throw new Error("Active-tab scans must be requested by the extension popup.");
  }
}

function requireScoringSender(sender) {
  if (sender?.id !== chrome.runtime.id) {
    throw new Error("Image scoring request has an invalid extension sender.");
  }
  if (!Number.isInteger(sender.tab?.id) || sender.frameId !== 0) {
    throw new Error("Image scoring requests must come from a webpage top frame.");
  }
  if (!activeScanTabs.has(sender.tab.id)) {
    throw new Error("Image scoring request is not bound to an active tab scan.");
  }
}

async function scanActiveTab(sender) {
  requirePopupSender(sender);
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!Number.isInteger(tab?.id)) {
    throw new Error("No active webpage tab is available.");
  }
  if (activeScanTabs.has(tab.id)) {
    throw new Error("This tab already has an active image scan.");
  }

  activeScanTabs.add(tab.id);
  try {
    await ensureOriginAccess(tab);
    await chrome.scripting.insertCSS({
      target: { tabId: tab.id },
      files: ["content.css"],
    });
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content.js"],
    });
    return await chrome.tabs.sendMessage(tab.id, { type: "SCAN_PAGE" });
  } finally {
    activeScanTabs.delete(tab.id);
  }
}

async function ensureOriginAccess(tab) {
  if (
    !chrome.permissions ||
    typeof chrome.permissions.contains !== "function" ||
    typeof chrome.permissions.request !== "function"
  ) {
    return;
  }
  const url = new URL(typeof tab.url === "string" ? tab.url : "");
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("The active page has no requestable web origin.");
  }
  const origin = `${url.origin}/*`;
  const details = { origins: [origin] };
  if (await chrome.permissions.contains(details)) {
    return;
  }
  if (!(await chrome.permissions.request(details))) {
    throw new Error(`Optional access to ${url.origin} was denied.`);
  }
}

async function loadRuntime() {
  try {
    const ort = globalThis.POIDH_ORT;
    if (!ort) {
      return createModelRuntime();
    }
    const backend = await loadOnnxBackend({
      ort,
      modelUrl: chrome.runtime.getURL("model/detector.onnx"),
      metadataUrl: chrome.runtime.getURL("model/metadata.json"),
    });
    return createModelRuntime({ backend });
  } catch (_error) {
    // The unpacked developer build remains usable before model files are installed.
    return createModelRuntime();
  }
}

function getRuntime() {
  if (runtimePromise === undefined) {
    runtimePromise = loadRuntime();
  }
  return runtimePromise;
}

async function routeMessage(message, sender) {
  switch (message?.type) {
    case "SCAN_ACTIVE_TAB":
      return scanActiveTab(sender);
    case "SCORE_IMAGES": {
      requireScoringSender(sender);
      const results = await (await getRuntime()).scoreImages(message.images);
      return { ok: true, results };
    }
    default:
      throw new Error("Unknown extension message.");
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  routeMessage(message, sender)
    .then(sendResponse)
    .catch((error) => {
      const detail = error instanceof Error ? error.message : "Unknown extension error.";
      sendResponse({ ok: false, error: detail });
    });
  return true;
});
