import {
  createModelRuntime,
  validateImageRequests,
} from "./runtime/model-runtime.mjs";
import { loadOnnxBackend } from "./runtime/onnx-backend.mjs";
import ort from "./runtime/ort-runtime.mjs";

const SUPPORTED_SOURCE = /^(?:https?:|blob:|data:image\/)/i;
const activeScanTabs = new Map();
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
  const scan = activeScanTabs.get(sender.tab.id);
  if (scan === undefined || typeof scan.pageOrigin !== "string") {
    throw new Error("Image scoring request is not bound to an active tab scan.");
  }
  return scan;
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

  const scan = { pageOrigin: null };
  activeScanTabs.set(tab.id, scan);
  try {
    scan.pageOrigin = await ensureOriginAccess(tab);
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
  const url = new URL(typeof tab.url === "string" ? tab.url : "");
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("The active page has no requestable web origin.");
  }
  if (
    !chrome.permissions ||
    typeof chrome.permissions.contains !== "function" ||
    typeof chrome.permissions.request !== "function"
  ) {
    return url.origin;
  }
  const origin = `${url.origin}/*`;
  const details = { origins: [origin] };
  if (await chrome.permissions.contains(details)) {
    return url.origin;
  }
  if (!(await chrome.permissions.request(details))) {
    throw new Error(`Optional access to ${url.origin} was denied.`);
  }
  return url.origin;
}

function partitionScoringSources(images, { pageOrigin }) {
  const allowed = [];
  const blocked = [];
  for (const image of images) {
    const source = image.source;
    if (/^data:image\//i.test(source)) {
      allowed.push(image);
      continue;
    }
    const url = new URL(source);
    if (url.origin === pageOrigin) {
      allowed.push(image);
      continue;
    }
    blocked.push({
      id: image.id,
      status: "error",
      code: "IMAGE_ORIGIN_NOT_ALLOWED",
      message: "Image source origin differs from the active page origin.",
    });
  }
  return { allowed, blocked };
}

function mergeScoringResults(requests, allowedResults, blockedResults) {
  const byId = new Map(
    [...allowedResults, ...blockedResults].map((result) => [result.id, result]),
  );
  return requests.map(({ id }) => byId.get(id));
}

function validateAndPartitionScoringSources(images, scan) {
  const requests = validateImageRequests(images);
  for (const image of requests) {
    if (!SUPPORTED_SOURCE.test(image.source)) {
      throw new TypeError("image source is unsupported");
    }
    try {
      const url = new URL(image.source);
      if (!["http:", "https:", "blob:"].includes(url.protocol) &&
          !/^data:image\//i.test(image.source)) {
        throw new TypeError("image source is unsupported");
      }
    } catch {
      throw new TypeError("image source is not a valid URL");
    }
  }
  return { requests, ...partitionScoringSources(requests, scan) };
}

async function loadRuntime() {
  try {
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
      const scan = requireScoringSender(sender);
      const { requests, allowed, blocked } = validateAndPartitionScoringSources(
        message.images,
        scan,
      );
      const scored = allowed.length === 0
        ? []
        : await (await getRuntime()).scoreImages(allowed);
      const results = mergeScoringResults(requests, scored, blocked);
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
