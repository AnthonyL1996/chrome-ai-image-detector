import { createModelRuntime } from "./runtime/model-runtime.mjs";

const runtime = createModelRuntime();
const activeScanTabs = new Set();

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

async function routeMessage(message, sender) {
  switch (message?.type) {
    case "SCAN_ACTIVE_TAB":
      return scanActiveTab(sender);
    case "SCORE_IMAGES": {
      requireScoringSender(sender);
      const results = await runtime.scoreImages(message.images);
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
