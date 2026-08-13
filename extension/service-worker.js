import { createModelRuntime } from "./runtime/model-runtime.mjs";

const runtime = createModelRuntime();

async function scanActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!Number.isInteger(tab?.id)) {
    throw new Error("No active webpage tab is available.");
  }

  await chrome.scripting.insertCSS({
    target: { tabId: tab.id },
    files: ["content.css"],
  });
  await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ["content.js"],
  });
  return chrome.tabs.sendMessage(tab.id, { type: "SCAN_PAGE" });
}

async function routeMessage(message, sender) {
  switch (message?.type) {
    case "SCAN_ACTIVE_TAB":
      return scanActiveTab();
    case "SCORE_IMAGES": {
      if (!sender.tab) {
        throw new Error("Image scoring requests must come from a webpage.");
      }
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
