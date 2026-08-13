"use strict";

const scanButton = document.querySelector("#scan-page");
const scanStatus = document.querySelector("#scan-status");

scanButton.addEventListener("click", async () => {
  scanButton.disabled = true;
  scanStatus.textContent = "Scanning page images locally…";

  try {
    const response = await chrome.runtime.sendMessage({ type: "SCAN_ACTIVE_TAB" });
    if (!response?.ok) {
      throw new Error(response?.error || "The page could not be scanned.");
    }
    const noun = response.count === 1 ? "image" : "images";
    scanStatus.textContent = `Found ${response.count} ${noun}. Local inference is not bundled yet.`;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown scan error.";
    scanStatus.textContent = `Scan failed: ${message}`;
  } finally {
    scanButton.disabled = false;
  }
});
