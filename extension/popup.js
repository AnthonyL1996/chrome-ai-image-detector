"use strict";

const scanButton = document.querySelector("#scan-page");
const scanStatus = document.querySelector("#scan-status");

function countFrom(response, field) {
  const value = response?.[field];
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`The scan returned an invalid ${field} count.`);
  }
  return value;
}

function plural(count, singular, pluralForm = `${singular}s`) {
  return count === 1 ? singular : pluralForm;
}

scanButton.addEventListener("click", async () => {
  scanButton.disabled = true;
  scanStatus.textContent = "Scanning page images locally…";

  try {
    const response = await chrome.runtime.sendMessage({ type: "SCAN_ACTIVE_TAB" });
    if (!response?.ok) {
      throw new Error(response?.error || "The page could not be scanned.");
    }
    const count = countFrom(response, "count");
    const errors = countFrom(response, "errors");
    const skipped = countFrom(response, "skipped");
    if (errors > count) {
      throw new Error("The scan returned more errors than scanned images.");
    }
    scanStatus.textContent =
      `Scanned ${count} ${plural(count, "image")} locally; ` +
      `${errors} ${plural(errors, "unavailable", "unavailable")}. ` +
      `Skipped ${skipped} unresolved or unsupported ${plural(skipped, "image")}.`;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown scan error.";
    scanStatus.textContent = `Scan failed: ${message}`;
  } finally {
    scanButton.disabled = false;
  }
});
