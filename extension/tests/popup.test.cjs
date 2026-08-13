"use strict";

const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const test = require("node:test");
const vm = require("node:vm");

function loadPopup(sendMessage) {
  const button = {
    disabled: false,
    addEventListener(type, callback) {
      assert.equal(type, "click");
      this.click = callback;
    },
  };
  const status = { textContent: "Ready to scan." };
  const document = {
    querySelector(selector) {
      return selector === "#scan-page" ? button : status;
    },
  };
  const source = readFileSync(new URL("../popup.js", `file://${__filename}`), "utf8");
  vm.runInNewContext(source, { chrome: { runtime: { sendMessage } }, document }, {
    filename: new URL("../popup.js", `file://${__filename}`).pathname,
  });
  return { button, status };
}

test("popup reports scanned, unavailable, and skipped image counts", async () => {
  const messages = [];
  const popup = loadPopup(async (message) => {
    messages.push(message);
    return { ok: true, count: 3, errors: 1, skipped: 2 };
  });

  await popup.button.click();

  assert.equal(messages.length, 1);
  assert.equal(messages[0].type, "SCAN_ACTIVE_TAB");
  assert.equal(
    popup.status.textContent,
    "Scanned 3 images locally; 1 unavailable. Skipped 2 unresolved or unsupported images.",
  );
  assert.equal(popup.button.disabled, false);
});

test("popup exposes service-worker failures and always re-enables scanning", async () => {
  const popup = loadPopup(async () => ({ ok: false, error: "No active tab." }));

  await popup.button.click();

  assert.equal(popup.status.textContent, "Scan failed: No active tab.");
  assert.equal(popup.button.disabled, false);
});
