"use strict";

((scope) => {
  const RESULT_CLASS = "poidh-ai-result";
  const SUPPORTED_SOURCE = /^(?:https?:|blob:|data:image\/)/i;

  function entriesFor(root) {
    const entries = [];

    for (const image of root.querySelectorAll("img")) {
      const source = String(image.currentSrc || image.src || "").trim();
      if (!source || !SUPPORTED_SOURCE.test(source)) {
        continue;
      }
      entries.push({
        element: image,
        request: {
          id: `image-${entries.length + 1}`,
          source,
          alt: String(image.alt || "").trim(),
        },
      });
    }
    return entries;
  }

  function collectImageCandidates(root) {
    return entriesFor(root).map(({ request }) => request);
  }

  function resultLabel(result) {
    if (result?.status === "pending") {
      return "Local AI confidence: pending";
    }
    if (
      result?.status === "ok" &&
      Number.isFinite(result.confidence) &&
      result.confidence >= 0 &&
      result.confidence <= 1
    ) {
      return `Local AI confidence: ${(result.confidence * 100).toFixed(1)}%`;
    }
    if (result?.status === "error" && typeof result.message === "string") {
      return `Local AI confidence unavailable: ${result.message}`;
    }
    return "Local AI confidence unavailable: Invalid local runtime result.";
  }

  function addAnnotation(root, image, id) {
    const annotation = root.createElement("span");
    annotation.className = RESULT_CLASS;
    annotation.dataset.poidhImageId = id;
    annotation.setAttribute("role", "status");
    annotation.setAttribute("aria-live", "polite");
    annotation.tabIndex = 0;
    annotation.textContent = resultLabel({ status: "pending" });
    image.after(annotation);
    return annotation;
  }

  async function scanPage({ root, sendMessage }) {
    for (const existing of root.querySelectorAll(`.${RESULT_CLASS}`)) {
      existing.remove();
    }

    const entries = entriesFor(root);
    const annotations = new Map(
      entries.map(({ element, request }) => [
        request.id,
        addAnnotation(root, element, request.id),
      ]),
    );
    if (entries.length === 0) {
      return { count: 0, errors: 0 };
    }

    try {
      const response = await sendMessage({
        type: "SCORE_IMAGES",
        images: entries.map(({ request }) => request),
      });
      if (!response?.ok || !Array.isArray(response.results)) {
        throw new Error(response?.error || "Invalid local runtime response.");
      }
      let errors = 0;
      const results = new Map(response.results.map((result) => [result.id, result]));
      for (const [id, annotation] of annotations) {
        const result = results.get(id) || {
          status: "error",
          message: "The local runtime returned no result.",
        };
        annotation.textContent = resultLabel(result);
        errors += result.status === "error" ? 1 : 0;
      }
      return { count: entries.length, errors };
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Unknown local error.";
      for (const annotation of annotations.values()) {
        annotation.textContent = `Local AI confidence unavailable: Scan failed locally: ${detail}`;
      }
      return { count: entries.length, errors: entries.length };
    }
  }

  scope.POIDHContent = Object.freeze({
    collectImageCandidates,
    resultLabel,
    scanPage,
  });

  if (
    typeof chrome !== "undefined" &&
    chrome.runtime?.onMessage &&
    typeof document !== "undefined" &&
    !scope.__POIDH_CONTENT_INSTALLED__
  ) {
    scope.__POIDH_CONTENT_INSTALLED__ = true;
    chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
      if (message?.type !== "SCAN_PAGE") {
        return false;
      }
      scanPage({
        root: document,
        sendMessage: (request) => chrome.runtime.sendMessage(request),
      })
        .then((summary) => sendResponse({ ok: true, ...summary }))
        .catch((error) => sendResponse({ ok: false, error: error.message }));
      return true;
    });
  }
})(globalThis);
