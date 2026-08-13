"use strict";

((scope) => {
  const RESULT_CLASS = "poidh-ai-result";
  const SUPPORTED_SOURCE = /^(?:https?:|blob:|data:image\/)/i;
  const states = new WeakMap();
  let scanSequence = 0;

  function canonicalSource(value, baseURI) {
    const source = String(value || "").trim();
    if (!source || !SUPPORTED_SOURCE.test(source)) {
      return null;
    }
    try {
      const url = new URL(source, baseURI);
      url.hash = "";
      return url.href;
    } catch {
      return null;
    }
  }

  function planFor(root) {
    const groupsBySource = new Map();
    let count = 0;
    let skipped = 0;

    for (const image of root.querySelectorAll("img")) {
      const source = canonicalSource(image.currentSrc || image.src, root.baseURI);
      if (source === null) {
        skipped += 1;
        continue;
      }
      count += 1;
      let group = groupsBySource.get(source);
      if (group === undefined) {
        group = {
          elements: [],
          request: {
            id: `image-${groupsBySource.size + 1}`,
            source,
            alt: String(image.alt || "").trim(),
          },
        };
        groupsBySource.set(source, group);
      }
      group.elements.push(image);
    }
    return { count, groups: [...groupsBySource.values()], skipped };
  }

  function collectImageCandidates(root) {
    return planFor(root).groups.map(({ request }) => request);
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
    if (
      result?.status === "error" &&
      typeof result.message === "string" &&
      result.message.trim()
    ) {
      return `Local AI confidence unavailable: ${result.message.trim()}`;
    }
    return "Local AI confidence unavailable: Invalid local runtime result.";
  }

  function elementFactory(root) {
    const factory = typeof root.createElement === "function" ? root : root.ownerDocument;
    if (!factory || typeof factory.createElement !== "function") {
      throw new TypeError("scan root must provide an owner document");
    }
    return factory;
  }

  function restoreDescription({ image, previous }) {
    if (previous === null) {
      image.removeAttribute("aria-describedby");
    } else {
      image.setAttribute("aria-describedby", previous);
    }
  }

  function cleanup(root) {
    const state = states.get(root);
    if (state === undefined) {
      return;
    }
    for (const relationship of state.relationships) {
      restoreDescription(relationship);
    }
    for (const node of state.nodes) {
      node.remove();
    }
    states.delete(root);
  }

  function addAnnotation(factory, image, id, state) {
    const annotation = factory.createElement("span");
    annotation.className = RESULT_CLASS;
    annotation.dataset.poidhOwned = "true";
    annotation.id = id;
    annotation.tabIndex = -1;

    const shadow = annotation.attachShadow({ mode: "closed" });
    const style = factory.createElement("style");
    style.textContent = `
      :host {
        all: initial !important;
        background: #17203a !important;
        border: 1px solid #7892e8 !important;
        border-radius: .25rem !important;
        color: #fff !important;
        display: inline-block !important;
        font: 600 .75rem/1.35 system-ui, sans-serif !important;
        margin: .2rem !important;
        max-width: min(24rem, 90vw) !important;
        padding: .3rem .45rem !important;
        vertical-align: middle !important;
      }
    `;
    shadow.append(style, factory.createElement("slot"));

    const previous = image.getAttribute("aria-describedby");
    const descriptions = previous ? previous.trim().split(/\s+/) : [];
    image.setAttribute("aria-describedby", [...descriptions, id].join(" "));
    state.relationships.push({ image, previous });
    state.nodes.push(annotation);
    image.after(annotation);
    return annotation;
  }

  function addLiveRegion(root, factory, state) {
    const live = factory.createElement("span");
    live.dataset.poidhScanStatus = "true";
    live.setAttribute("role", "status");
    live.setAttribute("aria-live", "polite");
    live.setAttribute("aria-atomic", "true");
    live.tabIndex = -1;
    const container = root.body || root.documentElement || root;
    if (typeof container.append !== "function") {
      throw new TypeError("scan root must provide an appendable container");
    }
    container.append(live);
    state.nodes.push(live);
    return live;
  }

  function setAnnotation(annotation, result) {
    const label = resultLabel(result);
    annotation.textContent = label;
    annotation.setAttribute("aria-label", label);
  }

  function validatedResults(response, groups) {
    if (!response?.ok || !Array.isArray(response.results)) {
      throw new Error(response?.error || "Invalid local runtime response.");
    }
    if (response.results.length !== groups.length) {
      throw new Error("Local runtime result cardinality does not match requests.");
    }
    const expected = new Set(groups.map(({ request }) => request.id));
    const results = new Map();
    for (const result of response.results) {
      if (!result || !expected.has(result.id) || results.has(result.id)) {
        throw new Error("Local runtime result IDs do not match requests.");
      }
      if (
        result.status !== "error" &&
        !(
          result.status === "ok" &&
          Number.isFinite(result.confidence) &&
          result.confidence >= 0 &&
          result.confidence <= 1
        )
      ) {
        throw new Error("Local runtime returned a non-terminal result.");
      }
      results.set(result.id, result);
    }
    return results;
  }

  function summaryLabel({ count, errors, skipped }) {
    return `${count} images scanned; ${errors} unavailable; ${skipped} skipped.`;
  }

  async function scanPage({ root, sendMessage }) {
    cleanup(root);
    const factory = elementFactory(root);
    const plan = planFor(root);
    const state = { nodes: [], relationships: [] };
    states.set(root, state);
    const live = addLiveRegion(root, factory, state);
    live.textContent = `Scanning ${plan.count} images locally; ${plan.skipped} skipped.`;

    const annotations = new Map();
    const sequence = ++scanSequence;
    for (const group of plan.groups) {
      const controls = group.elements.map((image, index) => {
        const annotation = addAnnotation(
          factory,
          image,
          `poidh-ai-result-${sequence}-${group.request.id}-${index + 1}`,
          state,
        );
        setAnnotation(annotation, { status: "pending" });
        return annotation;
      });
      annotations.set(group.request.id, controls);
    }

    if (plan.groups.length === 0) {
      const summary = { count: 0, errors: 0, skipped: plan.skipped };
      live.textContent = summaryLabel(summary);
      return summary;
    }

    try {
      const response = await sendMessage({
        type: "SCORE_IMAGES",
        images: plan.groups.map(({ request }) => request),
      });
      const results = validatedResults(response, plan.groups);
      let errors = 0;
      for (const [id, controls] of annotations) {
        const result = results.get(id);
        for (const annotation of controls) {
          setAnnotation(annotation, result);
        }
        if (result.status === "error") {
          errors += controls.length;
        }
      }
      const summary = { count: plan.count, errors, skipped: plan.skipped };
      live.textContent = summaryLabel(summary);
      return summary;
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Unknown local error.";
      const failure = {
        status: "error",
        message: `Scan failed locally: ${detail}`,
      };
      for (const controls of annotations.values()) {
        for (const annotation of controls) {
          setAnnotation(annotation, failure);
        }
      }
      const summary = {
        count: plan.count,
        errors: plan.count,
        skipped: plan.skipped,
      };
      live.textContent = summaryLabel(summary);
      return summary;
    }
  }

  const api = Object.freeze({ collectImageCandidates, resultLabel, scanPage });
  scope.POIDHContent = api;

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
