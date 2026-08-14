const SUPPORTED_SOURCE = /^(?:https?:|blob:|data:image\/)/i;

export function validateImageRequests(images) {
  if (!Array.isArray(images)) {
    throw new TypeError("images must be an array");
  }

  const ids = new Set();
  return images.map((image) => {
    if (!image || typeof image !== "object") {
      throw new TypeError("each image request must be an object");
    }
    const id = typeof image.id === "string" ? image.id.trim() : "";
    if (!id) {
      throw new TypeError("each image request requires a non-empty id");
    }
    if (ids.has(id)) {
      throw new TypeError(`duplicate image id: ${id}`);
    }
    ids.add(id);

    const source = typeof image.source === "string" ? image.source.trim() : "";
    if (!SUPPORTED_SOURCE.test(source)) {
      throw new TypeError(`image ${id} requires a supported source`);
    }
    const alt = typeof image.alt === "string" ? image.alt.trim() : "";
    return { id, source, alt };
  });
}

function validateBackend(backend) {
  if (
    !backend ||
    typeof backend.status !== "function" ||
    typeof backend.scoreImages !== "function"
  ) {
    throw new TypeError("backend must implement status() and scoreImages()");
  }
}

function validateBackendStatus(status, { requireReady }) {
  if (!status || typeof status !== "object" || Array.isArray(status)) {
    throw new TypeError("model backend status must be an object");
  }
  if (typeof status.engine !== "string" || !status.engine.trim()) {
    throw new TypeError("model backend status requires a non-empty engine");
  }
  if (status.localOnly !== true) {
    throw new TypeError("model backend must declare a local-only runtime");
  }
  if (typeof status.ready !== "boolean") {
    throw new TypeError("model backend ready status must be a boolean");
  }
  if (requireReady && status.ready !== true) {
    throw new TypeError("configured model backend must be ready");
  }
  return Object.freeze({
    engine: status.engine.trim(),
    localOnly: true,
    ready: status.ready,
  });
}

function hasExactKeys(value, expected) {
  const actual = Object.keys(value).sort();
  return (
    actual.length === expected.length &&
    actual.every((key, index) => key === expected[index])
  );
}

function validateTerminalResult(result, expectedIds, seenIds) {
  if (!result || typeof result !== "object" || Array.isArray(result)) {
    throw new TypeError("each backend result must be an object");
  }
  if (typeof result.id !== "string" || !expectedIds.has(result.id)) {
    throw new TypeError("backend result has an unknown image id");
  }
  if (seenIds.has(result.id)) {
    throw new TypeError(`duplicate backend result id: ${result.id}`);
  }
  seenIds.add(result.id);

  if (result.status === "ok") {
    if (!hasExactKeys(result, ["confidence", "id", "status"])) {
      throw new TypeError("successful backend result fields do not match schema");
    }
    if (
      !Number.isFinite(result.confidence) ||
      result.confidence < 0 ||
      result.confidence > 1
    ) {
      throw new TypeError("successful backend result requires confidence in [0, 1]");
    }
    return Object.freeze({
      id: result.id,
      status: "ok",
      confidence: result.confidence,
    });
  }

  if (result.status === "error") {
    if (!hasExactKeys(result, ["code", "id", "message", "status"])) {
      throw new TypeError("error backend result fields do not match schema");
    }
    const code = typeof result.code === "string" ? result.code.trim() : "";
    const message = typeof result.message === "string" ? result.message.trim() : "";
    if (!code || !message) {
      throw new TypeError("error backend result requires non-empty code and message");
    }
    return Object.freeze({ id: result.id, status: "error", code, message });
  }

  throw new TypeError("backend result must have terminal ok or error status");
}

export function validateImageResults(requests, results) {
  if (!Array.isArray(results)) {
    throw new TypeError("backend results must be an array");
  }
  if (results.length !== requests.length) {
    throw new TypeError("backend result cardinality must match image requests");
  }
  const expectedIds = new Set(requests.map(({ id }) => id));
  const seenIds = new Set();
  const byId = new Map(
    results.map((result) => {
      const validated = validateTerminalResult(result, expectedIds, seenIds);
      return [validated.id, validated];
    }),
  );
  return requests.map(({ id }) => byId.get(id));
}

export function createModelRuntime({ backend = null } = {}) {
  if (backend !== null) {
    validateBackend(backend);
    validateBackendStatus(backend.status(), { requireReady: true });
  }

  return Object.freeze({
    status() {
      if (backend === null) {
        return { engine: "unconfigured", localOnly: true, ready: false };
      }
      return validateBackendStatus(backend.status(), { requireReady: true });
    },

    async scoreImages(images) {
      const validated = validateImageRequests(images);
      if (backend !== null) {
        validateBackendStatus(backend.status(), { requireReady: true });
        const results = await backend.scoreImages(validated);
        validateBackendStatus(backend.status(), { requireReady: true });
        return validateImageResults(validated, results);
      }
      return validated.map(({ id }) => ({
        code: "MODEL_RUNTIME_UNAVAILABLE",
        id,
        message: "Local model runtime is unavailable.",
        status: "error",
      }));
    },
  });
}
