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

export function createModelRuntime({ backend = null } = {}) {
  if (backend !== null) {
    validateBackend(backend);
  }

  return Object.freeze({
    status() {
      if (backend === null) {
        return { engine: "unconfigured", localOnly: true, ready: false };
      }
      const status = backend.status();
      if (status?.localOnly !== true) {
        throw new TypeError("model backend must declare a local-only runtime");
      }
      return status;
    },

    async scoreImages(images) {
      const validated = validateImageRequests(images);
      if (backend !== null) {
        return backend.scoreImages(validated);
      }
      return validated.map(({ id }) => ({
        code: "MODEL_RUNTIME_UNAVAILABLE",
        id,
        message: "Local model runtime is not bundled yet.",
        status: "error",
      }));
    },
  });
}
