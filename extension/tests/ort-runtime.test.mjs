import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";

test("vendored ONNX Runtime Web initializes its local WASM backend", async () => {
  const vendorBase = pathToFileURL(resolve("extension/runtime/vendor") + "/").href;
  globalThis.chrome = {
    runtime: {
      getURL(path) {
        assert.equal(path, "runtime/vendor/");
        return vendorBase;
      },
    },
  };
  const { default: ort } = await import("../runtime/ort-runtime.mjs");

  await assert.rejects(
    ort.InferenceSession.create(new Uint8Array([0, 1, 2, 3]), {
      executionProviders: ["wasm"],
    }),
    (error) => {
      const message = String(error?.message || error);
      assert.match(message, /protobuf parsing failed|invalid model/i);
      assert.doesNotMatch(message, /no available backend|ERR_MODULE_NOT_FOUND|fetch failed/i);
      return true;
    },
  );
});
