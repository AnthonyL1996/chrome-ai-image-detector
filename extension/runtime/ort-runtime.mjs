import ort from "./vendor/ort.wasm.bundle.min.mjs";

// Keep the WebAssembly binary inside the unpacked extension. The runtime is
// configured explicitly so a future packaging change cannot redirect it to a
// network URL or a page-relative resource.
if (ort?.env?.wasm) {
  const vendorBase = chrome.runtime.getURL("runtime/vendor/");
  ort.env.wasm.wasmPaths = {
    mjs: `${vendorBase}ort-wasm-simd-threaded.mjs`,
    wasm: `${vendorBase}ort-wasm-simd-threaded.wasm`,
  };
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
}

export default ort;
