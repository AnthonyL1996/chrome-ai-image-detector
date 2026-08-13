# MV3 local detector extension

This directory contains the Manifest V3 browser surface for the detector. It
discovers ordinary `<img>` elements only after the user presses **Scan images
on this page** and adds an accessible status next to each discovered image.

The service worker loads a bundled ONNX Runtime Web WASM adapter and the
`model/detector.onnx` plus `model/metadata.json` bundle when those artifacts
are installed. The adapter verifies the model digest and fixed input/output
contract before scoring. If the bundle is absent or fails validation, each
image receives an explicit local-runtime-unavailable result.

Inference stays local to the browser; no image or result is sent to a remote
inference service and no telemetry is collected. To load ordinary page images
from the service worker, the first scan asks for optional access to the active
page origin. The request is per-origin and can be denied; persistent
`host_permissions` are not declared. For the same-origin privacy boundary,
images whose URL origin differs from the scanned page are reported unavailable
instead of being fetched as a privileged cross-origin request.

For local inspection, open `chrome://extensions`, enable Developer mode, choose
**Load unpacked**, and select this `extension` directory. This development
build expects the audited model bundle to be present; the ONNX Runtime Web WASM
files are vendored under `runtime/vendor/`.
