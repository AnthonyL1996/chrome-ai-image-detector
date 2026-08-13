# MV3 extension shell

This directory is a no-backend Manifest V3 scaffold. It discovers ordinary
`<img>` elements only after the user presses **Scan images on this page** and
adds an accessible status next to each discovered image.

No inference engine or model is bundled yet. Every current result therefore
states that local confidence is unavailable. The adapter in
`runtime/model-runtime.mjs` is the boundary for a later audited ONNX Runtime
Web backend and model bundle. It requires backends to identify themselves as
local-only.

The extension has no host permissions, remote inference, telemetry, or network
code. Chrome grants temporary access to the active tab when the user invokes
the extension action.

For local inspection, open `chrome://extensions`, enable Developer mode, choose
**Load unpacked**, and select this `extension` directory. This is a development
scaffold, not a working AI detector.
