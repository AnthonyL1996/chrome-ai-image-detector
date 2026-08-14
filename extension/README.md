# MV3 local detector extension

This directory contains the Manifest V3 browser surface for the detector. It
discovers ordinary `<img>` elements only after the user presses **Scan images
on this page** and adds an accessible status next to each discovered image.

The service worker loads a bundled ONNX Runtime Web WASM adapter and the
`model/detector.onnx` plus `model/metadata.json` bundle. The adapter verifies
the model digest and fixed input/output contract before scoring. The bundled
model is a Community Forensics ViT-S/16 @384 derivative with a refit head and
an embedded +2.29-logit calibration offset; its upstream lineage and notices
are recorded in the repository [NOTICE.md](../NOTICE.md).

Inference stays local to the browser; no image or result is sent to a remote
inference service and no telemetry is collected. The explicit action click uses
Chrome's temporary `activeTab` grant for the current page; no persistent
`host_permissions` or runtime permission prompt is needed. For the same-origin
privacy boundary, images whose URL origin differs from the scanned page are
reported unavailable instead of being fetched as a privileged cross-origin
request.

The browser preprocessing resizes the shorter side to 440px with the same
rounded dimensions and fractional center-crop draw convention as the source
pipeline, converts RGBA to RGB NCHW float32, and applies ImageNet normalization.
Decoded images are capped at 20 MiB, 16 megapixels, and 8192 pixels per axis.
Each image fetch/decode is bounded by a 30-second timeout; timed-out images are
reported unavailable and do not block the remaining scan.
Blob-backed images
are materialized in the webpage context before scoring; if that cannot be done,
they are reported unavailable rather than fetched from the service worker.
The ONNX Runtime Web WASM files are vendored under `runtime/vendor/`; no model
or runtime download, cloud inference request, or result upload occurs after
installation. Page image bytes are read only from the permitted page origin
during an explicit scan.

For local inspection, open `chrome://extensions`, enable Developer mode, choose
**Load unpacked**, and select this `extension` directory.

The public source model and graph-wrapping command are documented in the root
README. The upstream repository reports a proxy benchmark result, but that is
not evidence of a score on POIDH's private benchmark.
