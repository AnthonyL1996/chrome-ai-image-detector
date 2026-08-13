# Third-party notices

This repository contains original MIT-licensed source code and can optionally
interoperate with third-party software and data. The following upstream
licenses are not replaced by this repository's MIT license.

## Software

- **timm** provides the optional ConvNeXtV2 Nano model implementation and is
  licensed under Apache License 2.0. It is installed as an optional dependency;
  its source is not copied into this repository.
- **ONNX Runtime Web 1.27.0** provides the vendored WebAssembly inference
  runtime under `extension/runtime/vendor/` and is licensed under the MIT
  License. The vendored files are the upstream `ort.wasm.bundle.min.mjs`
  (SHA-256
  `1db5e1c5cd2b860eed85e6eeff23e2aaa7cffcc407f67093bcc888f631b94ba9`) and
  its `ort-wasm-simd-threaded.mjs` sidecar (SHA-256
  `0a1e718d99c41b22c21f2520ff4f9e883a6b5533856e398d21816ee8eb8185d3`) plus
  `ort-wasm-simd-threaded.wasm` (SHA-256
  `d1ab1b94b16a65b29d710d0b587b29e7bed336827577623913479b8afe8113e6`).
  Copyright Microsoft Corporation and contributors.

## MONET dataset metadata and selected subsets

The in-progress data preparation path targets a pinned revision of the MONET
dataset (`jasperai/monet`). Its upstream dataset card is declared
Apache-2.0. The selected sources have separate declared licenses:

- `commoncatalog-cc-by`: CC-BY-4.0. Redistribution requires the attribution and
  notices applicable to the source material.
- `synthetic-flux-schnell`: Apache-2.0.
- `synthetic-flux-klein`: Apache-2.0.
- `synthetic-z-image`: Apache-2.0.

The three synthetic entries are model-generator subsets. Their dataset license
labels do not waive any separate upstream model terms or restrictions that may
apply. Consumers must review the pinned MONET dataset card and source metadata
before use or redistribution.

## Bundled and unbundled third-party artifacts

The extension bundles only the ONNX Runtime Web files listed above. This
repository does not bundle third-party datasets, images, pretrained weights,
trained checkpoints, or detector model weights. Installing optional
dependencies or downloading data is a separate action governed by the relevant
upstream licenses and terms.
