# POIDH AI detector

> **Work in progress:** the repository contains a runnable local extension and
> a reproducible H100 training path, but no result here establishes the score
> on POIDH's private benchmark.

This repository is an experimental, auditable implementation for POIDH bounty
#323. The intended deliverable is a Manifest V3 browser extension that scores
images locally as real or AI-generated. No remote inference API is planned for
the extension.

## Target contract

The work is organized around these acceptance and deployment constraints:

- reach at least 75% on the bounty evaluation while keeping the decision
  threshold fixed at `0.65`;
- keep calibration data disjoint from the exposed holdouts and record dataset,
  split, checkpoint, configuration, code, and license provenance;
- keep the separate MONET research path reproducible, with random-init
  ConvNeXtV2 Nano experiments and explicit H100 provenance;
- ship one ONNX file, no external tensor data, at most 100 MiB, in the extension;
- expose one fixed `float32` input named `image` with shape
  `(1, 3, 384, 384)` and one calibrated `float32` output named
  `probability_ai` with shape `(1, 1)`; and
- run inference entirely inside a local-only Manifest V3 extension.

These are repository contracts and goals. They are not evidence that the
current model meets the bounty threshold or deployment requirements.

## Current status

The extension now bundles a hash-checked Community Forensics-derived ONNX
detector and the vendored ONNX Runtime Web WASM adapter. Its preprocessing and
model I/O contract are covered by browser-runtime tests. The upstream public
repository reports a proxy benchmark result, but the private POIDH benchmark is
not available here, so this repository makes no private-score claim.

The separate MONET path remains an auditable research and retraining route. The
default BCE-selected H100 pilot checkpoint was below the bounty gate; a final
epoch-30 EMA audit on the same frozen validation partition reached 77.36%
balanced accuracy at the fixed 0.65 threshold (AUC 0.8965). That is public
proxy evidence only, not a claim about POIDH's private benchmark. The training
CLI can select checkpoints by this fixed-threshold metric for follow-up runs.

The MONET preparation path pins an upstream dataset revision, applies
deterministic source quotas, verifies declared per-row licenses, prevents
content/provenance overlap with registered holdouts, and writes hashed
manifests. Those controls reduce accidental drift; they do not replace an
independent license or model-quality review.

## Development

Python 3.11 or newer is required. The core contracts use the standard library.
Create a virtual environment and install the developer tools with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the dependency-free test suite directly:

```bash
python -m unittest discover -s tests -v
```

Run the extension runtime and service-worker tests with Node's ESM VM support:

```bash
node --experimental-vm-modules --test extension/tests/*.mjs extension/tests/*.cjs
```

Or run the same tests and repository checks with the developer dependencies:

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

Install the optional training and dataset tooling separately when needed:

```bash
python -m pip install -e '.[training]'
```

The optional versions are intentionally broad compatibility ranges, not a
claim of byte-for-byte reproducibility. A real training run must capture its
resolved Python, PyTorch, CUDA, cuDNN, NumPy, and timm environment alongside
the checkpoint.

## Data and model licensing

The repository's original source code is MIT-licensed. That license does not
automatically apply to datasets, generated images, trained weights, model
implementations supplied by dependencies, or other upstream artifacts.

The in-progress MONET path uses a mixed-license selection described in
[NOTICE.md](NOTICE.md): the dataset card and three model-generator subsets are
declared Apache-2.0, while the CommonCatalog subset is CC-BY-4.0 and carries
attribution obligations. Before downloading, redistributing, training on, or
publishing any resulting artifact, verify the pinned upstream documentation,
the license recorded for every selected row, applicable attribution, generator
terms, and the intended use. A trained model's distributability must be
reviewed independently from this repository's code license.

The extension bundles one third-party-derived ONNX detector. Its source model,
graph transformation, digest, and upstream MIT notices are documented in
[NOTICE.md](NOTICE.md) and `tools/wrap_community_forensics_onnx.py`. The MONET
training path does not bundle its dataset, images, or checkpoints.

## Rebuilding the bundled model

The 83 MiB model is derived from the public MIT-licensed Community Forensics
export used by the permissively licensed reference implementation. To rebuild
the exact extension graph, install a current `onnx` and `numpy`, clone the
source repository, and run:

```bash
git clone https://github.com/pixilated730/local-ai-image-detector.git /tmp/local-ai-image-detector
git -C /tmp/local-ai-image-detector checkout --detach dddb57b
python3 -m pip install 'onnx==1.20.1' numpy
python3 tools/wrap_community_forensics_onnx.py \
  /tmp/local-ai-image-detector/extension/models/detector.onnx \
  --destination extension/model/detector.onnx
sha256sum extension/model/detector.onnx
```

The expected SHA-256 is
`89be5ba0b80dfa2e2fa6bbc3eea28562a07a067905d8407f8540ebfa13a565ba`.
The sidecar metadata records the same digest and the fixed 384px preprocessing
contract. This rebuild command is a supply-chain transform, not a claim that
the source repository's proxy evaluation is the POIDH private benchmark.

## License

Original source code is available under the [MIT License](LICENSE). See
[NOTICE.md](NOTICE.md) for third-party software and dataset notices.
