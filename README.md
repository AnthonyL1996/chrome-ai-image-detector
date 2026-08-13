# POIDH AI detector

> **Work in progress:** this repository does not contain a finished or
> bounty-winning detector. The current calibration baseline is **72.5%**, below
> the **75%** target.

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
- train the ConvNeXtV2 Nano detector from random initialization rather than
  redistribute third-party pretrained weights;
- export one ONNX file, opset 18, no external tensor data, at most 100 MiB;
- expose one fixed `float32` input named `image` with shape
  `(1, 3, 224, 224)` and one calibrated `float32` output named
  `probability_ai` with shape `(1, 1)`; and
- ultimately run inference entirely inside a local-only Manifest V3 extension.

These are repository contracts and goals. They are not evidence that the
current model meets the bounty threshold or deployment requirements.

## Current status

The repository currently focuses on reproducible dataset, leakage, training,
calibration, checkpoint, and export contracts. The existing 72.5% calibration
baseline is below the 75% target. A custom, permissively licensed training path
based on selected MONET subsets is in progress. There is no finished extension,
production ONNX model, or bundled model checkpoint here yet.

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

No third-party dataset, image, pretrained weight, trained checkpoint, or ONNX
model is bundled in this repository.

## License

Original source code is available under the [MIT License](LICENSE). See
[NOTICE.md](NOTICE.md) for third-party software and dataset notices.
