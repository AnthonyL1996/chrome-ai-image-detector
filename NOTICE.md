# Third-party notices

This repository contains original MIT-licensed source code and can optionally
interoperate with third-party software and data. The following upstream
licenses are not replaced by this repository's MIT license.

## Software

- **timm** provides the optional ConvNeXtV2 Nano model implementation and is
  licensed under Apache License 2.0. It is installed as an optional dependency;
  its source is not copied into this repository.

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

## No bundled third-party artifacts

This repository does not bundle third-party datasets, images, pretrained
weights, trained checkpoints, or other model weights. Installing optional
dependencies or downloading data is a separate action governed by the relevant
upstream licenses and terms.
