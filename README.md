# MSAWF: Multi-stage Augmentation Website Fingerprinting

## Overview

MSAWF is a research implementation for few-shot multi-tab website fingerprinting. It combines synthetic pretraining, synthetic-to-real transfer, and perturbation-aware target adaptation in a three-stage framework.

This repository provides the implementation of MSAWF and its evaluation interfaces. It is not distributed as a one-command reproduction package, and paper-reported results are not regenerated during repository packaging.

## Method

- **Stage I:** dual-view perturbation-robust synthetic pretraining.
- **Stage II:** progressive synthetic-to-real transfer bridging.
- **Stage III:** dual-objective few-shot target adaptation.

The paper defines the research method; the source code provides the corresponding implementation.

## Repository Structure

```text
configs/              Canonical and example experiment configurations
msawf/data/           Trace schemas, synthesis, manifests, and split utilities
msawf/augmentation/   Prefix and insertion-like transformations
msawf/models/         Encoder, classifier, and composed model interfaces
msawf/losses/         Training objective implementations
msawf/trainers/       Stage I, II, and III trainers
msawf/checkpoints/    Checkpoint serialization and lineage validation
msawf/runtime/        Runtime, batching, logging, and CLI composition
msawf/evaluation/     Evaluation interfaces and metrics
scripts/              Thin wrappers around installed console entry points
tests/                Unit and smoke tests
```

## Requirements

- Python 3.10 or newer
- PyTorch 2.3 or newer
- NumPy 1.24 or newer

CPU execution is supported for initialization and lightweight validation. The package does not require a specific CUDA release.

## Installation

From the repository root:

```bash
python -m pip install -e .
```

Dependencies are declared in `pyproject.toml` and `requirements.txt`.

## Data Format

Traces use packet direction values `-1` and `+1`, with `0` reserved for padding. The maximum observation length is 15,000 packets, and supported prefix lengths are 3,000, 5,000, 8,000, 10,000, and 15,000 packets.

The implementation supports multi-label traces and multi-label few-shot support sets. Synthetic multi-tab traces are constructed from single-tab direction sequences using the deterministic synthesis utilities provided under `msawf.data`.

## Basic Usage

```bash
msawf-train --help
msawf-eval --help
```

Experiments are controlled through JSON files under `configs/`. The command-line interfaces expose only the currently implemented package interfaces.

## Evaluation

The evaluation package provides interfaces for closed-world, unified open-world, insertion-noise robustness, fixed-prefix, and early-recognition settings. Implemented summaries include Precision, Recall, F1, A@K, degradation rate, average decision length, average observation ratio, and early decision rate.

Evaluation requires user-prepared data, a compatible checkpoint, and the corresponding manifest information. Evaluation is kept separate from training and checkpoint selection.

## Data and Defense Notes

Datasets and pretrained checkpoints are not bundled with this repository. Users must prepare traces compatible with the documented input schema and supply paths through configuration files.

The dataset used in our experiments is obtained from the dataset release associated with [FMWF](https://github.com/WW-Meng/FMWF). Please refer to the original project and [publication](https://doi.org/10.1145/3696410.3714811) for dataset availability and usage conditions.

Defense-specific traffic for WTF-PAD, FRONT, and Tamaraw is also not bundled. Defense datasets or adapters must be prepared separately; the repository does not provide or promise external dataset downloads.

## Citation

Please cite the accompanying MSAWF paper using its authoritative publication metadata. The publication record is not finalized in this repository, so this README intentionally does not infer authors, DOI, venue, volume, or page numbers.

## Acknowledgements

The MSAWF code and documentation are released under the MIT License. See `LICENSE` for scope and `DATASETS.md` for dataset attribution.
