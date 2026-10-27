# Climatesleuth

## About

MILES-ClimateSleuth is an open software platform to train and deploy AI models
that help diagnose climate-model divergence and support climate-model tuning.
The package is backed by the AI and atmospheric-science expertise of the MILES
group, the NSF National Center for Atmospheric Research, and Learning the Earth
with Artificial Intelligence and Physics (LEAP).

ClimateSleuth is organized around a common workflow:

1. prepare paired climate-model datasets;
2. train contrastive-learning encoders;
3. project model members and observations into latent space;
4. evaluate latent-space structure and model distance;
5. use explainable-AI analysis to interpret latent-distance diagnostics.

## Package layout

The reusable Python package lives under `climatesleuth/`.

| Folder | Purpose |
| --- | --- |
| `climatesleuth/datasets` | Dataset loading, paired-member batching, static variables, and preprocessing helpers used to build training inputs. |
| `climatesleuth/models` | Contrastive-learning model definitions and losses, including VGG/ResNet encoders, the channel-aware monthly radiation/TLWP VGG, and `NTXentLoss`. |
| `climatesleuth/trainer` | Training utilities for contrastive-learning experiments. |
| `climatesleuth/evaluation` | Latent-feature visualization, linear/logistic evaluation, and latent-distance calculations for model-vs-reference comparisons. |
| `climatesleuth/analysis` | Explainable-AI and attribution tools, including integrated gradients and backward optimization. |

Additional project folders:

| Folder | Purpose |
| --- | --- |
| `scripts` | Example workflows and experiment entry points for training, evaluation, and analysis. |
| `notebooks` | Interactive examples and exploratory workflows. |
| `docs/source` | Sphinx documentation source, including package overview, user guides, API pages, and development notes. |
| `tests` | Lightweight tests and smoke checks for package functionality. |

## Main capabilities

### Contrastive learning

`climatesleuth.models` provides neural encoders for learning latent
representations of climate fields. The package includes the original VGG and
ResNet contrastive encoders plus `MonthlyRadiationTLWPNoPolesVGG`, a
channel-aware VGG model adapted from recent monthly radiation/TLWP training.

`NTXentLoss` implements the symmetric SimCLR-style normalized temperature-scaled
cross entropy objective used for paired climate-field training.

### Latent-space evaluation

`climatesleuth.evaluation` includes tools to visualize latent features and
calculate distances between model and reference/observation embeddings. These
utilities can summarize monthly distances to member-mean latent distances.

### Explainable AI

`climatesleuth.analysis` contains attribution and optimization methods that help
interpret what the trained encoders use to distinguish climate-model behavior.

## Documentation and examples

The documentation source is in `docs/source/` and is configured for Read the Docs.
Start with:

- `docs/source/overview.rst` for the full package summary;
- `docs/source/api/` for API reference pages.

Sample scripts include:

- `scripts/train_monthly_radiation_tlwp_no_poles_sample.py`;
- `scripts/calculate_latent_distance_sample.py`.

## Recent package additions

Recent updates moved reusable components from monthly radiation/TLWP
contrastive-learning experiments into the
package:

- `climatesleuth.models.VGG_contrastive_channel_aware` for the channel-aware VGG
  encoder;
- updated `climatesleuth.models.NTXentLoss` with the symmetric SimCLR-style loss;
- `climatesleuth.evaluation.latent_distance` for cosine/Euclidean latent-distance
  utilities;
- Sphinx documentation and sample scripts following the package structure.

## Public release scope

This repository contains the contrastive learning, evaluation, and attribution
code. Parameter-calibration code is excluded from this public release.

## License

Original contributions not already covered by another license are available
for noncommercial academic research and teaching under `LICENSE`. Commercial
use of that material requires separate written permission. Upstream NSF NCAR
MILES-ClimateSleuth code remains MIT-licensed under `LICENSE-MIT`; its existing
permissions, including commercial use, are preserved.
