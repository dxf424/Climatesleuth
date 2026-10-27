Package overview
================

MILES-ClimateSleuth is a research package for learning and comparing
climate-model behavior in representation space.

Core subpackages
----------------

``climatesleuth.datasets``
    Dataset and preprocessing utilities used to build paired inputs for
    contrastive learning.

``climatesleuth.models``
    Neural-network encoders and losses. This includes the original VGG/ResNet
    contrastive encoders and the channel-aware monthly radiation/TLWP VGG model.

``climatesleuth.trainer``
    Training helpers for contrastive-learning experiments.

``climatesleuth.evaluation``
    Latent-feature visualization, linear/logistic evaluation, and latent-distance
    calculations. Latent distances can be summarized by month or member.

``climatesleuth.analysis``
    Explainable-AI and attribution tools, including integrated gradients and
    backward optimization.

Recommended workflow
--------------------

1. Build paired climate fields with ``climatesleuth.datasets`` or an experiment
   preprocessing script.
2. Train an encoder from ``climatesleuth.models`` with ``NTXentLoss``.
3. Project PPE members and observations into latent space.
4. Use ``climatesleuth.evaluation`` to compute latent distances.
What belongs in the package
---------------------------

The package should contain stable interfaces: models, losses, data transforms,
latent-distance metrics and file-format helpers.
Hard-coded NCAR campaign paths, one-off figures, and PBS job files should remain
as scripts or examples until they become reusable APIs.
