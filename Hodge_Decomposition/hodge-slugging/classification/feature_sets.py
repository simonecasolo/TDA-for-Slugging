"""
Feature set definitions for the 3W dataset classification experiment.

Four feature sets are compared (Section 5 of the paper):
  baseline_scalar  : 8 persistence indicators from the baseline paper
  baseline_image   : persistence images (best baseline)
  hodge_only       : 7 Hodge-derived features
  hodge_augmented  : baseline_scalar + hodge_only (14 features)
"""

from __future__ import annotations

BASELINE_SCALAR_FEATURES = [
    "max_pers_h0",
    "max_pers_h1",
    "mean_betti0",
    "mean_betti1",
    "entropy_norm_h0",
    "entropy_norm_h1",
]

HODGE_FEATURES = [
    "eta_harm",
    "eta_grad",
    "eta_curl",
    "harm_curl_ratio",
    "beta1_hodge",
    "lambda1",
    "spectral_gap",
]

HODGE_AUGMENTED_FEATURES = BASELINE_SCALAR_FEATURES + HODGE_FEATURES

FEATURE_SETS = {
    "baseline_scalar": BASELINE_SCALAR_FEATURES,
    "hodge_only": HODGE_FEATURES,
    "hodge_augmented": HODGE_AUGMENTED_FEATURES,
}

# Labels matching the 3W dataset convention
REGIME_LABELS = {
    0: "normal",
    1: "severe_slugging",
    2: "flow_instabilities",
}
