"""
Feature set definitions for the 3W dataset classification experiment.

Feature sets compared (Section 5 of the paper):
  baseline_scalar      : 6 persistence indicators from the baseline paper
  hodge_only           : 8 Hodge-derived features (incl. winding_number)
  hodge_augmented      : baseline_scalar + hodge_only (14 features)
  full_hodge           : hodge_only + L1 spectrum (18 features)
  hodge_augmented_full : baseline_scalar + full_hodge (24 features)
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
    "winding_number",
]

EIG_FEATURES = [f"eig_{i:02d}" for i in range(1, 11)]

HODGE_AUGMENTED_FEATURES      = BASELINE_SCALAR_FEATURES + HODGE_FEATURES
FULL_HODGE_FEATURES            = HODGE_FEATURES + EIG_FEATURES
HODGE_AUGMENTED_FULL_FEATURES  = BASELINE_SCALAR_FEATURES + FULL_HODGE_FEATURES

FEATURE_SETS = {
    "baseline_scalar":      BASELINE_SCALAR_FEATURES,
    "hodge_only":           HODGE_FEATURES,
    "hodge_augmented":      HODGE_AUGMENTED_FEATURES,
    "full_hodge":           FULL_HODGE_FEATURES,
    "hodge_augmented_full": HODGE_AUGMENTED_FULL_FEATURES,
}

# Labels matching the 3W dataset convention
REGIME_LABELS = {
    0: "normal",
    1: "severe_slugging",
    2: "flow_instabilities",
}
