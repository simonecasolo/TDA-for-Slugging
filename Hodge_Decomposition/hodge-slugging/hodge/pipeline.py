"""
Integrated GiottoTDA + Hodge pipeline for a single time window.

Runs both tracks in parallel:
  Track 1 (existing): GiottoTDA persistence features
  Track 2 (new):      Hodge decomposition + L1 spectrum

Returns a unified feature dict combining both tracks.
"""

from __future__ import annotations

import numpy as np
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, Amplitude, BettiCurve

from hodge.boundary_matrices import (
    SimplicialComplex,
    get_rips_simplices,
    build_boundary_matrices,
    pressure_to_1cochain,
)
from hodge.decomposition import hodge_decomposition
from hodge.spectrum import compute_l1_spectrum
from utils.epsilon_selection import epsilon_from_diagram


_VRP = VietorisRipsPersistence(
    homology_dimensions=[0, 1],
    infinity_values=1e10,
    n_jobs=1,
)


def run_window(
    embedding: np.ndarray,
    pressure_window: np.ndarray,
    n_eigs: int = 20,
    epsilon_strategy: str = "most_persistent",
) -> dict:
    """
    Run the full Hodge + persistence pipeline on one embedded window.

    Parameters
    ----------
    embedding : np.ndarray, shape (n_points, embed_dim)
        Takens-embedded pressure signal for this window.
    pressure_window : np.ndarray, shape (n_samples,)
        Raw pressure values for this window (used to build 1-cochain).
    n_eigs : int
        Number of L1 eigenvalues to compute.
    epsilon_strategy : str
        Strategy for selecting filtration scale from persistence diagram.
        One of {"most_persistent", "max_beta1"}.

    Returns
    -------
    dict
        Combined feature dict with keys from both tracks.
        Returns zero-filled Hodge features if no H1 loop is found.
    """
    # ------------------------------------------------------------------ #
    # Track 1: GiottoTDA persistence features (unchanged from baseline)   #
    # ------------------------------------------------------------------ #
    diagram = _VRP.fit_transform(embedding[np.newaxis])[0]

    persistence_features = _extract_persistence_features(diagram)

    # ------------------------------------------------------------------ #
    # Bridge: derive epsilon from persistence diagram                     #
    # ------------------------------------------------------------------ #
    epsilon = epsilon_from_diagram(diagram, strategy=epsilon_strategy)

    _zero_hodge = {
        "eta_harm": 0.0,
        "eta_grad": 0.0,
        "eta_curl": 0.0,
        "harm_curl_ratio": 0.0,
        "orthogonality_residual": 0.0,
        "beta1_hodge": 0,
        "lambda1": 0.0,
        "lambda2": 0.0,
        "spectral_gap": 0.0,
        "bulk_mean": 0.0,
    }

    if epsilon is None:
        return {**persistence_features, **_zero_hodge}

    # ------------------------------------------------------------------ #
    # Track 2: Hodge decomposition                                        #
    # ------------------------------------------------------------------ #
    sc = get_rips_simplices(embedding, epsilon, max_dim=2)

    if len(sc.edges) == 0 or len(sc.triangles) == 0:
        return {**persistence_features, **_zero_hodge}

    B1, B2 = build_boundary_matrices(sc)

    n_nodes = len(sc.nodes)
    p_on_nodes = pressure_window[:n_nodes]
    f1 = pressure_to_1cochain(p_on_nodes, sc)

    decomp = hodge_decomposition(f1, B1, B2)
    spec = compute_l1_spectrum(B1, B2, n_eigs=n_eigs)

    hodge_features = {
        "eta_harm": decomp.eta_harm,
        "eta_grad": decomp.eta_grad,
        "eta_curl": decomp.eta_curl,
        "harm_curl_ratio": decomp.harm_curl_ratio,
        "orthogonality_residual": decomp.orthogonality_residual,
        **spec.to_feature_dict(),
    }

    return {**persistence_features, **hodge_features}


# ------------------------------------------------------------------ #
# Private helpers                                                      #
# ------------------------------------------------------------------ #

def _extract_persistence_features(diagram: np.ndarray) -> dict:
    """
    Extract scalar persistence features from a GiottoTDA diagram.
    Replicates the 8-indicator feature set from the baseline paper.
    """
    features = {}
    diag_3d = diagram[np.newaxis]  # shape (1, n_pts, 3)

    try:
        PE = PersistenceEntropy(nan_fill_value=0.0)
        entropy = PE.fit_transform(diag_3d)[0]
        features["entropy_h0"] = float(entropy[0])
        features["entropy_h1"] = float(entropy[1])
    except Exception:
        features["entropy_h0"] = 0.0
        features["entropy_h1"] = 0.0

    try:
        AMP = Amplitude(metric="wasserstein", order=None)
        amp = AMP.fit_transform(diag_3d)[0]
        features["p_inf_h0"] = float(amp[0])
        features["p_inf_h1"] = float(amp[1])
    except Exception:
        features["p_inf_h0"] = 0.0
        features["p_inf_h1"] = 0.0

    try:
        BC = BettiCurve(n_bins=10)
        betti = BC.fit_transform(diag_3d)[0]
        features["mean_betti0"] = float(np.mean(betti[0]))
        features["mean_betti1"] = float(np.mean(betti[1]))
    except Exception:
        features["mean_betti0"] = 0.0
        features["mean_betti1"] = 0.0

    return features
