"""
Standalone worker script for per-file Hodge feature extraction in notebook 03.

Called by the notebook as:
    python nb03_feature_worker.py <pickle_in> <parquet_out>

<pickle_in>  : path to a pickle file containing (sig, embed_params, window_size, window_stride, embed_stride)
<parquet_out>: path where the resulting DataFrame should be written

Exit codes:
    0 — success, parquet written
    1 — error (printed to stderr)
"""
import sys
import pathlib
import pickle

import numpy as np
import pandas as pd

# Make the project root importable (worker is in utils/, project root is one level up)
_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from gtda.time_series import TakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, BettiCurve

from hodge.boundary_matrices import get_rips_simplices, build_boundary_matrices, pressure_to_1cochain
from hodge.decomposition import hodge_decomposition
from hodge.spectrum import compute_l1_spectrum
from utils.epsilon_selection import epsilon_from_diagram


def extract_features_one(signal, embed_params, embed_stride):
    te    = TakensEmbedding(stride=embed_stride, **embed_params)
    cloud = te.fit_transform(signal.reshape(1, -1))[0]

    VRP  = VietorisRipsPersistence(homology_dimensions=[0, 1], infinity_values=1e10, n_jobs=1)
    diag = VRP.fit_transform(cloud[np.newaxis])[0]
    diag_3d = diag[np.newaxis]

    PE_n = PersistenceEntropy(normalize=True, nan_fill_value=0.0)
    BC   = BettiCurve()
    ent_norm = PE_n.fit_transform(diag_3d)[0]
    betti    = BC.fit_transform(diag_3d)[0]

    pers_h0 = [abs(p[1] - p[0]) for p in diag if p[2] == 0]
    pers_h1 = [abs(p[1] - p[0]) for p in diag if p[2] == 1]

    pers_feats = {
        "max_pers_h0":     float(max(pers_h0)) if pers_h0 else 0.0,
        "max_pers_h1":     float(max(pers_h1)) if pers_h1 else 0.0,
        "mean_betti0":     float(np.mean(betti[0])),
        "mean_betti1":     float(np.mean(betti[1])),
        "entropy_norm_h0": float(ent_norm[0]),
        "entropy_norm_h1": float(ent_norm[1]),
    }

    eps = epsilon_from_diagram(diag, strategy="most_persistent")
    zero_hodge = dict(eta_harm=0., eta_grad=0., eta_curl=0.,
                      harm_curl_ratio=0., beta1_hodge=0,
                      lambda1=0., lambda2=0., spectral_gap=0.)
    if eps is None:
        return {**pers_feats, **zero_hodge}

    sc = get_rips_simplices(cloud, eps, max_dim=2)
    if len(sc.edges) < 2 or len(sc.triangles) == 0:
        return {**pers_feats, **zero_hodge}

    B1, B2 = build_boundary_matrices(sc)
    n_nodes = len(sc.nodes)
    f1      = pressure_to_1cochain(signal[:n_nodes], sc)
    decomp  = hodge_decomposition(f1, B1, B2)
    spec    = compute_l1_spectrum(B1, B2, n_eigs=min(15, len(sc.edges) - 2))

    return {**pers_feats,
            "eta_harm": decomp.eta_harm, "eta_grad": decomp.eta_grad,
            "eta_curl": decomp.eta_curl, "harm_curl_ratio": decomp.harm_curl_ratio,
            "beta1_hodge": spec.beta1_hodge,
            "lambda1": spec.lambda1, "lambda2": spec.lambda2,
            "spectral_gap": spec.spectral_gap}


def extract_features_windows(sig, embed_params, window_size, window_stride, embed_stride):
    τ, d    = embed_params["time_delay"], embed_params["dimension"]
    min_pts = (d - 1) * τ + embed_stride
    rows    = []
    n       = len(sig)
    for start in range(0, n - window_size + 1, window_stride):
        feats = extract_features_one(sig[start:start + window_size], embed_params, embed_stride)
        feats["window_start"] = start
        rows.append(feats)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: nb03_feature_worker.py <pickle_in> <parquet_out>", file=sys.stderr)
        sys.exit(1)

    pkl_path    = pathlib.Path(sys.argv[1])
    parquet_out = pathlib.Path(sys.argv[2])

    with open(pkl_path, "rb") as fh:
        sig, embed_params, window_size, window_stride, embed_stride = pickle.load(fh)

    df = extract_features_windows(sig, embed_params, window_size, window_stride, embed_stride)
    df.to_parquet(parquet_out)
