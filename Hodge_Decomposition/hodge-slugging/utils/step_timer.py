"""
Per-step timing diagnostic for the feature extraction pipeline.

Usage:
    .venv310/bin/python utils/step_timer.py [--file FILENAME] [--n-files N]

Loads the 3W signal files and times every sub-step of extract_features_one_file
independently, printing a breakdown table so the bottleneck is visible.
"""
import sys, pathlib, time, warnings, argparse
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import gudhi as _gudhi
import ripser as _ripser_mod

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from gtda.time_series import TakensEmbedding, Resampler, SingleTakensEmbedding
from gtda.diagrams import PersistenceEntropy, BettiCurve

from hodge.boundary_matrices import get_alpha_simplices, build_boundary_matrices, angle_to_1cochain
from hodge.decomposition import hodge_decomposition
from hodge.spectrum import compute_l1_spectrum
from utils.epsilon_selection import epsilon_from_diagram

# ── config ────────────────────────────────────────────────────────────────────
DATA_3W_ROOT = pathlib.Path("/Users/simo/Repos/TDA/Slugging/Data/3W_data")
LABEL_DIRS   = {0: DATA_3W_ROOT / "0", 1: DATA_3W_ROOT / "3", 2: DATA_3W_ROOT / "4"}
SENSOR_COL   = "P-TPT"
N_RESAMPLE   = 8000
PA_TO_BAR    = 1e-5
MIDDLE_FRAC   = 3 / 8   # 3000 samples — matches PH notebook signal length
MIN_WIN_LEN   = 200
EMBED_STRIDE  = 3       # stride 3; ~900 pts per file (matches PH notebook density)
ALPHA_PCA_DIM = 3
N_EIGS_OUT    = 10
MAX_TRIANGLES = 30_000


def _load_signal(csv_path):
    df = pd.read_csv(csv_path).infer_objects(copy=False).interpolate()
    if SENSOR_COL not in df.columns:
        return None
    sig = df[SENSOR_COL].dropna().values.astype(float)
    period = max(1, int(len(sig) / N_RESAMPLE))
    resampler = Resampler(period=period)
    _, sig_rs = resampler.fit_transform_resample(np.arange(len(sig)), sig)
    n = N_RESAMPLE
    if len(sig_rs) >= n:
        out = sig_rs[:n]
    else:
        out = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(sig_rs)), sig_rs)
    return out * PA_TO_BAR


def _get_middle_window(signal):
    n = len(signal)
    w = max(int(n * MIDDLE_FRAC), MIN_WIN_LEN)
    start = (n - w) // 2
    return signal[start:start + w]


def _pers_diag(cloud, maxdim=1):
    """Ripser VR persistence — fast in any dimension. For persistence features."""
    result = _ripser_mod.ripser(cloud, maxdim=maxdim)
    rows = []
    for dim, dgm in enumerate(result['dgms']):
        for b, d in dgm:
            rows.append([float(b), float(d), float(dim)])
    return np.array(rows) if rows else np.zeros((0, 3))


def _alpha_diag(cloud_low):
    """gudhi Alpha on PCA-3D cloud. For epsilon selection (scale-compatible with Hodge)."""
    ac = _gudhi.AlphaComplex(points=cloud_low.tolist())
    st = ac.create_simplex_tree()
    st.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
    rows = []
    for dim, (b, d) in st.persistence():
        if dim > 1:
            continue
        b_dist = float(np.sqrt(max(b, 0.0)))
        d_dist = float(np.sqrt(d)) if np.isfinite(d) else np.inf
        rows.append([b_dist, d_dist, float(dim)])
    return np.array(rows) if rows else np.zeros((0, 3))


def time_steps(signal, label=""):
    t = {}

    # 1. Middle window
    t0 = time.perf_counter()
    win = _get_middle_window(signal)
    t["1_window"] = time.perf_counter() - t0

    if np.std(win) < 1e-10:    # sensor flat-line (fp noise only) — skip
        print(f"  [SKIP] constant signal — {label}")
        return {}

    # 2. Embed param search
    t0 = time.perf_counter()
    ste = SingleTakensEmbedding(parameters_type='search', time_delay=50, dimension=9, n_jobs=1)
    ste.fit_transform(win)
    params = {"time_delay": int(ste.time_delay_), "dimension": max(int(ste.dimension_), 4)}
    t["2_param_search"] = time.perf_counter() - t0
    tau, d = params["time_delay"], params["dimension"]

    # 3. Takens embedding
    t0 = time.perf_counter()
    te    = TakensEmbedding(stride=EMBED_STRIDE, **params)
    cloud = te.fit_transform(win.reshape(1, -1))[0]
    n_pts = len(cloud)
    t["3_takens_embed"] = time.perf_counter() - t0

    # 4. StandardScaler
    t0 = time.perf_counter()
    cloud = StandardScaler().fit_transform(cloud)
    t["4_scaler"] = time.perf_counter() - t0

    # 5. Ripser VR on full-dimensional cloud → persistence diagram
    t0 = time.perf_counter()
    diag_full     = _pers_diag(cloud)
    diag_full_fin = diag_full[np.isfinite(diag_full[:, 1])]
    t["5_ripser_full"] = time.perf_counter() - t0

    # 6. PersistenceEntropy + BettiCurve from full-dim Ripser diagram — before PCA
    t0 = time.perf_counter()
    if len(diag_full_fin) > 0:
        diag_batch = diag_full_fin[np.newaxis]
        PersistenceEntropy(normalize=True, nan_fill_value=0.0).fit_transform(diag_batch)
        BettiCurve().fit_transform(diag_batch)
    t["6_pers_features"] = time.perf_counter() - t0

    # 7. PCA(3) — for Hodge complex and epsilon selection
    t0 = time.perf_counter()
    pca_dim   = min(ALPHA_PCA_DIM, d, n_pts - 1)
    cloud_low = PCA(n_components=pca_dim).fit_transform(cloud)
    t["7_pca"] = time.perf_counter() - t0

    # 8. Alpha on PCA-3D cloud → epsilon selection (scale-compatible with Hodge)
    t0 = time.perf_counter()
    diag_low     = _alpha_diag(cloud_low)
    diag_low_fin = diag_low[np.isfinite(diag_low[:, 1])]
    t["8_alpha_3d"] = time.perf_counter() - t0

    # 9. epsilon_from_diagram (from 3D Alpha diagram — Hodge-compatible scale)
    t0 = time.perf_counter()
    eps_rips = epsilon_from_diagram(diag_low_fin, strategy="most_persistent")
    t["9_epsilon"] = time.perf_counter() - t0

    # 10. winding number
    t0 = time.perf_counter()
    theta   = np.arctan2(cloud_low[:, 1], cloud_low[:, 0])
    theta_s = np.sort(theta)
    diffs_w = (np.diff(theta_s) + np.pi) % (2 * np.pi) - np.pi
    gap_w   = (theta_s[0] + 2 * np.pi - theta_s[-1] + np.pi) % (2 * np.pi) - np.pi
    int(round((float(np.sum(diffs_w)) + float(gap_w)) / (2 * np.pi)))
    t["10_winding"] = time.perf_counter() - t0

    # 11. Alpha complex for Hodge (O(n) simplices via Delaunay — no triangle explosion)
    if eps_rips is None:
        for k in ["11_alpha_complex", "12_boundary_mat", "13_hodge", "14_l1_spectrum"]:
            t[k] = float('nan')
        print_table(t, tau, d, n_pts, eps_rips, label)
        return t

    t0 = time.perf_counter()
    sc = get_alpha_simplices(cloud_low, alpha_sq=eps_rips ** 2, max_dim=2)
    n_edges = len(sc.edges)
    n_tri   = len(sc.triangles)
    t["11_alpha_complex"] = time.perf_counter() - t0

    if n_tri == 0 or n_tri > MAX_TRIANGLES:
        for k in ["12_boundary_mat", "13_hodge", "14_l1_spectrum"]:
            t[k] = float('nan')
        print_table(t, tau, d, n_pts, eps_rips, label, n_edges=n_edges, n_tri=n_tri)
        return t

    # 12. Boundary matrices
    t0 = time.perf_counter()
    B1, B2 = build_boundary_matrices(sc)
    t["12_boundary_mat"] = time.perf_counter() - t0

    # 13. Hodge decomposition
    t0 = time.perf_counter()
    f1     = angle_to_1cochain(cloud_low, sc, dims=(0, 1))
    hodge_decomposition(f1, B1, B2)
    t["13_hodge"] = time.perf_counter() - t0

    # 14. L1 spectrum
    t0 = time.perf_counter()
    n_req = min(N_EIGS_OUT + 5, max(N_EIGS_OUT, n_edges - 2))
    compute_l1_spectrum(B1, B2, n_eigs=n_req)
    t["14_l1_spectrum"] = time.perf_counter() - t0

    print_table(t, tau, d, n_pts, eps_rips, label, n_edges=n_edges, n_tri=n_tri)
    return t


def print_table(t, tau, d, n_pts, eps, label, n_edges=None, n_tri=None):
    total = sum(v for v in t.values() if not np.isnan(v))
    print(f"\n{'='*70}")
    print(f"  {label}")
    eps_str = f"{eps:.4f}" if eps is not None else "None"
    print(f"  τ={tau}  d={d}  n_pts={n_pts}  ε={eps_str}"
          + (f"  edges={n_edges}  triangles={n_tri}" if n_edges else ""))
    print(f"{'='*70}")
    print(f"  {'Step':<28} {'seconds':>8}  {'%':>6}")
    print(f"  {'-'*46}")
    for step, sec in t.items():
        if np.isnan(sec):
            print(f"  {step:<28} {'n/a':>8}")
        else:
            pct = 100 * sec / total if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {step:<28} {sec:8.3f}s  {pct:5.1f}%  {bar}")
    print(f"  {'─'*46}")
    print(f"  {'TOTAL':<28} {total:8.3f}s")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None,
                        help="Stem of a specific CSV file to time (e.g. WELL-00001_20170201070114)")
    parser.add_argument("--n-files", type=int, default=3,
                        help="Number of files per class to time if --file not given")
    args = parser.parse_args()

    if args.file:
        # Find the file across all label dirs
        for label, d in LABEL_DIRS.items():
            p = d / f"{args.file}.csv"
            if p.exists():
                print(f"\nLoading {p.name}  (class {label})")
                sig = _load_signal(p)
                if sig is not None:
                    time_steps(sig, label=f"class={label}  {p.stem}")
                return
        print(f"File not found: {args.file}")
        return

    # Time N files per class
    for label, d in LABEL_DIRS.items():
        files = sorted(d.glob("*.csv"))[:args.n_files]
        for p in files:
            print(f"\nLoading {p.name}  (class {label})")
            sig = _load_signal(p)
            if sig is None:
                print("  skip — no SENSOR_COL")
                continue
            time_steps(sig, label=f"class={label}  {p.stem}")


if __name__ == "__main__":
    main()
