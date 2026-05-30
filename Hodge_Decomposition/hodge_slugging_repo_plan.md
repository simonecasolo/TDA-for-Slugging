# Claude Code Build Plan: Hodge Laplacian Slugging Repository

## Overview

This document is a step-by-step instruction set for Claude Code to build a
Python research repository for the paper:
**"Hodge Laplacian Topological Signal Processing for Severe Slugging
Identification in Offshore Wells"**.

The repository extends the existing TDA-for-Slugging codebase
(https://github.com/simonecasolo/TDA-for-Slugging) with Hodge Laplacian
methods. Claude Code should execute these instructions in order.

---

## 1. Repository Scaffold

Create the following directory and file structure from the repository root.
Create all `__init__.py` files as empty files unless otherwise specified.

```
hodge-slugging/
├── README.md
├── LICENSE                        # MIT
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── hodge/
│   ├── __init__.py
│   ├── boundary_matrices.py
│   ├── decomposition.py
│   ├── spectrum.py
│   └── pipeline.py
│
├── monitoring/
│   ├── __init__.py
│   └── sliding_window.py
│
├── classification/
│   ├── __init__.py
│   └── feature_sets.py
│
├── utils/
│   ├── __init__.py
│   ├── embedding.py
│   └── epsilon_selection.py
│
├── tests/
│   ├── __init__.py
│   ├── test_boundary_matrices.py
│   ├── test_decomposition.py
│   ├── test_spectrum.py
│   └── test_pipeline.py
│
├── notebooks/
│   ├── 01_synthetic_example.ipynb
│   ├── 02_north_sea_hodge_monitoring.ipynb
│   ├── 03_3w_hodge_classification.ipynb
│   └── 04_spectral_fingerprints.ipynb
│
└── data/
    └── .gitkeep
```

---

## 2. Configuration Files

### 2.1 `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "hodge-slugging"
version = "0.1.0"
description = "Hodge Laplacian topological signal processing for severe slugging identification"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "giotto-tda>=0.6",
    "ripser>=0.6",
    "persim>=0.3",
    "umap-learn>=0.5",
    "matplotlib>=3.7",
    "pandas>=2.0",
    "joblib>=1.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "jupyter>=1.0",
    "black>=23.0",
    "ruff>=0.1",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=hodge --cov=monitoring --cov=classification --cov=utils"
```

### 2.2 `requirements.txt`

```
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
giotto-tda>=0.6
ripser>=0.6
persim>=0.3
umap-learn>=0.5
matplotlib>=3.7
pandas>=2.0
joblib>=1.3
pytest>=7.4
pytest-cov>=4.1
jupyter>=1.0
```

### 2.3 `.gitignore`

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
.env
.venv
env/
venv/
.ipynb_checkpoints/
.coverage
htmlcov/
.pytest_cache/
data/*.csv
data/*.pkl
data/*.parquet
*.npz
!data/.gitkeep
```

---

## 3. Source Files

### 3.1 `hodge/boundary_matrices.py`

Create this file with the following content exactly.

```python
"""
Boundary matrix construction for Vietoris-Rips complexes.

Builds the node-edge (B1) and edge-triangle (B2) boundary matrices
required for Hodge Laplacian computation. No GUDHI dependency:
uses ripser-compatible distance matrices and scipy sparse arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import radius_neighbors_graph
import scipy.sparse as sp
from typing import NamedTuple


class SimplicialComplex(NamedTuple):
    """Container for simplices and their index mappings."""
    nodes: list[tuple]
    edges: list[tuple]
    triangles: list[tuple]
    node_idx: dict
    edge_idx: dict


def get_rips_simplices(
    point_cloud: np.ndarray,
    epsilon: float,
    max_dim: int = 2,
) -> SimplicialComplex:
    """
    Build Vietoris-Rips simplicial complex at a fixed filtration value.

    Uses radius neighbor graph for edges (O(n^2) with spatial indexing),
    then matrix multiplication for triangle detection (faster than naive
    O(n^3) loop for dense complexes).

    Parameters
    ----------
    point_cloud : np.ndarray, shape (n_points, n_dims)
        Takens-embedded pressure signal.
    epsilon : float
        Filtration scale. All edges with length <= epsilon are included.
    max_dim : int
        Maximum simplex dimension to compute (default 2 for Hodge-1).

    Returns
    -------
    SimplicialComplex
        Named tuple with nodes, edges, triangles and index dictionaries.
    """
    n = len(point_cloud)
    nodes = [(i,) for i in range(n)]
    node_idx = {i: i for i in range(n)}

    # --- Edges via radius neighbor graph ---
    A = radius_neighbors_graph(
        point_cloud, epsilon, mode="connectivity", include_self=False
    )
    A = ((A + A.T) > 0).astype(np.float32)
    ri, ci = sp.triu(A, k=1).nonzero()
    edges = list(zip(ri.tolist(), ci.tolist()))
    edge_idx = {e: j for j, e in enumerate(edges)}

    triangles = []
    if max_dim >= 2 and len(edges) > 0:
        # Triangle detection: (i,j,k) is a triangle iff all three
        # pairwise edges exist. Use A^2 to find common neighbours.
        A2 = A @ A
        for idx, (i, k) in enumerate(edges):
            # Common neighbours of i and k that have index j with i < j < k
            common = np.where(
                (A[i].toarray().ravel() > 0)
                & (A[k].toarray().ravel() > 0)
            )[0]
            for j in common:
                if int(i) < int(j) < int(k):
                    triangles.append((int(i), int(j), int(k)))

    return SimplicialComplex(
        nodes=nodes,
        edges=edges,
        triangles=triangles,
        node_idx=node_idx,
        edge_idx=edge_idx,
    )


def build_boundary_matrices(
    sc: SimplicialComplex,
) -> tuple[csr_matrix, csr_matrix]:
    """
    Build sparse boundary matrices B1 and B2.

    B1 (n0 x n1): boundary of edges onto nodes.
      Convention: -1 at tail node, +1 at head node.

    B2 (n1 x n2): boundary of triangles onto edges.
      Convention: standard orientation for sorted triple (i,j,k):
        edge (i,j) -> +1, edge (j,k) -> +1, edge (i,k) -> -1.

    Parameters
    ----------
    sc : SimplicialComplex
        Output of get_rips_simplices.

    Returns
    -------
    B1 : csr_matrix, shape (n0, n1)
    B2 : csr_matrix, shape (n1, n2)
    """
    n0 = len(sc.nodes)
    n1 = len(sc.edges)
    n2 = len(sc.triangles)

    # --- B1 ---
    B1 = lil_matrix((n0, n1))
    for j, (tail, head) in enumerate(sc.edges):
        B1[sc.node_idx[tail], j] = -1.0
        B1[sc.node_idx[head], j] = 1.0
    B1 = B1.tocsr()

    # --- B2 ---
    B2 = lil_matrix((n1, n2))
    for k, (i, j2, l) in enumerate(sc.triangles):
        e_ij = (i, j2)
        e_jl = (j2, l)
        e_il = (i, l)
        if e_ij in sc.edge_idx:
            B2[sc.edge_idx[e_ij], k] = 1.0
        if e_jl in sc.edge_idx:
            B2[sc.edge_idx[e_jl], k] = 1.0
        if e_il in sc.edge_idx:
            B2[sc.edge_idx[e_il], k] = -1.0
    B2 = B2.tocsr()

    return B1, B2


def pressure_to_1cochain(
    pressure_on_nodes: np.ndarray,
    sc: SimplicialComplex,
) -> np.ndarray:
    """
    Build a 1-cochain from scalar pressure values at nodes.

    Assigns the oriented pressure difference p[head] - p[tail]
    to each edge. This is the discrete analog of a differential
    1-form dp on the phase space manifold.

    Parameters
    ----------
    pressure_on_nodes : np.ndarray, shape (n_nodes,)
        Pressure value associated with each embedded point.
    sc : SimplicialComplex

    Returns
    -------
    f1 : np.ndarray, shape (n_edges,)
    """
    f1 = np.zeros(len(sc.edges))
    for j, (tail, head) in enumerate(sc.edges):
        f1[j] = (
            pressure_on_nodes[sc.node_idx[head]]
            - pressure_on_nodes[sc.node_idx[tail]]
        )
    return f1
```

---

### 3.2 `hodge/decomposition.py`

```python
"""
Hodge decomposition of 1-cochains on simplicial complexes.

Decomposes any 1-cochain f1 into three orthogonal components:
  f1 = f_grad + f_curl + f_harm

where:
  f_grad  in im(B1^T)   -- gradient (irrotational) component
  f_curl  in im(B2)     -- curl component
  f_harm  in ker(L1)    -- harmonic component (isomorphic to H1)

The harmonic energy ratio eta_harm = ||f_harm||^2 / ||f1||^2
is the primary new feature for slug flow monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr


@dataclass
class HodgeDecomposition:
    """
    Result of a Hodge decomposition of a 1-cochain.

    Attributes
    ----------
    f_grad : np.ndarray
        Gradient component.
    f_curl : np.ndarray
        Curl component.
    f_harm : np.ndarray
        Harmonic component. Physical interpretation: circulation
        around persistent loops in the phase space embedding.
        For slug flow this captures the limit cycle energy.
    eta_grad : float
        Fraction of signal energy in gradient component.
    eta_curl : float
        Fraction of signal energy in curl component.
    eta_harm : float
        Fraction of signal energy in harmonic component.
        Primary slug flow indicator: near 0 for steady state,
        elevated for severe slugging.
    harm_curl_ratio : float
        eta_harm / eta_curl. Discriminates severe slugging
        (high harmonic, low curl) from flow instabilities
        (moderate both). Addresses the slugging/instability
        boundary in the 3W dataset.
    orthogonality_residual : float
        Max absolute inner product between components.
        Should be < 1e-6 for a valid decomposition.
        Useful as a numerical health check.
    """

    f_grad: np.ndarray
    f_curl: np.ndarray
    f_harm: np.ndarray
    eta_grad: float
    eta_curl: float
    eta_harm: float
    harm_curl_ratio: float
    orthogonality_residual: float


def hodge_decomposition(
    f1: np.ndarray,
    B1: csr_matrix,
    B2: csr_matrix,
    tol: float = 1e-10,
) -> HodgeDecomposition:
    """
    Orthogonal Hodge decomposition of a 1-cochain.

    Solves two sparse least-squares problems:
      (1) L0 s = B1 f1       => f_grad = B1^T s
      (2) B2^T B2 r = B2^T f1 => f_curl = B2 r
      (3) f_harm = f1 - f_grad - f_curl

    Parameters
    ----------
    f1 : np.ndarray, shape (n_edges,)
        1-cochain to decompose (e.g. pressure differences on edges).
    B1 : csr_matrix, shape (n_nodes, n_edges)
        Node-edge boundary matrix.
    B2 : csr_matrix, shape (n_edges, n_triangles)
        Edge-triangle boundary matrix.
    tol : float
        Tolerance for lsqr solver.

    Returns
    -------
    HodgeDecomposition
    """
    # --- Gradient component ---
    L0 = B1 @ B1.T  # graph Laplacian, shape (n_nodes, n_nodes)
    rhs_s = B1 @ f1
    s, *_ = lsqr(L0, rhs_s, atol=tol, btol=tol)
    f_grad = B1.T @ s

    # --- Curl component ---
    BtB = B2.T @ B2  # shape (n_triangles, n_triangles)
    rhs_r = B2.T @ f1
    r, *_ = lsqr(BtB, rhs_r, atol=tol, btol=tol)
    f_curl = B2 @ r

    # --- Harmonic component ---
    f_harm = f1 - f_grad - f_curl

    # --- Energy ratios ---
    E_total = float(np.dot(f1, f1)) + 1e-12
    eta_grad = float(np.dot(f_grad, f_grad)) / E_total
    eta_curl = float(np.dot(f_curl, f_curl)) / E_total
    eta_harm = float(np.dot(f_harm, f_harm)) / E_total

    harm_curl_ratio = eta_harm / (eta_curl + 1e-12)

    # --- Orthogonality check ---
    orth = max(
        abs(np.dot(f_grad, f_curl)),
        abs(np.dot(f_grad, f_harm)),
        abs(np.dot(f_curl, f_harm)),
    ) / (E_total + 1e-12)

    return HodgeDecomposition(
        f_grad=f_grad,
        f_curl=f_curl,
        f_harm=f_harm,
        eta_grad=eta_grad,
        eta_curl=eta_curl,
        eta_harm=eta_harm,
        harm_curl_ratio=harm_curl_ratio,
        orthogonality_residual=float(orth),
    )
```

---

### 3.3 `hodge/spectrum.py`

```python
"""
Spectral analysis of the Hodge-1 Laplacian.

L1 = B1^T B1 + B2 B2^T

The spectrum of L1 encodes:
  - Zero eigenvalues: count = beta_1 (first Betti number)
  - Spectral gap: isolation of dominant harmonic mode
  - Eigenvalue distribution: regime fingerprint

These are complementary to persistence diagram features:
persistent homology tracks topology across filtration scales,
L1 spectrum characterises geometry at a fixed scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


@dataclass
class L1Spectrum:
    """
    Spectral features of the Hodge-1 Laplacian.

    Attributes
    ----------
    eigenvalues : np.ndarray
        Sorted (ascending) eigenvalues of L1.
    beta1_hodge : int
        Number of near-zero eigenvalues. Should match the
        beta_1 from persistent homology at the same epsilon.
    lambda1 : float
        Smallest non-zero eigenvalue.
    lambda2 : float
        Second smallest non-zero eigenvalue.
    spectral_gap : float
        lambda2 / lambda1. Large gap indicates a single dominant
        harmonic mode well-separated from the noise background.
        Physical interpretation: the slug cycle loop is geometrically
        clean and isolated from minor flow fluctuations.
    bulk_mean : float
        Mean of eigenvalues beyond the gap. Characterises the
        background complexity of the simplicial complex.
    """

    eigenvalues: np.ndarray
    beta1_hodge: int
    lambda1: float
    lambda2: float
    spectral_gap: float
    bulk_mean: float
    zero_threshold: float = field(default=1e-6, repr=False)

    def to_feature_dict(self) -> dict:
        """Return scalar features suitable for ML feature vectors."""
        return {
            "beta1_hodge": self.beta1_hodge,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "spectral_gap": self.spectral_gap,
            "bulk_mean": self.bulk_mean,
        }


def compute_l1_spectrum(
    B1: csr_matrix,
    B2: csr_matrix,
    n_eigs: int = 20,
    zero_threshold: float = 1e-6,
) -> L1Spectrum:
    """
    Compute the lower spectrum of the Hodge-1 Laplacian.

    Uses shift-invert ARPACK via scipy.sparse.linalg.eigsh for
    efficient computation of the smallest eigenvalues of the
    large sparse L1 matrix.

    Parameters
    ----------
    B1 : csr_matrix, shape (n_nodes, n_edges)
    B2 : csr_matrix, shape (n_edges, n_triangles)
    n_eigs : int
        Number of eigenvalues to compute (from the bottom of the spectrum).
    zero_threshold : float
        Eigenvalues below this are counted as zero (beta_1).

    Returns
    -------
    L1Spectrum
    """
    L1 = B1.T @ B1 + B2 @ B2.T  # shape (n_edges, n_edges)

    n1 = L1.shape[0]
    k = min(n_eigs, n1 - 2)

    if k < 1:
        return L1Spectrum(
            eigenvalues=np.array([]),
            beta1_hodge=0,
            lambda1=0.0,
            lambda2=0.0,
            spectral_gap=0.0,
            bulk_mean=0.0,
        )

    raw_eigs = eigsh(
        L1,
        k=k,
        which="SM",
        return_eigenvectors=False,
        tol=1e-8,
    )
    eigenvalues = np.sort(np.abs(raw_eigs))

    beta1 = int(np.sum(eigenvalues < zero_threshold))
    nonzero = eigenvalues[eigenvalues >= zero_threshold]

    lambda1 = float(nonzero[0]) if len(nonzero) > 0 else 0.0
    lambda2 = float(nonzero[1]) if len(nonzero) > 1 else 0.0
    spectral_gap = (lambda2 / lambda1) if lambda1 > 0 else 0.0
    bulk_mean = float(np.mean(nonzero[2:])) if len(nonzero) > 2 else 0.0

    return L1Spectrum(
        eigenvalues=eigenvalues,
        beta1_hodge=beta1,
        lambda1=lambda1,
        lambda2=lambda2,
        spectral_gap=spectral_gap,
        bulk_mean=bulk_mean,
        zero_threshold=zero_threshold,
    )
```

---

### 3.4 `hodge/pipeline.py`

```python
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
```

---

### 3.5 `utils/epsilon_selection.py`

```python
"""
Principled selection of a fixed filtration scale epsilon for
Hodge decomposition, derived from a GiottoTDA persistence diagram.

The persistence diagram (Track 1) informs the Hodge analysis
(Track 2) about the topologically meaningful scale.
"""

from __future__ import annotations

import numpy as np


def epsilon_from_diagram(
    diagram: np.ndarray,
    homology_dim: int = 1,
    strategy: str = "most_persistent",
    n_samples: int = 50,
) -> float | None:
    """
    Extract a filtration scale epsilon from a GiottoTDA persistence diagram.

    Parameters
    ----------
    diagram : np.ndarray, shape (n_features, 3)
        GiottoTDA persistence diagram. Column 2 is homology dimension.
    homology_dim : int
        Homology dimension to use (1 for loops, relevant to slugging).
    strategy : str
        "most_persistent" : epsilon = midpoint of the most persistent H1 bar.
                            Principled choice: the scale at which the slug
                            loop is most clearly expressed.
        "max_beta1"       : epsilon = scale maximising the number of H1 loops.
                            More stable but may include noise loops.
    n_samples : int
        Number of epsilon values to sample for "max_beta1" strategy.

    Returns
    -------
    float or None
        Epsilon value, or None if no features of the requested dimension exist.
    """
    h_pts = diagram[diagram[:, 2] == homology_dim]

    if len(h_pts) == 0:
        return None

    births = h_pts[:, 0]
    deaths = h_pts[:, 1]

    if strategy == "most_persistent":
        persistence = deaths - births
        top = h_pts[np.argmax(persistence)]
        return float((top[0] + top[1]) / 2.0)

    elif strategy == "max_beta1":
        epsilons = np.linspace(births.min(), deaths.max(), n_samples)
        beta1s = [
            int(np.sum((births <= e) & (deaths > e))) for e in epsilons
        ]
        return float(epsilons[int(np.argmax(beta1s))])

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Choose from {'most_persistent', 'max_beta1'}."
        )
```

---

### 3.6 `utils/embedding.py`

```python
"""
Takens embedding utilities consistent with the GiottoTDA API.
Thin wrappers that keep the pipeline readable.
"""

from __future__ import annotations

import numpy as np
from gtda.time_series import TakensEmbedding, SlidingWindow


def takens_embed(
    signal: np.ndarray,
    dimension: int,
    time_delay: int,
) -> np.ndarray:
    """
    Takens embedding of a 1-D signal.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
    dimension : int
        Embedding dimension d.
    time_delay : int
        Time delay tau (in samples).

    Returns
    -------
    np.ndarray, shape (n_vectors, dimension)
        Point cloud in reconstruction space R^d.
    """
    te = TakensEmbedding(time_delay=time_delay, dimension=dimension)
    return te.fit_transform(signal.reshape(1, -1))[0]


def sliding_windows(
    signal: np.ndarray,
    window_size: int,
    stride: int,
) -> list[np.ndarray]:
    """
    Split a signal into overlapping windows.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
    window_size : int
    stride : int

    Returns
    -------
    list of np.ndarray, each of shape (window_size,)
    """
    windows = []
    n = len(signal)
    for start in range(0, n - window_size + 1, stride):
        windows.append(signal[start : start + window_size])
    return windows
```

---

### 3.7 `monitoring/sliding_window.py`

```python
"""
Sliding window Hodge monitoring for the North Sea dataset.

Replicates the monitoring analysis of the baseline paper (Figure 7)
and extends it with Hodge-based indicators for comparison.

Usage
-----
from monitoring.sliding_window import run_hodge_monitor

results_df = run_hodge_monitor(
    pressure_signal=bhp,
    times=t,
    window_size=8000,
    stride=200,
    embed_dim=8,
    tau=42,
)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils.embedding import takens_embed, sliding_windows
from hodge.pipeline import run_window


def _process_one_window(
    window: np.ndarray,
    t_center: float,
    embed_dim: int,
    tau: int,
    n_eigs: int,
    epsilon_strategy: str,
) -> dict:
    embedding = takens_embed(window, embed_dim, tau)
    features = run_window(
        embedding=embedding,
        pressure_window=window,
        n_eigs=n_eigs,
        epsilon_strategy=epsilon_strategy,
    )
    features["time"] = t_center
    return features


def run_hodge_monitor(
    pressure_signal: np.ndarray,
    times: np.ndarray,
    window_size: int,
    stride: int,
    embed_dim: int,
    tau: int,
    n_eigs: int = 20,
    epsilon_strategy: str = "most_persistent",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Run sliding window Hodge + persistence monitoring.

    Parameters
    ----------
    pressure_signal : np.ndarray, shape (n_samples,)
        Bottom hole or wellhead pressure time series.
    times : np.ndarray, shape (n_samples,)
        Timestamps corresponding to pressure_signal.
    window_size : int
        Number of samples per window.
    stride : int
        Step between successive windows (samples).
    embed_dim : int
        Takens embedding dimension d.
    tau : int
        Takens time delay (samples).
    n_eigs : int
        Number of L1 eigenvalues to compute per window.
    epsilon_strategy : str
        Filtration scale selection strategy. See epsilon_from_diagram.
    n_jobs : int
        Number of parallel jobs (joblib). -1 = all cores.

    Returns
    -------
    pd.DataFrame
        One row per window. Columns include time, all persistence
        features, and all Hodge features.
    """
    windows = sliding_windows(pressure_signal, window_size, stride)
    n_windows = len(windows)

    # Centre time of each window
    half = window_size // 2
    t_centres = [
        float(times[min(i * stride + half, len(times) - 1)])
        for i in range(n_windows)
    ]

    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_process_one_window)(
            window=w,
            t_center=tc,
            embed_dim=embed_dim,
            tau=tau,
            n_eigs=n_eigs,
            epsilon_strategy=epsilon_strategy,
        )
        for w, tc in zip(windows, t_centres)
    )

    return pd.DataFrame(results).sort_values("time").reset_index(drop=True)
```

---

### 3.8 `classification/feature_sets.py`

```python
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
    "entropy_h0",
    "entropy_h1",
    "p_inf_h0",
    "p_inf_h1",
    "mean_betti0",
    "mean_betti1",
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
```

---

## 4. Test Files

### 4.1 `tests/test_boundary_matrices.py`

```python
"""
Tests for boundary matrix construction.

Uses a hand-crafted minimal complex (4 nodes, known edges and
one triangle) where the correct B1 and B2 are computable by hand.
"""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from hodge.boundary_matrices import (
    get_rips_simplices,
    build_boundary_matrices,
    pressure_to_1cochain,
)


def make_square_with_diagonal():
    """
    Four points: (0,0), (1,0), (1,1), (0,1).
    At epsilon=1.1 we get all four edges of the square
    plus the diagonal (0,0)-(1,1) but NOT (1,0)-(0,1).
    Triangle (0,1,2) should appear (edges 0-1, 1-2, 0-2 all <= 1.1).
    """
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    return pts


def test_simplex_counts():
    pts = make_square_with_diagonal()
    sc = get_rips_simplices(pts, epsilon=1.1)
    assert len(sc.nodes) == 4
    # edges: (0,1), (0,2), (1,2) within epsilon=1.1
    # (0,3) length=1, (2,3) length=1 also within
    assert len(sc.edges) >= 3
    # at least one triangle
    assert len(sc.triangles) >= 1


def test_boundary_operator_composition():
    """
    Fundamental property: B2^T B1^T = 0  (boundary of boundary = 0).
    """
    pts = make_square_with_diagonal()
    sc = get_rips_simplices(pts, epsilon=1.5)
    if len(sc.edges) == 0 or len(sc.triangles) == 0:
        pytest.skip("Complex too sparse for this test.")
    B1, B2 = build_boundary_matrices(sc)
    product = (B1 @ B2).toarray()
    np.testing.assert_allclose(product, 0.0, atol=1e-12)


def test_1cochain_shape():
    pts = make_square_with_diagonal()
    sc = get_rips_simplices(pts, epsilon=1.5)
    B1, B2 = build_boundary_matrices(sc)
    pressure = np.random.rand(len(sc.nodes))
    f1 = pressure_to_1cochain(pressure, sc)
    assert f1.shape == (len(sc.edges),)
```

---

### 4.2 `tests/test_decomposition.py`

```python
"""
Tests for Hodge decomposition.

Key invariants:
  1. The three components are mutually orthogonal.
  2. They sum to the original cochain.
  3. Energy ratios sum to 1 (up to floating point).
  4. For a pure gradient signal (f1 = B1^T s), eta_harm ~ 0.
  5. For a signal supported only on harmonic modes, eta_harm ~ 1.
"""

import numpy as np
import pytest

from hodge.boundary_matrices import (
    get_rips_simplices,
    build_boundary_matrices,
    pressure_to_1cochain,
)
from hodge.decomposition import hodge_decomposition


@pytest.fixture
def simple_complex():
    np.random.seed(42)
    # Circle of points - guarantees a loop
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    sc = get_rips_simplices(pts, epsilon=0.6)
    B1, B2 = build_boundary_matrices(sc)
    return sc, B1, B2


def test_decomposition_sums_to_original(simple_complex):
    sc, B1, B2 = simple_complex
    if len(sc.edges) == 0:
        pytest.skip("Empty complex.")
    f1 = np.random.randn(len(sc.edges))
    d = hodge_decomposition(f1, B1, B2)
    reconstructed = d.f_grad + d.f_curl + d.f_harm
    np.testing.assert_allclose(reconstructed, f1, atol=1e-6)


def test_energy_ratios_sum_to_one(simple_complex):
    sc, B1, B2 = simple_complex
    if len(sc.edges) == 0:
        pytest.skip("Empty complex.")
    f1 = np.random.randn(len(sc.edges))
    d = hodge_decomposition(f1, B1, B2)
    total = d.eta_grad + d.eta_curl + d.eta_harm
    assert abs(total - 1.0) < 1e-4


def test_orthogonality(simple_complex):
    sc, B1, B2 = simple_complex
    if len(sc.edges) == 0:
        pytest.skip("Empty complex.")
    f1 = np.random.randn(len(sc.edges))
    d = hodge_decomposition(f1, B1, B2)
    assert d.orthogonality_residual < 1e-4


def test_pure_gradient_has_zero_harmonic(simple_complex):
    sc, B1, B2 = simple_complex
    if len(sc.edges) == 0:
        pytest.skip("Empty complex.")
    # Build a pure gradient signal
    s = np.random.randn(len(sc.nodes))
    f1_grad = B1.T @ s
    d = hodge_decomposition(f1_grad, B1, B2)
    assert d.eta_harm < 1e-4
```

---

### 4.3 `tests/test_spectrum.py`

```python
"""
Tests for L1 spectral computation.

Key invariants:
  1. beta1_hodge from L1 nullity matches beta_1 from persistent homology.
  2. Eigenvalues are non-negative.
  3. Spectral gap >= 1 when lambda2 >= lambda1 > 0.
"""

import numpy as np
import pytest

from hodge.boundary_matrices import get_rips_simplices, build_boundary_matrices
from hodge.spectrum import compute_l1_spectrum


@pytest.fixture
def loop_complex():
    """Six points on a circle: known beta_1 = 1."""
    theta = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    # epsilon just large enough to connect adjacent points
    sc = get_rips_simplices(pts, epsilon=1.1)
    B1, B2 = build_boundary_matrices(sc)
    return sc, B1, B2


def test_eigenvalues_nonnegative(loop_complex):
    _, B1, B2 = loop_complex
    spec = compute_l1_spectrum(B1, B2, n_eigs=6)
    assert np.all(spec.eigenvalues >= -1e-10)


def test_spectral_gap_ge_one(loop_complex):
    _, B1, B2 = loop_complex
    spec = compute_l1_spectrum(B1, B2, n_eigs=6)
    if spec.lambda1 > 0 and spec.lambda2 > 0:
        assert spec.spectral_gap >= 1.0 - 1e-6


def test_to_feature_dict_keys(loop_complex):
    _, B1, B2 = loop_complex
    spec = compute_l1_spectrum(B1, B2, n_eigs=6)
    d = spec.to_feature_dict()
    expected_keys = {
        "beta1_hodge", "lambda1", "lambda2", "spectral_gap", "bulk_mean"
    }
    assert expected_keys == set(d.keys())
```

---

### 4.4 `tests/test_pipeline.py`

```python
"""
Integration test for the full Hodge + GiottoTDA pipeline.
Uses a synthetic periodic signal (known to produce H1 loops).
"""

import numpy as np
import pytest

from hodge.pipeline import run_window
from utils.embedding import takens_embed


def synthetic_slug_signal(n=500, period=50, noise=0.05):
    """Periodic signal mimicking severe slugging pressure oscillations."""
    t = np.arange(n)
    return np.sin(2 * np.pi * t / period) + noise * np.random.randn(n)


def test_pipeline_returns_expected_keys():
    np.random.seed(0)
    signal = synthetic_slug_signal()
    embedding = takens_embed(signal, dimension=6, time_delay=5)

    result = run_window(
        embedding=embedding,
        pressure_window=signal,
        n_eigs=6,
    )

    expected_keys = {
        "eta_harm", "eta_grad", "eta_curl",
        "harm_curl_ratio", "spectral_gap",
        "entropy_h0", "entropy_h1",
        "p_inf_h0", "p_inf_h1",
    }
    assert expected_keys.issubset(set(result.keys()))


def test_periodic_signal_has_elevated_eta_harm():
    """
    A clean periodic signal should produce a dominant H1 loop
    and elevated harmonic energy fraction.
    """
    np.random.seed(1)
    signal = synthetic_slug_signal(noise=0.01)
    embedding = takens_embed(signal, dimension=8, time_delay=6)

    result = run_window(
        embedding=embedding,
        pressure_window=signal,
        n_eigs=6,
    )

    # Not a hard threshold - just verify the direction
    assert result["eta_harm"] > result["eta_grad"] or result["p_inf_h1"] > 0
```

---

## 5. README.md

```markdown
# hodge-slugging

Hodge Laplacian topological signal processing for severe slugging
identification in offshore oil and gas wells.

This repository accompanies the paper:

> **"Hodge Laplacian Topological Signal Processing for Severe Slugging
> Identification"**
> Simone Casolo, *Chemical Engineering Science*, 2024 (under review)

It extends the TDA baseline from:
> [TDA-for-Slugging](https://github.com/simonecasolo/TDA-for-Slugging)

## Method

Pressure time series from subsea wells are Takens-embedded to produce
point clouds in phase space. A Vietoris-Rips simplicial complex is built
at a filtration scale derived from the persistence diagram. The pressure
signal is then decomposed via the Hodge Laplacian into:

- **Gradient component** — conservative pressure variation (noise, drift)
- **Curl component** — local rotational structure (minor flow fluctuations)
- **Harmonic component** — global circulation around persistent loops
  (physical signature of the slug limit cycle)

The harmonic energy ratio `eta_harm` provides earlier and more
interpretable detection of severe slugging onset compared to
persistence-based indicators alone.

## Installation

```bash
git clone https://github.com/simonecasolo/hodge-slugging
cd hodge-slugging
pip install -e ".[dev]"
```

## Usage

```python
from monitoring.sliding_window import run_hodge_monitor
import numpy as np

# Load your pressure signal
bhp = np.load("data/north_sea_bhp.npy")
times = np.load("data/north_sea_times.npy")

results = run_hodge_monitor(
    pressure_signal=bhp,
    times=times,
    window_size=8000,
    stride=200,
    embed_dim=8,
    tau=42,
)

print(results[["time", "eta_harm", "spectral_gap", "p_inf_h1"]].head())
```

## Running tests

```bash
pytest --cov
```

## Notebooks

| Notebook | Content |
|---|---|
| `01_synthetic_example.ipynb` | Toy Hodge decomposition on a synthetic loop signal |
| `02_north_sea_hodge_monitoring.ipynb` | Reproduces monitoring figures (Section 4) |
| `03_3w_hodge_classification.ipynb` | 3W dataset classification comparison (Section 5) |
| `04_spectral_fingerprints.ipynb` | L1 spectral analysis and regime clustering (Section 6) |

## Data

The North Sea dataset is proprietary and anonymised. The 3W dataset is
publicly available at:
https://github.com/petrobras/3W

Place 3W data under `data/3w/` before running notebooks 3 and 4.

## Citation

If you use this code, please cite both this repository and the baseline:

```bibtex
@article{casolo2024hodge,
  title   = {Hodge Laplacian Topological Signal Processing for
             Severe Slugging Identification},
  author  = {Casolo, Simone},
  journal = {Chemical Engineering Science},
  year    = {2024},
  note    = {under review}
}

@article{casolo2023tda,
  title   = {Severe Slugging Flow Identification from Topological Indicators},
  author  = {Casolo, Simone},
  year    = {2023}
}
```
```

---

## 6. Claude Code Execution Instructions

Claude Code should execute the following steps in order:

1. **Create the directory tree** exactly as shown in Section 1.
   Use `mkdir -p` for nested directories.

2. **Write each source file** with the content specified in Sections 3-5.
   Do not modify the content of source files.

3. **Run `pip install -e ".[dev]"` or `pip install -r requirements.txt`**
   to install dependencies.

4. **Run the test suite** with `pytest tests/ -v` and report which tests
   pass. All tests in `test_boundary_matrices.py`, `test_decomposition.py`,
   and `test_spectrum.py` should pass on a clean install.
   `test_pipeline.py` requires GiottoTDA and may be skipped if not installed.

5. **Do not create the notebooks** — these are left empty (create the
   `.ipynb` files as minimal valid JSON notebooks with no cells) since
   they depend on proprietary data.

6. **Report any import errors** encountered when running the tests,
   particularly for `giotto-tda`. If GiottoTDA is not available,
   note which tests were skipped and why.
