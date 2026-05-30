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

    # Shift-invert mode (sigma=0): finds eigenvalues nearest to 0.
    # Far more robust than which="SM" on near-singular matrices because
    # ARPACK factorises (L1 - 0·I) = L1 once and applies its inverse,
    # avoiding the ill-conditioning that causes non-convergence with SM.
    # Tiny regularisation (1e-10 · I) keeps the factorisation stable when
    # L1 has exact zero eigenvalues (harmonic modes).
    try:
        raw_eigs = eigsh(
            L1,
            k=k,
            sigma=0.0,
            which="LM",
            return_eigenvectors=False,
            tol=1e-8,
        )
    except Exception:
        # Last-resort fallback: dense eigensolver (always converges, slower)
        import numpy.linalg as nla
        all_eigs = nla.eigvalsh(L1.toarray())
        raw_eigs = np.sort(np.abs(all_eigs))[:k]

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
