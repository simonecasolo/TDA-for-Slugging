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

    # Ratio is meaningful only when curl is non-negligible.
    # Cap at 1000 to avoid inf/nan when η_curl ≈ 0 (clean harmonic mode).
    harm_curl_ratio = min(eta_harm / (eta_curl + 1e-12), 1000.0)

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
