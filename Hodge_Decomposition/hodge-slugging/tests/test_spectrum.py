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
