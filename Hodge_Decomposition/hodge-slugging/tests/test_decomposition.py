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
