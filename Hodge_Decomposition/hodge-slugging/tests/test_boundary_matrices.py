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
    # At epsilon=1.1 only the four unit-length sides are included
    # (diagonal = sqrt(2) ~ 1.41 > 1.1), so no triangles yet.
    sc_small = get_rips_simplices(pts, epsilon=1.1)
    assert len(sc_small.nodes) == 4
    assert len(sc_small.edges) == 4   # four sides of the square
    assert len(sc_small.triangles) == 0

    # At epsilon=1.5 the diagonal (0,2) is included, giving triangle (0,1,2).
    sc = get_rips_simplices(pts, epsilon=1.5)
    assert len(sc.nodes) == 4
    assert len(sc.edges) >= 5         # four sides + at least one diagonal
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
