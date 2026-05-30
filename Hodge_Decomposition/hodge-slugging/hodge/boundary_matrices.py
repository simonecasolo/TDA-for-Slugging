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

    Assigns p[head] - p[tail] to each edge. This equals B₁ᵀ p, so
    it is an *exact* (pure-gradient) form: η_harm = η_curl = 0 always.
    Useful for visualising the gradient component but not for detecting
    topology. Use angle_to_1cochain for harmonic content.
    """
    f1 = np.zeros(len(sc.edges))
    for j, (tail, head) in enumerate(sc.edges):
        f1[j] = (
            pressure_on_nodes[sc.node_idx[head]]
            - pressure_on_nodes[sc.node_idx[tail]]
        )
    return f1


def angle_to_1cochain(
    pca_cloud: np.ndarray,
    sc: SimplicialComplex,
    dims: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """
    Winding-number 1-cochain: angular displacement along each edge.

    For a periodic signal whose Takens embedding traces a loop in phase
    space, this cochain assigns to each edge (u, v) the signed angular
    displacement θ(v) − θ(u) (wrapped to (−π, π]), where θ is the
    angle in the plane spanned by the two PCA components in *dims*.

    Unlike pressure_to_1cochain, this is NOT exact: the sum around
    any cycle homologous to the slug loop is ±2π (the winding number).
    After Hodge decomposition, the harmonic component captures this
    global circulation — the topological signature of the slug cycle.

    Parameters
    ----------
    pca_cloud : np.ndarray, shape (n_nodes, ≥2)
        Low-dimensional projection of the point cloud (e.g. PCA 3D).
    sc : SimplicialComplex
    dims : tuple of two ints
        Which PCA axes define the loop plane (default: (0, 1)).

    Returns
    -------
    f1 : np.ndarray, shape (n_edges,)
    """
    theta = np.arctan2(pca_cloud[:, dims[1]], pca_cloud[:, dims[0]])
    f1 = np.zeros(len(sc.edges))
    for j, (tail, head) in enumerate(sc.edges):
        dt = theta[sc.node_idx[head]] - theta[sc.node_idx[tail]]
        # Wrap to (−π, π]: short-arc angular step
        f1[j] = (dt + np.pi) % (2 * np.pi) - np.pi
    return f1
