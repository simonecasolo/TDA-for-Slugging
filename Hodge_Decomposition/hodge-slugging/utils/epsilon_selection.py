"""
Heuristic selection of a fixed filtration scale ε for Hodge decomposition,
derived from a persistence diagram.

**Neither strategy implemented here is a published standard.**

The philosophy of persistent homology is to deliberately avoid choosing a
single filtration scale — persistence diagrams encode topology *across all
scales* simultaneously (Carlsson, 2005; Edelsbrunner & Harer, 2010).
Choosing a fixed ε throws away that multi-scale information.

However, Hodge decomposition requires a single simplicial complex, so a
fixed scale must be chosen. The two strategies below are engineering
heuristics tailored to the slugging-detection use case.

The closest motivating reference is:
    Perea & Harer (2015). "Sliding windows and persistence: an application
    of topological methods to signal analysis." *Foundations of Computational
    Mathematics*, 15(3), 799–838.
They extract the *maximum H₁ persistence value* as a scalar periodicity
indicator, but do not select a fixed filtration scale for downstream analysis.
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
    Extract a heuristic filtration scale ε from a persistence diagram.

    This function implements two ad-hoc strategies for bridging persistent
    homology (Track 1) with Hodge decomposition (Track 2).  Neither is a
    published standard; both are practical choices for applications where a
    single simplicial complex is required.

    Parameters
    ----------
    diagram : np.ndarray, shape (n_features, 3)
        Persistence diagram. Column 0 = birth, 1 = death, 2 = homology dim.
    homology_dim : int
        Homology dimension to query (1 for loops).
    strategy : str
        "most_persistent":
            ε = (birth + death) / 2 of the most persistent H₁ bar.
            Rationale: the midpoint is a symmetric, convenient choice that
            sits in the interior of the bar's lifetime. It is *not* derived
            from any theoretical optimality criterion; it is simply the
            arithmetic mean of the two endpoints. Sensitive to a single
            dominant bar — works well when there is one clear loop (slug
            cycle) but may be misleading if several bars have similar
            persistence.

        "max_beta1":
            ε = the scale at which the number of active H₁ features is
            maximised. Slightly more robust because it aggregates information
            across all bars rather than relying on one. Still ad-hoc; no
            published justification.

        "second_persistent_half":
            ε = half of the second highest H₁ persistence value.
            Rationale: the most persistent bar often captures the dominant
            slug cycle; the second bar reflects secondary structure. Using
            half its lifetime as ε keeps the complex in a regime where that
            second topological feature is just becoming active, providing a
            more conservative scale than the midpoint of the top bar.
            Falls back to half the top persistence when fewer than two H₁
            bars are present.

    n_samples : int
        Number of ε values sampled when strategy == "max_beta1".

    Returns
    -------
    float or None
        Chosen ε, or None if no H₁ features exist in the diagram.

    Notes
    -----
    Sensitivity check: if your downstream results change substantially when
    switching between the two strategies, the Hodge features are not robust
    to scale selection and should be interpreted with caution.
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

    elif strategy == "second_persistent_half":
        persistence = deaths - births
        sorted_pers = np.sort(persistence)[::-1]
        second = sorted_pers[1] if len(sorted_pers) >= 2 else sorted_pers[0]
        return float(second / 2.0)

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Choose from {{'most_persistent', 'max_beta1', 'second_persistent_half'}}."
        )
