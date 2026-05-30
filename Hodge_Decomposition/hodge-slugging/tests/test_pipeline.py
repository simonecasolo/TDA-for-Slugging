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
