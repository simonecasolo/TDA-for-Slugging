"""
Takens embedding utilities — thin wrappers around giotto-tda.

Prefer giotto-tda's TakensEmbedding and SlidingWindow directly in notebooks.
These helpers are retained only for the monitoring module's internal use.
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
    Takens embedding of a 1-D signal via giotto-tda.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
    dimension : int
    time_delay : int

    Returns
    -------
    np.ndarray, shape (n_vectors, dimension)
    """
    te = TakensEmbedding(time_delay=time_delay, dimension=dimension)
    return te.fit_transform(signal.reshape(1, -1))[0]


def sliding_windows(
    signal: np.ndarray,
    window_size: int,
    stride: int,
) -> list[np.ndarray]:
    """
    Split a signal into overlapping windows via giotto-tda's SlidingWindow.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
    window_size : int
    stride : int

    Returns
    -------
    list of np.ndarray, each of shape (window_size,)
    """
    sw = SlidingWindow(size=window_size, stride=stride)
    # SlidingWindow expects shape (n_samples, n_features) or (1, n_samples)
    windows_2d = sw.fit_transform(signal.reshape(1, -1))[0]
    # windows_2d shape: (n_windows, window_size)
    return [windows_2d[i] for i in range(len(windows_2d))]
