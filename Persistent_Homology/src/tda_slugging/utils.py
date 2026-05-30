"""Shared utilities for TDA-based slugging detection."""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from gtda.time_series import (
    Resampler,
    SingleTakensEmbedding,
    takens_embedding_optimal_parameters,
)
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy


def read_files(path, label, limit):
    """Load CSV files from *path*, resample each to *limit* points, return (array, DataFrame).

    Parameters
    ----------
    path : str or Path
        Directory containing CSV files.
    label : str
        Column name of the time-series variable to extract.
    limit : int
        Number of points to resample each time series to.
    """
    data = []
    for file in sorted(os.listdir(path)):
        if not file.endswith(".csv"):
            continue
        data_df = pd.read_csv(os.path.join(path, file)).interpolate()
        period = max(1, int(len(data_df) / limit))
        resampler = Resampler(period=period)
        _, signal = resampler.fit_transform_resample(data_df.index, data_df[label])
        data.append(signal)

    data_T = list(map(list, zip(*data)))
    return np.array(data_T).T, pd.DataFrame.from_records(data_T)


def find_shortest_file(path, label):
    """Return the length of the shortest time series (by *label* column) in *path*."""
    lengths = []
    for file in os.listdir(path):
        if not file.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(path, file))
        lengths.append(len(df[label]))
    return min(lengths)


def batch_analyzer(input_df, stride, max_embedding_dimension, max_time_delay):
    """Run Takens embedding + persistent homology on every column of *input_df*.

    Parameters
    ----------
    input_df : pd.DataFrame
        Each column is one time series.
    stride : int
        Stride for the Takens embedder.
    max_embedding_dimension : int
        Upper bound for the FNN embedding-dimension search.
    max_time_delay : int
        Upper bound for the mutual-information time-delay search.

    Returns
    -------
    point_clouds_pca : list of ndarray
        PCA-projected point clouds (3 components).
    diagrams : list of ndarray
        Persistence diagrams (shape: 1 x n_features x 3).
    entropies : list of ndarray
        Persistence entropy per diagram.
    norm_entropies : list of ndarray
        Normalised persistence entropy per diagram.
    """
    max_time_delay = int(max_time_delay)
    max_embedding_dimension = int(max_embedding_dimension)
    homology_dimensions = (0, 1)

    VRP = VietorisRipsPersistence(homology_dimensions=homology_dimensions)
    pca = PCA(n_components=3)
    PE_signal = PersistenceEntropy()
    PE_norm = PersistenceEntropy(normalize=True)

    point_clouds_pca, diagrams, entropies, norm_entropies = [], [], [], []

    for i, col in enumerate(input_df.columns, start=1):
        opt_delay, opt_dim = takens_embedding_optimal_parameters(
            input_df[col], max_time_delay, max_embedding_dimension, stride=stride
        )
        opt_dim = max(opt_dim, 3)

        print(
            f"Analyzing {i}/{len(input_df.columns)} ({100*i//len(input_df.columns)}%)"
            f"  dim={opt_dim}  delay={opt_delay}"
        )

        embedder = SingleTakensEmbedding(
            parameters_type="fixed",
            n_jobs=6,
            time_delay=opt_delay,
            dimension=opt_dim,
            stride=stride,
        )
        embedded = embedder.fit_transform(input_df[col])
        embedded_pca = pca.fit_transform(embedded)

        diagram = VRP.fit_transform(embedded.reshape(1, *embedded.shape))

        point_clouds_pca.append(embedded_pca if opt_dim > 3 else embedded)
        diagrams.append(diagram)
        entropies.append(PE_signal.fit_transform(diagram))
        norm_entropies.append(PE_norm.fit_transform(diagram))

    return point_clouds_pca, diagrams, entropies, norm_entropies


def mutual_information(X, time_delay, n_bins):
    """Mutual information between X and its time-delayed version.

    Used to find the optimal time delay for Takens embedding (first minimum).
    """
    contingency = np.histogram2d(X[:-time_delay], X[time_delay:], bins=n_bins)[0]
    return mutual_info_score(None, None, contingency=contingency)
