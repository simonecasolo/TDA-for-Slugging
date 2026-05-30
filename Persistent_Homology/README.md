# TDA-for-Slugging

Topological data analysis (TDA) for severe slugging identification in offshore petroleum production systems.
Companion code for the paper *"Severe Slugging Flow Identification from Topological Indicators"* (Casolo, 2023).

## Method summary

Pressure time series from subsea sensors are embedded via Takens delay-coordinate embedding, then analysed with persistent homology (Vietoris–Rips filtration). The resulting persistence diagrams are summarised by topological indicators (persistence entropy, maximum persistence, Betti curves) and fed to standard machine-learning classifiers to distinguish normal flow, minor instabilities, and severe slugging.

## Repository layout

```
├── notebooks/              Jupyter notebooks (numbered in suggested reading order)
│   ├── 01_simple_shapes.ipynb       TDA on synthetic signals (pedagogical)
│   ├── 02_slug_autopsy.ipynb        Deep-dive: white noise vs. slug flow topology
│   ├── 03_slugging_analysis.ipynb   3W severe-slugging dataset analysis
│   ├── 04_normal_op_analysis.ipynb  3W normal-operation baseline
│   ├── 05_instabilities.ipynb       3W minor-instabilities analysis
│   ├── 06_machine_learning.ipynb    ML classification of flow regimes
│   └── 07_imperial_data.ipynb       Experimental slug data (Imperial College)
│
├── src/tda_slugging/
│   └── utils.py            Shared utilities (read_files, batch_analyzer, …)
│
├── data/
│   ├── 3W/                 Severe-slugging events from the public 3W dataset
│   │   ├── ALL/            All events (real + simulated)
│   │   ├── REAL/           Real events only
│   │   └── SIMULATED/      OLGA-simulated events only
│   └── well/               Anonymised North-Sea well data (proprietary)
│
├── images/                 Figures used in the paper
├── outputs/                Computed persistence data files (gitignored)
├── pyproject.toml
└── poetry.lock
```

## Data

The **type-3 (severe slugging)** subset of the [3W dataset](https://github.com/ricardovvargas/3w_dataset) is committed under `data/3W/`.
Notebooks that use **normal operation (type 0)** or **minor instabilities (type 4)** data require the full 3W dataset — download it and set `DATA_3W_NORMAL` / `DATA_3W_UNSTABLE` in the notebook preamble cells.

## Getting started

```bash
# install dependencies
poetry install

# launch notebooks
poetry run jupyter lab notebooks/
```

Each notebook starts with a preamble cell that sets `REPO_ROOT` and all data-path variables — adjust only that cell if your dataset location differs.

## Dependencies

- [giotto-tda](https://giotto-ai.github.io/gtda-docs/) — Takens embedding and persistent homology
- [scikit-tda](https://scikit-tda.org/) — additional TDA utilities
- scikit-learn — machine-learning classifiers
- pandas, numpy, matplotlib, plotly, seaborn

## Citation

```bibtex
@article{casolo2023slugging,
  title   = {Severe Slugging Flow Identification from Topological Indicators},
  author  = {Casolo, Simone},
  year    = {2023}
}
```
