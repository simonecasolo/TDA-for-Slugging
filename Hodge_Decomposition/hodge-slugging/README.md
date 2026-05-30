# hodge-slugging

Hodge Laplacian topological signal processing for severe slugging
identification in offshore oil and gas wells.

This repository accompanies the paper:

> **"Hodge Laplacian Topological Signal Processing for Severe Slugging
> Identification"**
> Simone Casolo, *Chemical Engineering Science*, 2024 (under review)

It extends the TDA baseline from:
> [TDA-for-Slugging](https://github.com/simonecasolo/TDA-for-Slugging)

## Method

Pressure time series from subsea wells are Takens-embedded to produce
point clouds in phase space. A Vietoris-Rips simplicial complex is built
at a filtration scale derived from the persistence diagram. The pressure
signal is then decomposed via the Hodge Laplacian into:

- **Gradient component** — conservative pressure variation (noise, drift)
- **Curl component** — local rotational structure (minor flow fluctuations)
- **Harmonic component** — global circulation around persistent loops
  (physical signature of the slug limit cycle)

The harmonic energy ratio `eta_harm` provides earlier and more
interpretable detection of severe slugging onset compared to
persistence-based indicators alone.

## Installation

```bash
git clone https://github.com/simonecasolo/hodge-slugging
cd hodge-slugging
pip install -e ".[dev]"
```

## Usage

```python
from monitoring.sliding_window import run_hodge_monitor
import numpy as np

# Load your pressure signal
bhp = np.load("data/north_sea_bhp.npy")
times = np.load("data/north_sea_times.npy")

results = run_hodge_monitor(
    pressure_signal=bhp,
    times=times,
    window_size=8000,
    stride=200,
    embed_dim=8,
    tau=42,
)

print(results[["time", "eta_harm", "spectral_gap", "p_inf_h1"]].head())
```

## Running tests

```bash
pytest --cov
```

## Notebooks

| Notebook | Content |
|---|---|
| `01_synthetic_example.ipynb` | Toy Hodge decomposition on a synthetic loop signal |
| `02_north_sea_hodge_monitoring.ipynb` | Reproduces monitoring figures (Section 4) |
| `03_3w_hodge_classification.ipynb` | 3W dataset classification comparison (Section 5) |
| `04_spectral_fingerprints.ipynb` | L1 spectral analysis and regime clustering (Section 6) |

## Data

The North Sea dataset is proprietary and anonymised. The 3W dataset is
publicly available at:
https://github.com/petrobras/3W

Place 3W data under `data/3w/` before running notebooks 3 and 4.

## Citation

If you use this code, please cite both this repository and the baseline:

```bibtex
@article{casolo2024hodge,
  title   = {Hodge Laplacian Topological Signal Processing for
             Severe Slugging Identification},
  author  = {Casolo, Simone},
  journal = {Chemical Engineering Science},
  year    = {2024},
  note    = {under review}
}

@article{casolo2023tda,
  title   = {Severe Slugging Flow Identification from Topological Indicators},
  author  = {Casolo, Simone},
  year    = {2023}
}
```
