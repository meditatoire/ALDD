# ALDD: Adaptive Local Domain Decomposition for Operator Regression

PyTorch implementation of the paper:

> **Energy-based feature extraction with adaptive local domain decomposition for prediction of transient and turbulence flow with operator regression models**  
> W. Xu, M. Karthikeyakannan, C. McComb, N. Grande Gutiérrez  
> *Computers & Fluids, 2026*  
> [DOI: 10.1016/j.compfluid.2025.106958](https://doi.org/10.1016/j.compfluid.2025.106958)

---

## What is it about?

This code trains **operator regression models** (FNO and DeepONet) to predict fluid dynamics on large 2D fields by decomposing the domain into smaller subdomains with similar regimes (Laminar / Turbulent / Transitional). Following the idea of **Adaptive Local Domain Decomposition (ALDD)**:

1. **Decompose** the full field into overlapping subdomains.
2. **Characterize** each subdomain by its energy spectrum (or PCA).
3. **Cluster** subdomains with Wasserstein k-means so that regions with similar physics share the same model.
4. **Train** a separate specialist model per cluster.
5. **Predict** autoregressively by assigning the right model to each subdomain at inference time.

Boundary information is encoded via a **BENO** (Boundary-Embedded Neural Operator) transformer to keep transitions smooth across subdomain interfaces.

---

## Project Structure

```
ALDD/
├── src/
│   ├── training.py          # Main training script
│   ├── validation.py        # Autoregressive validation + plotting
│   ├── models.py            # FNO_BENO & DeepONet_BENO architectures
│   ├── cluster.py           # Energy spectrum reduction & Wasserstein k-means
│   ├── data_loader.py       # Domain decomposition & JHTDB loading
│   ├── pipeline.py          # Checkpointing, grid setup, reconstruction
│   ├── cluster_eval.py      # Clustering quality metrics
│   └── training_jhtdb_aldd.ipynb  # Notebook variant
├── data/                    # Local datasets
├── JHTDB/                   # Johns Hopkins Turbulence Database files
├── models/                  # Saved checkpoints (created automatically)
├── pyproject.toml           # uv / pip dependencies
└── README.md
```

---

## Setup

This project uses [**uv**](https://docs.astral.sh/uv/) for environment management.

```bash
git clone https://github.com/meditatoire/ALDD.git
cd ALDD
uv run ./src/training.py
```

If you prefer using `pip`:

```bash
pip install -e .
python ./src/training.py
```

**Dependencies:** Python ≥3.12, PyTorch, NumPy, scikit-learn, h5py (for JHTDB data), matplotlib, seaborn.

---

## Quick Start

### 1. Training

Edit `src/training.py` to choose your dataset, model, and hyperparameters:

| Variable | Description | Example |
|---|---|---|
| `MODEL_TYPE` | `'fno'` or `'deeponet'` | `'fno'` |
| `DATASET` | `'jhtdb'` or `'cylinder'` | `'jhtdb'` |
| `JHTDB_PATH` | Path to HDF5 file | `./JHTDB/data/...` |
| `TRAIN_Z` | List of z-planes to train on | `[32, 64, 96, ...]` |
| `N` | Subdomain size (NxN) | `16` |
| `K_CLUSTER` | Number of physics clusters | `3` |
| `TOP_P` | Spectral bins kept for clustering | `25` |
| `EPOCHS` | Training epochs per cluster | `200` |

Then run:

```bash
uv run ./src/training.py
```

Checkpoints are saved to `./models/` with names like:
```
models/fno_e200_k3_n16_top25.pt
```

### 2. Validation

Edit `src/validation.py` to set `VAL_Z` (hold-out planes) and run:

```bash
uv run ./src/validation.py models/fno_e200_k3_n16_top25.pt
```

This performs autoregressive rollout, prints per-step R² scores, and shows a side-by-side plot of Ground Truth vs Prediction + Error map.

---

## Key Method Details

- **Energy Spectrum Encoder:** Each subdomain is reduced to its top-*p* Fourier energy components.
- **Wasserstein k-means:** Clusters subdomains by the distribution of their kinetic energy spectra rather than Euclidean distance.
- **BENO Boundary Encoding:** A 1D convolution + transformer encodes boundary node values into a latent vector that is added to the operator’s source term.
- **Overlap & Averaging:** Adjacent subdomains overlap by 1 pixel; predictions in overlap zones are averaged during reconstruction.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{xu2026aldd,
  title={Energy-based feature extraction with adaptive local domain decomposition for prediction of transient and turbulence flow with operator regression models},
  author={Xu, Wenzhuo and Karthikeyakannan, Madhav and McComb, Christopher and Grande Guti{\'e}rrez, Noelia},
  journal={Computers \& Fluids},
  volume={307},
  pages={106958},
  year={2026},
  publisher={Elsevier}
}
```

---
