# B-RISE

RISE algorithm reimplemented with a new calculation of the importance scores using the **Banzhaf value**. The code is based on the original RISE implementation by Petsiuk et al.

The main difference is in the way the importance scores are calculated. Instead of using the original RISE method, we use the **Banzhaf value** to calculate the importance scores for each pixel. This allows us to capture the interactions between pixels and provide a more accurate explanation of the model's predictions.

---

# Setup

This project uses **uv** for environment and dependency management.

## 1. Install uv

If you don't have `uv` installed:

```bash
pip install uv
```
## 2. Create and activate the virtual environment
From the project root:
```
uv venv
```
## 3. Install dependencies
```
uv sync
```
# Code Structure

```
src/
└── brise/
    ├── brise.py           # Implementation of the B-RISE algorithm
    ├── rise.py            # Implementation of the original RISE algorithm
    ├── exact_banzhaf.py   # Exact Banzhaf value computation
    ├── evaluation.py      # Insertion and deletion evaluation
    └── utils.py           # Helper functions
```

Notebooks:
- `B_RISE_code.ipynb` – Code snippets from B-RISE algorithm.
- `B_RISE_saveframes.ipynb` – Saving frames for BRISE algorithm visualization.
- `RESNET.ipynb` – ResNet50 Model on few images and results of B-RISE and RISE algorithms.
- `MNSIT.ipynb` – MNIST Model on few images and results of B-RISE and RISE algorithms.