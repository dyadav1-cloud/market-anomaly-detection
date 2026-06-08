# Market Anomaly Detection in S&P 500 Equities

**Status:** Work in progress (Summer 2026)

A comparative study of statistical, machine learning, and deep learning methods
for detecting anomalous price-volume events in S&P 500 equities, with an
interpretability layer using SHAP.

## Research Question

When a statistical rule, an Isolation Forest, and an LSTM autoencoder flag
anomalous events in S&P 500 equities, to what extent do the three methods agree,
and — when they disagree — which types of disagreement are most predictive of
subsequent abnormal returns?

## Project Structure

- `data/` — Raw and processed datasets (gitignored)
- `notebooks/` — Jupyter notebooks for exploration and experiments
- `src/` — Reusable Python modules
- `results/` — Output figures and tables
- `papers/` — PDF references (gitignored)

## Setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

Verify GPU access:
```python
import torch
print(torch.cuda.is_available())  # should print True
```

## Mentor

Dr. Cagla Yildirim, Caldwell University

## Program

STEM Advance Summer Research Program, Caldwell University, 2026