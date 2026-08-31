# Market Anomaly Detection in S&P 500 Equities

Undergraduate research comparing three anomaly detection methods on ten years of S&P 500 price data: a rolling z-score baseline, an Isolation Forest, and an LSTM autoencoder. The project measures how much the three detectors agree with each other, then tests whether the days they disagree on actually predict abnormal stock returns, and uses SHAP to explain what each detector is picking up on.

## Methodology

Three detectors are built independently, each tuned to flag roughly 0.70% of stock-days as anomalous so their outputs are directly comparable:

- **±3σ baseline** (`02_statistical_baseline.ipynb`): flags a stock-day when its return z-score, computed against a trailing 20-day mean and standard deviation, exceeds 3 standard deviations.
- **Isolation Forest** (`03_isolation_forest.ipynb`): fit on all five engineered features (return z-score, three rolling volatility windows, volume z-score) together, so it can flag a day based on its overall feature-space position rather than return alone.
- **LSTM autoencoder** (`04_lstm_autoencoder.ipynb`): trained on 30-day sequences from the calmer 2015-2019 period, then used to score every stock-day by reconstruction error. A day the model can't reconstruct well is flagged as a departure from normal temporal patterns.

The agreement analysis (`05_agreement_analysis.ipynb`) computes Cohen's kappa and Jaccard overlap between each pair of detectors and partitions every flagged day into agreement buckets. The event study (`06_event_study.ipynb`) uses a Fama-French three-factor model to estimate expected returns, then checks whether the days flagged uniquely by each detector are followed by abnormal cumulative returns over the next 5, 10, and 20 trading days. The interpretability notebook (`07_interpretability.ipynb`) applies SHAP to the Isolation Forest and LSTM to explain which features drive each one's flags.

## Repository Structure

```
notebooks/    Jupyter notebooks 01-07, run in order (see below)
data/         Generated datasets and model artifacts (gitignored, rebuilt by running the notebooks)
results/      Saved charts (PNG) produced by the notebooks
src/          Reserved for reusable Python modules
papers/       Reference material (gitignored)
```

## Setup

Requires Python 3.12.

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
```

`requirements.txt` pins `torch`, `torchaudio`, and `torchvision` to CUDA 12.8 builds, which live on PyTorch's own package index rather than the default PyPI. Install those three first, then the rest:

```bash
pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

This works whether or not you have a GPU. Check GPU availability with:

```python
import torch
print(torch.cuda.is_available())
```

Training still runs on CPU if no GPU is available, just slower.

`data/` is gitignored since the files it holds are large and fully reproducible. Running notebook 01 downloads the S&P 500 roster and price history from Yahoo Finance and rebuilds the dataset from scratch.

`notebooks/00_environment_check.py` is a small standalone script, separate from the numbered 01-07 pipeline, that confirms PyTorch, CUDA, and the core libraries are installed correctly. Run it first if you want a quick sanity check before opening the notebooks.

## Running the Notebooks

Run the notebooks in order, top to bottom, since each one depends on files saved by the ones before it:

1. `01_data_acquisition.ipynb` downloads S&P 500 price data and builds the feature matrix.
2. `02_statistical_baseline.ipynb` builds the ±3σ detector.
3. `03_isolation_forest.ipynb` builds the Isolation Forest detector.
4. `04_lstm_autoencoder.ipynb` builds the LSTM autoencoder detector.
5. `05_agreement_analysis.ipynb` compares the three detectors' agreement.
6. `06_event_study.ipynb` tests whether disagreements predict abnormal returns.
7. `07_interpretability.ipynb` explains the Isolation Forest and LSTM with SHAP.

## Key Findings

- All three detectors are tuned to flag about the same rate of stock-days (~0.70%), but they barely agree on which ones. Only 13 of 23,390 total flagged stock-days (0.1%) are flagged by all three detectors, and pairwise Cohen's kappa ranges from -0.002 to 0.132, near or below chance agreement.
- The disagreement turns out to be economically meaningful. Days flagged only by the Isolation Forest are followed by positive abnormal returns that grow over time, reaching about +0.70% over the next 20 trading days. Days flagged only by the ±3σ baseline or the LSTM are followed by mildly negative abnormal returns instead, around -0.24% and -0.14% over 20 days. The gap between the Isolation Forest and the baseline is statistically significant (p < 0.001 at the 10- and 20-day horizons).
- SHAP shows why. The Isolation Forest's flags are driven almost entirely by volatility (vol_5, vol_20, vol_60), while the LSTM's flags are driven mainly by return and volume surprise, the opposite pattern. The three detectors are picking up genuinely different kinds of anomalies, not three versions of the same signal.

## Program

This is undergraduate research conducted as part of the STEM Advance Summer Research Program at Caldwell University, under the mentorship of Dr. Cagla Yildirim.
