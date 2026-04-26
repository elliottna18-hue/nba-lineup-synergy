# Setup Instructions

## Requirements

- Python 3.10 or higher
- ~2 GB disk space for model artifacts and data

## Installation

### 1. Clone or download the repository

```bash
git clone <repo-url>
cd nba-lineup-synergy
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch is listed as `torch>=2.0.0`. If you want GPU support, install the CUDA build from [pytorch.org](https://pytorch.org) before running `pip install -r requirements.txt`. The CPU build works for inference; training will be slower.

## Running the App

Pre-trained model artifacts are included in `artifacts/`. No re-training is required to use the app or CLI.

### Streamlit Superteam Builder

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### CLI Prediction Tool

```bash
python predict.py "LeBron James" "Stephen Curry" "Kevin Durant" "Draymond Green" "Klay Thompson"
```

Optional flags:
- `--season 2023-24` — specify a season (default: most recent available for each player)

## Re-running the Full Data Pipeline

Only necessary if you want to retrain from scratch or add new seasons.

```bash
# 1. Collect raw data from NBA API (requires internet; ~5 min due to rate limiting)
python collect_data.py

# 2. Clean raw data (deduplication, missing value handling)
python clean_data.py

# 3. Build lineup-level feature matrices
python build_features.py

# 4. Train/val/test split + feature scaling
python preprocess.py

# 5. Train all models
python train_baseline.py    # Ridge regression + NeuralNet
python train_deepsets.py    # DeepSets + hyperparameter sweep
python train_synergy.py     # SynergyNet (transformer)
python train_synergy_v2.py  # SynergyNetV2 (role features)

# 6. Generate evaluation figures
python analyze_results.py
```

### Expected Training Times (CPU)

| Script | Approximate Time |
|---|---|
| train_baseline.py | < 1 min |
| train_deepsets.py | 5–10 min |
| train_synergy.py | 10–15 min |
| train_synergy_v2.py | 15–20 min |

## Data Directory Structure

```
data/
  raw/        — API responses as parquet files
  clean/      — cleaned lineups and player stats
  features/   — lineup-level feature matrix
  model_ready/ — scaled X/y train/val/test splits
```
