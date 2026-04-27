# Attribution

## Data Sources

- **NBA Statistics API** — All lineup and player statistics were collected from [stats.nba.com](https://stats.nba.com) via the [`nba_api`](https://github.com/swar/nba_api) Python library. Data covers 14 seasons (2012-13 through 2025-26). No data was manually annotated; the pipeline is fully automated via `collect_data.py`.

## Libraries and Frameworks

- **PyTorch** — neural network architectures (SynergyNet, SynergyNetV2, NeuralNet, DeepSets), training loops, custom DataLoaders
- **scikit-learn** — Ridge regression baseline, StandardScaler
- **pandas / pyarrow** — data wrangling and parquet I/O
- **Streamlit** — interactive web application (`app.py`)
- **matplotlib** — evaluation figures (`analyze_results.py`)
- **nba_api** — NBA statistics data collection

## AI Tool Usage

This project was built with substantial assistance from **Claude Code (Anthropic)** throughout the development process. Below is an account of what was generated, what was modified, and what required debugging or rework.

- **`collect_data.py`** — Initial structure generated with Claude Code and Claude Code also helped generate API call syntax.
- **`clean_data.py`, `build_features.py`, `preprocess.py`** — Scaffolding generated with Claude Code‚ I then implemented deisgn choices.
- **`train_synergy.py`** — Claude Code implemented the MultiheadAttention block, residual connections, LayerNorm, and training loop. 
- **`app.py`** — Initial app structure generated with Claude Code. UI designed by Claude Code with several iterations and changes based on my input.
- **`predict.py`** — CLI tool generated with Claude Code.
- **`analyze_results.py`** — Figure generation code scaffolded with Claude Code.


