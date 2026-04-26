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

### What was AI-generated or heavily AI-assisted

- **`collect_data.py`** — Initial structure generated with Claude Code; the choice of which endpoints (`LeagueDashLineups`, `LeagueDashPlayerStats`), which measure types, and the per-100 possessions normalization were guided by the student with AI providing the API call syntax.
- **`clean_data.py`, `build_features.py`, `preprocess.py`** — Scaffolding generated with Claude Code. The temporal split strategy (train on past seasons, predict future) was a student design decision; AI implemented it. The 1000-minute filter was proposed and specified by the student.
- **`train_synergy.py`** — SynergyNet architecture was designed collaboratively: student specified "transformer self-attention over a 5-player matrix" as the core idea; Claude Code implemented the MultiheadAttention block, residual connections, LayerNorm, and training loop. The 5-seed ensemble approach was AI-suggested; student approved it.
- **`train_synergy_v2.py`** — The idea to add explicit role features (spacing_score = FG3A × TS_PCT, creation_score = USG_PCT × AST_PCT) and inject a max_creator × avg_spacing interaction term into the head was generated collaboratively: student raised the goal of detecting offensive synergy between creators and floor spacers; Claude Code proposed these specific features. Student evaluated results and confirmed improvement.
- **`app.py`** — Initial app structure generated with Claude Code. Budget/pricing formula (PIE + USG_PCT + MIN blend, power 2.5 curve, $130M cap) was designed iteratively with student specifying goals and reviewing balance. Several UI fixes were needed: player cards went through multiple iterations after student reported that names were not rendering (a Streamlit rendering issue with `st.container` + `st.markdown` was the root cause; fixed by consolidating to a single HTML block).
- **`predict.py`** — CLI tool generated with Claude Code. Required one significant fix: initial feature construction was column-major (all means, then all stds) but training data was row-major (mean/std/min/max per stat). This was debugged and corrected. A second bug (selecting maximum-minutes row across all seasons instead of most recent season) was also found and fixed.
- **`analyze_results.py`** — Figure generation code scaffolded with Claude Code; student specified which comparisons to visualize.

### What was substantially modified or reworked

- The SynergyNet feature ordering bug in `predict.py` required careful debugging of the exact column order produced by `build_features.py` vs. the order constructed at inference time.
- The Streamlit player card rendering went through three design iterations before finding that consolidating all card content into a single `st.markdown()` HTML block was necessary to avoid Streamlit's container CSS hiding elements.
- The pricing formula in `app.py` was tuned manually across multiple iterations to achieve the desired economic balance (making it nearly impossible to afford three max players, ensuring budget constraints are meaningful).

### What was written from scratch without AI assistance

- All high-level design decisions: problem framing (predicting lineup synergy vs. individual stats), choice of model paradigm (set/permutation-invariant architecture), evaluation methodology (temporal train/val/test split to avoid season leakage), and the economic model for the Superteam Builder.
- Experimental design for the ablation study comparing SynergyNet v1 vs. v2.
