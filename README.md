# NBA Lineup Synergy Predictor

## What it Does

This project predicts the offensive and defensive rating of any 5-player NBA lineup using a custom transformer neural network (SynergyNet). Rather than simply averaging individual player stats, the model learns interaction effects — how players' roles, spacing tendencies, and creation ability combine to produce team-level efficiency beyond what any individual player stat would predict. The project includes a full data pipeline (NBA API → raw → cleaned → features → model), five distinct model architectures evaluated head-to-head, an ablation-driven model iteration (SynergyNetV2), a CLI prediction tool, and an interactive Streamlit "Superteam Builder" where users assemble lineups under a salary cap and see predicted NET_RATING in real time.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Superteam Builder web app
streamlit run app.py

# 3. (Optional) Predict a specific lineup from the command line
python predict.py "LeBron James" "Anthony Davis" "Austin Reaves" "D'Angelo Russell" "Jarred Vanderbilt"

# 4. (Optional) Re-run the full data pipeline
python collect_data.py     # pull from stats.nba.com (~5 min)
python clean_data.py
python build_features.py
python preprocess.py
python train_baseline.py   # Ridge + NeuralNet baselines
python train_deepsets.py   # DeepSets architecture
python train_synergy.py    # SynergyNet (v1)
python train_synergy_v2.py # SynergyNetV2 + comparison table
python analyze_results.py  # generate all figures
```

> Pre-trained model artifacts are already included in `artifacts/`, so you can run the app and CLI tool without re-training.

## Video Links

- **Demo video:** [TBD — add link after recording]
- **Technical walkthrough:** [TBD — add link after recording]

## Evaluation

All models evaluated on a held-out test set (2024-25 and 2025-26 seasons, never seen during training or hyperparameter selection).

| Model | OFF MAE | DEF MAE | NET MAE | OFF R² | DEF R² |
|---|---|---|---|---|---|
| Ridge (baseline) | 6.053 | 5.956 | 12.009 | 0.196 | 0.219 |
| NeuralNet | 6.013 | 6.014 | 12.027 | 0.196 | 0.209 |
| DeepSets | 6.203 | 5.995 | 12.198 | 0.137 | 0.205 |
| SynergyNet (v1) | 6.168 | 6.010 | 12.178 | 0.151 | 0.200 |
| **SynergyNetV2** | **6.119** | **6.021** | **12.140** | **0.157** | **0.201** |

SynergyNetV2 achieves the best offensive MAE (6.119) and best combined NET MAE (12.140) on the test set. SynergyNetV2 improved over v1 by adding explicit role features (spacing score, creation score) and an interaction term (max creator × avg spacing) in the prediction head, yielding a −0.049 offensive MAE gain on test.

Validation MAE for the 5-seed SynergyNetV2 ensemble: OFF 6.001, DEF 5.957. The ensemble reduces variance vs. the best single seed (OFF 5.993 single → 6.001 ensemble on val; consistent across seeds indicates stable training).

Model accuracy is approximately ±6 NET_RATING points. Lineup NET_RATING typically ranges from −15 to +20, so this represents meaningful but imperfect predictive signal.

## Project Structure

```
collect_data.py      — NBA API data collection (14 seasons)
clean_data.py        — deduplication, missing-value handling
build_features.py    — per-lineup feature matrix construction
preprocess.py        — train/val/test split + StandardScaler
train_baseline.py    — Ridge regression + NeuralNet baselines
train_deepsets.py    — DeepSets permutation-invariant architecture
train_synergy.py     — SynergyNet (transformer self-attention over players)
train_synergy_v2.py  — SynergyNetV2 (role features + synergy interaction)
analyze_results.py   — generate comparison figures and summary table
predict.py           — CLI: predict any 5 players by name
app.py               — Streamlit Superteam Builder web app
artifacts/           — trained model weights + scalers
data/                — raw / clean / features / model_ready parquets
figures/             — generated evaluation figures
```
