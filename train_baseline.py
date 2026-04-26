"""
Baseline models for lineup OFF_RATING / DEF_RATING prediction.

Three baselines in increasing order of sophistication:

  1. Global mean       — always predict the training-set mean; sets the floor.
  2. Mean player rating — predict the average of the 5 players' individual
                          on-court ratings; captures the "no synergy" assumption.
  3. Ridge regression  — linear model over all 84 features; alpha tuned via
                          5-fold CV on the training set.

Each baseline trains and evaluates two separate models (OFF and DEF).
Results are saved to artifacts/baseline_results.csv for later comparison
with the neural net.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA      = Path('data/model_ready')
FEATURES  = Path('data/features')
ARTIFACTS = Path('artifacts')

TARGETS = ['OFF_RATING', 'DEF_RATING']
ALPHAS  = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split(name: str):
    X    = pd.read_parquet(DATA / f'X_{name}.parquet')
    y    = pd.read_parquet(DATA / f'y_{name}.parquet')
    meta = pd.read_parquet(DATA / f'meta_{name}.parquet')
    return X, y, meta


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'MAE' : round(mean_absolute_error(y_true, y_pred), 3),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        'R2'  : round(r2_score(y_true, y_pred), 3),
    }


def print_results(rows: list[dict]) -> None:
    header = f"{'Model':<28} {'Target':<12} {'Split':<6}  {'MAE':>6}  {'RMSE':>6}  {'R2':>6}"
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f"{r['model']:<28} {r['target']:<12} {r['split']:<6}  "
              f"{r['MAE']:>6.3f}  {r['RMSE']:>6.3f}  {r['R2']:>6.3f}")


def record(results: list, model: str, target: str, split: str,
           y_true: np.ndarray, y_pred: np.ndarray) -> None:
    m = compute_metrics(y_true, y_pred)
    results.append({'model': model, 'target': target, 'split': split, **m})


# ---------------------------------------------------------------------------
# 1. Global mean baseline
# ---------------------------------------------------------------------------

def run_global_mean(splits: dict, results: list) -> None:
    X_tr, y_tr, _ = splits['train']
    train_means = {t: y_tr[t].mean() for t in TARGETS}

    for split_name, (X, y, _) in splits.items():
        for target in TARGETS:
            pred = np.full(len(y), train_means[target])
            record(results, 'GlobalMean', target, split_name, y[target].values, pred)


# ---------------------------------------------------------------------------
# 2. Mean player rating heuristic
# ---------------------------------------------------------------------------

def run_mean_player_rating(splits: dict, results: list) -> None:
    """
    For each lineup, predict the unscaled mean of the 5 players' individual
    ratings. Since X is already StandardScaler-transformed, we inverse-
    transform just the two columns we need.
    """
    with open(DATA / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    X_tr, _, _ = splits['train']
    feature_cols = list(X_tr.columns)
    off_idx = feature_cols.index('mean_OFF_RATING')
    def_idx = feature_cols.index('mean_DEF_RATING')

    def unscale_col(X_scaled: pd.DataFrame, col_idx: int) -> np.ndarray:
        mean  = scaler.mean_[col_idx]
        scale = scaler.scale_[col_idx]
        return X_scaled.iloc[:, col_idx].values * scale + mean

    target_to_col = {'OFF_RATING': off_idx, 'DEF_RATING': def_idx}

    for split_name, (X, y, _) in splits.items():
        for target in TARGETS:
            pred = unscale_col(X, target_to_col[target])
            record(results, 'MeanPlayerRating', target, split_name, y[target].values, pred)


# ---------------------------------------------------------------------------
# 3. Ridge regression
# ---------------------------------------------------------------------------

def run_ridge(splits: dict, results: list) -> dict:
    X_tr, y_tr, _ = splits['train']
    models = {}

    for target in TARGETS:
        ridge = RidgeCV(alphas=ALPHAS, cv=5, scoring='neg_mean_absolute_error')
        ridge.fit(X_tr.values, y_tr[target].values)
        models[target] = ridge
        print(f'  Ridge {target}: best alpha = {ridge.alpha_:.4g}')

    for split_name, (X, y, _) in splits.items():
        for target in TARGETS:
            pred = models[target].predict(X.values)
            record(results, 'Ridge', target, split_name, y[target].values, pred)

    return models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    splits = {name: load_split(name) for name in ['train', 'val', 'test']}

    results: list[dict] = []

    print('=== 1. Global Mean ===')
    run_global_mean(splits, results)

    print('\n=== 2. Mean Player Rating ===')
    run_mean_player_rating(splits, results)

    print('\n=== 3. Ridge Regression ===')
    ridge_models = run_ridge(splits, results)

    # Save ridge models
    with open(ARTIFACTS / 'ridge_off.pkl', 'wb') as f:
        pickle.dump(ridge_models['OFF_RATING'], f)
    with open(ARTIFACTS / 'ridge_def.pkl', 'wb') as f:
        pickle.dump(ridge_models['DEF_RATING'], f)

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(ARTIFACTS / 'baseline_results.csv', index=False)

    # Print summary tables — val split is what matters for model selection
    print('\n=== Results: Validation Set ===')
    val_rows = [r for r in results if r['split'] == 'val']
    print_results(val_rows)

    print('\n=== Results: Test Set ===')
    test_rows = [r for r in results if r['split'] == 'test']
    print_results(test_rows)

    # Ridge: top 10 most influential features per target
    X_tr, _, _ = splits['train']
    feature_cols = list(X_tr.columns)
    print()
    for target in TARGETS:
        coefs = pd.Series(ridge_models[target].coef_, index=feature_cols)
        top = coefs.abs().nlargest(10)
        print(f'Ridge top-10 features by |coef| — {target}:')
        for feat, val in top.items():
            direction = '+' if coefs[feat] > 0 else '-'
            print(f'  {direction}  {feat:<35}  {coefs[feat]:+.3f}')
        print()


if __name__ == '__main__':
    main()
