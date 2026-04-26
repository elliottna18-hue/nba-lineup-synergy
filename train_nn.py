"""
Neural network for lineup OFF_RATING / DEF_RATING prediction.

Two separate networks — one per target. A shared backbone was tried first
but created conflicting gradients on this small dataset (1,068 samples);
separate models allow each target to use its own feature weighting.

Architecture per model:
    Input(84) -> Linear(84->64) -> ReLU -> Dropout(0.4)
              -> Linear(64->32) -> ReLU -> Dropout(0.3)
              -> Linear(32->1)

Design notes for this dataset size:
  - Targets are z-score normalized (training stats). Without this, the model
    starts with ~110x too-large gradients and early-stops within 5 epochs.
  - L1Loss: directly minimizes MAE (the evaluation metric). MSE/SmoothL1
    over-weights the extreme-rating outliers that survived POSS>=100 filtering.
  - Dropout 0.4/0.3 + weight decay 5e-3: needed together; either alone is
    insufficient regularization for ~5k params / 1,068 samples.
  - LR 5e-4 (slower than 1e-3): prevents the model from immediately jumping
    to a near-mean solution and getting stuck there.
  - Train 5 seeds, ensemble by averaging: known to reduce variance 10-20%
    on small datasets with minimal added complexity.

Outputs (artifacts/):
  nn_off_seed{k}.pt / nn_def_seed{k}.pt  -- per-seed weights
  nn_target_stats.pkl                     -- y mean/std for inference
  nn_history.csv                          -- per-epoch metrics (seed 0)
  baseline_results.csv                    -- updated with NN rows
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA      = Path('data/model_ready')
ARTIFACTS = Path('artifacts')

EPOCHS       = 500
BATCH_SIZE   = 32
LR           = 5e-4
WEIGHT_DECAY = 5e-3
LR_PATIENCE  = 25
ES_PATIENCE  = 50
N_SEEDS      = 5


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RatingNet(nn.Module):
    """Single-target MLP: 84 -> 64 -> 32 -> 1."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split(name: str):
    X    = pd.read_parquet(DATA / f'X_{name}.parquet').values.astype(np.float32)
    y    = pd.read_parquet(DATA / f'y_{name}.parquet')
    return (torch.tensor(X),
            torch.tensor(y['OFF_RATING'].values.astype(np.float32)),
            torch.tensor(y['DEF_RATING'].values.astype(np.float32)))


def compute_metrics(y_true, y_pred) -> dict:
    return {
        'MAE' : round(float(mean_absolute_error(y_true, y_pred)), 3),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        'R2'  : round(float(r2_score(y_true, y_pred)), 3),
    }


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor,
            mean: float, std: float) -> np.ndarray:
    model.eval()
    return model(X).numpy() * std + mean


# ---------------------------------------------------------------------------
# Single-seed training loop
# ---------------------------------------------------------------------------

def train_one(X_tr_t: torch.Tensor, y_tr_n: torch.Tensor,
              X_val: torch.Tensor,  y_val: torch.Tensor,
              mean: float, std: float,
              seed: int, ckpt_path: Path,
              verbose: bool = False) -> tuple[nn.Module, list]:

    torch.manual_seed(seed)
    model = RatingNet(input_dim=X_tr_t.shape[1])
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_PATIENCE, factor=0.5, min_lr=1e-6
    )

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_n),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_val_mae = float('inf')
    best_epoch   = 0
    patience_ctr = 0
    history      = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        train_loss /= len(loader.dataset)

        pred_val = predict(model, X_val, mean, std)
        val_mae  = float(mean_absolute_error(y_val.numpy(), pred_val))
        scheduler.step(val_mae)

        history.append({'epoch': epoch, 'train_loss': round(train_loss, 5),
                        'val_mae': round(val_mae, 4),
                        'lr': optimizer.param_groups[0]['lr']})

        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            best_epoch   = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= ES_PATIENCE:
                if verbose:
                    print(f'    early stop ep {epoch} '
                          f'(best ep {best_epoch}, val MAE {best_val_mae:.3f})')
                break

        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f'    ep {epoch:>4}  train={train_loss:.4f}  '
                  f'val MAE={val_mae:.3f}  '
                  f'lr={optimizer.param_groups[0]["lr"]:.1e}')

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return model, history


# ---------------------------------------------------------------------------
# Ensemble: train N seeds, average predictions
# ---------------------------------------------------------------------------

def train_ensemble(target: str, X_tr: torch.Tensor, y_tr: torch.Tensor,
                   X_val: torch.Tensor, y_val_true: torch.Tensor,
                   mean: float, std: float) -> list[nn.Module]:

    y_tr_n = (y_tr - mean) / std  # normalized targets for training

    models = []
    val_maes = []
    print(f'  {target} ({N_SEEDS} seeds):')
    for seed in range(N_SEEDS):
        ckpt = ARTIFACTS / f'nn_{target.lower()[:3]}_seed{seed}.pt'
        verbose = (seed == 0)
        model, hist = train_one(X_tr, y_tr_n, X_val, y_val_true,
                                mean, std, seed, ckpt, verbose=verbose)
        if seed == 0:
            pd.DataFrame(hist).to_csv(
                ARTIFACTS / f'nn_history_{target.lower()[:3]}.csv', index=False
            )
        pred = predict(model, X_val, mean, std)
        mae  = mean_absolute_error(y_val_true.numpy(), pred)
        val_maes.append(mae)
        models.append(model)
        print(f'    seed {seed}: val MAE = {mae:.3f}')

    # Ensemble
    ens_pred = np.mean([predict(m, X_val, mean, std) for m in models], axis=0)
    ens_mae  = mean_absolute_error(y_val_true.numpy(), ens_pred)
    print(f'    ensemble: val MAE = {ens_mae:.3f}  '
          f'(vs best single: {min(val_maes):.3f})')
    return models


@torch.no_grad()
def ensemble_predict(models: list, X: torch.Tensor,
                     mean: float, std: float) -> np.ndarray:
    return np.mean([predict(m, X, mean, std) for m in models], axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    X_tr,  y_tr_off,  y_tr_def  = load_split('train')
    X_val, y_val_off, y_val_def = load_split('val')
    X_te,  y_te_off,  y_te_def  = load_split('test')

    # Target stats from training set
    target_stats = {
        'off_mean': float(y_tr_off.mean()), 'off_std': float(y_tr_off.std()),
        'def_mean': float(y_tr_def.mean()), 'def_std': float(y_tr_def.std()),
    }
    with open(ARTIFACTS / 'nn_target_stats.pkl', 'wb') as f:
        pickle.dump(target_stats, f)

    n_params = sum(p.numel() for p in RatingNet(X_tr.shape[1]).parameters())
    print(f'Model params per target: {n_params:,}  '
          f'(x2 = {2*n_params:,} total across both targets)')
    print(f'Samples: train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}\n')

    print('=== Training OFF_RATING models ===')
    off_models = train_ensemble(
        'OFF_RATING', X_tr, y_tr_off, X_val, y_val_off,
        target_stats['off_mean'], target_stats['off_std'],
    )

    print('\n=== Training DEF_RATING models ===')
    def_models = train_ensemble(
        'DEF_RATING', X_tr, y_tr_def, X_val, y_val_def,
        target_stats['def_mean'], target_stats['def_std'],
    )

    # Final evaluation
    results = []
    for split, X, y_off, y_def in [
        ('train', X_tr,  y_tr_off,  y_tr_def),
        ('val',   X_val, y_val_off, y_val_def),
        ('test',  X_te,  y_te_off,  y_te_def),
    ]:
        p_off = ensemble_predict(off_models, X,
                                 target_stats['off_mean'], target_stats['off_std'])
        p_def = ensemble_predict(def_models, X,
                                 target_stats['def_mean'], target_stats['def_std'])
        for target, y_true, pred in [('OFF_RATING', y_off, p_off),
                                      ('DEF_RATING', y_def, p_def)]:
            results.append({'model': 'NeuralNet', 'target': target,
                            'split': split, **compute_metrics(y_true.numpy(), pred)})

    df_nn = pd.DataFrame(results)

    baseline_path = ARTIFACTS / 'baseline_results.csv'
    if baseline_path.exists():
        existing = pd.read_csv(baseline_path)
        existing = existing[existing['model'] != 'NeuralNet']
        combined = pd.concat([existing, df_nn], ignore_index=True)
    else:
        combined = df_nn
    combined.to_csv(baseline_path, index=False)

    model_order = {'GlobalMean': 0, 'MeanPlayerRating': 1, 'Ridge': 2, 'NeuralNet': 3}
    combined['_ord'] = combined['model'].map(model_order)

    for split_name in ['val', 'test']:
        header = (f"{'Model':<20} {'Target':<12} {'Split':<6}"
                  f"  {'MAE':>6}  {'RMSE':>6}  {'R2':>6}")
        print(f'\n=== {split_name.capitalize()} Results ===')
        print(header)
        print('-' * len(header))
        for _, r in (combined[combined['split'] == split_name]
                     .sort_values(['target', '_ord'])).iterrows():
            print(f"{r['model']:<20} {r['target']:<12} {r['split']:<6}"
                  f"  {r['MAE']:>6.3f}  {r['RMSE']:>6.3f}  {r['R2']:>6.3f}")

    print('\n=== NN vs Ridge (val MAE) ===')
    for target in ['OFF_RATING', 'DEF_RATING']:
        def get_val_mae(mdl):
            return combined[(combined['model'] == mdl) &
                            (combined['target'] == target) &
                            (combined['split'] == 'val')]['MAE'].values[0]
        ridge_mae = get_val_mae('Ridge')
        nn_mae    = get_val_mae('NeuralNet')
        delta     = ridge_mae - nn_mae
        sign      = '+' if delta >= 0 else ''
        print(f'  {target}: Ridge {ridge_mae:.3f} -> NN {nn_mae:.3f}  '
              f'({sign}{delta:.3f})')


if __name__ == '__main__':
    main()
