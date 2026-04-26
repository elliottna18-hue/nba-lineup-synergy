"""
Deep Sets model for lineup OFF_RATING / DEF_RATING prediction.

Why Deep Sets over the MLP on aggregated features:
  The MLP sees only pre-computed mean/std/min/max of the 5 players — it cannot
  distinguish which specific player combination produced those aggregates.
  Deep Sets encodes every player independently through a shared phi network,
  then the model learns its own aggregation. This lets it discover synergy
  patterns like "a rim-protector + two high-AST_TO guards score better than
  their averages predict" — effects that vanish after hand-crafting aggregations.

Architecture:
  phi (shared per-player encoder, 21-dim input):
      Linear(21->embed) + LayerNorm + ReLU + Dropout
      Linear(embed->embed) + LayerNorm + ReLU + Dropout

  Aggregation: concat [mean(5 embeddings), max(5 embeddings)]  ->  2*embed dims
  Context: append SEASON_YEAR (scaled)

  rho_off / rho_def (separate heads):
      Linear(2*embed+1 -> embed) + ReLU + Dropout
      Linear(embed -> 32) + ReLU + Linear(32 -> 1)

Tuning: grid search over embed_dim x dropout, 1 seed each, pick best val MAE.
Final model: best config trained with 5 seeds, ensemble-averaged.

Outputs (artifacts/):
  ds_off_seed{k}.pt / ds_def_seed{k}.pt
  ds_target_stats.pkl
  ds_player_scaler.pkl
  ds_history_seed0_{off,def}.csv
  ds_sweep_results.csv
  baseline_results.csv  (updated with DeepSets row)
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CLEAN     = Path('data/clean')
ARTIFACTS = Path('artifacts')

TRAIN_SEASONS = ['2012-13', '2013-14', '2014-15', '2015-16', '2016-17',
                 '2017-18', '2018-19', '2019-20', '2020-21', '2021-22']
VAL_SEASONS   = ['2022-23', '2023-24']
TEST_SEASONS  = ['2024-25', '2025-26']

PLAYER_STAT_COLS = [
    'OFF_RATING', 'DEF_RATING', 'USG_PCT', 'AST_PCT', 'AST_TO', 'PIE',
    'EFG_PCT', 'TS_PCT', 'OREB_PCT', 'DREB_PCT', 'TM_TOV_PCT',
    'PTS', 'FGA', 'FG3A', 'FTA', 'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB',
]

N_PLAYERS    = 5
N_SEEDS      = 5
EPOCHS       = 500
BATCH_SIZE   = 32
LR           = 5e-4
WEIGHT_DECAY = 5e-3
LR_PATIENCE  = 25
ES_PATIENCE  = 50

# Grid search space
SWEEP_CONFIGS = [
    {'embed_dim': 32,  'dropout': (0.3, 0.2)},
    {'embed_dim': 64,  'dropout': (0.3, 0.2)},
    {'embed_dim': 128, 'dropout': (0.3, 0.2)},
    {'embed_dim': 32,  'dropout': (0.4, 0.3)},
    {'embed_dim': 64,  'dropout': (0.4, 0.3)},
    {'embed_dim': 128, 'dropout': (0.4, 0.3)},
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DeepSetsNet(nn.Module):
    """
    Permutation-invariant set network.
    phi processes each player independently (shared weights).
    rho decodes from the mean+max aggregation + era context.
    """
    def __init__(self, n_feats: int, embed_dim: int = 64,
                 dropout: tuple = (0.3, 0.2)):
        super().__init__()
        d0, d1 = dropout

        self.phi = nn.Sequential(
            nn.Linear(n_feats, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(d0),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(d1),
        )

        # mean + max concat + 1 era feature
        agg_dim = 2 * embed_dim + 1

        def head():
            return nn.Sequential(
                nn.Linear(agg_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(d1),
                nn.Linear(embed_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        self.off_head = head()
        self.def_head = head()

    def forward(self, x_players: torch.Tensor,
                x_season: torch.Tensor):
        # x_players : (B, 5, n_feats)
        # x_season  : (B, 1)
        emb      = self.phi(x_players)          # (B, 5, embed)
        agg_mean = emb.mean(dim=1)              # (B, embed)
        agg_max  = emb.max(dim=1).values        # (B, embed)
        h = torch.cat([agg_mean, agg_max, x_season], dim=1)
        return self.off_head(h).squeeze(-1), self.def_head(h).squeeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_player_matrices(lineups: pd.DataFrame,
                          player_lookup: dict,
                          n_feats: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X_players : (n_lineups, 5, n_feats)  float32
      X_season  : (n_lineups,)             int  (first year of season, e.g. 2022)
      valid     : (n_lineups,)             bool (all 5 players found)
    """
    X_players = np.zeros((len(lineups), N_PLAYERS, n_feats), dtype=np.float32)
    X_season  = np.zeros(len(lineups), dtype=np.int32)
    valid     = np.zeros(len(lineups), dtype=bool)

    for i, (_, row) in enumerate(lineups.iterrows()):
        season     = row['SEASON']
        player_ids = [int(p) for p in row['GROUP_ID'].strip('-').split('-')]
        X_season[i] = int(season[:4])

        feats = []
        for pid in player_ids:
            key = (pid, season)
            if key in player_lookup:
                feats.append(player_lookup[key])
            else:
                break
        else:
            X_players[i] = np.array(feats, dtype=np.float32)
            valid[i] = True

    return X_players, X_season, valid


def prepare_data():
    lineups = pd.read_parquet(CLEAN / 'lineups.parquet')
    players = pd.read_parquet(CLEAN / 'players.parquet')

    # Build fast player lookup: (PLAYER_ID, SEASON) -> feature array
    player_lookup = {
        (row.PLAYER_ID, row.SEASON): row[PLAYER_STAT_COLS].values.astype(np.float32)
        for _, row in players.iterrows()
    }

    # --- scale player features using training-set players ---
    train_lineups = lineups[lineups['SEASON'].isin(TRAIN_SEASONS)]
    train_pids    = set(
        int(p)
        for gid in train_lineups['GROUP_ID']
        for p in gid.strip('-').split('-')
    )
    train_player_rows = players[
        players['PLAYER_ID'].isin(train_pids) &
        players['SEASON'].isin(TRAIN_SEASONS)
    ][PLAYER_STAT_COLS].values

    player_scaler = StandardScaler()
    player_scaler.fit(train_player_rows)

    # Apply scaler to lookup
    scaled_lookup = {
        k: player_scaler.transform(v.reshape(1, -1)).squeeze(0)
        for k, v in player_lookup.items()
    }

    # --- build per-split matrices ---
    splits = {}
    for name, seasons in [('train', TRAIN_SEASONS),
                           ('val',   VAL_SEASONS),
                           ('test',  TEST_SEASONS)]:
        sub = lineups[lineups['SEASON'].isin(seasons)].reset_index(drop=True)
        X_pl, X_yr, valid = build_player_matrices(
            sub, scaled_lookup, len(PLAYER_STAT_COLS)
        )
        y_off = sub['OFF_RATING'].values.astype(np.float32)
        y_def = sub['DEF_RATING'].values.astype(np.float32)

        # Drop lineups with missing players
        n_dropped = (~valid).sum()
        if n_dropped:
            print(f'  {name}: dropped {n_dropped} lineups (player not in lookup)')
        X_pl, X_yr = X_pl[valid], X_yr[valid]
        y_off, y_def = y_off[valid], y_def[valid]
        splits[name] = (X_pl, X_yr, y_off, y_def)

    # Scale SEASON_YEAR using training years
    all_train_years = splits['train'][1].astype(np.float32)
    yr_mean  = all_train_years.mean()
    yr_std   = all_train_years.std() + 1e-8

    def scale_yr(yr):
        return ((yr.astype(np.float32) - yr_mean) / yr_std).reshape(-1, 1)

    tensors = {}
    for name, (X_pl, X_yr, y_off, y_def) in splits.items():
        tensors[name] = (
            torch.tensor(X_pl),
            torch.tensor(scale_yr(X_yr)),
            torch.tensor(y_off),
            torch.tensor(y_def),
        )

    target_stats = {
        'off_mean': float(splits['train'][2].mean()),
        'off_std' : float(splits['train'][2].std()),
        'def_mean': float(splits['train'][3].mean()),
        'def_std' : float(splits['train'][3].std()),
    }

    return tensors, target_stats, player_scaler, (yr_mean, yr_std)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> dict:
    return {
        'MAE' : round(float(mean_absolute_error(y_true, y_pred)), 3),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        'R2'  : round(float(r2_score(y_true, y_pred)), 3),
    }


@torch.no_grad()
def predict(model, X_pl, X_yr, mean, std):
    model.eval()
    pred_n, _ = model(X_pl, X_yr)
    return pred_n.numpy() * std + mean


@torch.no_grad()
def predict_both(model, X_pl, X_yr, ts):
    model.eval()
    off_n, def_n = model(X_pl, X_yr)
    return (off_n.numpy() * ts['off_std'] + ts['off_mean'],
            def_n.numpy() * ts['def_std'] + ts['def_mean'])


def train_one(model, X_pl_tr, X_yr_tr, y_tr_n,
              X_pl_val, X_yr_val, y_val_true,
              val_mean, val_std,
              seed, ckpt_path, verbose=False):

    torch.manual_seed(seed)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_PATIENCE, factor=0.5, min_lr=1e-6
    )

    loader = DataLoader(
        TensorDataset(X_pl_tr, X_yr_tr, y_tr_n),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_mae  = float('inf')
    best_ep   = 0
    pat_ctr   = 0
    history   = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_pl_b, X_yr_b, y_b in loader:
            optimizer.zero_grad()
            pred, _ = model(X_pl_b, X_yr_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_pl_b)
        train_loss /= len(loader.dataset)

        pred_val = predict(model, X_pl_val, X_yr_val, val_mean, val_std)
        val_mae  = float(mean_absolute_error(y_val_true.numpy(), pred_val))
        scheduler.step(val_mae)

        history.append({'epoch': epoch, 'train_loss': round(train_loss, 5),
                        'val_mae': round(val_mae, 4),
                        'lr': optimizer.param_groups[0]['lr']})

        if val_mae < best_mae - 1e-4:
            best_mae = val_mae
            best_ep  = epoch
            pat_ctr  = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            pat_ctr += 1
            if pat_ctr >= ES_PATIENCE:
                if verbose:
                    print(f'    early stop ep {epoch} '
                          f'(best ep {best_ep}, val MAE {best_mae:.3f})')
                break

        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f'    ep {epoch:>4}  loss={train_loss:.4f}  '
                  f'val={val_mae:.3f}  lr={optimizer.param_groups[0]["lr"]:.1e}')

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return model, history, best_mae


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def sweep(tensors, target_stats):
    X_pl_tr, X_yr_tr, y_tr_off, y_tr_def = tensors['train']
    X_pl_val, X_yr_val, y_val_off, y_val_def = tensors['val']

    y_tr_off_n = (y_tr_off - target_stats['off_mean']) / target_stats['off_std']
    y_tr_def_n = (y_tr_def - target_stats['def_mean']) / target_stats['def_std']

    n_feats = X_pl_tr.shape[2]
    results = []

    print(f'{"Config":<30} {"OFF val":>8} {"DEF val":>8} {"Combined":>9} {"Params":>8}')
    print('-' * 70)

    for cfg in SWEEP_CONFIGS:
        ed, dr = cfg['embed_dim'], cfg['dropout']
        label  = f'embed={ed} drop={dr[0]}/{dr[1]}'

        ckpt_off = ARTIFACTS / '_sweep_off.pt'
        ckpt_def = ARTIFACTS / '_sweep_def.pt'

        model_off = DeepSetsNet(n_feats, embed_dim=ed, dropout=dr)
        model_def = DeepSetsNet(n_feats, embed_dim=ed, dropout=dr)

        torch.manual_seed(0)
        _, _, off_mae = train_one(model_off,
                                  X_pl_tr, X_yr_tr, y_tr_off_n,
                                  X_pl_val, X_yr_val, y_val_off,
                                  target_stats['off_mean'], target_stats['off_std'],
                                  seed=0, ckpt_path=ckpt_off)
        torch.manual_seed(0)
        _, _, def_mae = train_one(model_def,
                                  X_pl_tr, X_yr_tr, y_tr_def_n,
                                  X_pl_val, X_yr_val, y_val_def,
                                  target_stats['def_mean'], target_stats['def_std'],
                                  seed=0, ckpt_path=ckpt_def)

        combined = (off_mae + def_mae) / 2
        n_params  = model_off.n_params()
        results.append({**cfg, 'off_mae': off_mae, 'def_mae': def_mae,
                        'combined': combined, 'n_params': n_params})
        print(f'{label:<30} {off_mae:>8.3f} {def_mae:>8.3f} {combined:>9.3f} {n_params:>8,}')

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print('Preparing data...')
    tensors, target_stats, player_scaler, (yr_mean, yr_std) = prepare_data()

    for name, (X_pl, X_yr, y_off, y_def) in tensors.items():
        print(f'  {name}: {len(X_pl)} lineups  '
              f'OFF mean={y_off.mean():.1f}  DEF mean={y_def.mean():.1f}')

    n_feats = tensors['train'][0].shape[2]
    print(f'  Player features per slot: {n_feats}\n')

    # Persist scalers needed for inference
    with open(ARTIFACTS / 'ds_target_stats.pkl', 'wb') as f:
        pickle.dump(target_stats, f)
    with open(ARTIFACTS / 'ds_player_scaler.pkl', 'wb') as f:
        pickle.dump({'scaler': player_scaler, 'yr_mean': yr_mean, 'yr_std': yr_std}, f)

    # --- grid search ---
    print('=== Grid search (1 seed each) ===')
    sweep_df = sweep(tensors, target_stats)
    sweep_df.to_csv(ARTIFACTS / 'ds_sweep_results.csv', index=False)

    best_cfg = sweep_df.loc[sweep_df['combined'].idxmin()]
    best_ed  = int(best_cfg['embed_dim'])
    best_dr  = tuple(best_cfg['dropout'])
    print(f'\nBest config: embed_dim={best_ed}  dropout={best_dr}  '
          f'combined MAE={best_cfg["combined"]:.3f}')

    # --- train best config with N seeds ---
    X_pl_tr, X_yr_tr, y_tr_off, y_tr_def = tensors['train']
    X_pl_val, X_yr_val, y_val_off, y_val_def = tensors['val']

    y_tr_off_n = (y_tr_off - target_stats['off_mean']) / target_stats['off_std']
    y_tr_def_n = (y_tr_def - target_stats['def_mean']) / target_stats['def_std']

    off_models, def_models = [], []
    off_val_maes, def_val_maes = [], []

    for target, y_tr_n, y_val, ts_mean, ts_std, model_list, mae_list, label in [
        ('OFF', y_tr_off_n, y_val_off,
         target_stats['off_mean'], target_stats['off_std'],
         off_models, off_val_maes, 'OFF_RATING'),
        ('DEF', y_tr_def_n, y_val_def,
         target_stats['def_mean'], target_stats['def_std'],
         def_models, def_val_maes, 'DEF_RATING'),
    ]:
        print(f'\n=== Training {label} ({N_SEEDS} seeds) ===')
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            model = DeepSetsNet(n_feats, embed_dim=best_ed, dropout=best_dr)
            ckpt  = ARTIFACTS / f'ds_{target.lower()}_seed{seed}.pt'
            verbose = (seed == 0)
            model, hist, val_mae = train_one(
                model,
                X_pl_tr, X_yr_tr, y_tr_n,
                X_pl_val, X_yr_val, y_val,
                ts_mean, ts_std,
                seed=seed, ckpt_path=ckpt, verbose=verbose,
            )
            if seed == 0:
                pd.DataFrame(hist).to_csv(
                    ARTIFACTS / f'ds_history_seed0_{target.lower()}.csv', index=False
                )
            model_list.append(model)
            mae_list.append(val_mae)
            print(f'  seed {seed}: val MAE = {val_mae:.3f}')

        ens_pred = np.mean(
            [predict(m, X_pl_val, X_yr_val, ts_mean, ts_std) for m in model_list], axis=0
        )
        ens_mae = float(mean_absolute_error(y_val.numpy(), ens_pred))
        print(f'  ensemble: val MAE = {ens_mae:.3f}  '
              f'(vs best single: {min(mae_list):.3f})')

    # --- final evaluation ---
    results = []
    for split_name, (X_pl, X_yr, y_off, y_def) in tensors.items():
        p_off = np.mean(
            [predict(m, X_pl, X_yr,
                     target_stats['off_mean'], target_stats['off_std'])
             for m in off_models], axis=0
        )
        p_def = np.mean(
            [predict(m, X_pl, X_yr,
                     target_stats['def_mean'], target_stats['def_std'])
             for m in def_models], axis=0
        )
        for tgt, y_true, pred in [('OFF_RATING', y_off, p_off),
                                   ('DEF_RATING', y_def, p_def)]:
            results.append({'model': 'DeepSets', 'target': tgt,
                            'split': split_name,
                            **compute_metrics(y_true.numpy(), pred)})

    df_ds = pd.DataFrame(results)

    # Merge with all prior results
    baseline_path = ARTIFACTS / 'baseline_results.csv'
    if baseline_path.exists():
        existing = pd.read_csv(baseline_path)
        existing = existing[existing['model'] != 'DeepSets']
        combined = pd.concat([existing, df_ds], ignore_index=True)
    else:
        combined = df_ds
    combined.to_csv(baseline_path, index=False)

    # Print final table
    order = {'GlobalMean': 0, 'MeanPlayerRating': 1,
             'Ridge': 2, 'NeuralNet': 3, 'DeepSets': 4}
    combined['_ord'] = combined['model'].map(order)

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

    print('\n=== DeepSets vs Ridge (val MAE) ===')
    for target in ['OFF_RATING', 'DEF_RATING']:
        def get(mdl):
            return combined[(combined['model'] == mdl) &
                            (combined['target'] == target) &
                            (combined['split'] == 'val')]['MAE'].values[0]
        r, d = get('Ridge'), get('DeepSets')
        sign = '+' if r - d >= 0 else ''
        print(f'  {target}: Ridge {r:.3f} -> DeepSets {d:.3f}  ({sign}{r-d:.3f})')


if __name__ == '__main__':
    main()
