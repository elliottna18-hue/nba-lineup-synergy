"""
SynergyNet: attention-based lineup rating prediction.

Why attention captures synergy that mean/max pooling misses:
  In Deep Sets (mean+max), each player's final embedding is the same regardless
  of who else is in the lineup. A point guard with high AST_PCT is encoded
  identically whether surrounded by four floor spacers or four non-shooters.
  Self-attention fixes this — each player's representation is updated by
  attending to the other four, so the point guard's embedding becomes
  "point guard who has shooters around him" vs. "point guard who doesn't."
  This is the mechanism through which spacing and role complementarity
  can emerge from data.

Architecture (per target):
    phi  : Linear(21->64) + LN + ReLU + Dropout(0.3)   # per-player encoder
    attn : MultiheadAttention(64, heads=4) + residual + LN   # player interactions
    ff   : Linear(64->128) + ReLU + Dropout(0.2) + Linear(128->64) + LN  # refine
    pool : mean + max  ->  (128,)
    ctx  : concat SEASON_YEAR  ->  (129,)
    head : Linear(129->64) + ReLU + Dropout(0.2) + Linear(64->32) + ReLU + Linear(32->1)

Also benchmarks the synergy interaction features (syn_*) added to build_features.py
against Ridge and the previous NeuralNet/DeepSets, to show the dual contribution.

Outputs (artifacts/):
  syn_off_seed{k}.pt / syn_def_seed{k}.pt
  syn_target_stats.pkl
  syn_player_scaler.pkl
  baseline_results.csv  (updated with SynergyNet row)
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CLEAN     = Path('data/clean')
FEATURES  = Path('data/features')
DATA      = Path('data/model_ready')
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SynergyNet(nn.Module):
    """
    Transformer encoder block over the 5 lineup members.
    Self-attention lets each player's embedding be conditioned on who else
    is in the lineup — the mechanism through which spacing and role
    complementarity can be learned from data.
    """
    def __init__(self, n_feats: int, embed_dim: int = 64,
                 n_heads: int = 4, dropout: tuple = (0.3, 0.2)):
        super().__init__()
        d0, d1 = dropout

        # Per-player encoder (shared weights)
        self.phi = nn.Sequential(
            nn.Linear(n_feats, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(d0),
        )

        # Self-attention: each player attends to the other four
        self.attn      = nn.MultiheadAttention(embed_dim, n_heads,
                                               dropout=d1, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Position-wise feed-forward (refines each player's context-aware embedding)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(d1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

        # Aggregate + era context + per-target heads
        agg_dim = 2 * embed_dim + 1   # mean + max + SEASON_YEAR

        def head():
            return nn.Sequential(
                nn.Linear(agg_dim, embed_dim), nn.ReLU(), nn.Dropout(d1),
                nn.Linear(embed_dim, 32),      nn.ReLU(),
                nn.Linear(32, 1),
            )

        self.off_head = head()
        self.def_head = head()

    def forward(self, x_players: torch.Tensor, x_season: torch.Tensor):
        # x_players : (B, 5, n_feats)
        # x_season  : (B, 1)

        h = self.phi(x_players)                        # (B, 5, embed)

        # Self-attention with residual + LN
        attn_out, _ = self.attn(h, h, h)
        h = self.attn_norm(h + attn_out)               # (B, 5, embed)

        # Feed-forward with residual + LN
        h = self.ff_norm(h + self.ff(h))               # (B, 5, embed)

        # Pool: mean + max  ->  concat era
        agg = torch.cat([h.mean(1), h.max(1).values, x_season], dim=1)

        return self.off_head(agg).squeeze(-1), self.def_head(agg).squeeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_player_matrices(lineups, player_lookup, n_feats):
    X_pl = np.zeros((len(lineups), N_PLAYERS, n_feats), dtype=np.float32)
    X_yr = np.zeros(len(lineups), dtype=np.int32)
    valid = np.zeros(len(lineups), dtype=bool)

    for i, (_, row) in enumerate(lineups.iterrows()):
        pids = [int(p) for p in row['GROUP_ID'].strip('-').split('-')]
        X_yr[i] = int(row['SEASON'][:4])
        feats = []
        for pid in pids:
            key = (pid, row['SEASON'])
            if key in player_lookup:
                feats.append(player_lookup[key])
            else:
                break
        else:
            X_pl[i] = np.array(feats)
            valid[i] = True

    return X_pl, X_yr, valid


def prepare_data():
    lineups = pd.read_parquet(CLEAN / 'lineups.parquet')
    players = pd.read_parquet(CLEAN / 'players.parquet')

    player_lookup = {
        (row.PLAYER_ID, row.SEASON): row[PLAYER_STAT_COLS].values.astype(np.float32)
        for _, row in players.iterrows()
    }

    # Fit player scaler on training-season players
    train_lup  = lineups[lineups['SEASON'].isin(TRAIN_SEASONS)]
    train_pids = set(int(p) for gid in train_lup['GROUP_ID']
                     for p in gid.strip('-').split('-'))
    train_rows = players[
        players['PLAYER_ID'].isin(train_pids) &
        players['SEASON'].isin(TRAIN_SEASONS)
    ][PLAYER_STAT_COLS].values

    player_scaler = StandardScaler().fit(train_rows)
    scaled_lookup = {
        k: player_scaler.transform(v.reshape(1, -1)).squeeze(0)
        for k, v in player_lookup.items()
    }

    splits = {}
    for name, seasons in [('train', TRAIN_SEASONS),
                           ('val',   VAL_SEASONS),
                           ('test',  TEST_SEASONS)]:
        sub = lineups[lineups['SEASON'].isin(seasons)].reset_index(drop=True)
        X_pl, X_yr, valid = build_player_matrices(sub, scaled_lookup,
                                                  len(PLAYER_STAT_COLS))
        dropped = (~valid).sum()
        if dropped:
            print(f'  {name}: dropped {dropped} lineups')
        splits[name] = (
            X_pl[valid], X_yr[valid],
            sub['OFF_RATING'].values.astype(np.float32)[valid],
            sub['DEF_RATING'].values.astype(np.float32)[valid],
        )

    # Scale SEASON_YEAR
    yr_mean = splits['train'][1].astype(np.float32).mean()
    yr_std  = splits['train'][1].astype(np.float32).std() + 1e-8

    def to_tensors(X_pl, X_yr, y_off, y_def):
        yr_s = ((X_yr.astype(np.float32) - yr_mean) / yr_std).reshape(-1, 1)
        return (torch.tensor(X_pl), torch.tensor(yr_s),
                torch.tensor(y_off), torch.tensor(y_def))

    tensors = {name: to_tensors(*vals) for name, vals in splits.items()}

    target_stats = {
        'off_mean': float(splits['train'][2].mean()),
        'off_std' : float(splits['train'][2].std()),
        'def_mean': float(splits['train'][3].mean()),
        'def_std' : float(splits['train'][3].std()),
    }

    return tensors, target_stats, player_scaler, (yr_mean, yr_std)


# ---------------------------------------------------------------------------
# Training helpers (same pattern as train_deepsets.py)
# ---------------------------------------------------------------------------

def metrics(y_true, y_pred):
    return {
        'MAE' : round(float(mean_absolute_error(y_true, y_pred)), 3),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        'R2'  : round(float(r2_score(y_true, y_pred)), 3),
    }


@torch.no_grad()
def predict(model, X_pl, X_yr, mean, std):
    model.eval()
    out, _ = model(X_pl, X_yr)
    return out.numpy() * std + mean


def train_one(model, X_pl_tr, X_yr_tr, y_tr_n,
              X_pl_val, X_yr_val, y_val, val_mean, val_std,
              seed, ckpt, verbose=False):

    torch.manual_seed(seed)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, 'min', patience=LR_PATIENCE, factor=0.5, min_lr=1e-6)
    crit = nn.L1Loss()
    loader = DataLoader(TensorDataset(X_pl_tr, X_yr_tr, y_tr_n),
                        batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    best_mae, best_ep, pat = float('inf'), 0, 0
    history = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        tl = 0.0
        for Xp, Xy, yb in loader:
            opt.zero_grad()
            out, _ = model(Xp, Xy)
            loss = crit(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(Xp)
        tl /= len(loader.dataset)

        pv  = predict(model, X_pl_val, X_yr_val, val_mean, val_std)
        vm  = float(mean_absolute_error(y_val.numpy(), pv))
        sched.step(vm)
        history.append({'epoch': ep, 'train_loss': round(tl, 5),
                        'val_mae': round(vm, 4),
                        'lr': opt.param_groups[0]['lr']})

        if vm < best_mae - 1e-4:
            best_mae, best_ep, pat = vm, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            pat += 1
            if pat >= ES_PATIENCE:
                if verbose:
                    print(f'    early stop ep {ep} '
                          f'(best ep {best_ep}, MAE {best_mae:.3f})')
                break

        if verbose and (ep % 50 == 0 or ep == 1):
            print(f'    ep {ep:>4}  loss={tl:.4f}  val={vm:.3f}  '
                  f'lr={opt.param_groups[0]["lr"]:.1e}')

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    return model, history, best_mae


def train_ensemble(label, tensors, target_stats, ts_key, n_feats):
    X_pl_tr, X_yr_tr, y_tr_off, y_tr_def = tensors['train']
    X_pl_val, X_yr_val, y_val_off, y_val_def = tensors['val']

    is_off  = 'OFF' in ts_key
    y_tr    = y_tr_off  if is_off else y_tr_def
    y_val   = y_val_off if is_off else y_val_def
    ts_mean = target_stats['off_mean' if is_off else 'def_mean']
    ts_std  = target_stats['off_std'  if is_off else 'def_std']
    y_tr_n  = (y_tr - ts_mean) / ts_std

    models, maes = [], []
    print(f'  {label} ({N_SEEDS} seeds):')
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        model = SynergyNet(n_feats)
        ckpt  = ARTIFACTS / f'syn_{label.lower()[:3]}_seed{seed}.pt'
        model, hist, vm = train_one(
            model, X_pl_tr, X_yr_tr, y_tr_n,
            X_pl_val, X_yr_val, y_val, ts_mean, ts_std,
            seed=seed, ckpt=ckpt, verbose=(seed == 0),
        )
        if seed == 0:
            pd.DataFrame(hist).to_csv(
                ARTIFACTS / f'syn_history_{label.lower()[:3]}.csv', index=False)
        models.append(model)
        maes.append(vm)
        print(f'    seed {seed}: val MAE = {vm:.3f}')

    ens = np.mean([predict(m, X_pl_val, X_yr_val, ts_mean, ts_std)
                   for m in models], axis=0)
    ens_mae = float(mean_absolute_error(y_val.numpy(), ens))
    print(f'    ensemble: val MAE = {ens_mae:.3f}  (best single: {min(maes):.3f})')
    return models


# ---------------------------------------------------------------------------
# Ridge with synergy features (shows feature contribution separately)
# ---------------------------------------------------------------------------

def benchmark_ridge_with_syn_features(tensors_dict):
    """Re-run Ridge on X that now includes syn_* features."""
    results = []
    for target in ['OFF_RATING', 'DEF_RATING']:
        Xtr = pd.read_parquet(DATA / 'X_train.parquet').values
        Xvl = pd.read_parquet(DATA / 'X_val.parquet').values
        Xte = pd.read_parquet(DATA / 'X_test.parquet').values
        ytr = pd.read_parquet(DATA / 'y_train.parquet')[target].values
        yvl = pd.read_parquet(DATA / 'y_val.parquet')[target].values
        yte = pd.read_parquet(DATA / 'y_test.parquet')[target].values

        r = RidgeCV(alphas=[0.1, 1, 10, 100, 500, 1000], cv=5)
        r.fit(Xtr, ytr)
        for split, X, y in [('train', Xtr, ytr), ('val', Xvl, yvl), ('test', Xte, yte)]:
            results.append({'model': 'Ridge+SynFeats', 'target': target,
                            'split': split, **metrics(y, r.predict(X))})
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print('Preparing player-matrix data...')
    tensors, target_stats, player_scaler, (yr_mean, yr_std) = prepare_data()
    for name, (X_pl, X_yr, y_off, y_def) in tensors.items():
        print(f'  {name}: {len(X_pl)} lineups  '
              f'OFF={y_off.mean():.1f}  DEF={y_def.mean():.1f}')

    n_feats = tensors['train'][0].shape[2]
    model_ex = SynergyNet(n_feats)
    print(f'\nSynergyNet params: {model_ex.n_params():,}\n')

    with open(ARTIFACTS / 'syn_target_stats.pkl', 'wb') as f:
        pickle.dump(target_stats, f)
    with open(ARTIFACTS / 'syn_player_scaler.pkl', 'wb') as f:
        pickle.dump({'scaler': player_scaler, 'yr_mean': yr_mean, 'yr_std': yr_std}, f)

    print('=== Training SynergyNet ===')
    off_models = train_ensemble('OFF_RATING', tensors, target_stats,
                                'OFF_RATING', n_feats)
    def_models = train_ensemble('DEF_RATING', tensors, target_stats,
                                'DEF_RATING', n_feats)

    # Final evaluation
    results = []
    for sname, (X_pl, X_yr, y_off, y_def) in tensors.items():
        p_off = np.mean([predict(m, X_pl, X_yr,
                                 target_stats['off_mean'], target_stats['off_std'])
                         for m in off_models], axis=0)
        p_def = np.mean([predict(m, X_pl, X_yr,
                                 target_stats['def_mean'], target_stats['def_std'])
                         for m in def_models], axis=0)
        for tgt, y, p in [('OFF_RATING', y_off, p_off), ('DEF_RATING', y_def, p_def)]:
            results.append({'model': 'SynergyNet', 'target': tgt,
                            'split': sname, **metrics(y.numpy(), p)})

    df_syn = pd.DataFrame(results)

    # Ridge re-run with new syn_ features
    print('\n=== Ridge with synergy features ===')
    df_ridge_syn = benchmark_ridge_with_syn_features(tensors)
    for _, r in df_ridge_syn[df_ridge_syn['split'] == 'val'].iterrows():
        print(f'  {r["target"]}: MAE={r["MAE"]:.3f}  R2={r["R2"]:.3f}')

    # Merge everything
    baseline_path = ARTIFACTS / 'baseline_results.csv'
    if baseline_path.exists():
        existing = pd.read_csv(baseline_path)
        existing = existing[~existing['model'].isin(['SynergyNet', 'Ridge+SynFeats'])]
        combined = pd.concat([existing, df_syn, df_ridge_syn], ignore_index=True)
    else:
        combined = pd.concat([df_syn, df_ridge_syn], ignore_index=True)
    combined.to_csv(baseline_path, index=False)

    order = {'GlobalMean': 0, 'MeanPlayerRating': 1, 'Ridge': 2,
             'Ridge+SynFeats': 3, 'NeuralNet': 4, 'DeepSets': 5, 'SynergyNet': 6}
    combined['_ord'] = combined['model'].map(order)

    for sname in ['val', 'test']:
        header = (f"{'Model':<20} {'Target':<12} {'Split':<6}"
                  f"  {'MAE':>6}  {'RMSE':>6}  {'R2':>6}")
        print(f'\n=== {sname.capitalize()} Results ===')
        print(header)
        print('-' * len(header))
        for _, r in (combined[combined['split'] == sname]
                     .sort_values(['target', '_ord'])).iterrows():
            print(f"{r['model']:<20} {r['target']:<12} {r['split']:<6}"
                  f"  {r['MAE']:>6.3f}  {r['RMSE']:>6.3f}  {r['R2']:>6.3f}")

    print('\n=== SynergyNet vs Ridge (val MAE) ===')
    for tgt in ['OFF_RATING', 'DEF_RATING']:
        def get(mdl):
            return combined[(combined['model'] == mdl) &
                            (combined['target'] == tgt) &
                            (combined['split'] == 'val')]['MAE'].values[0]
        r, s = get('Ridge'), get('SynergyNet')
        sign = '+' if r - s >= 0 else ''
        print(f'  {tgt}: Ridge {r:.3f} -> SynergyNet {s:.3f}  ({sign}{r-s:.3f})')


if __name__ == '__main__':
    main()
