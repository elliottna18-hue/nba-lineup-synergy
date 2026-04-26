"""
SynergyNetV2: targeted improvements for offensive synergy detection.

Changes vs. SynergyNet (v1):
  1. Role features added to player input (23-dim instead of 21):
       spacing_score  = FG3A  * TS_PCT   (quality 3-pt threat)
       creation_score = USG_PCT * AST_PCT (creates and distributes)
     Gives attention clear "spacer vs. creator" signals to interact on,
     rather than forcing it to discover roles from raw counting stats.

  2. Explicit creator x spacing interaction injected at the prediction head:
       max_creator       = max(creation_score across 5 players)
       avg_spacing       = mean(spacing_score across 5 players)
       creator_x_spacing = max_creator * avg_spacing
     Directly encodes the playmaker-spacer hypothesis: a great creator
     surrounded by floor spacers should score above their individual sum.

  3. Two Transformer encoder layers (was one):
     More capacity to learn second-order role interactions.

Outputs (artifacts/):
  syn_v2_off_seed{k}.pt / syn_v2_def_seed{k}.pt
  syn_v2_target_stats.pkl
  syn_v2_player_scaler.pkl
  baseline_results.csv  (updated with SynergyNetV2 row)
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
# Indices used to extract role features inside forward()
IDX_FG3A    = PLAYER_STAT_COLS.index('FG3A')
IDX_TS_PCT  = PLAYER_STAT_COLS.index('TS_PCT')
IDX_USG_PCT = PLAYER_STAT_COLS.index('USG_PCT')
IDX_AST_PCT = PLAYER_STAT_COLS.index('AST_PCT')
# Role feature columns appended after the 21 base stats
ROLE_SPACING_IDX  = 21   # spacing_score  = FG3A  * TS_PCT
ROLE_CREATION_IDX = 22   # creation_score = USG_PCT * AST_PCT
N_PLAYER_FEATS    = 23   # 21 base + 2 role

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

class SynergyNetV2(nn.Module):
    """
    Two-layer Transformer encoder with explicit creator × spacer interactions.

    The key addition over v1:
      - spacing_score / creation_score as input features → attention has
        clear role signals to align playmakers with spacers
      - max_creator × avg_spacing injected directly into the prediction head
        → the model can't miss the interaction even if attention doesn't
        discover it on its own
    """
    def __init__(self, n_feats: int = N_PLAYER_FEATS, embed_dim: int = 64,
                 n_heads: int = 4, dropout: tuple = (0.3, 0.2)):
        super().__init__()
        d0, d1 = dropout

        # Per-player encoder
        self.phi = nn.Sequential(
            nn.Linear(n_feats, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(d0),
        )

        # --- Two Transformer encoder layers ---
        self.attn1      = nn.MultiheadAttention(embed_dim, n_heads,
                                                dropout=d1, batch_first=True)
        self.attn1_norm = nn.LayerNorm(embed_dim)
        self.ff1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(),
            nn.Dropout(d1), nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ff1_norm = nn.LayerNorm(embed_dim)

        self.attn2      = nn.MultiheadAttention(embed_dim, n_heads,
                                                dropout=d1, batch_first=True)
        self.attn2_norm = nn.LayerNorm(embed_dim)
        self.ff2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(),
            nn.Dropout(d1), nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ff2_norm = nn.LayerNorm(embed_dim)

        # head input: mean(64) + max(64) + season(1) + interactions(3) = 132
        agg_dim = 2 * embed_dim + 1 + 3

        def head():
            return nn.Sequential(
                nn.Linear(agg_dim, embed_dim), nn.ReLU(), nn.Dropout(d1),
                nn.Linear(embed_dim, 32),      nn.ReLU(),
                nn.Linear(32, 1),
            )

        self.off_head = head()
        self.def_head = head()

    def forward(self, x_players: torch.Tensor, x_season: torch.Tensor):
        # x_players : (B, 5, 23)   x_season : (B, 1)

        # Extract scaled role features before encoding
        spacing  = x_players[:, :, ROLE_SPACING_IDX]    # (B, 5)
        creation = x_players[:, :, ROLE_CREATION_IDX]   # (B, 5)

        h = self.phi(x_players)                          # (B, 5, embed)

        # Layer 1
        attn_out, _ = self.attn1(h, h, h)
        h = self.attn1_norm(h + attn_out)
        h = self.ff1_norm(h + self.ff1(h))

        # Layer 2
        attn_out, _ = self.attn2(h, h, h)
        h = self.attn2_norm(h + attn_out)
        h = self.ff2_norm(h + self.ff2(h))

        # Explicit creator × spacer interactions
        max_creator       = creation.max(dim=1).values           # (B,)
        avg_spacing       = spacing.mean(dim=1)                   # (B,)
        creator_x_spacing = max_creator * avg_spacing             # (B,)

        interact = torch.stack(
            [max_creator, avg_spacing, creator_x_spacing], dim=1  # (B, 3)
        )

        agg = torch.cat(
            [h.mean(1), h.max(1).values, x_season, interact], dim=1  # (B, 132)
        )

        return self.off_head(agg).squeeze(-1), self.def_head(agg).squeeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Data — adds role features to player matrix
# ---------------------------------------------------------------------------

def add_role_features(raw_stats: np.ndarray) -> np.ndarray:
    """
    Appends 2 role features to a (N, 21) raw stat matrix → (N, 23).
    Computed before scaling so the product has physical meaning.
    """
    fg3a    = raw_stats[:, IDX_FG3A]
    ts_pct  = raw_stats[:, IDX_TS_PCT]
    usg_pct = raw_stats[:, IDX_USG_PCT]
    ast_pct = raw_stats[:, IDX_AST_PCT]

    spacing_score  = (fg3a  * ts_pct ).reshape(-1, 1)   # quality 3-pt threat
    creation_score = (usg_pct * ast_pct).reshape(-1, 1)  # creates + distributes

    return np.hstack([raw_stats, spacing_score, creation_score])


def build_player_matrices(lineups, player_lookup, n_feats):
    X_pl  = np.zeros((len(lineups), N_PLAYERS, n_feats), dtype=np.float32)
    X_yr  = np.zeros(len(lineups), dtype=np.int32)
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

    # Raw 21-feature lookup (before role features or scaling)
    raw_lookup = {
        (row.PLAYER_ID, row.SEASON): row[PLAYER_STAT_COLS].values.astype(np.float32)
        for _, row in players.iterrows()
    }

    # Fit player scaler on 23-feature training vectors
    train_lup  = lineups[lineups['SEASON'].isin(TRAIN_SEASONS)]
    train_pids = set(int(p) for gid in train_lup['GROUP_ID']
                     for p in gid.strip('-').split('-'))
    train_raw = players[
        players['PLAYER_ID'].isin(train_pids) &
        players['SEASON'].isin(TRAIN_SEASONS)
    ][PLAYER_STAT_COLS].values.astype(np.float32)

    train_with_roles = add_role_features(train_raw)          # (N, 23)
    player_scaler = StandardScaler().fit(train_with_roles)

    # Build scaled 23-feature lookup for every player
    scaled_lookup = {}
    for k, raw in raw_lookup.items():
        with_roles = add_role_features(raw.reshape(1, -1)).squeeze(0)  # (23,)
        scaled_lookup[k] = player_scaler.transform(
            with_roles.reshape(1, -1)).squeeze(0)

    splits = {}
    for name, seasons in [('train', TRAIN_SEASONS),
                           ('val',   VAL_SEASONS),
                           ('test',  TEST_SEASONS)]:
        sub = lineups[lineups['SEASON'].isin(seasons)].reset_index(drop=True)
        X_pl, X_yr, valid = build_player_matrices(sub, scaled_lookup, N_PLAYER_FEATS)
        dropped = (~valid).sum()
        if dropped:
            print(f'  {name}: dropped {dropped} lineups')
        splits[name] = (
            X_pl[valid], X_yr[valid],
            sub['OFF_RATING'].values.astype(np.float32)[valid],
            sub['DEF_RATING'].values.astype(np.float32)[valid],
        )

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
# Training helpers (identical to v1)
# ---------------------------------------------------------------------------

def metrics(y_true, y_pred):
    return {
        'MAE' : round(float(mean_absolute_error(y_true, y_pred)), 3),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        'R2'  : round(float(r2_score(y_true, y_pred)), 3),
    }


@torch.no_grad()
def predict(model, X_pl, X_yr, mean, std, use_off=True):
    model.eval()
    out_off, out_def = model(X_pl, X_yr)
    raw = out_off if use_off else out_def
    return raw.numpy() * std + mean


def train_one(model, X_pl_tr, X_yr_tr, y_tr_n,
              X_pl_val, X_yr_val, y_val, val_mean, val_std,
              seed, ckpt, use_off=True, verbose=False):

    torch.manual_seed(seed)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, 'min', patience=LR_PATIENCE, factor=0.5, min_lr=1e-6)
    crit  = nn.L1Loss()
    loader = DataLoader(
        TensorDataset(X_pl_tr, X_yr_tr, y_tr_n),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed))

    best_mae, best_ep, pat = float('inf'), 0, 0
    history = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        tl = 0.0
        for Xp, Xy, yb in loader:
            opt.zero_grad()
            out_off, out_def = model(Xp, Xy)
            out = out_off if use_off else out_def
            loss = crit(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(Xp)
        tl /= len(loader.dataset)

        pv  = predict(model, X_pl_val, X_yr_val, val_mean, val_std, use_off)
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


def train_ensemble(label, tensors, target_stats, use_off, n_feats):
    X_pl_tr, X_yr_tr, y_tr_off, y_tr_def = tensors['train']
    X_pl_val, X_yr_val, y_val_off, y_val_def = tensors['val']

    y_tr    = y_tr_off  if use_off else y_tr_def
    y_val   = y_val_off if use_off else y_val_def
    ts_mean = target_stats['off_mean' if use_off else 'def_mean']
    ts_std  = target_stats['off_std'  if use_off else 'def_std']
    y_tr_n  = (y_tr - ts_mean) / ts_std

    tag = 'OFF' if use_off else 'DEF'
    models, maes = [], []
    print(f'  {label} {tag} ({N_SEEDS} seeds):')
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        model = SynergyNetV2(n_feats)
        ckpt  = ARTIFACTS / f'syn_v2_{tag.lower()}_seed{seed}.pt'
        model, hist, vm = train_one(
            model, X_pl_tr, X_yr_tr, y_tr_n,
            X_pl_val, X_yr_val, y_val, ts_mean, ts_std,
            seed=seed, ckpt=ckpt, use_off=use_off, verbose=(seed == 0),
        )
        if seed == 0:
            pd.DataFrame(hist).to_csv(
                ARTIFACTS / f'syn_v2_history_{tag.lower()}.csv', index=False)
        models.append(model)
        maes.append(vm)
        print(f'    seed {seed}: val MAE = {vm:.3f}')

    ens = np.mean([predict(m, X_pl_val, X_yr_val, ts_mean, ts_std, use_off)
                   for m in models], axis=0)
    ens_mae = float(mean_absolute_error(y_val.numpy(), ens))
    print(f'    ensemble: val MAE = {ens_mae:.3f}  (best single: {min(maes):.3f})')
    return models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print('Preparing player-matrix data (23 features: 21 base + 2 role)...')
    tensors, target_stats, player_scaler, (yr_mean, yr_std) = prepare_data()
    for name, (X_pl, X_yr, y_off, y_def) in tensors.items():
        print(f'  {name}: {len(X_pl)} lineups  '
              f'player_feats={X_pl.shape[2]}  '
              f'OFF={y_off.mean():.1f}  DEF={y_def.mean():.1f}')

    n_feats = tensors['train'][0].shape[2]
    model_ex = SynergyNetV2(n_feats)
    print(f'\nSynergyNetV2 params: {model_ex.n_params():,}\n')

    with open(ARTIFACTS / 'syn_v2_target_stats.pkl', 'wb') as f:
        pickle.dump(target_stats, f)
    with open(ARTIFACTS / 'syn_v2_player_scaler.pkl', 'wb') as f:
        pickle.dump({'scaler': player_scaler, 'yr_mean': yr_mean, 'yr_std': yr_std}, f)

    print('=== Training SynergyNetV2 ===')
    off_models = train_ensemble('SynergyNetV2', tensors, target_stats,
                                use_off=True,  n_feats=n_feats)
    def_models = train_ensemble('SynergyNetV2', tensors, target_stats,
                                use_off=False, n_feats=n_feats)

    # Full evaluation
    results = []
    for sname, (X_pl, X_yr, y_off, y_def) in tensors.items():
        p_off = np.mean([predict(m, X_pl, X_yr,
                                 target_stats['off_mean'], target_stats['off_std'], True)
                         for m in off_models], axis=0)
        p_def = np.mean([predict(m, X_pl, X_yr,
                                 target_stats['def_mean'], target_stats['def_std'], False)
                         for m in def_models], axis=0)
        for tgt, y, p in [('OFF_RATING', y_off, p_off), ('DEF_RATING', y_def, p_def)]:
            results.append({'model': 'SynergyNetV2', 'target': tgt,
                            'split': sname, **metrics(y.numpy(), p)})

    df_v2 = pd.DataFrame(results)

    # Merge into baseline_results.csv
    baseline_path = ARTIFACTS / 'baseline_results.csv'
    if baseline_path.exists():
        existing = pd.read_csv(baseline_path)
        existing = existing[existing['model'] != 'SynergyNetV2']
        combined = pd.concat([existing, df_v2], ignore_index=True)
    else:
        combined = df_v2
    combined.to_csv(baseline_path, index=False)

    # Side-by-side comparison
    print('\n' + '=' * 65)
    print('COMPARISON: SynergyNet (v1) vs SynergyNetV2')
    print('=' * 65)
    order = {'Ridge': 0, 'NeuralNet': 1, 'DeepSets': 2,
             'SynergyNet': 3, 'SynergyNetV2': 4}
    for split in ['val', 'test']:
        print(f'\n  {split.upper()} SET')
        print(f'  {"Model":<16} {"OFF MAE":>8} {"DEF MAE":>8} {"NET MAE":>9} {"OFF R2":>7} {"DEF R2":>7}')
        print('  ' + '-' * 60)
        sub = combined[combined['split'] == split]
        for mdl in ['Ridge', 'NeuralNet', 'DeepSets', 'SynergyNet', 'SynergyNetV2']:
            rows = sub[sub['model'] == mdl]
            if rows.empty:
                continue
            off = rows[rows['target'] == 'OFF_RATING'].iloc[0]
            def_ = rows[rows['target'] == 'DEF_RATING'].iloc[0]
            print(f'  {mdl:<16} {off["MAE"]:>8.3f} {def_["MAE"]:>8.3f} '
                  f'{off["MAE"]+def_["MAE"]:>9.3f} {off["R2"]:>7.3f} {def_["R2"]:>7.3f}')

    print('\nDeltas (v2 - v1, negative = improvement):')
    for split in ['val', 'test']:
        sub = combined[combined['split'] == split]
        v1_off = sub[(sub['model']=='SynergyNet') & (sub['target']=='OFF_RATING')]['MAE'].values
        v1_def = sub[(sub['model']=='SynergyNet') & (sub['target']=='DEF_RATING')]['MAE'].values
        v2_off = sub[(sub['model']=='SynergyNetV2') & (sub['target']=='OFF_RATING')]['MAE'].values
        v2_def = sub[(sub['model']=='SynergyNetV2') & (sub['target']=='DEF_RATING')]['MAE'].values
        if len(v1_off) and len(v2_off):
            print(f'  {split}: OFF {v2_off[0]-v1_off[0]:+.3f}  '
                  f'DEF {v2_def[0]-v1_def[0]:+.3f}')


if __name__ == '__main__':
    main()
