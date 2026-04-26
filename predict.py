"""
Predict OFF_RATING, DEF_RATING, and NET_RATING for any user-specified 5-man lineup.

Usage:
    python predict.py "LeBron James" "Anthony Davis" "Austin Reaves" "D'Angelo Russell" "Jarred Vanderbilt"
    python predict.py --season 2023-24 "Stephen Curry" "Klay Thompson" "Draymond Green" "Andrew Wiggins" "Kevon Looney"

If no season is specified, uses the most recent season available for each player.

Models used:
    Ridge       -- linear baseline (92 mean/std/min/max features)
    SynergyNet  -- attention model (5x21 player matrix, best for DEF)
    NeuralNet   -- MLP ensemble   (92 features, best for OFF)
    Ensemble    -- avg(SynergyNet_DEF, NeuralNet_OFF)
"""

import sys
import pickle
import argparse
import difflib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# Force UTF-8 output on Windows so player names with accents print correctly
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CLEAN     = Path('data/clean')
DATA      = Path('data/model_ready')
ARTIFACTS = Path('artifacts')

PLAYER_STAT_COLS = [
    'OFF_RATING', 'DEF_RATING', 'USG_PCT', 'AST_PCT', 'AST_TO', 'PIE',
    'EFG_PCT', 'TS_PCT', 'OREB_PCT', 'DREB_PCT', 'TM_TOV_PCT',
    'PTS', 'FGA', 'FG3A', 'FTA', 'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB',
]

N_SEEDS = 5


# ── Model definitions (must match training code) ───────────────────────────────

class SynergyNet(nn.Module):
    def __init__(self, n_feats=21, embed_dim=64, n_heads=4, dropout=(0.3, 0.2)):
        super().__init__()
        d0, d1 = dropout
        self.phi = nn.Sequential(
            nn.Linear(n_feats, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(d0))
        self.attn      = nn.MultiheadAttention(embed_dim, n_heads, dropout=d1, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ff        = nn.Sequential(nn.Linear(embed_dim, embed_dim*2), nn.ReLU(),
                                       nn.Dropout(d1), nn.Linear(embed_dim*2, embed_dim))
        self.ff_norm   = nn.LayerNorm(embed_dim)
        agg_dim = 2*embed_dim + 1
        def head():
            return nn.Sequential(nn.Linear(agg_dim, embed_dim), nn.ReLU(), nn.Dropout(d1),
                                 nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.off_head = head()
        self.def_head = head()

    def forward(self, x_players, x_season):
        h = self.phi(x_players)
        attn_out, _ = self.attn(h, h, h)
        h = self.attn_norm(h + attn_out)
        h = self.ff_norm(h + self.ff(h))
        agg = torch.cat([h.mean(1), h.max(1).values, x_season], dim=1)
        return self.off_head(agg).squeeze(-1), self.def_head(agg).squeeze(-1)


class RatingNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Player lookup helpers ──────────────────────────────────────────────────────

def load_players():
    players = pd.read_parquet(CLEAN / 'players.parquet')
    name_col = next((c for c in players.columns if 'NAME' in c.upper()), None)
    if name_col is None:
        raise RuntimeError(f'No name column found in players.parquet. '
                           f'Columns: {list(players.columns)}')
    return players, name_col


def resolve_player(query: str, players: pd.DataFrame, name_col: str,
                   season: str | None) -> tuple[int, str, str]:
    """Return (player_id, matched_name, season) for the best match to `query`."""
    sub = players if season is None else players[players['SEASON'] == season]
    names = sub[name_col].unique().tolist()
    matches = difflib.get_close_matches(query, names, n=5, cutoff=0.5)
    if not matches:
        matches = [n for n in names if query.lower() in n.lower()]
    if not matches:
        raise ValueError(
            f'Could not find "{query}" in player data'
            + (f' for season {season}' if season else '') +
            f'.\nTry one of: {sorted(names[:20])}')
    best = matches[0]
    if len(matches) > 1 and matches[0].lower() != query.lower():
        alt = [m for m in matches[1:] if m != best]
        print(f'  Note: "{query}" -> "{best}" (alt: {alt})')

    if season is None:
        # Pick the player's most recent season in the data
        player_rows = sub[sub[name_col] == best].sort_values('SEASON', ascending=False)
        row = player_rows.iloc[0]
    else:
        row = sub[sub[name_col] == best]
        # if multiple rows (traded player), pick max MIN
        if len(row) > 1 and 'MIN' in row.columns:
            row = row.sort_values('MIN', ascending=False)
        row = row.iloc[0]
    return int(row['PLAYER_ID']), best, row['SEASON']


def get_player_stats(pid: int, season: str, players: pd.DataFrame) -> np.ndarray:
    row = players[(players['PLAYER_ID'] == pid) & (players['SEASON'] == season)]
    if row.empty:
        raise ValueError(f'No stats for player_id={pid} in season {season}')
    if len(row) > 1:
        row = row.sort_values('MIN', ascending=False)
    return row.iloc[0][PLAYER_STAT_COLS].values.astype(np.float32)


# ── Feature engineering for Ridge / NeuralNet ─────────────────────────────────

def build_92_features(player_rows: list[np.ndarray], season_year: int,
                      scaler_92) -> np.ndarray:
    """
    Replicates build_features.py logic.
    Column order matches X_train.parquet:
      for each stat: mean_STAT, std_STAT, min_STAT, max_STAT  (84 cols)
      then: syn_* (7 cols) + SEASON_YEAR (1 col) = 92 total.
    """
    mat = np.stack(player_rows)   # (5, 21)
    feat = []
    # stat-major, agg-minor: matches [mean_OFF, std_OFF, min_OFF, max_OFF, mean_DEF, ...]
    for col_idx in range(mat.shape[1]):
        col = mat[:, col_idx]
        feat.extend([col.mean(), col.std(), col.min(), col.max()])

    # syn_ features (same thresholds as build_features.py)
    usg   = mat[:, PLAYER_STAT_COLS.index('USG_PCT')]
    fg3a  = mat[:, PLAYER_STAT_COLS.index('FG3A')]
    ast   = mat[:, PLAYER_STAT_COLS.index('AST_PCT')]
    pie   = mat[:, PLAYER_STAT_COLS.index('PIE')]
    stl   = mat[:, PLAYER_STAT_COLS.index('STL')]
    blk   = mat[:, PLAYER_STAT_COLS.index('BLK')]

    n_spacers           = float((fg3a >= 4.0).sum())
    playmaker_x_spacing = float(ast.max() * fg3a.mean())
    shooter_x_passer    = float(fg3a.max() * ast.max())
    p = usg / (usg.sum() + 1e-8)
    usage_entropy       = float(-np.sum(p * np.log(p + 1e-8)))
    n_creators          = float((usg >= 0.22).sum())
    star_differential   = float(pie.max() - pie.mean())
    def_spread          = float(stl.std() + blk.std())

    feat.extend([n_spacers, playmaker_x_spacing, shooter_x_passer,
                 usage_entropy, n_creators, star_differential, def_spread])
    feat.append(float(season_year))   # SEASON_YEAR

    feature_names = (
        [f'{agg}_{stat}' for stat in PLAYER_STAT_COLS for agg in ('mean', 'std', 'min', 'max')]
        + ['syn_n_spacers', 'syn_playmaker_x_spacing', 'syn_shooter_x_passer',
           'syn_usage_entropy', 'syn_n_creators', 'syn_star_differential', 'syn_def_spread',
           'SEASON_YEAR']
    )
    raw = pd.DataFrame([feat], columns=feature_names, dtype=np.float32)
    return scaler_92.transform(raw)


# ── Load artifacts ─────────────────────────────────────────────────────────────

def load_ridge():
    with open(ARTIFACTS / 'ridge_off.pkl', 'rb') as f:
        r_off = pickle.load(f)
    with open(ARTIFACTS / 'ridge_def.pkl', 'rb') as f:
        r_def = pickle.load(f)
    return r_off, r_def


def load_scaler_92():
    with open(DATA / 'scaler.pkl', 'rb') as f:
        return pickle.load(f)


def load_syn_artifacts():
    with open(ARTIFACTS / 'syn_target_stats.pkl', 'rb') as f:
        ts = pickle.load(f)
    with open(ARTIFACTS / 'syn_player_scaler.pkl', 'rb') as f:
        ps = pickle.load(f)
    return ts, ps


def load_nn_artifacts():
    with open(ARTIFACTS / 'nn_target_stats.pkl', 'rb') as f:
        return pickle.load(f)


def load_synergy_models():
    ts, ps = load_syn_artifacts()
    off_models, def_models = [], []
    for seed in range(N_SEEDS):
        m = SynergyNet()
        m.load_state_dict(torch.load(ARTIFACTS / f'syn_off_seed{seed}.pt', weights_only=True))
        m.eval()
        off_models.append(m)

        m = SynergyNet()
        m.load_state_dict(torch.load(ARTIFACTS / f'syn_def_seed{seed}.pt', weights_only=True))
        m.eval()
        def_models.append(m)
    return off_models, def_models, ts, ps


def load_nn_models(input_dim):
    ts = load_nn_artifacts()
    off_models, def_models = [], []
    for seed in range(N_SEEDS):
        m = RatingNet(input_dim)
        m.load_state_dict(torch.load(ARTIFACTS / f'nn_off_seed{seed}.pt', weights_only=True))
        m.eval()
        off_models.append(m)

        m = RatingNet(input_dim)
        m.load_state_dict(torch.load(ARTIFACTS / f'nn_def_seed{seed}.pt', weights_only=True))
        m.eval()
        def_models.append(m)
    return off_models, def_models, ts


# ── Predict ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_synergy(off_models, def_models, ts, ps,
                    player_rows: list[np.ndarray], season_year: int):
    scaler = ps['scaler']
    yr_mean, yr_std = ps['yr_mean'], ps['yr_std']

    mat_raw = np.stack(player_rows)           # (5, 21)
    mat_sc  = scaler.transform(mat_raw)       # (5, 21) scaled
    X_pl = torch.tensor(mat_sc, dtype=torch.float32).unsqueeze(0)   # (1,5,21)
    yr_s = torch.tensor([[(season_year - yr_mean) / yr_std]], dtype=torch.float32)  # (1,1)

    off_preds = []
    for m in off_models:
        out, _ = m(X_pl, yr_s)
        off_preds.append(out.item() * ts['off_std'] + ts['off_mean'])

    def_preds = []
    for m in def_models:
        _, out = m(X_pl, yr_s)
        def_preds.append(out.item() * ts['def_std'] + ts['def_mean'])

    return float(np.mean(off_preds)), float(np.mean(def_preds))


@torch.no_grad()
def predict_nn(off_models, def_models, ts, X92: np.ndarray):
    x = torch.tensor(X92, dtype=torch.float32)   # (1, 92)

    off_preds = [m(x).item() * ts['off_std'] + ts['off_mean'] for m in off_models]
    def_preds = [m(x).item() * ts['def_std'] + ts['def_mean'] for m in def_models]
    return float(np.mean(off_preds)), float(np.mean(def_preds))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Predict NBA 5-man lineup ratings using trained models.')
    parser.add_argument('players', nargs='+',
                        help='5 player names (use quotes for multi-word names)')
    parser.add_argument('--season', default=None,
                        help='Season in format YYYY-YY, e.g. 2024-25 '
                             '(default: most recent season for each player)')
    args = parser.parse_args()

    if len(args.players) != 5:
        print(f'Error: expected 5 player names, got {len(args.players)}')
        sys.exit(1)

    print('\nLoading player data...')
    players, name_col = load_players()

    print('Resolving players...')
    resolved = []
    seasons_used = []
    for name in args.players:
        pid, matched, season = resolve_player(name, players, name_col, args.season)
        stats = get_player_stats(pid, season, players)
        resolved.append((matched, pid, season, stats))
        seasons_used.append(season)
        print(f'  {name!r:30s} -> {matched} ({season})')

    # Use the modal season as "the lineup season"
    from collections import Counter
    season = Counter(seasons_used).most_common(1)[0][0]
    season_year = int(season[:4])

    player_rows = [r[3] for r in resolved]

    print(f'\nLineup season: {season}  (year={season_year})')
    print()

    # ── Ridge ──────────────────────────────────────────────────────────────────
    scaler_92 = load_scaler_92()
    X92 = build_92_features(player_rows, season_year, scaler_92)
    r_off, r_def = load_ridge()
    ridge_off = float(r_off.predict(X92)[0])
    ridge_def = float(r_def.predict(X92)[0])

    # ── NeuralNet ──────────────────────────────────────────────────────────────
    input_dim = X92.shape[1]
    nn_off_models, nn_def_models, nn_ts = load_nn_models(input_dim)
    nn_off, nn_def = predict_nn(nn_off_models, nn_def_models, nn_ts, X92)

    # ── SynergyNet ─────────────────────────────────────────────────────────────
    syn_off_models, syn_def_models, syn_ts, syn_ps = load_synergy_models()
    syn_off, syn_def = predict_synergy(syn_off_models, syn_def_models,
                                        syn_ts, syn_ps, player_rows, season_year)

    # ── Best-model ensemble: NeuralNet for OFF, SynergyNet for DEF ─────────────
    best_off = nn_off   # NeuralNet has slight edge on OFF
    best_def = syn_def  # SynergyNet best on DEF

    # ── Print results ──────────────────────────────────────────────────────────
    lineup_str = ', '.join(r[0] for r in resolved)
    print('=' * 62)
    print(f'LINEUP: {lineup_str}')
    print(f'SEASON: {season}')
    print('=' * 62)
    print(f'{"Model":<16} {"OFF":>7} {"DEF":>7} {"NET":>7}')
    print('-' * 40)
    for label, off, def_ in [
        ('Ridge',      ridge_off, ridge_def),
        ('NeuralNet',  nn_off,    nn_def),
        ('SynergyNet', syn_off,   syn_def),
        ('Best-Model', best_off,  best_def),
    ]:
        net = off - def_
        marker = '  <--' if label == 'Best-Model' else ''
        print(f'{label:<16} {off:>7.1f} {def_:>7.1f} {net:>7.1f}{marker}')
    print('=' * 62)
    print()
    print('Notes:')
    print('  OFF = offensive rating (pts per 100 poss), higher is better')
    print('  DEF = defensive rating (pts allowed per 100 poss), lower is better')
    print('  NET = OFF - DEF, higher is better')
    print('  Best-Model uses NeuralNet OFF + SynergyNet DEF (best val MAE per target)')
    print()
    print('Context (val MAE for reference):')
    print('  Average NBA lineup spread: ~6 rating points per model')
    print('  Ridge   val MAE: OFF=6.00  DEF=6.10')
    print('  NeuralNet      : OFF=6.03  DEF=6.10')
    print('  SynergyNet     : OFF=6.04  DEF=5.96')


if __name__ == '__main__':
    main()
