"""
Feature engineering for lineup OFF_RATING / DEF_RATING prediction.

For each 5-man lineup, look up every player's individual stats and compute
four aggregations across the quintet: mean, std, min, max.
This gives the model signals for average quality, balance, floor, and ceiling.

Input:  data/clean/lineups.parquet, data/clean/players.parquet
Output: data/features/lineup_features.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

CLEAN    = Path('data/clean')
FEATURES = Path('data/features')

PLAYER_STAT_COLS = [
    # on-court efficiency
    'OFF_RATING', 'DEF_RATING',
    # role / usage
    'USG_PCT', 'AST_PCT', 'AST_TO', 'PIE',
    # shooting
    'EFG_PCT', 'TS_PCT',
    # rebounding
    'OREB_PCT', 'DREB_PCT',
    # ball security
    'TM_TOV_PCT',
    # per-100 rates
    'PTS', 'FGA', 'FG3A', 'FTA',
    'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB',
]

AGGS = ['mean', 'std', 'min', 'max']


def parse_player_ids(group_id: pd.Series) -> pd.DataFrame:
    """Strip leading/trailing hyphens, split on '-', return 5 int columns."""
    split = group_id.str.strip('-').str.split('-', expand=True)
    split.columns = [f'P{i+1}_ID' for i in range(5)]
    return split.astype('int64')


def build_features() -> pd.DataFrame:
    lineups = pd.read_parquet(CLEAN / 'lineups.parquet')
    players = pd.read_parquet(CLEAN / 'players.parquet')

    # --- parse lineup composition ---
    pid_cols = [f'P{i+1}_ID' for i in range(5)]
    lineups = pd.concat([lineups, parse_player_ids(lineups['GROUP_ID'])], axis=1)

    # --- melt to long: one row per (lineup × player slot) ---
    id_vars = ['GROUP_ID', 'SEASON']
    long = lineups[id_vars + pid_cols].melt(
        id_vars=id_vars,
        value_vars=pid_cols,
        var_name='SLOT',
        value_name='PLAYER_ID',
    )

    # --- join player stats (same season) ---
    lookup = players[['PLAYER_ID', 'SEASON'] + PLAYER_STAT_COLS]
    long = long.merge(lookup, on=['PLAYER_ID', 'SEASON'], how='left')

    # --- filter to lineups where all 5 players are found in player table ---
    n_found = long.groupby('GROUP_ID')['OFF_RATING'].count()
    complete = n_found[n_found == 5].index
    n_dropped = lineups['GROUP_ID'].nunique() - len(complete)
    if n_dropped:
        print(f'  Dropped {n_dropped} lineups with unmatched player(s) '
              f'(player below MPG/GP threshold)')
    long = long[long['GROUP_ID'].isin(complete)]

    # --- aggregate: mean / std / min / max across 5 players ---
    agg = long.groupby('GROUP_ID')[PLAYER_STAT_COLS].agg(AGGS)
    # flatten MultiIndex columns: (stat, agg) -> agg_stat
    agg.columns = [f'{fn}_{col}' for col, fn in agg.columns]
    agg = agg.reset_index()

    # --- synergy interaction features (computed before info is lost to aggregation) ---
    # These require the per-player granularity that mean/std/min/max throws away.
    FG3A_SPACER_THRESHOLD = 4.0   # FG3A per 100 to count as a floor spacer
    USG_CREATOR_THRESHOLD = 0.22  # USG_PCT to count as a shot creator

    def synergy(grp):
        usg   = grp['USG_PCT'].values
        fg3a  = grp['FG3A'].values
        ast   = grp['AST_PCT'].values
        ts    = grp['TS_PCT'].values
        pie   = grp['PIE'].values
        dreb  = grp['DREB_PCT'].values
        stl   = grp['STL'].values
        blk   = grp['BLK'].values

        # How many lineup members are credible floor spacers?
        n_spacers = float((fg3a >= FG3A_SPACER_THRESHOLD).sum())

        # Best passer × average spacing: captures "does the playmaker have shooters?"
        playmaker_x_spacing = float(ast.max() * fg3a.mean())

        # Best shooter × best passer: "do we have both a creator and a target?"
        shooter_x_passer = float(fg3a.max() * ast.max())

        # Usage entropy: how balanced is offensive load? (max = log(5) ≈ 1.61)
        p = usg / (usg.sum() + 1e-8)
        usage_entropy = float(-np.sum(p * np.log(p + 1e-8)))

        # Shot creators: players who can make their own offense
        n_creators = float((usg >= USG_CREATOR_THRESHOLD).sum())

        # Star power differential: best player vs. lineup average
        star_differential = float(pie.max() - pie.mean())

        # Defensive versatility: spread of individual DEF contributions
        def_spread = float(stl.std() + blk.std())

        return pd.Series({
            'syn_n_spacers'          : n_spacers,
            'syn_playmaker_x_spacing': playmaker_x_spacing,
            'syn_shooter_x_passer'   : shooter_x_passer,
            'syn_usage_entropy'      : usage_entropy,
            'syn_n_creators'         : n_creators,
            'syn_star_differential'  : star_differential,
            'syn_def_spread'         : def_spread,
        })

    syn = long.groupby('GROUP_ID').apply(synergy, include_groups=False).reset_index()
    agg = agg.merge(syn, on='GROUP_ID', how='left')

    # --- join targets + metadata ---
    meta = lineups[['GROUP_ID', 'GROUP_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
                    'SEASON', 'GP', 'MIN', 'POSS',
                    'OFF_RATING', 'DEF_RATING', 'NET_RATING']]
    df = meta.merge(agg, on='GROUP_ID', how='inner')

    # Numeric era feature: first calendar year of the season (2012-13 -> 2012)
    df['SEASON_YEAR'] = df['SEASON'].str[:4].astype(int)

    return df


def main() -> None:
    FEATURES.mkdir(parents=True, exist_ok=True)

    print('Building lineup features...')
    df = build_features()

    feature_cols = [c for c in df.columns if any(c.startswith(f'{fn}_') for fn in AGGS)]
    print(f'  Lineups: {len(df)}')
    print(f'  Feature columns: {len(feature_cols)}  ({len(PLAYER_STAT_COLS)} stats × {len(AGGS)} aggs)')
    print(f'  Seasons: {sorted(df.SEASON.unique())}')
    print(f'  OFF_RATING  range: [{df.OFF_RATING.min():.1f}, {df.OFF_RATING.max():.1f}]  '
          f'mean={df.OFF_RATING.mean():.1f}')
    print(f'  DEF_RATING  range: [{df.DEF_RATING.min():.1f}, {df.DEF_RATING.max():.1f}]  '
          f'mean={df.DEF_RATING.mean():.1f}')
    print(f'  NET_RATING  range: [{df.NET_RATING.min():.1f}, {df.NET_RATING.max():.1f}]  '
          f'mean={df.NET_RATING.mean():.1f}')
    nulls = df[feature_cols].isnull().sum()
    print(f'  Null feature values: {nulls.sum()} total across {(nulls > 0).sum()} columns')

    out = FEATURES / 'lineup_features.parquet'
    df.to_parquet(out, index=False)
    print(f'\nSaved {out}')
    print(f'  Shape: {df.shape}')
    print(f'  Columns: meta+targets ({len(df.columns) - len(feature_cols)})  +  features ({len(feature_cols)})')


if __name__ == '__main__':
    main()
