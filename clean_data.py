"""
Clean raw NBA data for lineup OFF_RATING / DEF_RATING prediction.

Inputs  (data/raw/):
  lineups_advanced_totals.parquet
  players_advanced_totals.parquet
  players_base_per100possessions.parquet

Outputs (data/clean/):
  lineups.parquet   — one row per (lineup, season); targets + identifiers only
  players.parquet   — one row per (player, season); features for ML
"""

import pandas as pd
from pathlib import Path

RAW  = Path('data/raw')
CLEAN = Path('data/clean')

# Minimum sample-size thresholds
MIN_LINEUP_POSS = 150  # ~6 full games of lineup usage
MIN_PLAYER_MPG  = 10   # minimum minutes per game (filters garbage-time players)
MIN_PLAYER_GP   = 10   # minimum games played (ensures meaningful sample)


# ---------------------------------------------------------------------------
# Lineups
# ---------------------------------------------------------------------------

LINEUP_KEEP = [
    'GROUP_ID',           # hyphen-separated player IDs — join key
    'GROUP_NAME',         # hyphen-separated player names — human readable
    'TEAM_ID',
    'TEAM_ABBREVIATION',
    'SEASON',
    'GP',
    'MIN',
    'POSS',
    'OFF_RATING',         # target 1
    'DEF_RATING',         # target 2
    'NET_RATING',         # derived target (OFF - DEF)
]


def clean_lineups() -> pd.DataFrame:
    df = pd.read_parquet(RAW / 'lineups_advanced_totals.parquet')

    df = df[LINEUP_KEEP].copy()
    df = df[df['POSS'] >= MIN_LINEUP_POSS].reset_index(drop=True)

    # Recompute NET_RATING to avoid 1-decimal API rounding inconsistencies
    df['NET_RATING'] = (df['OFF_RATING'] - df['DEF_RATING']).round(1)

    print(f'Lineups: {len(df)} rows after filtering (POSS >= {MIN_LINEUP_POSS})')
    print(f'  seasons: {sorted(df.SEASON.unique())}')
    print(f'  OFF_RATING  range: [{df.OFF_RATING.min():.1f}, {df.OFF_RATING.max():.1f}]')
    print(f'  DEF_RATING  range: [{df.DEF_RATING.min():.1f}, {df.DEF_RATING.max():.1f}]')
    return df


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

# From players_advanced_totals — efficiency / role metrics
ADV_KEEP = [
    'PLAYER_ID',
    'PLAYER_NAME',
    'SEASON',
    'GP',
    'MIN',
    'OFF_RATING',    # individual on-court offensive rating
    'DEF_RATING',    # individual on-court defensive rating
    'USG_PCT',       # usage rate — how much offense runs through this player
    'AST_PCT',       # assist percentage — playmaking load
    'AST_TO',        # assist-to-turnover ratio — decision quality
    'OREB_PCT',      # offensive rebounding rate
    'DREB_PCT',      # defensive rebounding rate
    'TM_TOV_PCT',    # team turnover rate when player is on court
    'EFG_PCT',       # effective field goal % — shooting efficiency
    'TS_PCT',        # true shooting % — shooting efficiency incl. FTs
    'PIE',           # player impact estimate
]

# From players_base_per100possessions — per-possession rates
PER100_KEEP = [
    'PLAYER_ID',
    'SEASON',
    'FGA',      # shot volume
    'FG3A',     # 3-point attempt rate
    'FTA',      # free throw attempt rate (ability to draw fouls)
    'AST',      # assists per 100
    'TOV',      # turnovers per 100
    'STL',      # steals per 100 — defensive activity
    'BLK',      # blocks per 100 — rim protection
    'OREB',     # offensive rebounds per 100
    'DREB',     # defensive rebounds per 100
    'PTS',      # points per 100
]


def clean_players() -> pd.DataFrame:
    adv  = pd.read_parquet(RAW / 'players_advanced_totals.parquet')[ADV_KEEP].copy()
    p100 = pd.read_parquet(RAW / 'players_base_per100possessions.parquet')[PER100_KEEP].copy()

    # Deduplicate traded players: keep the row with the most minutes per (player, season).
    # Players traded mid-season appear once per team in the API response.
    adv  = adv.sort_values('MIN', ascending=False).drop_duplicates(['PLAYER_ID', 'SEASON'])
    # Borrow MIN from adv just for dedup ordering, then discard it
    p100 = p100.merge(
        adv[['PLAYER_ID', 'SEASON', 'MIN']],
        on=['PLAYER_ID', 'SEASON'], how='left'
    )
    p100 = p100.sort_values('MIN', ascending=False).drop_duplicates(['PLAYER_ID', 'SEASON'])
    p100 = p100.drop(columns='MIN')

    df = adv.merge(p100, on=['PLAYER_ID', 'SEASON'], how='inner')

    # MIN here is minutes per game; GP is games played
    df = df[(df['MIN'] >= MIN_PLAYER_MPG) & (df['GP'] >= MIN_PLAYER_GP)].reset_index(drop=True)

    print(f'Players: {len(df)} rows after filtering (MPG >= {MIN_PLAYER_MPG}, GP >= {MIN_PLAYER_GP})')
    print(f'  seasons: {sorted(df.SEASON.unique())}')
    print(f'  OFF_RATING range: [{df.OFF_RATING.min():.1f}, {df.OFF_RATING.max():.1f}]')
    print(f'  DEF_RATING range: [{df.DEF_RATING.min():.1f}, {df.DEF_RATING.max():.1f}]')
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    print(f'  null counts: {nulls.to_dict() if len(nulls) else "none"}')
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)

    lineups = clean_lineups()
    lineups.to_parquet(CLEAN / 'lineups.parquet', index=False)
    print()

    players = clean_players()
    players.to_parquet(CLEAN / 'players.parquet', index=False)

    print('\nSaved to data/clean/')


if __name__ == '__main__':
    main()
