"""
Collect NBA per-possession data for lineup synergy ML.

Goal: predict 5-man lineup OFF_RATING and DEF_RATING from player composition.

Pulls from stats.nba.com via nba_api:
  - 5-man lineup advanced stats  -> targets (OFF_RATING, DEF_RATING, NET_RATING)
  - Player advanced stats         -> individual on-court efficiency / role metrics
  - Player base per-100 stats     -> per-possession rates (STL, BLK, AST, TOV, ...)

Outputs parquet files to data/raw/.
"""

import time
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import LeagueDashLineups, LeagueDashPlayerStats

SEASONS = [
    '2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2017-18',
    '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24',
    '2024-25', '2025-26',
]
DATA_DIR = Path('data/raw')
SLEEP = 1.0  # seconds between requests to avoid rate limiting

# Only the combinations needed for the prediction goal
LINEUP_CONFIGS = [
    ('Advanced', 'Totals'),          # OFF_RATING, DEF_RATING, NET_RATING, POSS (targets + filter)
]

PLAYER_CONFIGS = [
    ('Advanced', 'Totals'),          # individual on-court efficiency, role metrics
    ('Base',     'Per100Possessions'), # per-100 rates: STL, BLK, AST, TOV, FG3A, FTA, PTS
]


def fetch_lineup_stats(season: str, measure_type: str, per_mode: str) -> pd.DataFrame:
    resp = LeagueDashLineups(
        season=season,
        measure_type_detailed_defense=measure_type,
        per_mode_detailed=per_mode,
        group_quantity=5,
        timeout=60,
    )
    df = resp.get_data_frames()[0]
    df['SEASON'] = season
    return df


def fetch_player_stats(season: str, measure_type: str, per_mode: str) -> pd.DataFrame:
    resp = LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense=measure_type,
        per_mode_detailed=per_mode,
        timeout=60,
    )
    df = resp.get_data_frames()[0]
    df['SEASON'] = season
    return df


def collect(entity: str, configs: list, fetch_fn) -> None:
    for measure, per_mode in configs:
        tag = f'{entity}_{measure.lower()}_{per_mode.lower()}'
        out = DATA_DIR / f'{tag}.parquet'

        existing = pd.read_parquet(out) if out.exists() else None
        existing_seasons = set(existing['SEASON'].unique()) if existing is not None else set()
        seasons_to_fetch = [s for s in SEASONS if s not in existing_seasons]

        if not seasons_to_fetch:
            print(f'  skip (all seasons present): {out.name}')
            continue

        frames = [existing] if existing is not None else []
        for season in seasons_to_fetch:
            print(f'  {entity} {measure}/{per_mode} {season} ...', end=' ', flush=True)
            try:
                df = fetch_fn(season, measure, per_mode)
                frames.append(df)
                print(f'{len(df)} rows')
            except Exception as e:
                print(f'ERROR: {e}')
            time.sleep(SLEEP)

        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
            print(f'  -> saved {out.name}')


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print('=== Lineups ===')
    collect('lineups', LINEUP_CONFIGS, fetch_lineup_stats)

    print('\n=== Players ===')
    collect('players', PLAYER_CONFIGS, fetch_player_stats)

    print('\nDone.')


if __name__ == '__main__':
    main()
