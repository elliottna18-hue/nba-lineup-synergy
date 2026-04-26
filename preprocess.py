"""
Preprocessing and train/val/test splits for lineup rating prediction.

Split strategy: temporal by season (no random shuffling — avoids leakage
across seasons and reflects real deployment: train on past, predict future).

    Train : 2020-21 through 2023-24  (~67%)
    Val   : 2024-25                  (~17%)
    Test  : 2025-26                  (~16%)

Scaler is fit on training data only, then applied to val and test.

Outputs (data/model_ready/):
    X_train.parquet, X_val.parquet, X_test.parquet   — scaled features
    y_train.parquet, y_val.parquet, y_test.parquet   — OFF_RATING, DEF_RATING
    meta_train.parquet, meta_val.parquet, meta_test.parquet  — identifiers
    scaler.pkl   — fitted StandardScaler (reuse for new data / inference)
    split_info.txt  — split summary for reference
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

FEATURES_DIR  = Path('data/features')
OUT_DIR       = Path('data/model_ready')

TRAIN_SEASONS = ['2012-13', '2013-14', '2014-15', '2015-16', '2016-17',
                 '2017-18', '2018-19', '2019-20', '2020-21', '2021-22']
VAL_SEASONS   = ['2022-23', '2023-24']
TEST_SEASONS  = ['2024-25', '2025-26']

TARGETS = ['OFF_RATING', 'DEF_RATING']

META_COLS = ['GROUP_ID', 'GROUP_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
             'SEASON', 'GP', 'MIN', 'POSS', 'NET_RATING']


def load_and_split(df: pd.DataFrame) -> tuple:
    train = df[df['SEASON'].isin(TRAIN_SEASONS)].reset_index(drop=True)
    val   = df[df['SEASON'].isin(VAL_SEASONS)].reset_index(drop=True)
    test  = df[df['SEASON'].isin(TEST_SEASONS)].reset_index(drop=True)
    return train, val, test


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURES_DIR / 'lineup_features.parquet')

    feature_cols = [c for c in df.columns
                    if any(c.startswith(f'{fn}_') for fn in ['mean', 'std', 'min', 'max'])
                    or c.startswith('syn_')]
    feature_cols = feature_cols + ['SEASON_YEAR']

    # --- split ---
    train, val, test = load_and_split(df)

    X_train_raw = train[feature_cols]
    X_val_raw   = val[feature_cols]
    X_test_raw  = test[feature_cols]

    y_train = train[TARGETS]
    y_val   = val[TARGETS]
    y_test  = test[TARGETS]

    meta_train = train[META_COLS]
    meta_val   = val[META_COLS]
    meta_test  = test[META_COLS]

    # --- fit scaler on training data only ---
    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    X_train = pd.DataFrame(scaler.transform(X_train_raw),
                           columns=feature_cols, index=X_train_raw.index)
    X_val   = pd.DataFrame(scaler.transform(X_val_raw),
                           columns=feature_cols, index=X_val_raw.index)
    X_test  = pd.DataFrame(scaler.transform(X_test_raw),
                           columns=feature_cols, index=X_test_raw.index)

    # --- save ---
    for name, X, y, meta in [('train', X_train, y_train, meta_train),
                              ('val',   X_val,   y_val,   meta_val),
                              ('test',  X_test,  y_test,  meta_test)]:
        X.to_parquet(OUT_DIR / f'X_{name}.parquet', index=False)
        y.to_parquet(OUT_DIR / f'y_{name}.parquet', index=False)
        meta.to_parquet(OUT_DIR / f'meta_{name}.parquet', index=False)

    with open(OUT_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # --- report ---
    total = len(df)
    lines = [
        'Lineup Rating Prediction — Preprocessing Summary',
        '=' * 50,
        f'Total lineups : {total}',
        f'Feature cols  : {len(feature_cols)}',
        '',
        'Split (temporal by season):',
        f'  Train  {TRAIN_SEASONS[0]}–{TRAIN_SEASONS[-1]} : {len(train):>4} rows  ({100*len(train)/total:.1f}%)',
        f'  Val    {VAL_SEASONS[0]}           : {len(val):>4} rows  ({100*len(val)/total:.1f}%)',
        f'  Test   {TEST_SEASONS[0]}           : {len(test):>4} rows  ({100*len(test)/total:.1f}%)',
        '',
        'Targets:',
        f'  OFF_RATING  train mean={y_train.OFF_RATING.mean():.1f}  std={y_train.OFF_RATING.std():.1f}'
        f'  val mean={y_val.OFF_RATING.mean():.1f}  test mean={y_test.OFF_RATING.mean():.1f}',
        f'  DEF_RATING  train mean={y_train.DEF_RATING.mean():.1f}  std={y_train.DEF_RATING.std():.1f}'
        f'  val mean={y_val.DEF_RATING.mean():.1f}  test mean={y_test.DEF_RATING.mean():.1f}',
        '',
        'Scaling: StandardScaler fit on training set only.',
        f'  Feature means (train): min={scaler.mean_.min():.3f}  max={scaler.mean_.max():.3f}',
        f'  Feature stds  (train): min={scaler.scale_.min():.3f}  max={scaler.scale_.max():.3f}',
        '',
        'Files saved to data/model_ready/:',
        '  X_train/val/test.parquet   — scaled features',
        '  y_train/val/test.parquet   — OFF_RATING, DEF_RATING',
        '  meta_train/val/test.parquet — GROUP_ID, GROUP_NAME, SEASON, etc.',
        '  scaler.pkl                 — fitted StandardScaler',
    ]
    report = '\n'.join(lines)
    print(report)
    with open(OUT_DIR / 'split_info.txt', 'w') as f:
        f.write(report + '\n')

    # quick sanity checks
    assert X_train.isnull().sum().sum() == 0, 'nulls in X_train'
    assert X_val.isnull().sum().sum() == 0,   'nulls in X_val'
    assert X_test.isnull().sum().sum() == 0,  'nulls in X_test'
    assert abs(X_train.mean().mean()) < 1e-6, 'X_train not zero-mean after scaling'
    assert abs(X_train.std().mean() - 1.0) < 1e-3, 'X_train not unit-variance after scaling'
    print('\nAll sanity checks passed.')


if __name__ == '__main__':
    main()
