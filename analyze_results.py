"""
Final analysis and visualization for NBA Lineup Synergy project.

Generates:
  figures/fig1_model_comparison.png   -- val MAE bar chart for all models
  figures/fig2_off_vs_def.png         -- OFF vs DEF MAE scatter
  figures/fig3_synfeats_ablation.png  -- Ridge vs Ridge+SynFeats (shows zero delta)
  figures/fig4_deepsets_sweep.png     -- DeepSets hyperparameter grid
  figures/fig5_train_curves.png       -- NN training curve (representative seed)
  figures/fig6_r2_comparison.png      -- R^2 across splits
  summary_table.csv                   -- full results table
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path('figures')
OUT.mkdir(exist_ok=True)

RESULTS = Path('artifacts/baseline_results.csv')
DS_SWEEP = Path('artifacts/ds_sweep_results.csv')
NN_HIST_OFF = Path('artifacts/nn_history_off.csv')
NN_HIST_DEF = Path('artifacts/nn_history_def.csv')
DS_HIST_OFF = Path('artifacts/ds_history_seed0_off.csv')
DS_HIST_DEF = Path('artifacts/ds_history_seed0_def.csv')
SYN_HIST_OFF = Path('artifacts/syn_history_off.csv')
SYN_HIST_DEF = Path('artifacts/syn_history_def.csv')

# ── Load main results ──────────────────────────────────────────────────────────
df = pd.read_csv(RESULTS)

MODEL_ORDER = ['GlobalMean', 'MeanPlayerRating', 'Ridge', 'Ridge+SynFeats',
               'NeuralNet', 'DeepSets', 'SynergyNet', 'SynergyNetV2']
COLORS = {
    'GlobalMean':      '#aaaaaa',
    'MeanPlayerRating':'#888888',
    'Ridge':           '#4e79a7',
    'Ridge+SynFeats':  '#a0cbe8',
    'NeuralNet':       '#f28e2b',
    'DeepSets':        '#76b7b2',
    'SynergyNet':      '#e15759',
    'SynergyNetV2':    '#b07aa1',
}

val = df[df['split'] == 'val'].copy()
test = df[df['split'] == 'test'].copy()

# ── Fig 1: Val MAE bar chart (primary comparison) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.suptitle('Validation MAE by Model', fontsize=14, fontweight='bold')

for ax, target, title in zip(axes, ['OFF_RATING', 'DEF_RATING'],
                             ['Offensive Rating (OFF)', 'Defensive Rating (DEF)']):
    sub = val[val['target'] == target].set_index('model')
    models = [m for m in MODEL_ORDER if m in sub.index]
    maes = [sub.loc[m, 'MAE'] for m in models]
    colors = [COLORS[m] for m in models]
    bars = ax.bar(range(len(models)), maes, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('MAE (rating points)')
    ax.set_title(title)
    # annotate values
    for i, (bar, mae) in enumerate(zip(bars, maes)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mae:.3f}', ha='center', va='bottom', fontsize=8)
    best_idx = int(np.argmin(maes))
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2.0)
    ymin = min(maes) - 0.4
    ymax = max(maes) + 0.4
    ax.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig(OUT / 'fig1_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig1')

# ── Fig 2: OFF vs DEF MAE scatter ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for model in MODEL_ORDER:
    sub = val[val['model'] == model]
    if len(sub) < 2:
        continue
    off_mae = sub[sub['target'] == 'OFF_RATING']['MAE'].values
    def_mae = sub[sub['target'] == 'DEF_RATING']['MAE'].values
    if len(off_mae) and len(def_mae):
        ax.scatter(off_mae[0], def_mae[0], color=COLORS[model], s=100, zorder=3,
                   label=model)
        ax.annotate(model, (off_mae[0], def_mae[0]),
                    textcoords='offset points', xytext=(6, 4), fontsize=8)

ax.set_xlabel('OFF_RATING Val MAE')
ax.set_ylabel('DEF_RATING Val MAE')
ax.set_title('OFF vs DEF Validation MAE\n(lower-left = better on both)')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'fig2_off_vs_def.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig2')

# ── Fig 3: SynFeats ablation (Ridge vs Ridge+SynFeats) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Synergy Features Ablation: Ridge vs Ridge+SynFeats', fontsize=13, fontweight='bold')

for ax, split in zip(axes, ['val', 'test']):
    sub = df[df['split'] == split]
    ridge = sub[sub['model'] == 'Ridge'].set_index('target')['MAE']
    ridge_syn = sub[sub['model'] == 'Ridge+SynFeats'].set_index('target')['MAE']
    targets = ['OFF_RATING', 'DEF_RATING']
    labels = ['OFF', 'DEF']
    x = np.arange(len(targets))
    w = 0.35
    b1 = ax.bar(x - w/2, [ridge.get(t, 0) for t in targets], w,
                label='Ridge', color='#4e79a7', alpha=0.9)
    b2 = ax.bar(x + w/2, [ridge_syn.get(t, 0) for t in targets], w,
                label='Ridge+SynFeats', color='#a0cbe8', alpha=0.9)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('MAE')
    ax.set_title(f'{split.capitalize()} Set')
    ax.legend(fontsize=9)
    ymin = min(ridge.values.min(), ridge_syn.values.min()) - 0.15
    ymax = max(ridge.values.max(), ridge_syn.values.max()) + 0.15
    ax.set_ylim(ymin, ymax)
    ax.text(0.5, 0.95, 'Zero difference — synergy is nonlinear',
            transform=ax.transAxes, ha='center', va='top', fontsize=8,
            style='italic', color='#555555')

plt.tight_layout()
plt.savefig(OUT / 'fig3_synfeats_ablation.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig3')

# ── Fig 4: DeepSets hyperparameter sweep ──────────────────────────────────────
sweep = pd.read_csv(DS_SWEEP)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle('DeepSets Hyperparameter Grid Search (Val MAE)', fontsize=13, fontweight='bold')

for ax, col, title in zip(axes, ['off_mae', 'def_mae'], ['OFF MAE', 'DEF MAE']):
    pivoted = sweep.pivot_table(index='embed_dim', columns='dropout', values=col)
    for drop_col in pivoted.columns:
        ax.plot(pivoted.index, pivoted[drop_col], marker='o',
                label=f'drop={drop_col}')
    best_idx = sweep[col].idxmin()
    best_row = sweep.loc[best_idx]
    ax.scatter(best_row['embed_dim'], best_row[col], color='red', s=150, zorder=5,
               marker='*', label=f'Best ({best_row[col]:.3f})')
    ax.set_xlabel('embed_dim')
    ax.set_ylabel('Val MAE')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(sweep['embed_dim'].unique()))

plt.tight_layout()
plt.savefig(OUT / 'fig4_deepsets_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig4')

# ── Fig 5: Training curves ─────────────────────────────────────────────────────
nn_off = pd.read_csv(NN_HIST_OFF)
nn_def = pd.read_csv(NN_HIST_DEF)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('MLP Neural Net Training Curves', fontsize=13, fontweight='bold')

for ax, hist, target in zip(axes, [nn_off, nn_def], ['OFF_RATING', 'DEF_RATING']):
    epochs = hist['epoch']
    ax.plot(epochs, hist['train_loss'], label='Train loss (normalized)', color='#4e79a7', alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(epochs, hist['val_mae'], label='Val MAE', color='#e15759', alpha=0.8)
    best_epoch = hist.loc[hist['val_mae'].idxmin(), 'epoch']
    best_mae = hist['val_mae'].min()
    ax2.axvline(best_epoch, color='grey', linestyle='--', alpha=0.6, label=f'Best epoch={best_epoch}')
    ax2.scatter([best_epoch], [best_mae], color='#e15759', s=60, zorder=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss')
    ax2.set_ylabel('Val MAE')
    ax.set_title(target.replace('_', ' '))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUT / 'fig5_train_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig5')

# ── Fig 6: R^2 comparison across models ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('R² by Model and Split', fontsize=13, fontweight='bold')

for ax, target in zip(axes, ['OFF_RATING', 'DEF_RATING']):
    sub = df[df['target'] == target]
    models = [m for m in MODEL_ORDER if m in sub['model'].values]
    x = np.arange(len(models))
    w = 0.28
    for i, (split, color) in enumerate(zip(['train', 'val', 'test'],
                                            ['#4e79a7', '#f28e2b', '#59a14f'])):
        r2s = []
        for m in models:
            row = sub[(sub['model'] == m) & (sub['split'] == split)]
            r2s.append(row['R2'].values[0] if len(row) else 0)
        ax.bar(x + (i - 1) * w, r2s, w, label=split, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('R²')
    ax.set_title(target.replace('_', ' '))
    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(OUT / 'fig6_r2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig6')

# ── Summary table ─────────────────────────────────────────────────────────────
pivot = df.pivot_table(index=['model', 'split'], columns='target', values=['MAE', 'R2'])
pivot.columns = [f'{col[0]}_{col[1].replace("_RATING", "")}' for col in pivot.columns]
pivot = pivot.reset_index()
pivot.to_csv('summary_table.csv', index=False, float_format='%.4f')
print('Saved summary_table.csv')

# ── Console summary ────────────────────────────────────────────────────────────
print()
print('=' * 70)
print('VAL SET RESULTS (primary comparison)')
print('=' * 70)
print(f'{"Model":<20} {"OFF MAE":>8} {"DEF MAE":>8} {"NET MAE":>9} {"OFF R²":>7} {"DEF R²":>7}')
print('-' * 70)
for model in MODEL_ORDER:
    sub = val[val['model'] == model]
    if sub.empty:
        continue
    off = sub[sub['target'] == 'OFF_RATING']
    def_ = sub[sub['target'] == 'DEF_RATING']
    if off.empty or def_.empty:
        continue
    off_mae = off['MAE'].values[0]
    def_mae = def_['MAE'].values[0]
    off_r2  = off['R2'].values[0]
    def_r2  = def_['R2'].values[0]
    net_mae = off_mae + def_mae  # upper bound on NET_RATING MAE
    print(f'{model:<20} {off_mae:>8.3f} {def_mae:>8.3f} {net_mae:>9.3f} {off_r2:>7.3f} {def_r2:>7.3f}')

print()
print('KEY DELTAS (vs. Ridge baseline, val set):')
ridge_off = val[(val['model']=='Ridge') & (val['target']=='OFF_RATING')]['MAE'].values[0]
ridge_def = val[(val['model']=='Ridge') & (val['target']=='DEF_RATING')]['MAE'].values[0]
for model in ['Ridge+SynFeats', 'NeuralNet', 'DeepSets', 'SynergyNet', 'SynergyNetV2']:
    sub = val[val['model'] == model]
    if sub.empty: continue
    off = sub[sub['target']=='OFF_RATING']['MAE'].values
    def_ = sub[sub['target']=='DEF_RATING']['MAE'].values
    if not len(off) or not len(def_): continue
    d_off = off[0] - ridge_off
    d_def = def_[0] - ridge_def
    print(f'  {model:<20} OFF {d_off:+.3f}  DEF {d_def:+.3f}')

print()
print('TEST SET RESULTS')
print('=' * 70)
print(f'{"Model":<20} {"OFF MAE":>8} {"DEF MAE":>8} {"OFF R²":>7} {"DEF R²":>7}')
print('-' * 70)
for model in MODEL_ORDER:
    sub = test[test['model'] == model]
    if sub.empty: continue
    off = sub[sub['target']=='OFF_RATING']
    def_ = sub[sub['target']=='DEF_RATING']
    if off.empty or def_.empty: continue
    print(f'{model:<20} {off["MAE"].values[0]:>8.3f} {def_["MAE"].values[0]:>8.3f} '
          f'{off["R2"].values[0]:>7.3f} {def_["R2"].values[0]:>7.3f}')
