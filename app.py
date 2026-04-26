"""
NBA Superteam Builder — Streamlit app.

$150M budget. Players priced by individual stats (PIE + USG + MIN blend).
Pick 5 players, SynergyNet predicts the lineup's NET_RATING.
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
from pathlib import Path

CLEAN     = Path('data/clean')
ARTIFACTS = Path('artifacts')

PLAYER_STAT_COLS = [
    'OFF_RATING', 'DEF_RATING', 'USG_PCT', 'AST_PCT', 'AST_TO', 'PIE',
    'EFG_PCT', 'TS_PCT', 'OREB_PCT', 'DREB_PCT', 'TM_TOV_PCT',
    'PTS', 'FGA', 'FG3A', 'FTA', 'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB',
]

BUDGET    = 130
MIN_PRICE = 3
MAX_PRICE = 50
N_SEEDS   = 5

SEASONS = ['2025-26', '2024-25', '2023-24', '2022-23', '2021-22',
           '2020-21', '2019-20', '2018-19', '2017-18', '2016-17',
           '2015-16', '2014-15', '2013-14', '2012-13']

TIER_COLOR = {'MAX': '#c0392b', 'STAR': '#d35400', 'STARTER': '#2980b9',
              'ROLE': '#27ae60', 'MIN': '#7f8c8d'}
TIER_RANGE = {'MAX': '$38–50M', 'STAR': '$25–38M', 'STARTER': '$13–25M',
              'ROLE': '$7–13M', 'MIN': '$3–7M'}


# ── Model ──────────────────────────────────────────────────────────────────────

class SynergyNet(nn.Module):
    def __init__(self, n_feats=21, embed_dim=64, n_heads=4, dropout=(0.3, 0.2)):
        super().__init__()
        d0, d1 = dropout
        self.phi = nn.Sequential(
            nn.Linear(n_feats, embed_dim), nn.LayerNorm(embed_dim),
            nn.ReLU(), nn.Dropout(d0))
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


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    with open(ARTIFACTS / 'syn_target_stats.pkl', 'rb') as f:
        ts = pickle.load(f)
    with open(ARTIFACTS / 'syn_player_scaler.pkl', 'rb') as f:
        ps = pickle.load(f)
    off_models, def_models = [], []
    for seed in range(N_SEEDS):
        for lst, prefix, use_off in [(off_models, 'off', True), (def_models, 'def', False)]:
            m = SynergyNet()
            m.load_state_dict(torch.load(ARTIFACTS / f'syn_{prefix}_seed{seed}.pt',
                                         weights_only=True))
            m.eval()
            lst.append(m)
    return off_models, def_models, ts, ps


@st.cache_data
def load_players():
    return pd.read_parquet(CLEAN / 'players.parquet')


# ── Pricing ────────────────────────────────────────────────────────────────────

def compute_prices(df: pd.DataFrame) -> pd.Series:
    pie_pct = df['PIE'].rank(pct=True)
    usg_pct = df['USG_PCT'].rank(pct=True)
    min_pct = df['MIN'].rank(pct=True)
    quality = 0.55 * pie_pct + 0.30 * usg_pct + 0.15 * min_pct
    # power=2.5 creates a steep curve: elites cost ~$45-50M, solid starters ~$12M,
    # role players ~$7M — getting 2 elites consumes ~73% of the $130M budget
    return (MIN_PRICE + (MAX_PRICE - MIN_PRICE) * (quality ** 2.5)).round(1)


def tier_label(price: float) -> str:
    # Thresholds calibrated to power=2.5 / max=$50 curve
    if price >= 38: return 'MAX'       # top ~12%  — true superstars
    if price >= 25: return 'STAR'      # top 12–25% — all-star caliber
    if price >= 13: return 'STARTER'   # top 25–50% — solid starters
    if price >= 7:  return 'ROLE'      # top 50–66% — role players
    return 'MIN'


# ── Prediction ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_lineup(player_rows, season_year, off_models, def_models, ts, ps):
    scaler = ps['scaler']
    yr_mean, yr_std = ps['yr_mean'], ps['yr_std']
    mat_sc = scaler.transform(np.stack(player_rows))
    X_pl = torch.tensor(mat_sc, dtype=torch.float32).unsqueeze(0)
    yr_s = torch.tensor([[(season_year - yr_mean) / yr_std]], dtype=torch.float32)
    off = float(np.mean([m(X_pl, yr_s)[0].item() * ts['off_std'] + ts['off_mean']
                         for m in off_models]))
    def_ = float(np.mean([m(X_pl, yr_s)[1].item() * ts['def_std'] + ts['def_mean']
                          for m in def_models]))
    return {'off': round(off, 1), 'def': round(def_, 1), 'net': round(off - def_, 1)}


# ── UI helpers ─────────────────────────────────────────────────────────────────

def net_color(net: float) -> str:
    if net >= 12: return '#27ae60'
    if net >= 7:  return '#2ecc71'
    if net >= 3:  return '#f39c12'
    if net >= 0:  return '#e67e22'
    return '#e74c3c'


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render_player_card(col, row, key_suffix):
    tier   = row['Tier']
    color  = TIER_COLOR.get(tier, '#7f8c8d')
    nc     = net_color(row['NET'])
    bg     = _rgba(color, 0.07)
    border = _rgba(color, 0.4)
    name   = str(row['PLAYER_NAME'])
    with col:
        st.markdown(
            f"<div style='border:1.5px solid {border};border-radius:8px;"
            f"background:{bg};padding:11px 13px 9px;margin-bottom:4px;'>"
            f"<div style='font-size:0.95em;font-weight:700;"
            f"margin-bottom:4px;letter-spacing:0.01em;'>{name}</div>"
            f"<div style='font-size:0.75em;font-weight:600;color:{color};"
            f"margin-bottom:6px;letter-spacing:0.04em;text-transform:uppercase;'>"
            f"{tier} &nbsp;·&nbsp; ${row['Price ($M)']:.0f}M</div>"
            f"<div style='font-size:0.82em;'>"
            f"<span style='color:#2980b9;font-weight:600;cursor:default;' "
            f"title='Offensive rating — points scored per 100 possessions on court'>OFF {row['OFF_RATING']:.0f}</span>"
            f"&ensp;<span style='opacity:0.4;'>|</span>&ensp;"
            f"<span style='color:#c0392b;font-weight:600;cursor:default;' "
            f"title='Defensive rating — points allowed per 100 possessions on court (lower is better)'>DEF {row['DEF_RATING']:.0f}</span>"
            f"&ensp;<span style='opacity:0.4;'>|</span>&ensp;"
            f"<span style='color:{nc};font-weight:600;cursor:default;' "
            f"title='Net rating — offensive minus defensive rating'>NET {row['NET']:+.0f}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        if st.button('Remove', key=f'rm_{key_suffix}', use_container_width=True):
            st.session_state.lineup.remove(int(row['PLAYER_ID']))
            st.rerun()


def render_empty_slot(col, n):
    with col:
        st.markdown(
            f"<div style='border:1.5px dashed rgba(128,128,128,0.35);border-radius:8px;"
            f"padding:26px 8px;text-align:center;opacity:0.5;min-height:110px;"
            f"display:flex;flex-direction:column;align-items:center;justify-content:center;'>"
            f"<div style='font-size:1.4em;line-height:1;'>+</div>"
            f"<div style='font-size:0.75em;margin-top:5px;letter-spacing:0.05em;"
            f"text-transform:uppercase;'>Slot {n}</div></div>",
            unsafe_allow_html=True,
        )


# ── App ────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title='NBA Lineup Synergy', page_icon='🏀', layout='wide')

    st.markdown("""
    <style>
      .block-container { padding-top: 1.2rem; }
      .stDataFrame { border-radius: 8px; }
      button[kind="secondary"] { font-size: 0.78em; }
    </style>
    """, unsafe_allow_html=True)

    players_all = load_players()
    off_models, def_models, ts, ps = load_models()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('### Settings')
        season = st.selectbox('Season', SEASONS)
        st.divider()
        st.markdown('**Salary Tiers**')
        for tier, color in TIER_COLOR.items():
            rng = TIER_RANGE[tier]
            st.markdown(
                f'<span style="color:{color};font-weight:600;">{tier}</span>'
                f'<span style="font-size:0.85em;"> &nbsp;{rng}</span>',
                unsafe_allow_html=True)

    # ── Season data ────────────────────────────────────────────────────────────
    p = players_all[players_all['SEASON'] == season].copy().reset_index(drop=True)
    if p.empty:
        st.error(f'No data for {season}.')
        return

    # Require 1000+ total minutes — filters bench players and injury-shortened seasons
    p = p[p['MIN'] * p['GP'] >= 1000].reset_index(drop=True)

    p['Price ($M)']  = compute_prices(p)
    p['Tier']        = p['Price ($M)'].apply(tier_label)
    p['NET']         = (p['OFF_RATING'] - p['DEF_RATING']).round(1)
    p['Tier Label']  = p['Tier']

    # ── Session state ──────────────────────────────────────────────────────────
    if 'lineup' not in st.session_state:
        st.session_state.lineup = []
    if 'history' not in st.session_state:
        st.session_state.history = []

    lineup_ids = st.session_state.lineup
    lineup_rows = p[p['PLAYER_ID'].isin(lineup_ids)]
    spent     = lineup_rows['Price ($M)'].sum()
    remaining = BUDGET - spent

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown('# NBA Superteam Builder')

    hc1, hc2, hc3, hc4 = st.columns([1.2, 1.2, 1.2, 3])
    hc1.metric('Budget', f'${BUDGET}M')
    hc2.metric('Spent', f'${spent:.1f}M',
               delta=f'${remaining:.1f}M left', delta_color='inverse')
    hc3.metric('Players', f'{len(lineup_ids)} / 5')
    if len(lineup_ids) == 5:
        pred = predict_lineup(
            [p.loc[p['PLAYER_ID'] == pid, PLAYER_STAT_COLS].values[0] for pid in lineup_ids],
            int(season[:4]), off_models, def_models, ts, ps)
        nc = net_color(pred['net'])
        hc4.markdown(
            f'<div style="padding:6px 14px;border-radius:8px;border:1.5px solid {nc};">'
            f'<span style="font-size:0.8em;font-weight:600;letter-spacing:0.05em;'
            f'text-transform:uppercase;opacity:0.6;">Predicted &nbsp;</span>'
            f'<span style="color:#2980b9;font-weight:600;">OFF {pred["off"]:.1f}</span>'
            f'<span style="opacity:0.4;"> &nbsp;·&nbsp; </span>'
            f'<span style="color:#e74c3c;font-weight:600;">DEF {pred["def"]:.1f}</span>'
            f'<span style="opacity:0.4;"> &nbsp;·&nbsp; </span>'
            f'<span style="color:{nc};font-weight:700;font-size:1.1em;">NET {pred["net"]:+.1f}</span>'
            f'</div>',
            unsafe_allow_html=True)
    else:
        hc4.info(f'Select {5 - len(lineup_ids)} more player(s) to see prediction.')

    # ── Lineup slots ───────────────────────────────────────────────────────────
    slot_cols = st.columns(5)
    for i, col in enumerate(slot_cols):
        if i < len(lineup_ids):
            pid = lineup_ids[i]
            row = p[p['PLAYER_ID'] == pid].iloc[0]
            render_player_card(col, row, key_suffix=pid)
        else:
            render_empty_slot(col, i + 1)

    st.divider()

    # ── Main columns ───────────────────────────────────────────────────────────
    left, right = st.columns([3, 1.6])

    # ── LEFT: Player browser ───────────────────────────────────────────────────
    with left:
        st.markdown('### Player Browser')

        fc1, fc2, fc3 = st.columns([2, 2, 1.5])
        search      = fc1.text_input('Search', placeholder='Search players…',
                                     label_visibility='collapsed')
        tier_filter = fc2.multiselect('Tier', ['MAX','STAR','STARTER','ROLE','MIN'],
                                      default=['MAX','STAR','STARTER','ROLE','MIN'],
                                      label_visibility='collapsed',
                                      placeholder='Filter by tier…')
        sort_col    = fc3.selectbox('Sort', ['Price ($M)','NET','OFF_RATING',
                                             'DEF_RATING','PIE','USG_PCT'],
                                    label_visibility='collapsed')

        show = p.copy()
        if search:
            show = show[show['PLAYER_NAME'].str.contains(search, case=False, na=False)]
        if tier_filter:
            show = show[show['Tier'].isin(tier_filter)]
        show = show.sort_values(sort_col, ascending=(sort_col == 'DEF_RATING'))
        show['In Lineup'] = show['PLAYER_ID'].isin(lineup_ids)
        show['USG%']      = (show['USG_PCT'] * 100).round(1)
        show['AST%']      = (show['AST_PCT'] * 100).round(1)

        edit_cols = {
            'In Lineup':   st.column_config.CheckboxColumn('Add', width=50),
            'PLAYER_NAME': st.column_config.TextColumn('Player', width=170),
            'Tier Label':  st.column_config.TextColumn('Tier', width=90,
                help='Salary tier based on individual efficiency. MAX=$38–50M · STAR=$25–38M · STARTER=$13–25M · ROLE=$7–13M · MIN=$3–7M'),
            'Price ($M)':  st.column_config.NumberColumn('Price', format='$%.1fM', width=75,
                help='Salary cap value derived from PIE (55%), usage rate (30%), and minutes (15%). Budget is $130M for 5 players.'),
            'OFF_RATING':  st.column_config.NumberColumn('OFF', format='%.1f', width=60,
                help='Offensive rating — points scored per 100 possessions while this player was on court. League average is ~110.'),
            'DEF_RATING':  st.column_config.NumberColumn('DEF', format='%.1f', width=60,
                help='Defensive rating — points allowed per 100 possessions while this player was on court. Lower is better.'),
            'NET':         st.column_config.NumberColumn('NET', format='%+.1f', width=65,
                help='Net rating — offensive minus defensive rating. Positive means the team outscored opponents while this player was on court.'),
            'PIE':         st.column_config.NumberColumn('PIE', format='%.3f', width=65,
                help="Player Impact Estimate — share of total positive game events (pts, reb, ast, stl, blk minus misses/turnovers) attributed to this player. League average is ~0.10."),
            'USG%':        st.column_config.NumberColumn('USG%', format='%.1f%%', width=60,
                help='Usage rate — percentage of team possessions used by this player (via shot, free throw, or turnover) while on court.'),
            'FG3A':        st.column_config.NumberColumn('3PA/100', format='%.1f', width=75,
                help='Three-point attempts per 100 possessions. Indicates floor spacing ability.'),
            'AST%':        st.column_config.NumberColumn('AST%', format='%.1f%%', width=65,
                help='Assist percentage — share of teammate field goals this player assisted while on court. Measures playmaking and creation.'),
        }
        display_cols = ['In Lineup', 'PLAYER_NAME', 'Tier Label', 'Price ($M)',
                        'OFF_RATING', 'DEF_RATING', 'NET', 'PIE',
                        'USG%', 'FG3A', 'AST%']

        # Stable key: changes only when filter/sort changes, so edits don't carry over
        editor_key = f'browser_{search}_{sorted(tier_filter)}_{sort_col}'

        edited = st.data_editor(
            show[display_cols].reset_index(drop=True),
            column_config=edit_cols,
            disabled=[c for c in display_cols if c != 'In Lineup'],
            hide_index=True,
            use_container_width=True,
            height=560,
            key=editor_key,
        )

        # Process checkbox changes (only for players currently shown)
        shown_ids      = set(show['PLAYER_ID'].tolist())
        edited_pids    = set(show['PLAYER_ID'].iloc[edited[edited['In Lineup']].index].tolist())
        current_pids   = set(lineup_ids)

        added   = (edited_pids - current_pids) & shown_ids
        removed = (current_pids - edited_pids) & shown_ids

        rerun_needed = False
        for pid in removed:
            st.session_state.lineup.remove(pid)
            rerun_needed = True

        for pid in added:
            price = float(p.loc[p['PLAYER_ID'] == pid, 'Price ($M)'].values[0])
            if len(st.session_state.lineup) >= 5:
                st.warning('Lineup is full — remove a player first.')
            elif remaining < price:
                st.warning(
                    f'Over budget — that player costs ${price:.1f}M '
                    f'but only ${remaining:.1f}M remaining.')
            else:
                st.session_state.lineup.append(pid)
                rerun_needed = True

        if rerun_needed:
            st.rerun()

        n_shown = len(show)
        n_total = len(p)
        st.caption(f'Showing {n_shown} of {n_total} players · Check to add, uncheck to remove')

    # ── RIGHT: Prediction + history ────────────────────────────────────────────
    with right:
        st.markdown('### Prediction')

        if len(lineup_ids) == 5:
            pred = predict_lineup(
                [p.loc[p['PLAYER_ID'] == pid, PLAYER_STAT_COLS].values[0] for pid in lineup_ids],
                int(season[:4]), off_models, def_models, ts, ps)

            avg_ind_net = lineup_rows['NET'].mean()
            syn_delta   = pred['net'] - avg_ind_net
            nc = net_color(pred['net'])

            st.markdown(
                f'<div style="padding:16px;border-radius:8px;'
                f'border:1.5px solid {nc};text-align:center;">'
                f'<div style="font-size:0.72em;font-weight:600;letter-spacing:0.08em;'
                f'text-transform:uppercase;opacity:0.6;">Predicted NET Rating</div>'
                f'<div style="font-size:2.6em;font-weight:800;color:{nc};line-height:1.2;">'
                f'{pred["net"]:+.1f}</div>'
                f'<div style="font-size:0.85em;opacity:0.7;margin-top:2px;">'
                f'OFF {pred["off"]:.1f} &nbsp;·&nbsp; DEF {pred["def"]:.1f}</div>'
                f'</div>',
                unsafe_allow_html=True)

            syn_color = '#27ae60' if syn_delta >= 0 else '#e74c3c'
            st.markdown(
                f"""<div style="display:flex;justify-content:space-between;
                    text-align:center;margin-top:8px;gap:6px;">
                  <div style="flex:1;border:1px solid rgba(128,128,128,0.2);
                              border-radius:8px;padding:10px 4px;">
                    <div style="font-size:0.68em;font-weight:600;letter-spacing:0.06em;
                                text-transform:uppercase;opacity:0.5;">Avg Individual</div>
                    <div style="font-size:1.15em;font-weight:700;">{avg_ind_net:+.1f}</div>
                  </div>
                  <div style="flex:1;background:{_rgba(syn_color,0.08)};
                              border:1px solid {_rgba(syn_color,0.4)};
                              border-radius:8px;padding:10px 4px;">
                    <div style="font-size:0.68em;font-weight:600;letter-spacing:0.06em;
                                text-transform:uppercase;opacity:0.5;">Synergy Delta</div>
                    <div style="font-size:1.15em;font-weight:700;color:{syn_color};">{syn_delta:+.1f}</div>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

            verdict_text, verdict_color = (
                ('Elite lineup',   '#27ae60') if pred['net'] >= 12 else
                ('Strong lineup',  '#2ecc71') if pred['net'] >= 7  else
                ('Solid build',    '#f39c12') if pred['net'] >= 3  else
                ('Below average',  '#e67e22') if pred['net'] >= 0  else
                ('Weak lineup',    '#e74c3c')
            )
            st.markdown(
                f"<p style='font-size:0.9em;font-weight:600;color:{verdict_color};"
                f"margin-top:10px;'>{verdict_text}</p>",
                unsafe_allow_html=True)

            # Save to history
            names = ' / '.join(
                p.loc[p['PLAYER_ID'] == pid, 'PLAYER_NAME'].values[0]
                for pid in lineup_ids)
            cost = spent
            entry = {'Lineup': names, 'NET': pred['net'],
                     'OFF': pred['off'], 'DEF': pred['def'],
                     'Cost': f'${cost:.0f}M'}
            if not any(e['Lineup'] == names for e in st.session_state.history):
                st.session_state.history.append(entry)

            if st.button('Clear Lineup', use_container_width=True):
                st.session_state.lineup = []
                st.rerun()

        else:
            st.markdown(
                f'<div style="padding:28px 16px;border-radius:8px;'
                f'border:1px solid rgba(128,128,128,0.2);text-align:center;opacity:0.6;">'
                f'<div style="font-size:0.8em;letter-spacing:0.06em;text-transform:uppercase;">'
                f'Select {5 - len(lineup_ids)} more player(s)</div>'
                f'</div>',
                unsafe_allow_html=True)

        # ── Leaderboard ────────────────────────────────────────────────────────
        if st.session_state.history:
            st.divider()
            st.markdown('### Best Lineups')
            hist_df = (pd.DataFrame(st.session_state.history)
                       .sort_values('NET', ascending=False)
                       .head(8)
                       .reset_index(drop=True))
            hist_df.index += 1

            st.dataframe(
                hist_df[['Lineup', 'NET', 'OFF', 'DEF', 'Cost']],
                column_config={
                    'NET': st.column_config.NumberColumn('NET', format='%+.1f'),
                    'OFF': st.column_config.NumberColumn('OFF', format='%.1f'),
                    'DEF': st.column_config.NumberColumn('DEF', format='%.1f'),
                },
                use_container_width=True,
                height=min(36 * len(hist_df) + 38, 320),
            )

            if st.button('Clear history', use_container_width=True):
                st.session_state.history = []
                st.rerun()


if __name__ == '__main__':
    main()
