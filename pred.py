import pandas as pd
import numpy as np
from datetime import date, datetime
from nba_api.stats.endpoints import scoreboardv2, leaguedashplayerstats, commonallplayers
from nba_api.stats.static import teams as static_teams
import joblib as jl
import requests
from zoneinfo import ZoneInfo
import ssl, urllib.request, certifi
import warnings
from scipy.special import logit, expit
from nba_api.stats.endpoints import leaguegamefinder
from env_loader import load_env, get_env
from utils import load_state
from models.stacker import TwoStageStackTS, PurgedGroupTimeSeriesSplit
import re
import os
import unicodedata

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

load_env()
THE_ODDS_API_KEY = get_env("THE_ODDS_API_KEY", required=False)


"""
Ignoring redundant warnings
"""


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but PolynomialFeatures was fitted with feature names"
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names"
)
warnings.filterwarnings(
    "ignore",
    message="DataFrame is highly fragmented."
)
warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns."
)
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but StandardScaler was fitted without feature names"
)


"""
Maps
"""


TEAM_ID_TO_ABBR = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}

TEAM_MAP = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}


"""
Utility functions
"""


def decimal_odds(odds):
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100/abs(odds)) + 1
    
def beta_apply(p, theta):
    p = _clip(p)
    z = theta[0]*_logit(p) + theta[1]*np.log(1-p) + theta[2]
    return _clip(1/(1+np.exp(-z)))


def american_to_decimal(odds):
    odds = np.asarray(odds, dtype=float)
    return np.where(odds > 0, odds / 100 + 1, 100 / np.abs(odds) + 1)

EPS = 1e-6
def _clip(p):  return np.clip(np.asarray(p, float), EPS, 1 - EPS)
def _logit(p): p = _clip(p); return np.log(p/(1-p))

def apply_logit_pool(p_cal, p_mkt, w, b):
    z = w*logit(_clip(p_cal)) + (1-w)*logit(_clip(p_mkt)) + b
    return _clip(expit(z))

def norm_name(s: str) -> str:
    s = str(s)
    # 1) remove accents / diacritics (e.g., Dončić → Doncic)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")

    # 2) drop punctuation, keep letters/spaces (kills zero-width junk too)
    s = re.sub(r"[^A-Za-z\s]", " ", s)

    # 3) remove common suffixes
    s = re.sub(r"\b(JR|SR|II|III|IV|V)\b\.?", "", s, flags=re.I)

    # 4) squeeze spaces + uppercase for consistency
    s = re.sub(r"\s+", " ", s).strip().upper()
    return s

def load_bundle(path, use_cloudpickle=True):
    if use_cloudpickle:
        import cloudpickle as cp
        with open(path, "rb") as f:
            obj = cp.load(f)
    else:
        obj = jl.load(path)
    return obj["model"], obj["theta"], obj['FEATS']

def no_vig_probs(price_home, price_away):
    dec_home = american_to_decimal(price_home)
    dec_away = american_to_decimal(price_away)

    p_home_raw = 1 / dec_home
    p_away_raw = 1 / dec_away

    total = p_home_raw + p_away_raw
    p_home_nv = p_home_raw / total
    p_away_nv = p_away_raw / total

    return p_home_nv, p_away_nv

def apply_injury_adjustment(games, lost_elo_dict, beta_elo_per_pct=3.5, verbose=True):
    """
    Adjust home win probs using injury info from update_injuries().
    lost_elo_dict values may be proportions (0..1) or percentage points (0..100).
    beta_elo_per_pct: Elo points per 1 percentage point injury edge.
    """
    new_ps = []
    for _, row in games.iterrows():
        # missing-strength (proportions from your function)
        h = float(lost_elo_dict.get(row['team_home'], 0.0) or 0.0)
        a = float(lost_elo_dict.get(row['team_away'], 0.0) or 0.0)

        # convert to percentage points if values look like proportions
        if max(abs(h), abs(a)) <= 1.0:
            h *= 100.0
            a *= 100.0

        # injury edge in pp: positive => AWAY healthier than HOME (hurts HOME)
        diff_pp = a - h

        # model prob -> implied Elo diff
        p = float(np.clip(row['p_home'], 1e-6, 1 - 1e-6))
        elo_model = 400.0 * np.log10(p / (1.0 - p))

        # apply injury Elo adjustment
        elo_adj = elo_model + beta_elo_per_pct * diff_pp

        # back to probability
        new_p = 1.0 / (1.0 + 10.0 ** (-elo_adj / 400.0))
        new_ps.append(new_p)

        if verbose:
            print(f"{row['team_home']} vs {row['team_away']}: Δpp={diff_pp:+.1f}, "
                  f"p {p:.3f} -> {new_p:.3f} (Δ={new_p - p:+.3f})")

    return new_ps


"""
Updating 2025-26 values
"""


def get_games_so_far(season="2025-26"):
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00"
    )
    df = finder.get_data_frames()[0]
    df = df[df['PTS'].notna()]

    return df


def merge_games(df):
    df['HOME'] = df['MATCHUP'].str.contains('vs.')
    home_df = df[df['HOME']]
    away_df = df[~df['HOME']]

    merged = home_df.merge(
        away_df,
        on='GAME_ID',
        suffixes=('_home', '_away')
    )

    games_df = merged.rename(columns={
        'GAME_DATE_home': 'date',
        'TEAM_ABBREVIATION_home': 'team_home',
        'TEAM_ABBREVIATION_away': 'team_away',
        'PTS_home': 'score_home',
        'PTS_away': 'score_away'
    })
    games_df['date'] = pd.to_datetime(games_df['date'])
    games_df = games_df.sort_values('date').reset_index(drop=True)
    drop_cols = [
    'TEAM_ID_home', 'TEAM_ID_away', 'TEAM_NAME_home', 'TEAM_NAME_away',
    'MATCHUP_home', 'MATCHUP_away', 'GAME_ID',
    'WL_home', 'WL_away', 'HOME_home', 'HOME_away']
    games_df.drop(drop_cols, axis=1, inplace=True)

    return games_df


def update_elo_table(elo_dict, games_df, K, HCA, scale):
    for idx, row in games_df.iterrows():
        home, away = row['team_home'], row['team_away']
        score_home, score_away = row['score_home'], row['score_away']
        win_home = 1 if score_home > score_away else 0

        elo_dict.setdefault(home, 1500)
        elo_dict.setdefault(away, 1500)

        Ra, Rb = elo_dict[home], elo_dict[away]
        Ea = 1 / (1 + 10 ** ((Rb - (Ra + HCA)) / scale))
        margin = abs(score_home - score_away)
        mult = np.log1p(margin) * 2.2 / ((Ra+HCA - Rb) * 0.001 + 2.2)
        delta = K * mult * (win_home - Ea)

        elo_dict[home] += delta
        elo_dict[away] -= delta        

        games_df.loc[idx, 'elo_pre_home'] = Ra
        games_df.loc[idx, 'elo_pre_away'] = Rb
        games_df.loc[idx, 'elo_post_home'] = elo_dict[home]
        games_df.loc[idx, 'elo_post_away'] = elo_dict[away]

    return elo_dict


def team_state_as_of_today_df(
    games: pd.DataFrame,
    win_window: int = 10,
    fill_rest: int = 7,
    default_rw: float = 0.5,
    today: pd.Timestamp | None = None,
    roll_windows=(3,5),
) -> pd.DataFrame:
    # --- dates / wins / season filter ---
    if today is None:
        today = pd.Timestamp.today().normalize()
    df = games.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    if "win_home" not in df.columns:
        if {"score_home", "score_away"}.issubset(df.columns):
            df["win_home"] = (df["score_home"] > df["score_away"]).astype(int)
        else:
            raise ValueError("Need win_home or scores to compute wins.")

    cutoff = today - pd.Timedelta(days=1)
    df = df[df["date"] <= cutoff].copy()

    if "sid" in df.columns:
        cur_sid = df["sid"].max()
        df = df[df["sid"] == cur_sid].copy()
    else:
        season_year = np.where(df["date"].dt.month >= 7, df["date"].dt.year, df["date"].dt.year - 1)
        cur_season = season_year.max()
        df = df[season_year == cur_season].copy()

    df = df.reset_index(drop=False).rename(columns={"index": "row_id"})

    # --- box-score bases (same as add_features) ---
    df["off_eff_home"] = df["score_home"] / df["FGA_home"]
    df["off_eff_away"] = df["score_away"] / df["FGA_away"]
    df['def_eff_home'] = (df['BLK_home'] + df['STL_home'] - df['PF_home']) / df['score_away']
    df['def_eff_away'] = (df['BLK_away'] + df['STL_away'] - df['PF_away']) / df['score_home']

    df['poss_home'] = 0.96 * (df['FGA_home'] + 0.44*df['FTA_home'] - df['OREB_home'] + df['TOV_home'])
    df['poss_away'] = 0.96 * (df['FGA_away'] + 0.44*df['FTA_away'] - df['OREB_away'] + df['TOV_away'])
    df['pace_home'] = (df['poss_home'] + df['poss_away']) / (2 * df['MIN_home'])
    df['pace_away'] = (df['poss_home'] + df['poss_away']) / (2 * df['MIN_away'])

    df['off_rtg_home'] = df['score_home'] / df['poss_home'] * 100
    df['def_rtg_home'] = df['score_away'] / df['poss_away'] * 100
    df['off_rtg_away'] = df['score_away'] / df['poss_away'] * 100
    df['def_rtg_away'] = df['score_home'] / df['poss_home'] * 100

    df['fgp_home'] = df['FGM_home'] / df['FGA_home']
    df['fgp_away'] = df['FGM_away'] / df['FGA_away']
    df['tpp_home'] = df['FG3M_home'] / df['FG3A_home']
    df['tpp_away'] = df['FG3M_away'] / df['FG3A_away']

    df["efg_home"]     = (df["FGM_home"] + 0.5 * df["FG3M_home"]) / df["FGA_home"]
    df["efg_away"]     = (df["FGM_away"] + 0.5 * df["FG3M_away"]) / df["FGA_away"]

    df["tov_rate_home"] = df["TOV_home"] / (df["FGA_home"] + 0.44 * df["FTA_home"] + df["TOV_home"])
    df["tov_rate_away"] = df["TOV_away"] / (df["FGA_away"] + 0.44 * df["FTA_away"] + df["TOV_away"])
    df['ast_tov_ratio_home'] = df['AST_home'] / df['TOV_home']
    df['ast_tov_ratio_away'] = df['AST_away'] / df['TOV_away']

    df["reb_rate_home"] = df["REB_home"] / (df["REB_home"] + df["REB_away"])
    df["reb_rate_away"] = 1 - df["reb_rate_home"]

    df['orbp_home'] = df['OREB_home'] / (df['OREB_home'] + df['DREB_away']) 
    df['orbp_away'] = df['OREB_away'] / (df['OREB_away'] + df['DREB_home']) 
    df['drbp_home'] = df['DREB_home'] / (df['DREB_home'] + df['OREB_away'])
    df['drbp_away'] = df['DREB_away'] / (df['DREB_away'] + df['OREB_home'])

    df["ft_eff_home"]   = df["FTM_home"] / df["FTA_home"].replace(0, np.nan)
    df["ft_eff_away"]   = df["FTM_away"] / df["FTA_away"].replace(0, np.nan)

    df['ff_home'] = 0.4*df['efg_home'] + 0.25*df['tov_rate_home'] + 0.2 * (df['orbp_home'] + df['drbp_home']) / 2 + 0.15*df['ft_eff_home']
    df['ff_away'] = 0.4*df['efg_away'] + 0.25*df['tov_rate_away'] + 0.2 * (df['orbp_away'] + df['drbp_away']) / 2 + 0.15*df['ft_eff_away']

    # --- per-team logs for season state / venue / streak / rest ---
    home = df[["date","team_home","score_home","score_away","win_home","row_id"]].rename(
        columns={"team_home":"team","score_home":"pf","score_away":"pa","win_home":"win"}
    ); home["is_home"]=1
    away = df[["date","team_away","score_home","score_away","win_home","row_id"]].rename(
        columns={"team_away":"team","score_home":"pa","score_away":"pf","win_home":"win"}
    ); away["is_home"]=0; away["win"]=1-away["win"]

    logs = pd.concat([home,away], ignore_index=True)
    logs = logs.sort_values(["team","date","row_id"], kind="mergesort").reset_index(drop=True)
    logs["pt_diff"] = (logs["pf"] - logs["pa"]).astype(float)
    g = logs.groupby("team", sort=False)

    last_dates = g["date"].max()
    rest = ((today - last_dates).dt.days - 1).clip(lower=0)
    b2b_today = (rest == 0).astype(int)

    def _last_n_mean(s,n): return s.tail(n).mean() if len(s) else np.nan
    rw = g["win"].apply(lambda s: _last_n_mean(s, win_window)).fillna(default_rw)

    def _ending_streak(s):
        if s.empty: return 0
        last = int(s.iloc[-1]); cnt = 0
        for x in reversed(s.to_list()):
            if x == last: cnt += 1
            else: break
        return cnt if last==1 else -cnt
    streak = g["win"].apply(_ending_streak).fillna(0).astype(int)

    gp_incl = g.cumcount() + 1
    gp_prev = gp_incl - 1
    pf_cum  = g["pf"].cumsum()
    pa_cum  = g["pa"].cumsum()
    off_eff = ( (pf_cum - logs["pf"]).groupby(logs["team"]).last()
                / gp_prev.groupby(logs["team"]).last() ).replace([np.inf,-np.inf],np.nan)
    def_eff = ( (pa_cum - logs["pa"]).groupby(logs["team"]).last()
                / gp_prev.groupby(logs["team"]).last() ).replace([np.inf,-np.inf],np.nan)

    incl_denom = gp_incl.groupby(logs["team"]).last()
    off_eff = off_eff.fillna(pf_cum.groupby(logs["team"]).last()/incl_denom)
    def_eff = def_eff.fillna(pa_cum.groupby(logs["team"]).last()/incl_denom)

    spdiff = (g["pt_diff"].cumsum() - logs["pt_diff"]).groupby(logs["team"]).last()

    logs["home_game"]=(logs["is_home"]==1).astype(int); logs["away_game"]=(logs["is_home"]==0).astype(int)
    logs["home_win"]=logs["win"]*logs["home_game"]; logs["away_win"]=logs["win"]*logs["away_game"]
    logs["home_pdiff"]=logs["pt_diff"]*logs["home_game"]; logs["away_pdiff"]=logs["pt_diff"]*logs["away_game"]
    h_games_in=g["home_game"].cumsum(); a_games_in=g["away_game"].cumsum()
    h_wins_in=g["home_win"].cumsum(); a_wins_in=g["away_win"].cumsum()
    h_pd_in=g["home_pdiff"].cumsum(); a_pd_in=g["away_pdiff"].cumsum()
    h_games=(h_games_in-logs["home_game"]).groupby(logs["team"]).last()
    a_games=(a_games_in-logs["away_game"]).groupby(logs["team"]).last()
    h_wins=(h_wins_in-logs["home_win"]).groupby(logs["team"]).last()
    a_wins=(a_wins_in-logs["away_win"]).groupby(logs["team"]).last()
    h_pd  =(h_pd_in-logs["home_pdiff"]).groupby(logs["team"]).last()
    a_pd  =(a_pd_in-logs["away_pdiff"]).groupby(logs["team"]).last()

    home_win_pct=(h_wins/h_games).replace([np.inf,-np.inf],np.nan)
    away_win_pct=(a_wins/a_games).replace([np.inf,-np.inf],np.nan)
    home_point_diff=(h_pd/h_games).replace([np.inf,-np.inf],np.nan)
    away_point_diff=(a_pd/a_games).replace([np.inf,-np.inf],np.nan)

    # --- Elo pre rolling (same as add_features) ---
    if {"elo_pre_home","elo_pre_away"}.issubset(df.columns):
        logs["elo_pre"] = np.where(
            logs["is_home"]==1,
            df.loc[logs["row_id"],"elo_pre_home"].values,
            df.loc[logs["row_id"],"elo_pre_away"].values
        ).astype(float)
        logs["opp_elo_pre"] = np.where(
            logs["is_home"]==1,
            df.loc[logs["row_id"],"elo_pre_away"].values,
            df.loc[logs["row_id"],"elo_pre_home"].values
        ).astype(float)
        logs["elo_vs_opp"] = logs["elo_pre"] - logs["opp_elo_pre"]
        def _elo_roll_feats(e, w=5):
            ma5 = e.rolling(w, min_periods=2).mean()
            slope5 = e.rolling(w, min_periods=2).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=False)
            de = e.diff()
            mom = de.rolling(w, min_periods=2).mean()
            vol = de.rolling(w, min_periods=2).std()
            return pd.DataFrame({"elo_pre_ma5":ma5,"elo_pre_slope5":slope5,"elo_momentum":mom,"elo_volatility":vol})
        elo_df = g["elo_pre"].apply(_elo_roll_feats).reset_index(level=0, drop=True)
        logs = pd.concat([logs, elo_df], axis=1)
        logs["elo_vs_opp_roll5"] = g["elo_vs_opp"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        elo_pre_ma5     = logs.groupby("team")["elo_pre_ma5"].last()
        elo_pre_slope5  = logs.groupby("team")["elo_pre_slope5"].last()
        elo_momentum    = logs.groupby("team")["elo_momentum"].last()
        elo_volatility  = logs.groupby("team")["elo_volatility"].last()
        elo_vs_opp_roll5 = logs.groupby("team")["elo_vs_opp_roll5"].last()
    else:
        teams_all = logs["team"].unique()
        elo_pre_ma5     = pd.Series(np.nan, index=teams_all)
        elo_pre_slope5  = pd.Series(np.nan, index=teams_all)
        elo_momentum    = pd.Series(np.nan, index=teams_all)
        elo_volatility  = pd.Series(np.nan, index=teams_all)
        logs["elo_vs_opp_roll5"] = np.nan
        elo_vs_opp_roll5 = pd.Series(np.nan, index=teams_all)

    # --- per-team lag & rolling (3,5) for ALL numeric *_home/*_away bases ---
    num_cols = df.select_dtypes(include=[np.number]).columns
    home_bases = {c[:-5] for c in num_cols if c.endswith("_home")}
    away_bases = {c[:-5] for c in num_cols if c.endswith("_away")}
    bases = sorted(home_bases & away_bases)

    # make long per-team frame of those bases
    H = df[["date","team_home"] + [f"{b}_home" for b in bases]].rename(
        columns={"team_home":"team", **{f"{b}_home": b for b in bases}}
    )
    A = df[["date","team_away"] + [f"{b}_away" for b in bases]].rename(
        columns={"team_away":"team", **{f"{b}_away": b for b in bases}}
    )
    long = (
        pd.concat([H, A], ignore_index=True)
        .sort_values(["team","date"])
    )

    def _make_prev_roll(g):
        out = g.copy()
        for b in bases:
            out[f"{b}_prev"] = g[b].shift(1)
            for w in roll_windows:
                out[f"{b}_roll{w}"] = g[b].ewm(w, min_periods=1).mean().shift(1)
        cols = ["team","date"] + [c for c in out.columns if re.search(r"_(prev|roll\d+)$", c)]
        return out[cols]

    long_fr = long.groupby("team", group_keys=False).apply(_make_prev_roll)
    last_fr = long_fr.groupby("team").last().reset_index()

    base_cols = [b for b in ("off_rtg", "def_rtg") if b in long.columns]
    current_base = None
    if base_cols:
        current_base = (
            long.groupby("team")[base_cols]
            .last()
            .reset_index()
        )

    # --- assemble per-team state row ---
    teams = pd.Index(logs["team"].unique(), name="team")
    out = pd.DataFrame(index=teams)
    out["rest"] = rest.reindex(teams).fillna(fill_rest)
    out["rw_pct"] = rw.reindex(teams).fillna(default_rw)
    out["streak"] = streak.reindex(teams).fillna(0).astype(int)
    out["spdiff"] = spdiff.reindex(teams).fillna(0.0)
    out["off_eff"] = off_eff.reindex(teams).fillna(0.0)
    out["def_eff"] = def_eff.reindex(teams).fillna(0.0)
    out["home_win_pct"] = home_win_pct.reindex(teams).fillna(0.5)
    out["away_win_pct"] = away_win_pct.reindex(teams).fillna(0.5)
    out["home_point_diff"] = home_point_diff.reindex(teams).fillna(0.0)
    out["away_point_diff"] = away_point_diff.reindex(teams).fillna(0.0)
    out["elo_pre_ma5"] = elo_pre_ma5.reindex(teams).fillna(0.0)
    out["elo_pre_slope5"] = elo_pre_slope5.reindex(teams).fillna(0.0)
    out["elo_momentum"] = elo_momentum.reindex(teams).fillna(0.0)
    out["elo_volatility"] = elo_volatility.reindex(teams).fillna(0.0)
    out["elo_vs_opp_roll5"] = elo_vs_opp_roll5.reindex(teams).fillna(0.0)
    out["b2b"] = b2b_today.reindex(teams).fillna(0).astype(int)

    out = out.reset_index().merge(last_fr, on="team", how="left")
    if current_base is not None:
        out = out.merge(current_base, on="team", how="left")
    return out


# === Schedule-adjusted ratings (ridge) ===
def _fit_atk_def_ridge(gsub: pd.DataFrame,
                       teams: list[str],
                       lam: float = 50.0,
                       constraint_w: float = 100.0):
    """
    Solve for team attack/defense and a single HCA using ridge regression.
    Model:
        pts_home = atk[h] - def[a] + hca
        pts_away = atk[a] - def[h] - hca
    Identifiability via soft mean-zero constraints on atk/def.
    """
    import numpy as np
    T = len(teams)
    idx = {t:i for i,t in enumerate(teams)}
    G = len(gsub)

    # variables = [atk_0..atk_{T-1}, def_0..def_{T-1}, hca]
    P = 2*T + 1
    rows = 2*G + 2  # 2 per game + 2 constraints
    X = np.zeros((rows, P), float)
    y = np.zeros(rows, float)

    r = 0
    for h, a, ph, pa in zip(gsub["team_home"], gsub["team_away"], gsub["score_home"], gsub["score_away"]):
        ih, ia = idx[h], idx[a]
        # home points
        X[r, ih] = 1.0; X[r, T+ia] = -1.0; X[r, 2*T] = 1.0; y[r] = float(ph); r += 1
        # away points
        X[r, ia] = 1.0; X[r, T+ih] = -1.0; X[r, 2*T] = -1.0; y[r] = float(pa); r += 1

    # soft constraints: mean(atk)=0, mean(def)=0
    X[r, :T]      = constraint_w / np.sqrt(T); y[r] = 0.0; r += 1
    X[r, T:2*T]   = constraint_w / np.sqrt(T); y[r] = 0.0; r += 1

    XtX = X.T @ X
    Xty = X.T @ y
    XtX += lam * np.eye(P)

    coef = np.linalg.solve(XtX, Xty)
    atk = coef[:T]; deff = coef[T:2*T]; hca = float(coef[2*T])

    # tiny numeric cleanup
    atk  -= atk.mean()
    deff -= deff.mean()
    return atk, deff, hca, idx

def sched_ratings_as_of_today(hist_games: pd.DataFrame,
                              today: pd.Timestamp,
                              lam: float = 50.0,
                              constraint_w: float = 100.0):
    """
    Fit schedule-adjusted ratings using ONLY games with date < today (same season).
    Returns dicts: atk_d, def_d, hca_est
    """
    df = hist_games.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # choose current season
    if "sid" in df.columns:
        cur_sid = df["sid"].max()
        df = df[df["sid"] == cur_sid]
    else:
        season_year = np.where(df["date"].dt.month >= 7, df["date"].dt.year, df["date"].dt.year - 1)
        cur_season = season_year.max()
        df = df[season_year == cur_season]

    # restrict strictly before "today" (no same-day leakage)
    df = df[df["date"] < today]
    if df.empty:
        return {}, {}, 0.0

    teams = pd.Index(pd.unique(pd.concat([df["team_home"], df["team_away"]], ignore_index=True))).tolist()
    if len(df) < max(10, len(teams)):   # very early season guard
        return {}, {}, 0.0

    atk, deff, hca, idx = _fit_atk_def_ridge(df, teams, lam=lam, constraint_w=constraint_w)
    atk_d = {t: float(atk[i]) for t,i in idx.items()}
    def_d = {t: float(deff[i]) for t,i in idx.items()}
    return atk_d, def_d, float(hca)


def build_today_features(
    hist_games: pd.DataFrame,
    today_games: pd.DataFrame,
    elo_dict: dict[str, float],
    HCA: float,
    win_window: int = 10,
    fill_rest: int = 7,
    roll_windows=(3,5),
) -> pd.DataFrame:
    state = team_state_as_of_today_df(
        hist_games, win_window=win_window, fill_rest=fill_rest, today=pd.Timestamp.today().normalize(), roll_windows=roll_windows
    )

    H = today_games.rename(columns={"team_home":"team"})[["date","team"]].merge(
        state.add_suffix("_h"), left_on="team", right_on="team_h", how="left"
    ).drop(columns=["team_h"])
    A = today_games.rename(columns={"team_away":"team"})[["date","team"]].merge(
        state.add_suffix("_a"), left_on="team", right_on="team_a", how="left"
    ).drop(columns=["team_a"])

    feat = today_games.copy()
    feat = feat.merge(H, left_on=["team_home","date"], right_on=["team","date"], how="left").drop(columns=["team"])
    feat = feat.merge(A, left_on=["team_away","date"], right_on=["team","date"], how="left").drop(columns=["team"])

    _today = pd.Timestamp.today().normalize()
    atk_d, def_d, hca_est = sched_ratings_as_of_today(hist_games, today=_today, lam=50.0, constraint_w=100.0)

    # map ratings into today's pairings
    feat["atk_pre_home"] = feat["team_home"].map(atk_d).fillna(0.0)
    feat["def_pre_home"] = feat["team_home"].map(def_d).fillna(0.0)
    feat["atk_pre_away"] = feat["team_away"].map(atk_d).fillna(0.0)
    feat["def_pre_away"] = feat["team_away"].map(def_d).fillna(0.0)
    feat["hca_est_pre"]  = float(hca_est)

    # engineered diffs you reference in FEATURE_COLS
    feat["atk_diff"]            = feat["atk_pre_home"] - feat["atk_pre_away"]
    feat["def_diff"]            = feat["def_pre_home"] - feat["def_pre_away"]
    feat["atk_minus_oppdef"]    = feat["atk_pre_home"] - feat["def_pre_away"]
    feat["def_minus_oppatk"]    = feat["def_pre_home"] - feat["atk_pre_away"]

    # --- Elo & basic diffs ---
    feat["elo_home"] = feat["team_home"].map(elo_dict).fillna(1500.0) + HCA
    feat["elo_away"] = feat["team_away"].map(elo_dict).fillna(1500.0)
    feat["elo_diff"] = feat["elo_home"] - feat["elo_away"]

    feat["rest_diff"]   = feat["rest_h"] - feat["rest_a"]
    feat["rw_pct_diff"] = feat["rw_pct_h"] - feat["rw_pct_a"]
    feat["streak_diff"] = feat["streak_h"] - feat["streak_a"]
    feat["spdiff_diff"] = feat["spdiff_h"] - feat["spdiff_a"]
    feat["off_eff_diff"]= feat["off_eff_h"] - feat["off_eff_a"]
    feat["def_eff_diff"]= feat["def_eff_h"] - feat["def_eff_a"]
    if {"off_rtg_h", "def_rtg_a", "def_rtg_h", "off_rtg_a"}.issubset(feat.columns):
        feat["off_def_rtg_diff"] = feat["off_rtg_h"] - feat["def_rtg_a"]
        feat["def_off_rtg_diff"] = feat["def_rtg_h"] - feat["off_rtg_a"]
    else:
        zero = pd.Series(0.0, index=feat.index)
        feat["off_def_rtg_diff"] = feat.get("off_rtg_prev_h", zero) - feat.get("def_rtg_prev_a", zero)
        feat["def_off_rtg_diff"] = feat.get("def_rtg_prev_h", zero) - feat.get("off_rtg_prev_a", zero)
    feat["net_off_diff"]= feat["off_eff_h"] - feat["def_eff_a"]
    feat["net_def_diff"]= feat["def_eff_h"] - feat["off_eff_a"]
    feat["hvenue_winpct_diff"] = feat["home_win_pct_h"] - feat["away_win_pct_a"]
    feat["hvenue_pdiff_diff"]  = feat["home_point_diff_h"] - feat["away_point_diff_a"]
    feat["b2b_home"] = (feat["rest_h"]==0).astype(int)
    feat["b2b_away"] = (feat["rest_a"]==0).astype(int)

    feat["elo_pre_ma5_diff"]    = feat["elo_pre_ma5_h"]    - feat["elo_pre_ma5_a"]
    feat["elo_pre_slope5_diff"] = feat["elo_pre_slope5_h"] - feat["elo_pre_slope5_a"]
    feat["elo_momentum_diff"]   = feat["elo_momentum_h"]   - feat["elo_momentum_a"]
    feat["elo_volatility_diff"] = feat["elo_volatility_h"] - feat["elo_volatility_a"]
    feat["elo_vs_opp_roll5_diff"] = feat["elo_vs_opp_roll5_h"] - feat["elo_vs_opp_roll5_a"]

    # --- box-score prev/rolling diffs for ALL bases coming from state ---
    pattern = re.compile(r"^(?P<base>.+)_(?P<suf>prev|roll\d+)_(?P<side>[ha])$")

    # build an index of available home/away columns by (base, suf)
    pairs = {}
    for c in feat.columns:
        m = pattern.match(c)
        if m:
            key = (m.group("base"), m.group("suf"))
            side = m.group("side")  # 'h' or 'a'
            pairs.setdefault(key, {})[side] = c

    for (base, suf), sides in pairs.items():
        if "h" in sides and "a" in sides:
            feat[f"{base}_{suf}_diff"] = feat[sides["h"]] - feat[sides["a"]]

    # --- engineered combos ---
    feat["momentum_index"] = 0.4*feat["rw_pct_diff"] + 0.3*feat["streak_diff"]/5 + 0.3*np.tanh(feat["spdiff_diff"]/20)
    feat["elo_x_momentum"] = feat["elo_diff"] * feat["momentum_index"]
    feat["elo_x_rest"]     = feat["elo_diff"] * feat["rest_diff"]
    feat["elo_momentum_x_rest"] = feat["elo_momentum_diff"] * feat["rest_diff"]
    feat["off_def_product"]= feat["off_eff_diff"] * feat["def_eff_diff"]
    feat["venue_x_streak"] = feat["hvenue_winpct_diff"] * feat["streak_diff"]
    feat['elo_vs_opp_x_streak'] = feat['elo_vs_opp_roll5_diff'] * feat['streak_diff']
    feat['elo_vs_opp_x_winpct'] = feat['elo_vs_opp_roll5_diff'] * feat['rw_pct_diff']
    feat['elo_vs_opp_x_PLUS_MINUS'] = feat['elo_vs_opp_roll5_diff'] * feat['PLUS_MINUS_roll5_diff']

    return feat


"""
Getting injury update
"""


def update_injuries(t_games, T=None):
    import jpype
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), "--enable-native-access=ALL-UNNAMED")
    from nbainjuries import injury

    _ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    urllib.request.install_opener(
        urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_ctx))
    )

    lost_elo_dict = {team: 0 for team in pd.unique(t_games[['team_home', 'team_away']].values.ravel())}
    mult = {
        'Out': 1,
        'Doubtful' : 4/5,
        'Questionable' : 2/3,
        'Probable' : 1/3,
        'Available': 0
    }

    now_et = datetime.now(ZoneInfo("America/New_York"))
    injuries = injury.get_reportdata(now_et.replace(tzinfo=None), return_df=True)
    injuries = injuries[injuries['Game Date'] == now_et.date().strftime('%m/%d/%Y')]
    injuries['Player Name'] = injuries['Player Name'].str.replace(
        r'^\s*([^,]+),\s*(.+)\s*$',  
        r'\2 \1',                  
        regex=True
    ).str.replace(r'\s+', ' ', regex=True).str.strip()
    injuries.dropna(inplace=True)
    injuries = injuries[~injuries["Reason"].str.lower().isin([
    "not with team",
    'g league - two-way',
    "g league - on assignment"
    ])]
    stats_25 = leaguedashplayerstats.LeagueDashPlayerStats(season='2024-25', per_mode_detailed='PerGame')
    mins_25 = stats_25.get_data_frames()[0]
    mins_25 = mins_25[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'NBA_FANTASY_PTS']]

    stats_26 = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26', per_mode_detailed='PerGame')
    mins_26 = stats_26.get_data_frames()[0]
    mins_26 = mins_26[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'NBA_FANTASY_PTS']]    
   
    teams = commonallplayers.CommonAllPlayers(
        season='2025-26',
        is_only_current_season=1
    )
    teams = teams.get_data_frames()[0]
    teams["NAME_NORM"]   = teams["DISPLAY_FIRST_LAST"].map(norm_name)
    mins_26['NAME_NORM'] = mins_26['PLAYER_NAME'].map(norm_name)
    mins_25['NAME_NORM'] = mins_25['PLAYER_NAME'].map(norm_name)
    injuries['NAME_NORM'] = injuries['Player Name'].map(norm_name)

    for p in injuries['NAME_NORM']:
        try: team = teams.loc[teams['NAME_NORM'] == p, 'TEAM_ABBREVIATION'].iloc[0]
        except: 
            print(f'no team data for {p}')
            continue
        try: 
            try: 
                fp = mins_26.loc[mins_26['NAME_NORM'] == p, 'NBA_FANTASY_PTS'].iloc[0]
                total = mins_26.loc[mins_26['TEAM_ABBREVIATION'] == team, 'NBA_FANTASY_PTS'].sum()
            except: 
                fp = mins_25.loc[mins_25['NAME_NORM'] == p, 'NBA_FANTASY_PTS'].iloc[0]
                total = mins_25.loc[mins_25['TEAM_ABBREVIATION'] == team, 'NBA_FANTASY_PTS'].sum()
        except: 
            print(f'no mins data for {p}')
            continue
        
        status = injuries.loc[injuries['NAME_NORM'] == p, 'Current Status'].iloc[0]
        ratio = fp/total * mult[status]
        lost_elo_dict[team] += ratio

    if T:    
        vals = {k: float(v) for k, v in lost_elo_dict.items()}
        if not vals:
            return {}

        # Diminishing returns with hard cap T

        factors = {}
        for team, value in vals.items():
            v = max(0.0, value)           # no negatives
            adjusted = T * np.tanh(v / max(T, 1e-9))
            factors[team] = float(adjusted)

        return factors
    else:
        return lost_elo_dict


"""
Getting NBA lines for odds comparison
"""


def get_nba_lines(the_odds_api_key=None, region="us", markets="h2h", odds_format="american"):
    api_key = the_odds_api_key or THE_ODDS_API_KEY or get_env("THE_ODDS_API_KEY")
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "regions": region,       
        "markets": markets,       
        "oddsFormat": odds_format, 
        "dateFormat": "iso",
        "apiKey": api_key
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    rows = []
    for g in data:
        gid = g["id"]
        start = g["commence_time"]
        home = g["home_team"]
        away = g["away_team"]
        for bk in g.get("bookmakers", []):
            book = bk["title"]
            # find the h2h market
            for mk in bk.get("markets", []):
                if mk["key"] != "h2h": 
                    continue
                prices = {o["name"]: int(o["price"]) for o in mk["outcomes"]}
                if home in prices and away in prices:
                    rows.append({
                        "game_id": gid,
                        "commence_time": start,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "price_home": prices[home],
                        "price_away": prices[away],
                    })
    data = pd.DataFrame(rows)

    data = data[data['book'] == 'DraftKings'].reset_index(drop=True)
    data["tip_utc"] = pd.to_datetime(data["commence_time"], utc=True)
    data["tip_pst"] = data["tip_utc"].dt.tz_convert(ZoneInfo("America/Los_Angeles"))
    data["date_pst"] = data["tip_pst"].dt.date
    data = data[data['date_pst'] == datetime.now(ZoneInfo("America/Los_Angeles")).date()]

    data = data[['home_team', 'away_team', 'price_home', 'price_away']]

    return data


"""
Getting todays games and prediciting proability
"""


def get_todays_games():
    sb = scoreboardv2.ScoreboardV2(game_date=date.today().strftime("%m/%d/%Y"))
    frames = sb.get_data_frames()
    if not frames or frames[0].empty:
        return pd.DataFrame(columns=["date","game_id","home_team","away_team","home_score","away_score","status"])

    header = frames[0]

    df = header[[
        "GAME_ID","GAME_DATE_EST","GAME_STATUS_TEXT","HOME_TEAM_ID","VISITOR_TEAM_ID"
    ]].copy()
    df["team_home"] = df["HOME_TEAM_ID"].map(TEAM_ID_TO_ABBR)
    df["team_away"] = df["VISITOR_TEAM_ID"].map(TEAM_ID_TO_ABBR)

    if len(frames) > 1 and not frames[1].empty:
        lines = frames[1]
        home = lines[lines["HOME_TEAM_ID"].notna()][["GAME_ID","PTS"]].rename(columns={"PTS":"score_home"})
        away = lines[lines["HOME_TEAM_ID"].isna()][["GAME_ID","PTS"]].rename(columns={"PTS":"score_away"})
        df = df.merge(home, on="GAME_ID", how="left").merge(away, on="GAME_ID", how="left")
    else:
        df["score_home"] = pd.NA
        df["score_away"] = pd.NA

    df["date"] = pd.to_datetime(df["GAME_DATE_EST"]).dt.date
    df = df.rename(columns={"GAME_STATUS_TEXT":"status"})[
        ["date","GAME_ID","team_home","team_away","score_home","score_away","status"]
    ]
    return df


def predict_games(model, theta, elo_dict, hist_games, HCA, FEATS, 
                 INJURIES=False, T=None, b=0.8, target_date=None):
    """
    Predict today's games with consistent date handling.
    
    Parameters
    ----------
    target_date : date, optional
        The date to predict for. If None, uses today in Pacific time.
    """
    # Establish single source of truth for "today"
    if target_date is None:
        # Default: today in Pacific timezone (where NBA operates)
        target_date = datetime.now(ZoneInfo("America/Los_Angeles")).date()
    elif isinstance(target_date, (str, int, float, datetime)):
        # Convert to date if needed
        if isinstance(target_date, str):
            target_date = datetime.fromisoformat(target_date).date()
        elif isinstance(target_date, (int, float)):
            target_date = datetime.utcfromtimestamp(target_date).date()
        elif isinstance(target_date, datetime):
            target_date = target_date.date()
    
    print(f"[predict_games] Predicting for date: {target_date}", flush=True)
    
    # --- Get today's games (with consistent date filter) ---
    sb = scoreboardv2.ScoreboardV2(game_date=target_date.strftime("%m/%d/%Y"))
    frames = sb.get_data_frames()
    if not frames or frames[0].empty:
        print(f"[predict_games] No games found for {target_date}", flush=True)
        return pd.DataFrame()

    header = frames[0]
    today_games = header[[
        "GAME_ID","GAME_DATE_EST","GAME_STATUS_TEXT","HOME_TEAM_ID","VISITOR_TEAM_ID"
    ]].copy()
    today_games["team_home"] = today_games["HOME_TEAM_ID"].map(TEAM_ID_TO_ABBR)
    today_games["team_away"] = today_games["VISITOR_TEAM_ID"].map(TEAM_ID_TO_ABBR)

    if len(frames) > 1 and not frames[1].empty:
        lines = frames[1]
        home = lines[lines["HOME_TEAM_ID"].notna()][["GAME_ID","PTS"]].rename(
            columns={"PTS":"score_home"}
        )
        away = lines[lines["HOME_TEAM_ID"].isna()][["GAME_ID","PTS"]].rename(
            columns={"PTS":"score_away"}
        )
        today_games = today_games.merge(home, on="GAME_ID", how="left").merge(
            away, on="GAME_ID", how="left"
        )
    else:
        today_games["score_home"] = pd.NA
        today_games["score_away"] = pd.NA

    today_games["date"] = pd.to_datetime(today_games["GAME_DATE_EST"]).dt.date
    today_games = today_games.rename(columns={"GAME_STATUS_TEXT":"status"})[
        ["date","GAME_ID","team_home","team_away","score_home","score_away","status"]
    ]
    
    print(f"[predict_games] Found {len(today_games)} games", flush=True)
    
    # --- Injuries (if requested) ---
    if INJURIES:
        lost_elo_dict = update_injuries(today_games, T)

    # --- Features for today ---
    feat_today = build_today_features(
        hist_games=hist_games,
        today_games=today_games,
        elo_dict=elo_dict,
        HCA=HCA
    )

    def _predict_proba_binary(model, X):
        X_array = np.asarray(X, dtype=np.float64)
        
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_array)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    return proba[:, 1]
                if proba.ndim == 1:
                    return proba
        except (SystemError, AttributeError, RuntimeError, ValueError) as e:
            print(f"[ERROR] predict_proba failed ({type(e).__name__}): {e}", flush=True)
            try:
                print(f"[RETRY] Trying with float32...", flush=True)
                proba = model.predict_proba(X_array.astype(np.float32))
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    return proba[:, 1]
            except Exception as e2:
                print(f"[ERROR] float32 retry also failed: {e2}", flush=True)
        
        try:
            if hasattr(model, "decision_function"):
                print(f"[FALLBACK] Using decision_function", flush=True)
                return expit(model.decision_function(X_array))
        except Exception as e:
            print(f"[ERROR] decision_function failed: {e}", flush=True)
        
        try:
            print(f"[FALLBACK] Using predict()", flush=True)
            yhat = np.asarray(model.predict(X_array), dtype=float)
            rng = yhat.max() - yhat.min()
            if rng > 0:
                return (yhat - yhat.min()) / rng
            return np.full(len(X_array), 0.5)
        except Exception as e:
            print(f"[ERROR] All prediction methods failed: {e}", flush=True)
            return np.full(len(X_array), 0.5)
    
    # Build feature matrix
    X = feat_today.reindex(columns=FEATS)
    missing = [c for c in FEATS if c not in feat_today.columns]
    if missing:
        print(f"[predict_games] Missing features filled with 0: {missing}", flush=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Predict
    feat_today["p_home"] = _predict_proba_binary(model, X)

    # --- Get odds (with consistent date filter) ---
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    api_key = THE_ODDS_API_KEY or get_env("THE_ODDS_API_KEY")
    params = {
        "regions": "us",       
        "markets": "h2h",       
        "oddsFormat": "american", 
        "dateFormat": "iso",
        "apiKey": api_key
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    rows = []
    for g in data:
        gid = g["id"]
        start = g["commence_time"]
        home = g["home_team"]
        away = g["away_team"]
        
        # Parse game start time and check if it's on target_date (Pacific time)
        game_time = pd.to_datetime(start, utc=True).tz_convert(ZoneInfo("America/Los_Angeles"))
        game_date = game_time.date()
        
        if game_date != target_date:
            continue  # Skip games not on target date
        
        for bk in g.get("bookmakers", []):
            book = bk["title"]
            if book != "DraftKings":
                continue
            
            for mk in bk.get("markets", []):
                if mk["key"] != "h2h": 
                    continue
                prices = {o["name"]: int(o["price"]) for o in mk["outcomes"]}
                if home in prices and away in prices:
                    rows.append({
                        "game_id": gid,
                        "commence_time": start,
                        "home_team": home,
                        "away_team": away,
                        "book": book,
                        "price_home": prices[home],
                        "price_away": prices[away],
                    })
    
    if not rows:
        print(f"[predict_games] No odds found for {target_date}", flush=True)
        return pd.DataFrame()
    
    lines = pd.DataFrame(rows)
    lines["team_home"] = lines["home_team"].map(TEAM_MAP)
    lines["team_away"] = lines["away_team"].map(TEAM_MAP)
    
    print(f"[predict_games] Found odds for {len(lines)} games", flush=True)

    # --- Merge predictions with odds ---
    merged = lines.merge(
        feat_today[["status", "team_home","team_away","elo_home","elo_away","elo_diff","p_home"]],
        on=["team_home","team_away"],
        how="inner",
        suffixes=("_odds","_feat")
    )
    
    if merged.empty:
        print(f"[predict_games] No matching games after merge", flush=True)
        return pd.DataFrame()

    merged["nv_p_home"], merged["nv_p_away"] = no_vig_probs(
        merged["price_home"], merged["price_away"]
    )
    
    if INJURIES:
        merged['p_home'] = apply_injury_adjustment(merged, lost_elo_dict)
    
    merged["p_home"] = beta_apply(merged["p_home"], theta)
    merged["p_home"] = apply_logit_pool(merged["p_home"], merged["nv_p_home"], b, 0)

    # --- EV & Kelly ---
    merged["dec_home"] = merged["price_home"].apply(decimal_odds)
    merged["dec_away"] = merged["price_away"].apply(decimal_odds)

    p_home = merged["p_home"].astype(float)
    p_away = 1.0 - p_home
    b_home = merged["dec_home"] - 1.0
    b_away = merged["dec_away"] - 1.0

    merged["EV_home"]   = p_home * b_home - (1 - p_home)
    merged["EV_away"]   = p_away * b_away - (1 - p_away)
    merged["kelly_home"] = merged["EV_home"] / b_home
    merged["kelly_away"] = merged["EV_away"] / b_away

    # --- Final output ---
    out = merged[[
        'status', "team_home","team_away",
        "elo_home","elo_away",
        "p_home","nv_p_home","price_home","price_away",
        "EV_home","EV_away","kelly_home","kelly_away"
    ]].rename(columns={
        "elo_home":"adj_elo_home",
        "elo_away":"adj_elo_away"
    })

    out['status'] = out['status'].str.rstrip('ET')
    out['status'] = out['status'].apply(
        lambda x: (str(int(x.split(':')[0])-3) + ':' + x.split(':')[1]) + 'PST'
    )

    return out


"""
Getting betting candidates
"""


def get_candidates(df, kelly_frac, bankroll, ev_thresh=0, kelly_thresh=0, cap=0.03):
    # filter value
    mask = ((df["EV_home"] > ev_thresh) & (df["kelly_home"] > kelly_thresh)) | \
           ((df["EV_away"] > ev_thresh) & (df["kelly_away"] > kelly_thresh))
    candidates = df.loc[mask].copy()   # FIX: copy to avoid SettingWithCopy

    # choose side
    candidates["bet_side"] = np.where(
        (candidates["EV_home"] > candidates["EV_away"]),
        candidates["team_home"],
        candidates["team_away"]
    )

    # side-specific Kelly
    kelly = np.where(
        candidates["bet_side"] == candidates["team_home"],
        candidates["kelly_home"],
        candidates["kelly_away"]
    )

    # fractional Kelly + cap (e.g., 3% of bankroll)
    raw_stake = np.maximum(0, kelly) * kelly_frac * bankroll
    candidates["bet_amount"] = np.minimum(raw_stake, bankroll * cap)

    return candidates


"""
Main function
"""


def main():
    # Use Pacific timezone for consistency (NBA operates in Pacific time)
    from zoneinfo import ZoneInfo
    target_date = datetime.now(ZoneInfo("America/Los_Angeles")).date()
    
    print(f"Running predictions for: {target_date}")
    
    try:
        state = load_state('./states/2025-26_season_state.pkl')
        print('State loaded')
    except FileNotFoundError:
        print('State not found, using defaults')
        class State:
            pass
        state = State()
        state.init_elo = {team: 1500 for team in TEAM_MAP.values()}
        state.params = {"K": 20, "HCA": 65, "scale": 400}
    
    model, theta, FEATS = load_bundle(
        "./models/elo_model_ensemble_prod.pkl", 
        use_cloudpickle=True
    )
    print('Model loaded')
    
    games = get_games_so_far()
    games = merge_games(games)
    print(f'History: {len(games)} games')

    elo = update_elo_table(
        state.init_elo, games, 
        state.params['K'], 
        state.params['HCA'], 
        state.params['scale']
    )
    print('Elo updated')

    preds = predict_games(
        model=model, 
        theta=theta, 
        elo_dict=elo, 
        hist_games=games, 
        HCA=state.params['HCA'], 
        FEATS=FEATS, 
        INJURIES=True, 
        T=0.25, 
        b=1,
        target_date=target_date  # ← Pass explicit date
    )
    
    if preds.empty:
        print("No games found for today")
        return
    
    bets = get_candidates(preds, 0.1, 100)
    
    print("\n=== BETS ===")
    if not bets.empty:
        print(bets[['status', 'bet_side', 'bet_amount']].to_markdown(index=False))
    else:
        print("No bets meeting criteria")
    
    print("\n=== PREDICTIONS ===")
    print(preds[['team_home', 'team_away', 'adj_elo_home', 'adj_elo_away', 'p_home', 'nv_p_home', 'price_home', 'price_away', 'EV_home', 'EV_away']].to_markdown(index=False))


if __name__ == "__main__":
    main()
