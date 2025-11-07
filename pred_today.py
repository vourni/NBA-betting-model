# pred_today.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, date
from zoneinfo import ZoneInfo

# ---- import your existing functions/types from your script ----
# If you rename your big script to nbapred_core.py, import like this:
from pred import (
    load_bundle, load_state, merge_games, get_games_so_far,
    update_elo_table, predict_games, get_candidates,
    # we will call patched versions below; if you keep originals,
    # this wrapper provides date/env-aware versions.
)

# -----------------------------
# Small patched helpers (date/env)
# -----------------------------

def _parse_target_date(target_date: str | None) -> date:
    """Accepts 'today' or 'YYYY-MM-DD' (local Pacific time)."""
    tz = ZoneInfo("America/Los_Angeles")
    if (not target_date) or (str(target_date).lower() == "today"):
        return datetime.now(tz).date()
    return datetime.fromisoformat(target_date).date()

# -----------------------------
# Public function your API will call
# -----------------------------

def predict_today(
    target_date: str = "today",
    bankroll: float = 100.0,
    kelly_frac: float = 0.10,
    ev_thresh: float = 0.0,
    kelly_thresh: float = 0.02,
    cap: float = 0.03,
    injury_T: float = 0.25,
    pool_w: float = 1,      # w in logit pooling against no-vig market
    injuries: bool = False,
    model_path: str = "./models/elo_model_ensemble_prod.pkl",
    state_path: str = "./states/2025-26_season_state.pkl",
):
    """
    Runs your full pipeline for a given date (default: 'today' in PT).
    Returns JSON-serializable dict with 'bets' and 'preds' lists.
    """
    # --- date & env
    dt = _parse_target_date(target_date)
    # --- load model/state
    try:
        state = load_state(state_path)
    except FileNotFoundError:
        # fallback Elo init if state missing
        from pred import TEAM_MAP
        state = type("State", (), {})()
        state.init_elo = {team: 1500 for team in TEAM_MAP.values()}
        state.params = {"K": 20, "HCA": 65, "scale": 400}
    model, theta, FEATS = load_bundle(model_path, use_cloudpickle=True)

    # --- history & ELO rebuild
    games_hist = get_games_so_far(season="2025-26")
    games_hist = merge_games(games_hist)
    elo_dict = update_elo_table(state.init_elo, games_hist, state.params["K"], state.params["HCA"], state.params["scale"])

    # --- prediction step (patched get_nba_lines inside predict_games via env/date)
    # We need to pass the date and the odds key into predict_games; we do that by
    # monkey-patching a tiny closure around your existing function signature.
    from pred import get_nba_lines as _orig_get_nba_lines
    from pred import get_todays_games as _orig_get_todays_games

    # wrap odds getter to force date and env key
    def _get_nba_lines_for_date(*args, **kwargs):
        # enforce our key/date; ignore positional if provided
        df = _orig_get_nba_lines("9427c941ebf6ab6828ca2582f82cd24b", region="us", markets="h2h", odds_format="american")
        if "tip_utc" in df.columns:
            df["tip_utc"] = pd.to_datetime(df["commence_time"], utc=True)
            df["tip_pst"] = df["tip_utc"].dt.tz_convert(ZoneInfo("America/Los_Angeles"))
            df["date_pst"] = df["tip_pst"].dt.date
            df = df[df["date_pst"] == dt]
        return df

    # wrap today games to force date
    def _get_games_for_date():
        from nba_api.stats.endpoints import scoreboardv2
        sb = scoreboardv2.ScoreboardV2(game_date=dt.strftime("%m/%d/%Y"))
        frames = sb.get_data_frames()
        if not frames or frames[0].empty:
            return pd.DataFrame(columns=["date","GAME_ID","team_home","team_away","score_home","score_away","status"])
        header = frames[0]
        from pred import TEAM_ID_TO_ABBR
        df = header[["GAME_ID","GAME_DATE_EST","GAME_STATUS_TEXT","HOME_TEAM_ID","VISITOR_TEAM_ID"]].copy()
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
        # explicitly filter to dt (ScoreboardV2 already does, but this is defensive)
        return df[df["date"] == dt]

    # --- temporarily swap the functions your pipeline calls
    # NOTE: this is safe inside a short-lived request; for long-running,
    # refactor predict_games to accept function hooks.
    import pred
    _bak_lines = pred.get_nba_lines
    _bak_games = pred.get_todays_games
    pred.get_nba_lines = lambda *a, **k: _get_nba_lines_for_date()
    pred.get_todays_games = lambda: _get_games_for_date()

    try:
        preds_df = predict_games(
            model=model,
            theta=theta,
            elo_dict=elo_dict,
            hist_games=games_hist,
            HCA=state.params["HCA"],
            FEATS=FEATS,
            INJURIES=injuries,
            T=injury_T,
            b=pool_w,
        )
        bets_df = get_candidates(preds_df, kelly_frac, bankroll, ev_thresh=ev_thresh, kelly_thresh=kelly_thresh, cap=cap)
    finally:
        # restore originals
        pred.get_nba_lines = _bak_lines
        pred.get_todays_games = _bak_games

    # JSON-friendly outputs
    def _jsonable(df: pd.DataFrame):
        if df is None or df.empty:
            return []
        out = df.copy()
        for c in out.columns:
            if isinstance(out[c].dtype, pd.DatetimeTZDtype) or np.issubdtype(out[c].dtype, np.datetime64):
                out[c] = out[c].astype(str)
        return out.to_dict(orient="records")

    return {
        "date": dt.isoformat(),
        "bets": _jsonable(bets_df[["bet_side", "bet_amount"]].join(
            bets_df[["team_home","team_away","EV_home","EV_away","kelly_home","kelly_away"]], how="left"
        ) if not bets_df.empty else bets_df),
        "preds": _jsonable(preds_df),
    }
