# pred_today.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import Optional, Union
import threading

# Thread lock for predictions (critical for FastAPI)
_prediction_lock = threading.Lock()

# ---- import your existing functions/types from your script ----
from pred import (
    load_bundle, load_state, merge_games, get_games_so_far,
    update_elo_table, predict_games, get_candidates, TEAM_MAP
)

def _parse_target_date(target_date: Optional[Union[str, int, float, date, datetime]]) -> date:
    """Robustly convert many incoming types to a date (always Pacific timezone)."""
    if target_date is None:
        return datetime.now(ZoneInfo("America/Los_Angeles")).date()

    if isinstance(target_date, date) and not isinstance(target_date, datetime):
        return target_date
    if isinstance(target_date, datetime):
        return target_date.date()

    if isinstance(target_date, (int, float)):
        return datetime.utcfromtimestamp(target_date).date()

    if isinstance(target_date, str):
        s = target_date.strip()
        if s == "":
            return datetime.now(ZoneInfo("America/Los_Angeles")).date()
        if s.lower() in {"today", "now"}:
            return datetime.now(ZoneInfo("America/Los_Angeles")).date()

        try:
            return datetime.fromisoformat(s).date()
        except ValueError:
            pass

        for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue

    return datetime.now(ZoneInfo("America/Los_Angeles")).date()


def predict_today_with_preloaded(
    model,
    theta,
    FEATS,
    state,
    target_date: str = "today",
    bankroll: float = 100.0,
    kelly_frac: float = 0.10,
    ev_thresh: float = 0.0,
    kelly_thresh: float = 0,
    cap: float = 0.03,
    injury_T: float = 0.25,
    pool_w: float = 1,
    injuries: bool = False,
):
    """
    Prediction using pre-loaded model and state (avoids reloading).
    CRITICAL: Uses thread lock to prevent sklearn threading issues.
    """
    dt = _parse_target_date(target_date)
    
    # --- history & ELO rebuild ---
    games_hist = get_games_so_far(season="2025-26")
    games_hist = merge_games(games_hist)
    elo_dict = update_elo_table(
        state.init_elo, 
        games_hist, 
        state.params["K"], 
        state.params["HCA"], 
        state.params["scale"]
    )

    # --- prediction step (no patching needed - pass date directly) ---
    # CRITICAL: Lock during prediction to avoid sklearn threading bug
    with _prediction_lock:
        print(f"[PREDICT] Locked, calling predict_games for {dt}", flush=True)
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
            target_date=dt,  # ← Pass explicit date
        )
        print(f"[PREDICT] ✓ predict_games completed", flush=True)
    
    bets_df = get_candidates(
        preds_df, kelly_frac, bankroll, 
        ev_thresh=ev_thresh, 
        kelly_thresh=kelly_thresh, 
        cap=cap
    )

    # JSON-friendly outputs
    def _jsonable(df: pd.DataFrame):
        if df is None or df.empty:
            return []
        out = df.copy()
        for c in out.columns:
            if isinstance(out[c].dtype, pd.DatetimeTZDtype) or \
               np.issubdtype(out[c].dtype, np.datetime64):
                out[c] = out[c].astype(str)
        return out.to_dict(orient="records")

    return {
        "date": dt.isoformat(),
        "bets": _jsonable(
            bets_df[["bet_side", "bet_amount"]].join(
                bets_df[[
                    "team_home","team_away","EV_home","EV_away",
                    "kelly_home","kelly_away"
                ]], 
                how="left"
            ) if not bets_df.empty else bets_df
        ),
        "preds": _jsonable(preds_df),
    }


# Keep original function for backward compatibility
def predict_today(
    target_date: str = "today",
    bankroll: float = 100.0,
    kelly_frac: float = 0.10,
    ev_thresh: float = 0.0,
    kelly_thresh: float = 0.02,
    cap: float = 0.03,
    injury_T: float = 0.25,
    pool_w: float = 1,
    injuries: bool = False,
    model_path: str = "./models/elo_model_ensemble_prod.pkl",
    state_path: str = "./states/2025-26_season_state.pkl",
):
    """Original function that loads model each time (for CLI use)."""
    try:
        state = load_state(state_path)
    except FileNotFoundError:
        state = type("State", (), {})()
        state.init_elo = {team: 1500 for team in TEAM_MAP.values()}
        state.params = {"K": 20, "HCA": 65, "scale": 400}
    
    model, theta, FEATS = load_bundle(model_path, use_cloudpickle=True)
    
    return predict_today_with_preloaded(
        model=model,
        theta=theta,
        FEATS=FEATS,
        state=state,
        target_date=target_date,
        bankroll=bankroll,
        kelly_frac=kelly_frac,
        ev_thresh=ev_thresh,
        kelly_thresh=kelly_thresh,
        cap=cap,
        injury_T=injury_T,
        pool_w=pool_w,
        injuries=injuries,
    )