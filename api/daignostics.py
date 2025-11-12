#!/usr/bin/env python3
"""
Comprehensive debugging to find WHY local and API differ.
This will trace EVERY step and show where divergence happens.
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

def detailed_local_trace():
    """Run local prediction with detailed tracing."""
    print("=" * 80)
    print("LOCAL PREDICTION - DETAILED TRACE")
    print("=" * 80)
    
    from pred import (
        load_bundle, load_state, TEAM_MAP,
        get_games_so_far, merge_games, update_elo_table,
        predict_games, get_candidates
    )
    
    # Step 1: Date
    target_date = datetime.now(ZoneInfo("America/Los_Angeles")).date()
    print(f"\n[STEP 1] Target date: {target_date}")
    print(f"  Timezone: America/Los_Angeles")
    print(f"  ISO format: {target_date.isoformat()}")
    
    # Step 2: Load model
    print(f"\n[STEP 2] Loading model...")
    try:
        state = load_state('./states/2025-26_season_state.pkl')
        print(f"  ✓ State loaded from file")
    except FileNotFoundError:
        class State:
            pass
        state = State()
        state.init_elo = {team: 1500 for team in TEAM_MAP.values()}
        state.params = {"K": 20, "HCA": 65, "scale": 400}
        print(f"  ✓ State created (default)")
    
    model, theta, FEATS = load_bundle("./models/elo_model_ensemble_prod.pkl", use_cloudpickle=True)
    print(f"  ✓ Model loaded: {type(model).__name__}")
    print(f"  ✓ Theta: {theta}")
    print(f"  ✓ Features: {len(FEATS)}")
    
    # Step 3: History
    print(f"\n[STEP 3] Loading game history...")
    games = get_games_so_far()
    games = merge_games(games)
    print(f"  ✓ Loaded {len(games)} games")
    print(f"  Latest game: {games['date'].max()}")
    
    # Step 4: Elo
    print(f"\n[STEP 4] Updating Elo...")
    elo = update_elo_table(state.init_elo, games, state.params['K'], state.params['HCA'], state.params['scale'])
    print(f"  ✓ Elo updated for {len(elo)} teams")
    print(f"  Sample (GSW): {elo.get('GSW', 'N/A')}")
    
    # Step 5: Predict with explicit trace
    print(f"\n[STEP 5] Running predict_games...")
    print(f"  Parameters:")
    print(f"    target_date: {target_date}")
    print(f"    HCA: {state.params['HCA']}")
    print(f"    INJURIES: False")
    print(f"    T: 0.25")
    print(f"    b: 1")
    
    preds = predict_games(
        model=model,
        theta=theta,
        elo_dict=elo,
        hist_games=games,
        HCA=state.params['HCA'],
        FEATS=FEATS,
        INJURIES=False,
        T=0.25,
        b=1,
        target_date=target_date
    )
    
    if preds.empty:
        print("  ❌ NO GAMES FOUND!")
        return None
    
    print(f"  ✓ Found {len(preds)} games")
    
    # Step 6: Bets
    print(f"\n[STEP 6] Finding betting candidates...")
    bets = get_candidates(preds, 0.1, 100)
    print(f"  ✓ Found {len(bets)} bets")
    
    # Step 7: Detailed output
    print(f"\n[STEP 7] Results:")
    for _, row in preds.iterrows():
        print(f"\n  {row['team_home']} vs {row['team_away']}")
        print(f"    Status: {row.get('status', 'N/A')}")
        print(f"    p_home: {row['p_home']:.4f}")
        print(f"    nv_p_home: {row['nv_p_home']:.4f}")
        print(f"    price_home: {row['price_home']}")
        print(f"    price_away: {row['price_away']}")
        print(f"    EV_home: {row['EV_home']:.4f}")
        print(f"    EV_away: {row['EV_away']:.4f}")
    
    return {
        "date": target_date.isoformat(),
        "num_games": len(preds),
        "num_bets": len(bets),
        "games": preds[['team_home', 'team_away', 'p_home', 'price_home', 'price_away']].to_dict('records')
    }


def detailed_api_trace():
    """Run API prediction with detailed tracing."""
    print("\n" + "=" * 80)
    print("API PREDICTION - DETAILED TRACE")
    print("=" * 80)
    
    from pred_today import predict_today_with_preloaded
    from pred import load_bundle, load_state, TEAM_MAP, get_games_so_far, merge_games, update_elo_table
    
    # Step 1: Date
    target_date = datetime.now(ZoneInfo("America/Los_Angeles")).date()
    print(f"\n[STEP 1] Target date: {target_date}")
    print(f"  Timezone: America/Los_Angeles")
    print(f"  ISO format: {target_date.isoformat()}")
    
    # Step 2: Load model
    print(f"\n[STEP 2] Loading model...")
    try:
        state = load_state('./states/2025-26_season_state.pkl')
        print(f"  ✓ State loaded from file")
    except FileNotFoundError:
        class State:
            pass
        state = State()
        state.init_elo = {team: 1500 for team in TEAM_MAP.values()}
        state.params = {"K": 20, "HCA": 65, "scale": 400}
        print(f"  ✓ State created (default)")
    
    model, theta, FEATS = load_bundle("./models/elo_model_ensemble_prod.pkl", use_cloudpickle=True)
    print(f"  ✓ Model loaded: {type(model).__name__}")
    print(f"  ✓ Theta: {theta}")
    print(f"  ✓ Features: {len(FEATS)}")
    
    # Step 3: Call API function
    print(f"\n[STEP 3] Calling predict_today_with_preloaded...")
    print(f"  Parameters:")
    print(f"    target_date: {target_date}")
    print(f"    bankroll: 100.0")
    print(f"    kelly_frac: 0.1")
    print(f"    ev_thresh: 0.0")
    print(f"    kelly_thresh: 0")
    print(f"    cap: 0.03")
    print(f"    injury_T: 0.25")
    print(f"    injuries: False")
    print(f"    pool_w: 1")
    
    result = predict_today_with_preloaded(
        model=model,
        theta=theta,
        FEATS=FEATS,
        state=state,
        target_date='today',
        bankroll=100.0,
        kelly_frac=0.1,
        ev_thresh=0.0,
        kelly_thresh=0,
        cap=0.03,
        injury_T=0.25,
        injuries=False,
        pool_w=1
    )
    
    print(f"  ✓ Returned successfully")
    print(f"  ✓ Found {len(result['preds'])} games")
    print(f"  ✓ Found {len(result['bets'])} bets")
    
    # Step 4: Detailed output
    print(f"\n[STEP 4] Results:")
    for game in result['preds']:
        print(f"\n  {game['team_home']} vs {game['team_away']}")
        print(f"    Status: {game.get('status', 'N/A')}")
        print(f"    p_home: {game['p_home']:.4f}")
        print(f"    nv_p_home: {game['nv_p_home']:.4f}")
        print(f"    price_home: {game['price_home']}")
        print(f"    price_away: {game['price_away']}")
        print(f"    EV_home: {game['EV_home']:.4f}")
        print(f"    EV_away: {game['EV_away']:.4f}")
    
    return {
        "date": result["date"],
        "num_games": len(result["preds"]),
        "num_bets": len(result["bets"]),
        "games": [{k: v for k, v in g.items() if k in ['team_home', 'team_away', 'p_home', 'price_home', 'price_away']} 
                  for g in result["preds"]]
    }


def compare_detailed(local, api):
    """Compare with detailed breakdown."""
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    if local is None or api is None:
        print("❌ Cannot compare - one result is None")
        return False
    
    all_match = True
    
    # Date
    print(f"\n[DATE]")
    if local["date"] != api["date"]:
        print(f"  ❌ MISMATCH")
        print(f"    Local: {local['date']}")
        print(f"    API:   {api['date']}")
        all_match = False
    else:
        print(f"  ✓ Match: {local['date']}")
    
    # Counts
    print(f"\n[COUNTS]")
    if local["num_games"] != api["num_games"]:
        print(f"  ❌ Game count MISMATCH")
        print(f"    Local: {local['num_games']}")
        print(f"    API:   {api['num_games']}")
        all_match = False
    else:
        print(f"  ✓ Game count: {local['num_games']}")
    
    if local["num_bets"] != api["num_bets"]:
        print(f"  ❌ Bet count MISMATCH")
        print(f"    Local: {local['num_bets']}")
        print(f"    API:   {api['num_bets']}")
        all_match = False
    else:
        print(f"  ✓ Bet count: {local['num_bets']}")
    
    # Game-by-game
    if local["num_games"] > 0 and api["num_games"] > 0:
        print(f"\n[GAME-BY-GAME]")
        
        # Create lookups
        local_games = {(g['team_home'], g['team_away']): g for g in local["games"]}
        api_games = {(g['team_home'], g['team_away']): g for g in api["games"]}
        
        all_keys = set(local_games.keys()) | set(api_games.keys())
        
        for key in sorted(all_keys):
            lg = local_games.get(key)
            ag = api_games.get(key)
            
            matchup = f"{key[0]} vs {key[1]}"
            
            if lg is None:
                print(f"  ❌ {matchup}: ONLY in API")
                all_match = False
            elif ag is None:
                print(f"  ❌ {matchup}: ONLY in LOCAL")
                all_match = False
            else:
                # Compare values
                p_diff = abs(lg['p_home'] - ag['p_home'])
                price_h_diff = abs(lg['price_home'] - ag['price_home'])
                price_a_diff = abs(lg['price_away'] - ag['price_away'])
                
                if p_diff > 0.001 or price_h_diff > 0 or price_a_diff > 0:
                    print(f"  ❌ {matchup}:")
                    print(f"      p_home: local={lg['p_home']:.4f}, api={ag['p_home']:.4f}, diff={p_diff:.4f}")
                    print(f"      price_home: local={lg['price_home']}, api={ag['price_home']}, diff={price_h_diff}")
                    print(f"      price_away: local={lg['price_away']}, api={ag['price_away']}, diff={price_a_diff}")
                    all_match = False
                else:
                    print(f"  ✓ {matchup}: All values match")
    
    print("\n" + "=" * 80)
    if all_match:
        print("✅ PERFECT MATCH - Local and API are identical!")
    else:
        print("❌ DIFFERENCES FOUND - See details above")
    print("=" * 80)
    
    return all_match


if __name__ == "__main__":
    # Show environment
    now_utc = datetime.now(ZoneInfo("UTC"))
    now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    print(f"Current UTC:  {now_utc}")
    print(f"Current PT:   {now_pt}")
    print(f"Today (PT):   {now_pt.date()}")
    
    try:
        local_result = detailed_local_trace()
        api_result = detailed_api_trace()
        compare_detailed(local_result, api_result)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)