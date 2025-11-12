# api.py
from __future__ import annotations

import os, sys, traceback
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import date, datetime
from typing import Optional, Union, Any
import threading
from models.stacker import TwoStageStackTS, PurgedGroupTimeSeriesSplit  # ensure pickle-defined classes are importable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from env_loader import load_env, get_env

# Avoid unused-import lint errors in tooling
_STACKER_TYPES = (TwoStageStackTS, PurgedGroupTimeSeriesSplit)

# ===== CRITICAL FIX: Force single-threaded numpy/sklearn =====
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

load_env()

app = FastAPI(title="NBA Bets API", version="1.0")

# ===== MODEL SINGLETON =====
class ModelSingleton:
    """
    Load model once at startup, reuse across requests.
    Thread-safe with explicit locking.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.model = None
        self.theta = None
        self.FEATS = None
        self.state = None
        self.loaded = False
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def load(self):
        """Load model, state, and everything needed for prediction."""
        if self.loaded:
            return
        
        with self._lock:
            if self.loaded:  # Double-check after acquiring lock
                return
            
            print("[STARTUP] Loading model bundle...", flush=True)
            from pred import load_bundle, load_state, TEAM_MAP
            
            try:
                self.model, self.theta, self.FEATS = load_bundle(
                    "./models/elo_model_ensemble_prod.pkl", 
                    use_cloudpickle=True
                )
                print(f"[STARTUP] ✓ Model loaded: {type(self.model).__name__}", flush=True)
            except Exception as e:
                print(f"[STARTUP] ✗ Model load failed: {e}", flush=True)
                raise
            
            try:
                self.state = load_state('./states/2025-26_season_state.pkl')
                print(f"[STARTUP] ✓ State loaded", flush=True)
            except FileNotFoundError:
                print(f"[STARTUP] State not found, using defaults", flush=True)
                # Fallback Elo init
                class State:
                    pass
                self.state = State()
                self.state.init_elo = {team: 1500 for team in TEAM_MAP.values()}
                self.state.params = {"K": 20, "HCA": 65, "scale": 400}
            
            self.loaded = True
            print(f"[STARTUP] ✓ Ready for predictions", flush=True)

# Initialize at module load
_model_singleton = ModelSingleton.get_instance()

@app.on_event("startup")
async def startup_event():
    """Load model when server starts."""
    print("[APP] Starting up...", flush=True)
    try:
        _model_singleton.load()
        print("[APP] ✓ Startup complete", flush=True)
    except Exception as e:
        print(f"[APP] ✗ Startup failed: {e}", flush=True)
        traceback.print_exc()
        # Don't crash the server, but predictions will fail

# ---- Auth ----
API_TOKEN = get_env("API_TOKEN", required=False) or ""
if not API_TOKEN:
    print("[WARN] API_TOKEN not set. Set it in Cloud Run -> Edit & Deploy.")

def require_bearer(auth: Optional[str]):
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

# ---- Models ----
class PredictReq(BaseModel):
    date: Optional[Union[str, int, float, date, datetime]] = None
    bankroll: float
    kelly_frac: float
    ev_thresh: float
    kelly_thresh: float
    cap: float
    injury_T: float
    injuries: bool = False  # ← MISSING FIELD!
    pool_w: float

# ---- Helpers ----
def parse_date(s: Optional[Union[str, date, datetime]]) -> date:
    """Parse date with Pacific timezone as default (NBA operates in PT)."""
    from zoneinfo import ZoneInfo
    
    if s in (None, "", "today"):
        # Return today in Pacific time (where NBA operates)
        return datetime.now(ZoneInfo("America/Los_Angeles")).date()
    
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    if isinstance(s, datetime):
        return s.date()
    
    if isinstance(s, str):
        try:
            return datetime.fromisoformat(s).date()
        except ValueError:
            # Fallback to today if parsing fails
            return datetime.now(ZoneInfo("America/Los_Angeles")).date()
    
    return datetime.now(ZoneInfo("America/Los_Angeles")).date()

# ---- Error middleware: always return JSON on 500s ----
@app.middleware("http")
async def capture_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print("UNHANDLED ERROR:\n", tb, flush=True)
        return JSONResponse(
            {"error": str(e), "trace_tail": tb.splitlines()[-7:]},
            status_code=500
        )

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---- Endpoints ----
@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": _model_singleton.loaded
    }

@app.post("/echo")
def echo(payload: dict, authorization: Optional[str] = Header(None)):
    require_bearer(authorization)
    return payload

@app.get("/jvm-info")
def jvm_info(authorization: Optional[str] = Header(None)):
    require_bearer(authorization)
    import pathlib
    info: dict[str, Any] = {
        "JAVA_HOME": os.environ.get("JAVA_HOME"),
        "JPYPE_JVM": os.environ.get("JPYPE_JVM"),
    }
    try:
        import jpype
        info["default_jvm_path"] = jpype.getDefaultJVMPath()
        info["env_exists"] = pathlib.Path(info["JPYPE_JVM"] or "").exists()
        info["default_exists"] = pathlib.Path(info.get("default_jvm_path") or "").exists()
    except Exception as e:
        info["error"] = str(e)
    return info

@app.post("/predict")
def predict(req: PredictReq, authorization: Optional[str] = Header(None)):
    require_bearer(authorization)
    
    # Check model is loaded
    if not _model_singleton.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    # Import prediction function (lighter than whole module)
    from pred_today import predict_today_with_preloaded
    
    d = parse_date(req.date)
    
    # Use preloaded model/state
    return predict_today_with_preloaded(
        model=_model_singleton.model,
        theta=_model_singleton.theta,
        FEATS=_model_singleton.FEATS,
        state=_model_singleton.state,
        target_date=d,
        bankroll=req.bankroll,
        kelly_frac=req.kelly_frac,
        ev_thresh=req.ev_thresh,
        kelly_thresh=req.kelly_thresh,
        cap=req.cap,
        injury_T=req.injury_T,
        pool_w=req.pool_w
    )
