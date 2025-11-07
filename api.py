# api.py
from __future__ import annotations

import os, traceback
from datetime import date, datetime
from typing import Optional, Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="NBA Bets API", version="1.0")

# ---- Auth ----
API_TOKEN = os.environ.get("API_TOKEN", "")
if not API_TOKEN:
    # Still allow the service to boot; you'll get 401/403 as expected
    print("[WARN] API_TOKEN not set. Set it in Cloud Run -> Edit & Deploy.")

def require_bearer(auth: Optional[str]):
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

# ---- Models ----
class PredictReq(BaseModel):
    date: Optional[str] = None            # "today" or "YYYY-MM-DD"
    bankroll: float
    kelly_frac: float
    ev_thresh: float
    kelly_thresh: float
    cap: float
    injury_T: float
    pool_w: float

# ---- Helpers ----
def parse_date(s: Optional[str]) -> date:
    if s in (None, "", "today"):
        return date.today()
    return datetime.fromisoformat(s).date()

# ---- Error middleware: always return JSON on 500s ----
@app.middleware("http")
async def capture_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except HTTPException:
        # Let FastAPI render 4xx as-is
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print("UNHANDLED ERROR:\n", tb, flush=True)
        return JSONResponse(
            {"error": str(e), "trace_tail": tb.splitlines()[-7:]},
            status_code=500
        )

# ---- CORS (optional; safe defaults) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---- Endpoints ----
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/echo")
def echo(payload: dict, authorization: Optional[str] = Header(None)):
    require_bearer(authorization)
    return payload

@app.get("/jvm-info")
def jvm_info(authorization: Optional[str] = Header(None)):
    require_bearer(authorization)
    import os, pathlib
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
    # Defer heavy imports so the container always boots
    from pred_today import predict_today
    d = parse_date(req.date)
    return predict_today(
        d, req.bankroll, req.kelly_frac, req.ev_thresh,
        req.kelly_thresh, req.cap, req.injury_T, req.pool_w
    )
