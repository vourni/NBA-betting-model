# api.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from datetime import date, datetime


app = FastAPI()
API_TOKEN = os.environ.get("API_TOKEN", "")

class PredictReq(BaseModel):
    date: Optional[str] = None
    bankroll: float
    kelly_frac: float
    ev_thresh: float
    kelly_thresh: float
    cap: float
    injury_T: float
    INJURIES: bool
    pool_w: float

def require_bearer(auth: Optional[str]):
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if auth.split(" ", 1)[1] != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

def parse_date(s: Optional[str]) -> date:
    if s in (None, "", "today"):
        return date.today()
    return datetime.fromisoformat(s).date()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(req: PredictReq, authorization: Optional[str] = Header(None)):
    require_bearer(authorization)
    # 👉 defer heavy imports to here
    from pred_today import predict_today
    d = parse_date(req.date)
    return predict_today(d, req.bankroll, req.kelly_frac, req.ev_thresh, req.kelly_thresh, req.cap, req.injury_T, req.pool_w)
