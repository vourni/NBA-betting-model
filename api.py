# api.py
import os
import json
from typing import Any
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from pred_today import predict_today

API_TOKEN = 'bigjgondoittoem'

app = FastAPI(title="PredToday API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictIn(BaseModel):
    date: str = "today"        # "today" or "YYYY-MM-DD"
    bankroll: float = 100.0
    kelly_frac: float = 0.10
    ev_thresh: float = 0.0
    kelly_thresh: float = 0.02
    cap: float = 0.03
    injury_T: float = 0.25
    pool_w: float = 0.80

def _auth_or_403(token_header: str | None):
    if not API_TOKEN:
        return
    if not token_header or token_header != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=403, detail="Forbidden")

@app.get("/health")
def health(): return {"ok": True}

@app.post("/predict")
def predict(body: PredictIn, authorization: str | None = Header(None)):
    _auth_or_403(authorization)
    out = predict_today(
        target_date=body.date,
        bankroll=body.bankroll,
        kelly_frac=body.kelly_frac,
        ev_thresh=body.ev_thresh,
        kelly_thresh=body.kelly_thresh,
        cap=body.cap,
        injury_T=body.injury_T,
        pool_w=body.pool_w,
    )
    return out

@app.get("/predict")
def predict_get(
    date: str = "today",
    bankroll: float = 100.0,
    kelly_frac: float = 0.10,
    ev_thresh: float = 0.0,
    kelly_thresh: float = 0.02,
    cap: float = 0.03,
    injury_T: float = 0.25,
    pool_w: float = 0.80,
    authorization: str | None = Header(None),
):
    _auth_or_403(authorization)
    out = predict_today(
        target_date=date,
        bankroll=bankroll, kelly_frac=kelly_frac,
        ev_thresh=ev_thresh, kelly_thresh=kelly_thresh, cap=cap,
        injury_T=injury_T, pool_w=pool_w,
    )
    return out
