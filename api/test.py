import os
import sys
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from env_loader import load_env, get_env

load_env()
api_token = get_env("API_TOKEN", required=False)
if not api_token:
    raise RuntimeError("Set API_TOKEN in your environment or .env before running this test script.")

url = "https://nba-api-905297400519.us-central1.run.app/predict"
headers = {"Authorization": f"Bearer {api_token}"}
payload = {
  "date": "today",
  "bankroll": 100,
  "kelly_frac": 0.1,
  "ev_thresh": 0,
  "kelly_thresh": 0,
  "cap": 0.03,
  "injury_T": 0.25,
  'injuries' : False,
  "pool_w": 1,
}
print(requests.post(url, json=payload, headers=headers, timeout=30).json())
