import requests
url = "https://nba-api-905297400519.us-central1.run.app/predict"
headers = {"Authorization": "Bearer bigjgondoittoem"}
payload = {
  "date": "today",
  "bankroll": 100,
  "kelly_frac": 0.1,
  "ev_thresh": 0,
  "kelly_thresh": 0,
  "cap": 0.03,
  "injury_T": 0.25,
  'INJURIES' : False,
  "pool_w": 0.8,
}
print(requests.post(url, json=payload, headers=headers, timeout=30))