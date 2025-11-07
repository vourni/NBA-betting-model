# 🏀 NBA Elo & Game Prediction Model (2025–26)

This project builds a **live NBA prediction engine** that updates **team Elo ratings** daily using real game results and player data.  
It combines **quantitative modeling**, **data engineering**, and **sports analytics** to estimate win probabilities for every matchup during the 2025–26 season.

---

## ⚙️ Overview

- **Dynamic Elo model** recalculates team strength after each game  
- **Player data integration** via `nba_api` for minutes and availability  
- **Daily prediction script** generates win probabilities for upcoming games  
- **State tracking** preserves each day’s model for consistent updates  

---

## 🧩 Core Scripts

### `update_25_26_elo.py`
Fetches all 2025–26 season games and updates team Elo ratings using:
Elo_{t+1} = 0.75 × Elo_t + 0.25 × 1500

Saves updated season state to:
/states/2025-26_season_state.pkl


### `pred_today.py`
Predicts today’s games using:
- Latest Elo ratings  
- Player minutes per team  
- Home-court advantage adjustments  

Outputs a daily table of win probabilities.

### `utilities/utils.py`
Utility module with:
- `SeasonState` — stores Elo data, metadata, and parameters  
- `save_state()` / `load_state()` — handle serialized model checkpoints  

---

## 🧠 Tech Stack

- **Python 3.13+**
- **nba_api**, **pandas**, **numpy**
- **joblib**, **pickle** for model persistence  
- Optional: **tabula**, **nbainjuries** for injury data parsing

---

## 🚀 Why It Matters

This project demonstrates:
- Real-time integration of public NBA data  
- Automated predictive modeling with historical carry-over  
- Application of quantitative methods (Elo, logistic probabilities)  
- Scalable architecture for continuous model evolution  

---

## 👨‍💻 Author

**James Vournakis**  
Quantitative Researcher | Financial Mathematics & Statistics  
Exploring the intersection of sports analytics, data science, and predictive modeling.

