from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
import joblib

@dataclass
class SeasonState:
    season: str
    last_updated: str
    init_elo: Dict[str, float]   = field(default_factory=dict)
    features: Dict[str, list]  = field(default_factory=dict)
    params: Dict[str, Any]  = field(default_factory=dict)

def load_state(path: str) -> SeasonState:
    return joblib.load(path)

def save_state(state: SeasonState, path: str):
    state.last_updated = datetime.now().isoformat(timespec="seconds")
    tmp = path + ".tmp"
    joblib.dump(state, tmp)   
    import os; os.replace(tmp, path)
 
