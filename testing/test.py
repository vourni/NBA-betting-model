from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd

import ssl, urllib.request, certifi
_ssl_ctx = ssl.create_default_context(cafile=certifi.where())
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_ctx))
)

from nbainjuries import injury
from datetime import datetime

df_output = injury.get_reportdata(datetime.now(), return_df=True)
print(df_output)

# pull current season (e.g., '2024-25')
stats = leaguedashplayerstats.LeagueDashPlayerStats(season='2024-25', per_mode_detailed='PerGame')
df   = stats.get_data_frames()[0]

# select relevant columns
mins     = df[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'MIN']]
avg_mins = df.groupby('TEAM_ABBREVIATION')['MIN'].mean().to_dict()


