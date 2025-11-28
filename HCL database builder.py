"""
Builds Historical Crowd Levels (HCL) database selecting columns from multiple datasets
"""

import requests
import pandas as pd
from datetime import timedelta
from collections import Counter
import pandas as pd
import sqlite3

# OpenData BCN - Cultural Interest Points
POIs = pd.DataFrame(requests.get("https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search?resource_id=31431b23-d5b9-42b8-bcd0-a84da9d8c7fa&limit=32000").json()["result"]["records"])

# OpenData BCN - Events Agenda
agenda = pd.DataFrame(requests.get("https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search?resource_id=877ccf66-9106-4ae2-be51-95a9f6469e4c&limit=32000").json()["result"]["records"])

# OpenData BCN - Crowd levels Geopackages
r = requests.get("https://opendata-ajuntament.barcelona.cat/data/dataset/fb8d92bd-13ca-42c8-b364-1e05b03d8c08/resource/6f6f5330-e9ef-42d3-82f5-d6739d3469a0/download/2016_turisme_oci.gpkg", stream=True)
r.raise_for_status()
with open("2016_turisme_oci.gpkg", "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
conn = sqlite3.connect("2016_turisme_oci.gpkg")
layer = pd.read_sql("SELECT * FROM '2016_turisme_oci';", conn)
print(layer.head())

# --------------------------------------------------------

# Our Database - Historical Crowd Levels
HCL = pd.DataFrame(columns=["place", "longitude", "latitude", "events", "weathers", "crowd_levels"])

# Adding places names and their locations
HCL['place'] = POIs['name']
HCL['longitude'] = pd.to_numeric(POIs['geo_epgs_4326_lon'])
HCL['latitude'] = pd.to_numeric(POIs['geo_epgs_4326_lat'])

# Adding timestamped number of events
agenda = agenda.dropna(subset=['start_date'])
agenda['start_date'] = pd.to_datetime(agenda['start_date'])
agenda['end_date'] = pd.to_datetime(agenda['end_date']).fillna(agenda['start_date'])

agenda['latitude'] = pd.to_numeric(agenda['geo_epgs_4326_lat'])
agenda['longitude'] = pd.to_numeric(agenda['geo_epgs_4326_lon'])

def is_near(lat1, lon1, lat2, lon2, threshold=0.001):
    return abs(lat1 - lat2) < threshold and abs(lon1 - lon2) < threshold

for i, place in HCL.iterrows():
    # find nearby events
    nearby_events = agenda[
        agenda.apply(lambda x: is_near(place['latitude'], place['longitude'], x['latitude'], x['longitude']), axis=1)
    ]
    
    all_dates = [] # list to collect all dates of nearby events
    
    for _, event in nearby_events.iterrows():
        num_days = (event['end_date'] - event['start_date']).days + 1
        event_dates = [(event['start_date'] + timedelta(days=d)).strftime('%Y-%m-%d') for d in range(num_days)]
        all_dates.extend(event_dates)
    
    date_counts = dict(Counter(all_dates)) # count how many events occur on each date
    HCL.at[i, 'events'] = date_counts

# Adding timestamped weather conditions

# Adding timestamped crowd levels

# Check results
print(HCL)

# Save HCL as csv
HCL.to_csv("HCL.csv", index=False)