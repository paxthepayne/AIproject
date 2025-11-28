"""
Builds Historical Crowd Levels (HCL) database selecting columns from multiple datasets
"""
import requests
import pandas as pd

# OpenData BCN - Cultural Interest Points
url = "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search?resource_id=31431b23-d5b9-42b8-bcd0-a84da9d8c7fa&limit=32000"
POIs = pd.DataFrame(requests.get(url).json()["result"]["records"])

# Our Database - Historical Crowd Levels
HCL = pd.DataFrame(columns=["place", "longitude", "latitude", "events", "weathers", "crowd_levels"])
HCL['place'] = POIs['name']
HCL['longitude'] =POIs['geo_epgs_4326_lon']
HCL['latitude'] =POIs['geo_epgs_4326_lat']

# Check results
print(HCL)

# Save HCL as csv
HCL.to_csv("HCL.csv", index=False)