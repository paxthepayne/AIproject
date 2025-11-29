"""
Builds Crowd Levels (CL) database selecting columns from multiple outsourced datasets
"""

import requests
import pandas as pd
from datetime import timedelta
from collections import Counter
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------
# Load Cultural Interest Points (POIs)
url_pois = "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search?resource_id=31431b23-d5b9-42b8-bcd0-a84da9d8c7fa&limit=32000"
POIs = pd.DataFrame(requests.get(url_pois).json()["result"]["records"])
POIs['longitude'] = pd.to_numeric(POIs['geo_epgs_4326_lon'])
POIs['latitude'] = pd.to_numeric(POIs['geo_epgs_4326_lat'])

# Convert POIs to GeoDataFrame
gdf_pois = gpd.GeoDataFrame(POIs, geometry=gpd.points_from_xy(POIs['longitude'], POIs['latitude']), crs="EPSG:4326")

# --------------------------------------------------------
# Load Events Agenda
url_agenda = "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search?resource_id=877ccf66-9106-4ae2-be51-95a9f6469e4c&limit=32000"
agenda = pd.DataFrame(requests.get(url_agenda).json()["result"]["records"])

# Clean dates and coordinates
agenda = agenda.dropna(subset=['start_date'])
agenda['start_date'] = pd.to_datetime(agenda['start_date'])
agenda['end_date'] = pd.to_datetime(agenda['end_date']).fillna(agenda['start_date'])
agenda['latitude'] = pd.to_numeric(agenda['geo_epgs_4326_lat'])
agenda['longitude'] = pd.to_numeric(agenda['geo_epgs_4326_lon'])

# Function to check nearby events using a small distance threshold
def is_near(lat1, lon1, lat2, lon2, threshold=0.001):
    return abs(lat1 - lat2) < threshold and abs(lon1 - lon2) < threshold

# Compute total number of events per POI
def get_total_events(lat, lon):
    nearby = agenda[agenda.apply(lambda x: is_near(lat, lon, x['latitude'], x['longitude']), axis=1)]
    return len(nearby)  # total number of events

# --------------------------------------------------------
# Load geospatial datasets
datasets = {
    "turisme_oci": ("datasets/turisme_oci.geojson", "DN"),
    "turisme_allotjament": ("datasets/turisme_allotjament.geojson", "DN"),
    "turisme_atractius": ("datasets/turisme_atractius.geojson", "DN"),
    "densitat_poblacio": ("datasets/densitat_poblacio.geojson", "d_pobTotal"),
    "turisme_intensitat": ("datasets/turisme_intensitat.geojson", "gridcode"),
    "turisme_huts": ("datasets/turisme_huts.geojson", "gridcode")
}

geo_layers = []

for name, (path, prop) in datasets.items():
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs("EPSG:4326")  # match POIs CRS
    gdf = gdf.dropna(subset=[prop])
    
    # Normalize feature 0-100
    scaler = MinMaxScaler(feature_range=(0, 100))
    gdf[prop + "_norm"] = scaler.fit_transform(gdf[[prop]].astype(float))
    
    geo_layers.append((gdf, prop + "_norm"))

# --------------------------------------------------------
# Build CL DataFrame
CL = gdf_pois[['name', 'longitude', 'latitude', 'geometry']].copy()

# Compute combined normalized crowd levels
def get_max_for_point(point, gdf, prop):
    inside = gdf[gdf.contains(point)]
    return inside[prop].max() if not inside.empty else 0

def compute_combined_crowd(point):
    total = sum(get_max_for_point(point, gdf, prop) for gdf, prop in geo_layers)
    # Normalize sum back to 0-100
    return (total / (len(geo_layers) * 100)) * 100

CL['crowd_index'] = CL['geometry'].apply(compute_combined_crowd)

# Compute events count
CL['events_index'] = CL.apply(lambda row: get_total_events(row['latitude'], row['longitude']), axis=1)
# Normalize events to 0-100
min_val = CL['events_index'].min()
max_val = CL['events_index'].max()
if max_val > min_val: CL['events_index'] = ((CL['events_index'] - min_val) / (max_val - min_val)) * 100

# Placeholder for weathers
CL['weather_index'] = None

# Drop geometry
CL = CL.drop(columns='geometry')

# --------------------------------------------------------
# Save CL
CL.to_csv("CL.csv", index=False)
print(CL)
