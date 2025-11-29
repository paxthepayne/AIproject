"""
Builds Crowd Levels (CL) database with Weighted Scoring based on 'Pressio Turistica' Report.
"""

import requests
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------
# Load Cultural Interest Points (POIs) - OpenData BCN
# --------------------------------------------------------
POIs = pd.DataFrame(requests.get("https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search?resource_id=31431b23-d5b9-42b8-bcd0-a84da9d8c7fa&limit=32000").json()["result"]["records"])

# Clean and convert numeric
POIs['longitude'] = pd.to_numeric(POIs['geo_epgs_4326_lon'], errors='coerce')
POIs['latitude'] = pd.to_numeric(POIs['geo_epgs_4326_lat'], errors='coerce')
POIs = POIs.dropna(subset=['longitude', 'latitude'])

# Convert POIs to GeoDataFrame
gdf_pois = gpd.GeoDataFrame(
    POIs, 
    geometry=gpd.points_from_xy(POIs['longitude'], POIs['latitude']), 
    crs="EPSG:4326"
)

# --------------------------------------------------------
# Load Events Agenda - OpenData BCN
# --------------------------------------------------------
agenda = pd.DataFrame(requests.get("https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search?resource_id=877ccf66-9106-4ae2-be51-95a9f6469e4c&limit=32000").json()["result"]["records"])

# Clean and convert numeric
agenda['latitude'] = pd.to_numeric(agenda['geo_epgs_4326_lat'], errors='coerce')
agenda['longitude'] = pd.to_numeric(agenda['geo_epgs_4326_lon'], errors='coerce')
agenda = agenda.dropna(subset=['latitude', 'longitude'])

# Convert Events to GeoDataFrame
gdf_events = gpd.GeoDataFrame(
    agenda,
    geometry=gpd.points_from_xy(agenda['longitude'], agenda['latitude']),
    crs="EPSG:4326"
)

# --------------------------------------------------------
# Load GeoData & Apply Weights
# --------------------------------------------------------
# Mapped according to ATLAS research (https://coneixement-eu.bcn.cat/widget/atles-resiliencia/en_index_pressio_turistica.html) 
# Category: Accommodation (Allotjament + HUTs) -> Weight 0.6
# Category: Leisure (Oci) -> Weight 0.8
# Category: Attractions   -> Weight 1.0
# NEW: Population Density -> Weight 0.5

datasets_config = {
    # (Filepath, Column Name, Weight)
    "turisme_oci": ("datasets/turisme_oci.geojson", "DN", 0.8),
    "turisme_allotjament": ("datasets/turisme_allotjament.geojson", "DN", 0.6),
    "turisme_huts": ("datasets/turisme_huts.geojson", "gridcode", 0.6),
    "turisme_atractius": ("datasets/turisme_atractius.geojson", "DN", 1.0),
    "densitat_poblacio": ("datasets/densitat_poblacio.geojson", "d_pobTotal", 0.5) 
}

# Initialize crowd index with 0
gdf_pois['crowd_index'] = 0.0

for name, (path, col_name, weight) in datasets_config.items():
    # Load Layer
    gdf_layer = gpd.read_file(path).to_crs("EPSG:4326")
    
    # 1. Normalize the raw data column (0-100) before weighting
    scaler = MinMaxScaler(feature_range=(0, 100))
    gdf_layer[f"{name}_norm"] = scaler.fit_transform(gdf_layer[[col_name]].fillna(0))
    
    # 2. Spatial Join (Optimization)
    # 'predicate=within' checks which polygon the POI falls into
    joined = gpd.sjoin(gdf_pois, gdf_layer[[f"{name}_norm", 'geometry']], how="left", predicate="within")
    
    # 3. Fill NaNs (POIs that aren't inside any polygon of this layer)
    joined[f"{name}_norm"] = joined[f"{name}_norm"].fillna(0)
    
    # Add to the composite Weighted Score: Value * Weight
    gdf_pois['crowd_index'] += (joined[f"{name}_norm"] * weight)

# --------------------------------------------------------
# 4. Final Calculation & Events
# --------------------------------------------------------

# Calculate Events Density (Project -> Buffer -> Join)
# Reproject to EPSG:25831 (UTM Zone 31N) to use meters instead of degrees
gdf_pois_metric = gdf_pois.to_crs("EPSG:25831")
gdf_events_metric = gdf_events.to_crs("EPSG:25831")

# Buffer POIs by 100 meters
gdf_pois_metric['geometry'] = gdf_pois_metric.geometry.buffer(100)

# Check if an event is within 100m of the POI
events_nearby = gpd.sjoin(gdf_pois_metric, gdf_events_metric, how="inner", predicate="intersects")

# Count events per POI
event_counts = events_nearby.index.value_counts()
gdf_pois['events_count'] = gdf_pois.index.map(event_counts).fillna(0)

# Normalize values
scaler_final = MinMaxScaler(feature_range=(0, 100))
gdf_pois['crowd_index'] = scaler_final.fit_transform(gdf_pois[['crowd_index']])
gdf_pois['events_index'] = scaler_final.fit_transform(gdf_pois[['events_count']])

# --------------------------------------------------------
# Output
# --------------------------------------------------------
# Select final columns for the AI model
final_df = gdf_pois[[
    'name', 
    'latitude', 
    'longitude', 
    'crowd_index',
    'events_index'
]]

final_df.to_csv("CL.csv", index=False)
print(final_df)