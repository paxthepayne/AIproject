"""
Map Builder

- If map.json.gz exists -> do nothing
- Else:
    • Fetch POIs from OpenData BCN
    • Enrich with Google Data (popular times)
    • Build street network from OpenStreetMap
    • Aggregate POIs onto streets
    • Export map.json.gz
"""

# --- IMPORTS ---

# System
import os
import json
import gzip

# Data Handling
import numpy as np
import pandas as pd
import geopandas as gpd

# APIs
import requests
import googlemaps
import populartimes
import osmnx as ox

# Progress Bar and Multithreading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# --- CONFIGURATION ---

TARGET_LOCATION = "Barcelona, Spain"
OUTPUT_FILE = "map.json.gz"
GOOGLE_API_KEY = ""

# OpenData BCN Resource
BCN_API_URL = (
    "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search?"
    "resource_id=31431b23-d5b9-42b8-bcd0-a84da9d8c7fa&limit=32000"
)


# --- MAP BUILDER ---

def build_map():
    print("> Fetching POIs from OpenData BCN...")
    
    # Load Raw Data
    data = requests.get(BCN_API_URL).json()["result"]["records"]
    pois = pd.DataFrame(data)

    # Clean Data
    places = pd.DataFrame({
        "name": pois["name"],
        "lat": pd.to_numeric(pois["geo_epgs_4326_lat"], errors="coerce"),
        "lon": pd.to_numeric(pois["geo_epgs_4326_lon"], errors="coerce"),
        "popular_times": None
    })
    
    # Drop rows with missing values
    places = places.dropna(subset=["name", "lat", "lon"]).reset_index(drop=True)


    print("> Enriching POIs with Google popular times...")
    
    # Initialize Google Maps Client
    gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

    def fetch_popular_times(index, name, lat, lon):
        """Helper to fetch popular times for a single row."""
        try:
            # Find place Google ID
            res = gmaps.find_place(name, "textquery", location_bias=f"circle:200@{lat},{lon}", fields=["place_id"])
            if not res["candidates"]:
                return index, None
            
            google_id = res["candidates"][0]["place_id"]
            
            # Fetch populartimes data
            pop_data = populartimes.get_id(GOOGLE_API_KEY, google_id).get("populartimes")
            if not pop_data:
                return index, None

            # Convert to 7x24 numpy array
            popular_times = np.array([d["data"] for d in pop_data], dtype=float)
            
            if popular_times.shape != (7, 24):
                return index, None

            return index, popular_times
            
        except Exception:
            # Fail silently for individual items
            return index, None

    # Multithreaded enrichment of places
    tasks = [(i, r["name"], r["lat"], r["lon"]) for i, r in places.iterrows()]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_popular_times, *t) for t in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Enriching"):
            i, result = future.result()
            if result is not None:
                places.at[i, "popular_times"] = result

    count_enriched = places['popular_times'].notna().sum()
    print(f"> POIs enriched with crowd data: {count_enriched}")


    print("> Building OpenStreetMap walking network...")

    # Download graph and project to metric system
    G = ox.graph_from_place(TARGET_LOCATION, network_type="walk")
    G = ox.project_graph(G)
    
    # Convert streets to a GeoDataFrame
    _, streets = ox.graph_to_gdfs(G)
    streets = streets.reset_index()
    streets["id"] = range(len(streets)) # Assign unique ID to every street segment

    # Calculate centroids for the middle point of the street
    centroids = streets.centroid.to_crs("EPSG:4326")
    streets["center"] = list(zip(centroids.y, centroids.x))

    # Create Neighbors Map: Node ID (intersection) -> Set of Street IDs
    adj = {}
    for street_id, u, v in streets[["id", "u", "v"]].itertuples(index=False):
        adj.setdefault(u, set()).add(street_id)
        adj.setdefault(v, set()).add(street_id)
    street_lookup = streets.set_index(["u", "v", "key"])["id"].to_dict()


    print("> Linking POIs to nearest streets...")

    # Convert POIs to GeoDataFrame and project to same CRS as the streets
    gdf_places = gpd.GeoDataFrame(
        places, 
        geometry=gpd.points_from_xy(places.lon, places.lat), 
        crs="EPSG:4326"
    ).to_crs(G.graph["crs"])
    
    # Find nearest street edge for every POI
    places["street_edge"] = ox.nearest_edges(
        G, gdf_places.geometry.x, gdf_places.geometry.y, return_dist=False
    )


    print("> Aggregating final JSON structure...")
    
    nodes = {}

    # Process Streets (Type 0)

    for row in streets.itertuples(index=False):
        # Handle OSM names
        street_name = row.name
        if isinstance(street_name, list) and street_name:
            safe_name = street_name[0]
        elif isinstance(street_name, str):
            safe_name = street_name
        else:
            safe_name = "Calle Sin Nombre"

        # Find connected streets (neighbors)
        neighbors = list((adj[row.u] | adj[row.v]) - {row.id})

        nodes[row.id] = {
            "id": row.id,
            "type": 0,  # 0 = Street
            "name": safe_name,
            "coords": row.center,
            "len": float(row.length or 0),
            "conns": neighbors,
            "popular_times": None, # Will be aggregated from POIs
        }

    # Process POIs (Type 1)
    next_id = len(nodes)

    for _, r in places.iterrows():
        parent_id = street_lookup.get(r["street_edge"])
        
        # If POI didn't snap to a valid street, skip
        if parent_id is None:
            continue

        pid = next_id
        next_id += 1
        
        p_times = r["popular_times"]

        # Create POI Node
        nodes[pid] = {
            "id": pid,
            "type": 1,  # 1 = POI
            "name": r["name"],
            "coords": (r["lat"], r["lon"]),
            "len": 0,
            "conns": [parent_id], # Connects only to its parent street
            "popular_times": p_times.tolist() if p_times is not None else None,
        }

        # Add POI to parent's connections
        nodes[parent_id]["conns"].append(pid)

        # Aggregate Popular Times up to the Street
        if p_times is not None:
            existing = nodes[parent_id]["popular_times"]
            
            if existing is None:
                nodes[parent_id]["popular_times"] = p_times.tolist()
            else:
                current_arr = np.array(existing)
                if current_arr.shape == p_times.shape:
                    nodes[parent_id]["popular_times"] = (current_arr + p_times).tolist()


    print(f"> Saving {OUTPUT_FILE}...")

    with gzip.open(OUTPUT_FILE, "wt", encoding="utf-8") as f:
        json.dump(list(nodes.values()), f)

    print(f"> Map built successfully with {len(nodes)} total nodes.")


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    if os.path.exists(OUTPUT_FILE):
        print(f"> '{OUTPUT_FILE}' already exists.")
    else:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set.")
        
        build_map()