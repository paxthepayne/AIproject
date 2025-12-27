'''
Fetches Points of Interest from OpenData Barcelona,
enriches them with Google data (name, type, popular times), 
builds a street network (OpenStreetMap), integrates POIs into the network, 
and exports the final map as a compressed JSON file.
'''

# Data Handling
import gzip
import json
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import requests

# OpenStreetMap
import osmnx as ox

# Multithreading and Progress Bar 
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm

# Google Data (Outscraper)
from outscraper import ApiClient
outscraper = ApiClient(api_key="")

# Configuration
TARGET_LOCATION = "Barcelona, Spain"
OUTPUT_FILE = "map.json.gz"

# --- HELPER FUNCTIONS ---------------------------------------------------------

def convert_sets(obj):
    ''' Convert complex types to Lists to save as JSON '''
    if isinstance(obj, set): return list(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError

def save_map(nodes_dict):
    print(f"Saving map to '{OUTPUT_FILE}'...")
    with gzip.open(OUTPUT_FILE, "wt", encoding="UTF-8") as f:
        json.dump(list(nodes_dict.values()), f, default=convert_sets)

def load_map():
    print(f"Loading existing map from '{OUTPUT_FILE}'...")
    with gzip.open(OUTPUT_FILE, "rt", encoding="UTF-8") as f:
        nodes_list = json.load(f)
    return {n['id']: n for n in nodes_list} # Dict {id: node} for easy processing

# --- PHASE 1: BUILD STRUCTURE -------------------------------------------------

def build_structure():
    print(f"Building base map for {TARGET_LOCATION}:")
    
    # Fetch POIs
    print(f"> Fetching 'Points of Interest' Data (OpenData BCN)...")
    POIs = pd.DataFrame(requests.get(
    "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search?resource_id=31431b23-d5b9-42b8-bcd0-a84da9d8c7fa&limit=32000"
    ).json()["result"]["records"])

    # Initialize Places DataFrame
    places_df = pd.DataFrame({
        "name": POIs["name"],
        "lat": pd.to_numeric(POIs["geo_epgs_4326_lat"], errors="coerce"),
        "lon": pd.to_numeric(POIs["geo_epgs_4326_lon"], errors="coerce"),
        "is_open": False,
        "popular_times": None
    })
    places_df.dropna(subset=['lat', 'lon'], inplace=True)
    places_df.reset_index(drop=True, inplace=True)

    # Build Street Network
    print(f"> Fetching Street Network (OpenStreetMap)...")
    network = ox.project_graph(ox.graph_from_place(TARGET_LOCATION, network_type='walk'))

    # Map POIs to Street Edges
    print("> Linking POIs to Streets...")
    places_gdf = gpd.GeoDataFrame(
        places_df, 
        geometry=gpd.points_from_xy(places_df.lon, places_df.lat), 
        crs="EPSG:4326"
    ).to_crs(network.graph['crs'])
    nearest_edges = ox.nearest_edges(network, places_gdf.geometry.x, places_gdf.geometry.y, return_dist=False)
    places_df['street_edge'] = nearest_edges

    # Prepare Street Data
    _, streets = ox.graph_to_gdfs(network)
    streets = streets.reset_index()
    streets['id'] = range(len(streets))
    centroids = streets.centroid.to_crs("EPSG:4326")
    streets['center'] = list(zip(centroids.y, centroids.x))

    street_lookup = streets.set_index(['u', 'v', 'key'])['id'].to_dict()

    # Adjacency List
    adj_list = {}
    for sid, u, v in streets[['id', 'u', 'v']].itertuples(index=False):
        adj_list.setdefault(u, set()).add(sid)
        adj_list.setdefault(v, set()).add(sid)

    # Create Nodes Dictionary
    final_nodes = {}

    # Create Street Nodes (Type 0)
    for row in streets[['id', 'u', 'v', 'name', 'center', 'length']].itertuples(index=False):
        sid = row.id
        u, v = row.u, row.v
        name = row.name
        safe_name = (name[0] if isinstance(name, list) and name else name if isinstance(name, str) else "Calle Sin Nombre")
        
        final_nodes[sid] = {
            'id': sid,
            'type': 0,
            'name': safe_name,
            'coords': row.center,
            'len': float(row.length or 0),
            'conns': list((adj_list.get(u, set()) | adj_list.get(v, set())) - {sid}),
            'pop_open': None,
            'pop_closed': None
        }

    # Create POI Nodes (Type 1)
    next_id = len(final_nodes)
    for _, row in places_df.iterrows():
        parent_sid = street_lookup.get(row['street_edge'])
        if parent_sid is None: continue

        pid = next_id
        next_id += 1
        
        # Initial POI Node (Empty Data)
        final_nodes[pid] = {
            'id': pid,
            'type': 1,
            'name': row['name'],
            'coords': (row['lat'], row['lon']),
            'len': 0,
            'conns': [parent_sid],
            'pop_open': None,
            'pop_closed': None
        }
        # Link Parent -> POI
        final_nodes[parent_sid]['conns'].append(pid)

    print(f"Base map built: {len(final_nodes)} nodes.")
    return final_nodes

# --- PHASE 2: ENRICH & AGGREGATE ----------------------------------------------

def enrich_place_worker(query):
    """ Worker function for ThreadPool """
    nid, name, lat, lon = query
    try:
        # Simplified query
        results = outscraper.google_maps_search(query=f"{name}, Barcelona", limit=1)
        if not results:
            return nid, None, None, None, None, None
        place = results[0]
        if isinstance(place, list): place = place[0]
        
        google_id = place.get("place_id")
        results = outscraper.google_maps_search(query=google_id, limit=1)
        if not results:
            return nid, None, None, None, None, None
        place = results[0]
        if isinstance(place, list): place = place[0]

        google_name = place.get("name", name)
        new_lat = place.get("latitude", lat)
        new_lon = place.get("longitude", lon)
        
        # Open Logic
        place_types = set(place.get("place_types") or [])
        valid_types = {
            'park', 'cemetery', 'town_square', 'tourist_attraction', 'stadium', 
            'amusement_park', 'zoo', 'natural_feature', 'point_of_interest', 
            'neighborhood', 'route', 'street_address', 'transit_station', 
            'bus_station', 'train_station', 'subway_station'
        }
        is_open = 0 < len(place_types & valid_types)

        # Popular Times Logic
        pop = place.get("popular_times")
        popular_times = None
        if isinstance(pop, dict):
            popular_times = np.array(list(pop.values())) / 100.0
        elif isinstance(pop, list):
            try:
                popular_times = np.array([d["data"] for d in pop]) / 100.0
            except:
                popular_times = None

        return nid, google_name, new_lat, new_lon, is_open, popular_times
    except Exception:
        return nid, None, None, None, None, None

def enrich_and_aggregate(nodes):
    print(f"\n--- PHASE 2: ENRICHING & AGGREGATING DATA ---")
    
    # 1. Reset Street Aggregations (Clear old data)
    print("Resetting street crowd data...")
    for n in nodes.values():
        if n['type'] == 0:
            n['pop_open'] = None
            n['pop_closed'] = None

    # 2. Identify POIs to enrich
    poi_tasks = []
    for n in nodes.values():
        if n['type'] == 1:
            # Task: (id, name, lat, lon)
            poi_tasks.append((n['id'], n['name'], n['coords'][0], n['coords'][1]))

    print(f"Enriching {len(poi_tasks)} POIs via Outscraper...")

    # 3. Fetch Data
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(enrich_place_worker, t) for t in poi_tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Google Data"):
            try:
                res = future.result(timeout=60)
                nid, g_name, lat, lon, is_open, pop_times = res
                
                if g_name:
                    # Update POI Node
                    nodes[nid]['name'] = g_name
                    nodes[nid]['coords'] = (lat, lon)
                    # We store the raw data in the POI node relative to its type
                    # Using 'pop_open' if it's an 'open' type place, else 'pop_closed'
                    # Note: POIs usually just hold the data, aggregation happens on streets
                    nodes[nid]['pop_open'] = pop_times if is_open else None
                    nodes[nid]['pop_closed'] = pop_times if not is_open else None

                    # 4. Immediate Aggregation to Parent Street
                    if pop_times is not None:
                        parent_id = nodes[nid]['conns'][0] # POI has only 1 conn (the street)
                        street = nodes[parent_id]
                        
                        target_key = 'pop_open' if is_open else 'pop_closed'
                        current_pop = np.array(pop_times)

                        existing_raw = street[target_key]
                        
                        if existing_raw is None:
                            street[target_key] = current_pop.tolist()
                        else:
                            # Sum arrays
                            existing = np.array(existing_raw)
                            if existing.shape == current_pop.shape:
                                street[target_key] = (existing + current_pop).tolist()

            except TimeoutError:
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    return nodes

# --- MAIN EXECUTION -----------------------------------------------------------

if __name__ == "__main__":
    
    final_nodes = {}

    # Step 1: Check if map exists
    if os.path.exists(OUTPUT_FILE):
        print(f"Map file '{OUTPUT_FILE}' found.")
        final_nodes = load_map()
    else:
        print(f"Map file '{OUTPUT_FILE}' NOT found. Building from scratch...")
        final_nodes = build_structure()
        # Save skeletal map immediately
        save_map(final_nodes)

    # Step 2: Enrich and Recalculate popular times
    final_nodes = enrich_and_aggregate(final_nodes)

    # Step 3: Export Final map
    save_map(final_nodes)

    # STATS ----------------------------------------------------------------------
    street_nodes = [n for n in final_nodes.values() if n["type"] == 0]
    total_streets = len(street_nodes)
    
    streets_w_open = sum(n["pop_open"] is not None for n in street_nodes)
    streets_w_closed = sum(n["pop_closed"] is not None for n in street_nodes)

    print(f"\nFinal Stats:")
    print(f"Total Nodes: {len(final_nodes)}")
    print(f"Total Streets: {total_streets}")
    print(f"Streets with 'Open' Crowd Data: {streets_w_open}")
    print(f"Streets with 'Closed' Crowd Data: {streets_w_closed}")