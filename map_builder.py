import os
import sys
import json
import ast
import math
import pickle
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import googlemaps
import populartimes
from shapely.geometry import Point

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
GOOGLE_API_KEY = ""  # 🔑 Insert your Google Maps API Key here
TARGET_LOCATION = "Barcelona, Spain"
PLACES_DATABASE_FILE = "places.csv"
OUTPUT_MAP_FILE = "map.pkl"

PUBLIC_ACCESS_CATEGORIES = {
    'tourist_attraction', 'park', 'town_square', 'point_of_interest', 
    'natural_feature', 'neighborhood', 'cemetery', 'street_address', 'route'
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R_EARTH_METERS = 6371e3 
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R_EARTH_METERS * c

def convert_json_schedule_to_numpy(json_string):
    try:
        schedule_list = json.loads(json_string)
        if not schedule_list: return None
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        schedule_list.sort(key=lambda x: day_order.index(x['name']))
        return np.array([day['data'] for day in schedule_list])
    except: return None

def convert_numpy_schedule_to_json(numpy_matrix):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return json.dumps([{"name": d, "data": m.tolist()} for d, m in zip(days, numpy_matrix)])

# ==========================================
# 3. DATA ACQUISITION PIPELINE
# ==========================================

def fetch_and_process_places():
    if os.path.exists(PLACES_DATABASE_FILE):
        print(f"✅ Found existing '{PLACES_DATABASE_FILE}'. Loading data...")
        return pd.read_csv(PLACES_DATABASE_FILE)

    if not GOOGLE_API_KEY: sys.exit("❌ Error: No database found and no GOOGLE_API_KEY provided.")
    print("⚠️ Database not found. Starting data collection pipeline...")

    # Fetch Base POIs
    print("   [1/4] Fetching base location data...")
    url = "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search?resource_id=31431b23-d5b9-42b8-bcd0-a84da9d8c7fa&limit=32000"
    raw = pd.DataFrame(requests.get(url).json()["result"]["records"])
    places_df = pd.DataFrame({
        'name': raw['name'],
        'latitude': pd.to_numeric(raw['geo_epgs_4326_lat'], errors='coerce'),
        'longitude': pd.to_numeric(raw['geo_epgs_4326_lon'], errors='coerce')
    }).dropna()

    # Google Enrichment
    print("   [2/4] Enriching with Google Maps data...")
    gmaps = googlemaps.Client(key=GOOGLE_API_KEY)
    for i, row in places_df.iterrows():
        try:
            find = gmaps.find_place(f"{row['name']}, Barcelona", "textquery")
            if find['candidates']:
                pid = find['candidates'][0]['place_id']
                places_df.at[i, 'google_id'] = pid
                details = gmaps.place(pid, fields=['name', 'type'])['result']
                places_df.at[i, 'attributes'] = json.dumps(details.get('types', []))
                pt = populartimes.get_id(GOOGLE_API_KEY, pid)
                places_df.at[i, 'popular_times'] = json.dumps(pt.get('populartimes', []))
        except: pass

    # Spatial Imputation
    print("   [3/4] Imputing missing data...")
    places_df['has_data'] = places_df['popular_times'].apply(lambda x: len(str(x)) > 20)
    sources = places_df[places_df['has_data']].copy()
    targets = places_df[~places_df['has_data']].copy()
    src_matrices = {i: convert_json_schedule_to_numpy(r['popular_times']) for i, r in sources.iterrows()}

    for i, t in targets.iterrows():
        dists = np.sqrt((sources['latitude'] - t['latitude'])**2 + (sources['longitude'] - t['longitude'])**2)
        weighted_sum, total_w = np.zeros((7, 24)), 0
        for n_idx, dist in dists.nsmallest(3).items():
            sched = src_matrices.get(n_idx)
            if sched is not None:
                w = 1 / (max(dist, 0.0001) ** 2)
                weighted_sum += sched * w
                total_w += w
        if total_w > 0:
            places_df.at[i, 'popular_times'] = convert_numpy_schedule_to_json((weighted_sum / total_w).astype(int))

    places_df.to_csv(PLACES_DATABASE_FILE, index=False, quoting=1)
    return places_df

# ==========================================
# 4. MAP CONSTRUCTION PIPELINE
# ==========================================

def generate_street_network(places_df):
    print(f"🚀 Building Unified Network (Streets + Places) for: {TARGET_LOCATION}")
    
    # 1. Download Graph
    raw_graph = ox.graph_from_place(TARGET_LOCATION, network_type='walk')
    projected_graph = ox.project_graph(raw_graph)

    # 2. Prepare Place Data (Memory Optimized)
    print("   > Processing place data...")
    def parse_attrs(row):
        try:
            clean_json = str(row['popular_times']).replace('""', '"')
            if clean_json.startswith('"') and clean_json.endswith('"'):
                clean_json = clean_json[1:-1]
            data = ast.literal_eval(clean_json)
            # OPTIMIZATION: uint16 (2 bytes)
            return np.stack([d['data'] for d in data]).astype(np.uint16)
        except: 
            return None # OPTIMIZATION: Store None instead of zeros

    places_df['time_matrix'] = places_df.apply(parse_attrs, axis=1)

    def get_cat(row):
        try:
            attrs = ast.literal_eval(str(row['attributes']))
            return 'open' if not set(attrs).isdisjoint(PUBLIC_ACCESS_CATEGORIES) else 'closed'
        except: return 'closed'
    places_df['cat_type'] = places_df.apply(get_cat, axis=1)

    # 3. Snap Places to Streets
    print("   > Snapping places to network...")
    places_gdf = gpd.GeoDataFrame(
        places_df, 
        geometry=[Point(xy) for xy in zip(places_df.longitude, places_df.latitude)], 
        crs="EPSG:4326"
    ).to_crs(projected_graph.graph['crs'])

    nearest_edges = ox.nearest_edges(projected_graph, places_gdf.geometry.x, places_gdf.geometry.y)
    matches = pd.DataFrame(list(nearest_edges), columns=['u', 'v', 'key'], index=places_gdf.index)
    places_gdf = pd.concat([places_gdf, matches], axis=1)

    # 4. Process Streets Geometry
    print("   > Processing street geometry...")
    _, streets_gdf = ox.graph_to_gdfs(raw_graph)
    streets_gdf = streets_gdf.reset_index()
    streets_gdf['node_id'] = range(len(streets_gdf))
    
    cents = streets_gdf.to_crs(streets_gdf.estimate_utm_crs()).centroid.to_crs(streets_gdf.crs)
    streets_gdf['center_coordinates'] = list(zip(cents.y, cents.x))
    streets_gdf['length'] = streets_gdf['length'].astype(float)
    streets_gdf['name'] = streets_gdf['name'].fillna("Calle Sin Nombre").apply(lambda x: x[0] if isinstance(x, list) else x)

    # Connectivity Map
    node_to_streets = {}
    for _, row in streets_gdf.iterrows():
        sid = row['node_id']
        for n in (row['u'], row['v']):
            node_to_streets.setdefault(n, set()).add(sid)

    # 5. Build Unified Node List
    print("   > Assembling unified graph...")
    street_lookup = streets_gdf.set_index(['u', 'v', 'key'])['node_id'].to_dict()
    final_nodes = {} 

    # --- Initialize Street Nodes ---
    for _, row in streets_gdf.iterrows():
        sid = row['node_id']
        # Find intersections (exclude self)
        conns = (node_to_streets.get(row['u'], set()) | node_to_streets.get(row['v'], set())) - {sid}
        
        final_nodes[sid] = {
            'id': sid,
            'type': 0, # 0 = Street
            'name': row['name'],
            'coords': row['center_coordinates'],
            'len': row['length'],
            'conns': list(conns),
            'pop_open': None,
            'pop_closed': None
        }

    # --- Initialize Place Nodes ---
    place_id_counter = len(streets_gdf)
    
    for _, row in places_gdf.iterrows():
        pid = place_id_counter
        place_id_counter += 1
        
        street_key = (row['u'], row['v'], row['key'])
        parent_id = street_lookup.get(street_key)
        
        if parent_id is None: continue
        
        matrix = row['time_matrix']
        p_open = matrix if (matrix is not None and row['cat_type'] == 'open') else None
        p_closed = matrix if (matrix is not None and row['cat_type'] == 'closed') else None

        final_nodes[pid] = {
            'id': pid,
            'type': 1, # 1 = Place
            'name': row['name'],
            'coords': (row['latitude'], row['longitude']),
            'len': 0.0,
            'conns': [parent_id],
            'pop_open': p_open,
            'pop_closed': p_closed
        }
        
        # Update Parent Street
        final_nodes[parent_id]['conns'].append(pid)
        
        if matrix is not None:
            target_key = 'pop_open' if row['cat_type'] == 'open' else 'pop_closed'
            
            if final_nodes[parent_id][target_key] is None:
                final_nodes[parent_id][target_key] = matrix.copy()
            else:
                final_nodes[parent_id][target_key] += matrix

    # 6. Export (Standard Pickle)
    final_output = list(final_nodes.values())
    print(f"   > Exporting {len(final_output)} nodes to {OUTPUT_MAP_FILE}...")
    
    with open(OUTPUT_MAP_FILE, "wb") as f:
        pickle.dump(final_output, f)
        
    print("🎉 Success! Optimized Map built.")
    print("ℹ️  Keys: 'coords', 'len', 'conns', 'pop_open', 'pop_closed'")
    print("ℹ️  Types: 0=Street, 1=Place")

if __name__ == "__main__":
    places_data = fetch_and_process_places()
    generate_street_network(places_data)