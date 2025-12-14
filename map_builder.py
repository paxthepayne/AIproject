# ==========================================
# map_builder.py
# ==========================================

# General
import json
import pickle
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Data Handling
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point

# Google Services
import googlemaps
import populartimes


# CONFIGURATION ------------------------------------------------------------------------------------
GOOGLE_API_KEY = ""
TARGET_LOCATION = "Barcelona, Spain"
OUTPUT_FILE = "map.pkl"
MAX_WORKERS = 10

# Categories considered "Open Venues"
OPEN_CATS = [
    'park',
    'cemetery',
    'town_square',
    'tourist_attraction',
    'stadium',
    'amusement_park',
    'zoo',
    'natural_feature',
    'point_of_interest',
    'neighborhood',
    'route',
    'street_address',
    'transit_station',
    'bus_station',
    'train_station',
    'subway_station',
]


def parse_schedule(json_str):
    """Parse Google's populartimes format into a 7x24 matrix"""
    data = json.loads(str(json_str))
    if not data:
        return None
    sorter = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    data.sort(key=lambda x: sorter.index(x['name']))
    return np.array([d['data'] for d in data])


# DATA ACQUISITION ---------------------------------------------------------------------------------
print(f"Fetching 'Points of Interest' Data for {TARGET_LOCATION}")

raw = pd.DataFrame(
    requests.get(
        "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search"
        "?resource_id=31431b23-d5b9-42b8-bcd0-a84da9d8c7fa&limit=32000"
    ).json()["result"]["records"]
)

places_df = pd.DataFrame({
    'name': raw['name'],
    'lat': pd.to_numeric(raw['geo_epgs_4326_lat'], errors='coerce'),
    'lon': pd.to_numeric(raw['geo_epgs_4326_lon'], errors='coerce')
}).dropna()

gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

# Initialize columns
places_df['attributes'] = "[]"
places_df['popular_times'] = "[]"


# PARALLEL GOOGLE ENRICHMENT -----------------------------------------------------------------------
def enrich_place(task):
    """
    Worker function for Google enrichment.
    Returns (index, attributes_json, populartimes_json)
    """
    i, name = task
    try:
        find = gmaps.find_place(f"{name}, Barcelona", "textquery")
        if not find['candidates']:
            return i, None, None

        pid = find['candidates'][0]['place_id']
        details = gmaps.place(pid, fields=['type'])
        pt = populartimes.get_id(GOOGLE_API_KEY, pid)

        return (
            i,
            json.dumps(details['result'].get('types', [])),
            json.dumps(pt.get('populartimes', []))
        )
    except Exception:
        return i, None, None


tasks = [(i, row['name']) for i, row in places_df.iterrows()]

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(enrich_place, t) for t in tasks]

    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Google Maps enrichment"
    ):
        i, attrs, pops = future.result()
        if attrs is not None:
            places_df.at[i, 'attributes'] = attrs
            places_df.at[i, 'popular_times'] = pops


# FILL MISSING DATA --------------------------------------------------------------------------------
print("Computing Missing Popular Times")

places_df['matrix'] = places_df['popular_times'].apply(parse_schedule)

sources = places_df[places_df['matrix'].notnull()].copy()
targets = places_df[places_df['matrix'].isnull()].copy()

if not sources.empty:
    for i, t in targets.iterrows():
        dists = np.sqrt(
            (sources['lat'] - t['lat'])**2 +
            (sources['lon'] - t['lon'])**2
        )

        nearest = dists.nsmallest(3).index
        weights = 1 / (dists[nearest]**2 + 1e-6)

        weighted = np.sum(
            [sources.at[idx, 'matrix'] * w for idx, w in zip(nearest, weights)],
            axis=0
        )
        places_df.at[i, 'matrix'] = (weighted / weights.sum()).astype(int)


# STREET NETWORK -----------------------------------------------------------------------------------
print("Building Street Network")

G = ox.graph_from_place(TARGET_LOCATION, network_type='walk')
G_proj = ox.project_graph(G)

places_gdf = gpd.GeoDataFrame(
    places_df,
    geometry=[Point(xy) for xy in zip(places_df.lon, places_df.lat)],
    crs="EPSG:4326"
).to_crs(G_proj.graph['crs'])

nearest_edges = ox.nearest_edges(
    G_proj,
    places_gdf.geometry.x,
    places_gdf.geometry.y
)
places_df['street_edge'] = list(nearest_edges)

nodes, edges = ox.graph_to_gdfs(G_proj)
edges = edges.reset_index()
edges['node_id'] = range(len(edges))
edges['center'] = list(
    zip(
        edges.centroid.to_crs("EPSG:4326").y,
        edges.centroid.to_crs("EPSG:4326").x
    )
)

edge_lookup = edges.set_index(['u', 'v', 'key'])['node_id'].to_dict()
adj_list = {}

for _, row in edges.iterrows():
    sid = row['node_id']
    for n in (row['u'], row['v']):
        adj_list.setdefault(n, set()).add(sid)


# FINAL ASSEMBLY & EXPORT --------------------------------------------------------------------------
print(f"Exporting to '{OUTPUT_FILE}'")

final_nodes = {}

# Streets
for _, row in edges.iterrows():
    sid = row['node_id']
    neighbors = (adj_list[row['u']] | adj_list[row['v']]) - {sid}

    final_nodes[sid] = {
        'id': sid,
        'type': 0,
        'name': row.get('name', 'Street'),
        'coords': row['center'],
        'conns': list(neighbors),
        'pop_open': None,
        'pop_closed': None
    }

# Places
next_id = len(edges)
for _, row in places_df.iterrows():
    pid = next_id
    next_id += 1

    parent_sid = edge_lookup.get(row['street_edge'])
    if parent_sid is None:
        continue

    attrs = str(row['attributes'])
    is_open = any(cat in attrs for cat in OPEN_CATS)
    matrix = row['matrix']

    final_nodes[pid] = {
        'id': pid,
        'type': 1,
        'name': row['name'],
        'coords': (row['lat'], row['lon']),
        'conns': [parent_sid],
        'pop_open': matrix if is_open else None,
        'pop_closed': matrix if not is_open else None
    }

    final_nodes[parent_sid]['conns'].append(pid)

    target_key = 'pop_open' if is_open else 'pop_closed'
    if matrix is not None:
        if final_nodes[parent_sid][target_key] is None:
            final_nodes[parent_sid][target_key] = matrix.copy()
        else:
            final_nodes[parent_sid][target_key] += matrix


with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(list(final_nodes.values()), f)

print(f"> final map built with {len(final_nodes)} nodes.")
