'''
Fetches Points of Interest from OpenData Barcelona and OpenStreetMap,
enriches them with Google data (standard name, type, popular times), 
builds a street network (OpenStreetMap), integrates POIs into the network, 
and exports the final map as a compressed JSON file.
'''

# Data Handling
import gzip
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import requests

# OpenStreetMap
import osmnx as ox

# Google Services
import googlemaps
import populartimes

# Multithreading and Progress Bar 
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
TARGET_LOCATION = "Barcelona, Spain"
OUTPUT_FILE = "map.json.gz"
GOOGLE_API_KEY = ""


if __name__ == "__main__":

# Fetch POIs -------------------------------------------------------------------------------------------------
    print(f"Fetching 'Points of Interest' Data for {TARGET_LOCATION}... (OpenData BCN, OpenStreetMap)")

    # Load POIs (OpenData Barcelona)
    POIs = pd.DataFrame(requests.get(
        "https://opendata-ajuntament.barcelona.cat/data/api/action/datastore_search?resource_id=31431b23-d5b9-42b8-bcd0-a84da9d8c7fa&limit=32000"
    ).json()["result"]["records"])

    # Build DataFrame from OpenData BCN
    df_bcn = pd.DataFrame({
        "name": POIs["name"],
        "lat": pd.to_numeric(POIs["geo_epgs_4326_lat"], errors="coerce"),
        "lon": pd.to_numeric(POIs["geo_epgs_4326_lon"], errors="coerce"),
        "is_open": False,
        "popular_times": [np.zeros((7, 24), dtype=int).tolist() for _ in range(len(POIs))]
    })

    # Load POIs (OpenStreetMap)
    pois = ox.features_from_place(TARGET_LOCATION, tags={"amenity": ["cinema", "theatre", "place_of_worship", "hospital", "university"], "tourism": ["attraction", "museum", "theme_park", "viewpoint"], "shop": ["mall"]})
    pois = pois.reset_index(drop=True)

    # Filter malformed geometries
    pois = pois[pois.geometry.notnull() & pois.geometry.type.isin(["Point", "Polygon", "MultiPolygon"])]

    # Calculate coordinates (project to metric system and back to degrees)
    pois["geometry"] = pois.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)

    # Build DataFrame from OSM
    df_osm = pd.DataFrame({
        "name": pois["name"],
        "lat": pois.geometry.y,
        "lon": pois.geometry.x,
        "is_open": False,
        "popular_times": [np.zeros((7, 24), dtype=int).tolist() for _ in range(len(pois))]
    })

    # Merge the two datasets
    places_df = pd.concat([df_osm, df_bcn], ignore_index=True)

    # Remove entries with no name
    places_df = places_df[places_df["name"].notna() & (places_df["name"].str.strip() != "")]

    # Filter points with wrong coordinates 
    places_df = places_df.dropna(subset=['lat','lon']).reset_index(drop=True)
    places_df = places_df[(places_df.lat.between(41.0, 42.0)) & (places_df.lon.between(1.5, 2.5))].copy()
    places_df = places_df.reset_index(drop=True)


# Enrich POIs with Google Places Data ------------------------------------------------------------------------
    gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

    def enrich_place(task):
        i, name, lat, lon = task
        open_types = {'park', 'cemetery', 'town_square', 'tourist_attraction', 'stadium', 'amusement_park', 'zoo', 'natural_feature', 'point_of_interest', 'neighborhood', 'route', 'street_address', 'transit_station', 'bus_station', 'train_station', 'subway_station'}
        try:
            find = gmaps.find_place(f"{name}, Barcelona", "textquery", location_bias=f"point:{lat},{lon}")
            if not find.get('candidates'):
                return i, None, None, None
            place = find['candidates'][0]
            google_id = place['place_id']

            details = gmaps.place(google_id, fields=['name', 'type'])
            google_name = details.get('result', {}).get('name')
            is_open = bool(set(details.get("types", [])) & open_types)
            
            pt = populartimes.get_id(GOOGLE_API_KEY, google_id)
            popular_times = np.array([d['data'] for d in pt.get('populartimes', [])])

            return i, google_name, is_open, popular_times

        except Exception as e:
            print(e)
            return i, None, None, None

        
    # Enrich places
    tasks = [(i, row['name'], row['lat'], row['lon']) for i, row in places_df.iterrows()]
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(enrich_place, t) for t in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="> Enriching Places with Google Data"):
            i, google_name, is_open, popular_times = future.result()

            if google_name is not None:
                places_df.at[i, 'name'] = google_name
            places_df.at[i, 'is_open'] = bool(is_open)
            if popular_times.sum() != 0:
                places_df.at[i, 'popular_times'] = popular_times
                print(f"> Enriched Place: {places_df.at[i, 'name']}")

                

    
    # Remove entries that couldn't be enriched
    places_df = places_df[places_df['name'].notnull()]


# Build Street Network and Integrate POIs --------------------------------------------------------------------
    print(f"\nBuilding Street Network for {TARGET_LOCATION}... (OpenStreetMap)")

    # Load walkable street network
    network = ox.project_graph(ox.graph_from_place(TARGET_LOCATION, network_type='walk'))

    # Map POIs to nearest street edges
    places_gdf = gpd.GeoDataFrame(places_df, geometry=gpd.points_from_xy(places_df.lon, places_df.lat), crs="EPSG:4326").to_crs(network.graph['crs'])
    nearest_edges = ox.nearest_edges(network, places_gdf.geometry.x, places_gdf.geometry.y, return_dist=False)
    places_df['street_edge'] = nearest_edges

    # Assign unique ids to street segments and get coordinates (centroids)
    _, streets = ox.graph_to_gdfs(network)
    streets = streets.reset_index()
    streets['id'] = range(len(streets))
    centroids = streets.centroid.to_crs("EPSG:4326")
    streets['center'] = list(zip(centroids.y, centroids.x))

    # Build lookup for street edges
    street_lookup = streets.set_index(['u', 'v', 'key'])['id'].to_dict()

    # Build adjacency list (node -> streets ids)
    adj_list = {}
    for sid, u, v in streets[['id', 'u', 'v']].itertuples(index=False):
        adj_list.setdefault(u, set()).add(sid)
        adj_list.setdefault(v, set()).add(sid)

    # Build street nodes
    final_nodes = {
        sid: {
            'id': sid,
            'type': 0,
            'name': (name[0] if isinstance(name, list) and name else name if isinstance(name, str) else "Calle Sin Nombre"),
            'coords': center,
            'len': float(length or 0),
            'conns': list((adj_list[u] | adj_list[v]) - {sid}),
            'pop_open': None,
            'pop_closed': None
        }
        for sid, u, v, name, center, length
        in streets[['id', 'u', 'v', 'name', 'center', 'length']].itertuples(index=False)
    }

    # Attach POIs
    next_id = len(final_nodes)
    for _, row in places_df.iterrows():
        parent_sid = street_lookup.get(row['street_edge']) # Get corresponding street id
        if parent_sid is None: continue

        pid = next_id
        next_id += 1

        is_open = row['is_open']
        popular_times = row['popular_times']
        final_nodes[pid] = {
            'id': pid,
            'type': 1,
            'name': row['name'],
            'coords': (row['lat'], row['lon']),
            'len': 0,
            'conns': [parent_sid],
            'pop_open': popular_times if is_open else None,
            'pop_closed': popular_times if not is_open else None
        }

        final_nodes[parent_sid]['conns'].append(pid)

        target_key = 'pop_open' if is_open else 'pop_closed'
        
        if popular_times is not None:
            current_pop = np.array(popular_times)
            if final_nodes[parent_sid][target_key] is None:
                final_nodes[parent_sid][target_key] = current_pop.tolist()
            else:
                existing_raw = final_nodes[parent_sid][target_key]
                if existing_raw is None or len(existing_raw) == 0:
                    final_nodes[parent_sid][target_key] = current_pop.tolist()
                else:
                    existing = np.array(existing_raw)
                    final_nodes[parent_sid][target_key] = (existing + current_pop).tolist()


# Export Final Map (JSON + Gzip) -----------------------------------------------------------------------------
    print(f"\nExporting to '{OUTPUT_FILE}'...")

    def convert_sets(obj):
        '''
            Convert complex types to Lists so JSON can handle them
        '''
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError

    # Export final nodes to compressed JSON
    with gzip.open(f"{OUTPUT_FILE}", "wt", encoding="UTF-8") as f:
        json.dump(list(final_nodes.values()), f, default=convert_sets)

    print(f"> final map built with {len(final_nodes)} nodes.")


# SOME STATS -------------------------------------------------------------------------------------------------
    total_pois = len(places_df)
    pois_with_pop = places_df["popular_times"].notnull().sum()
    pois_without_pop = total_pois - pois_with_pop
    open_pois_with_pop = places_df[(places_df["is_open"] == True) & (places_df["popular_times"].notnull())].shape[0]
    closed_pois_with_pop = places_df[(places_df["is_open"] == False) & (places_df["popular_times"].notnull())].shape[0]

    print(f"\nTotal places: {total_pois}")
    print(f"Places with crowd data: {pois_with_pop}")
    print(f" ├─ Open: {open_pois_with_pop}")
    print(f" └─ Closed: {closed_pois_with_pop}")

    street_nodes = [n for n in final_nodes.values() if n["type"] == 0]
    total_streets = len(street_nodes)
    streets_with_crowd = sum((n["pop_open"] is not None) or (n["pop_closed"] is not None) for n in street_nodes)
    streets_with_open_crowd = sum(n["pop_open"] is not None for n in street_nodes)
    streets_with_closed_crowd = sum(n["pop_closed"] is not None for n in street_nodes)

    print(f"\nTotal streets: {total_streets}")
    print(f"Streets with crowd data: {streets_with_crowd}")
    print(f" ├─ Open: {streets_with_open_crowd}")
    print(f" └─ Closed: {streets_with_closed_crowd}")

