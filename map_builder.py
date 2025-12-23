# Data Handling
import gzip
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point

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
    if GOOGLE_API_KEY == "": raise ValueError("No Google Cloud API key found.")

# Fetch POIs from OpenStreetMap ------------------------------------------------------------------------------
    print(f"Fetching 'Points of Interest' Data for {TARGET_LOCATION} (OpenStreetMap)")

    # Desired venue types
    TAGS = {
        "amenity": ["bar", "cafe", "restaurant", "pub", "fast_food", "marketplace", "nightclub", "cinema", "theatre", "place_of_worship", "hospital", "university", "school"],
        "tourism": ["attraction", "museum", "gallery", "zoo", "theme_park", "viewpoint"], "leisure": ["park", "stadium", "sports_centre", "beach_resort"],
        "natural": ["beach"], "shop": ["mall"], "historic": True, "public_transport": ["station", "platform"], "railway": ["station", "subway_entrance"]
    }

    # Load POIs
    pois = ox.features_from_place(TARGET_LOCATION, tags=TAGS)
    pois = pois.reset_index(drop=True)

    # Filter malformed geometries
    pois = pois[pois.geometry.notnull() & pois.geometry.type.isin(["Point", "Polygon", "MultiPolygon"])]

    # Calculate coordinates (project to metric system and back to degrees)
    pois["geometry"] = pois.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)

    # Initialize DataFrame and filter invalid entries
    places_df = pd.DataFrame({"name": pois["name"], "lat": pois.geometry.y, "lon": pois.geometry.x, "is_open": False, "popular_times": None})
    places_df = places_df.dropna(subset=["name", "lat", "lon"])
    places_df = places_df[places_df["name"].str.strip() != ""]


# Enrich POIs with Google Places Data ------------------------------------------------------------------------
    gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

    def enrich_place(query):
        """
        Enrich a single place with Google Maps data.
        Returns: (index, google_name, lat, lon, is_open, popular_times)
        """
        open_types = ['park', 'cemetery', 'town_square', 'tourist_attraction', 'stadium', 'amusement_park', 'zoo', 'natural_feature', 'point_of_interest', 'neighborhood', 'route', 'street_address', 'transit_station', 'bus_station', 'train_station', 'subway_station']
        i, osm_name, lat, lon = query
        try:
            # Search place by name and get its Google id
            results = gmaps.find_place(input=osm_name, input_type="textquery", location_bias=f"circle:200@{lat},{lon}", fields=['place_id'])
            if not results['candidates']: 
                return i, None, None, None, None, None
            place_id = results['candidates'][0]['place_id']

            # Fetch place details with populartimes
            details = populartimes.get_id(GOOGLE_API_KEY, place_id)

            google_name = details.get('name')
            lat = details.get('coordinates', {}).get('lat', lat)
            lon = details.get('coordinates', {}).get('lng', lon)
            is_open = any(t in open_types for t in details.get('types', [])) 
            popular_times = (np.array([d["data"] for d in details["populartimes"]]) if details.get("populartimes") else None)

            return i, google_name, lat, lon, is_open, popular_times
        except Exception:
            return i, None, None, None, None, None
        
    # Concurrent enrichment of places
    places = [(i, row['name'], row['lat'], row['lon']) for i, row in places_df.iterrows()]
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = [executor.submit(enrich_place, p) for p in places]
        
        # Collect results as they complete (tqdm for progress bar)
        updates = []
        pbar = tqdm(total=len(results), desc="Enriching POIs with Google Data")
        for fut in as_completed(results):
            updates.append(fut.result())
            pbar.update(1)
        pbar.close()

    # Update DataFrame with enriched data
    for i, google_name, lat, lon, is_open, popular_times in updates:
        if google_name is not None:
            places_df.at[i, 'name'] = google_name
        if lat is not None and lon is not None:
            places_df.at[i, 'lat'] = lat
            places_df.at[i, 'lon'] = lon
        places_df.at[i, 'is_open'] = bool(is_open)
        places_df.at[i, 'popular_times'] = popular_times


    # Remove entries that couldn't be enriched
    places_df = places_df[places_df['name'].notnull()]


# Build Street Network and Integrate POIs --------------------------------------------------------------------
    print(f"Building Street Network for {TARGET_LOCATION} (OpenStreetMap)")

    # Load walkable street network
    network = ox.project_graph(ox.graph_from_place(TARGET_LOCATION, network_type='walk'))

    # Map POIs to nearest street edges
    places_gdf = gpd.GeoDataFrame(places_df, geometry=[Point(xy) for xy in zip(places_df.lon, places_df.lat)], crs="EPSG:4326").to_crs(network.graph['crs'])
    nearest_edges = ox.nearest_edges(network, places_gdf.geometry.x, places_gdf.geometry.y)
    places_df['street_edge'] = list(nearest_edges)

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
            if final_nodes[parent_sid][target_key] is None:
                final_nodes[parent_sid][target_key] = popular_times.copy()
            else:
                final_nodes[parent_sid][target_key] += popular_times



# Export Final Map (JSON + Gzip) -----------------------------------------------------------------------------
    print(f"Exporting to '{OUTPUT_FILE}'")

    # Convert complex types to Lists so JSON can handle them
    def convert_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError


    with gzip.open(f"{OUTPUT_FILE}", "wt", encoding="UTF-8") as f:
        json.dump(list(final_nodes.values()), f, default=convert_sets)

    print(f"> final map built with {len(final_nodes)} nodes.")
