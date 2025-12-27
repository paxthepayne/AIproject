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

# IMPORTS AND CONFIGURATION

# system
import os, json, gzip

# data handling
import numpy as np, pandas as pd, geopandas as gpd

# external APIs
import requests
import osmnx as ox, googlemaps, populartimes

# multithreading and progress bar
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

TARGET_LOCATION = "Barcelona, Spain"
OUTPUT_FILE = "map.json.gz"
GOOGLE_API_KEY = ""


# MAP BUILDER

def build_map():
    print("> Fetching POIs from OpenData BCN")
    
    # Fetch POIs from OpenData BCN
    pois = pd.DataFrame(requests.get(
        "https://opendata-ajuntament.barcelona.cat/data/api/action/"
        "datastore_search?resource_id=31431b23-d5b9-42b8-bcd0-a84da9d8c7fa&limit=32000"
        ).json()["result"]["records"])

    # Initialize places DataFrame
    places = pd.DataFrame({
        "name": pois["name"],
        "lat": pd.to_numeric(pois["geo_epgs_4326_lat"], errors="coerce"),
        "lon": pd.to_numeric(pois["geo_epgs_4326_lon"], errors="coerce"),
        "popular_times": None,
    })

    # Drop invalid entries
    places = places.dropna(subset=["name", "lat", "lon"]).reset_index(drop=True)


    print("> Enriching POIs with Google popular times")
    
    # Google Maps client
    gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

    def enrich_place(i, name, lat, lon):
        try:
            # Find place Google ID
            res = gmaps.find_place(name, "textquery", location_bias=f"circle:200@{lat},{lon}", fields=["place_id"])
            if not res["candidates"]:
                return i, None
            google_id = res["candidates"][0]["place_id"]
            
            # Fetch popular times
            pop = populartimes.get_id(GOOGLE_API_KEY, google_id).get("populartimes")
            if not pop:
                return i, None

            # Transform populartimes to array
            popular_times = np.array([d["data"] for d in pop], dtype=float)
            if popular_times.shape != (7, 24):
                return i, None

            return i, popular_times
        except Exception:
            return i, None

    # Enrich places in parallel
    tasks = [(i, r["name"], r["lat"], r["lon"]) for i, r in places.iterrows()]
    with ThreadPoolExecutor(max_workers=10) as ex:
        results = [ex.submit(enrich_place, *t) for t in tasks]
        for r in tqdm(as_completed(results), total=len(results)):
            i, popular_times = r.result()
            if popular_times is not None: places.at[i, "popular_times"] = popular_times

    print(f"> POIs with crowd data: {(places['popular_times'].notna()).sum()}")


    print("> Building OpenStreetMap walking network")

    # Build street network from OpenStreetMap (as GeoDataFrame)
    G = ox.project_graph(ox.graph_from_place(TARGET_LOCATION, network_type="walk"))
    _, streets = ox.graph_to_gdfs(G)
    streets = streets.reset_index()
    streets["id"] = range(len(streets))

    # Compute street center points
    centroids = streets.centroid.to_crs("EPSG:4326")
    streets["center"] = list(zip(centroids.y, centroids.x))

    # Create a street neighbor lookup
    street_lookup = streets.set_index(["u", "v", "key"])["id"].to_dict()
    adj = {}
    for sid, u, v in streets[["id", "u", "v"]].itertuples(index=False):
        adj.setdefault(u, set()).add(sid)
        adj.setdefault(v, set()).add(sid)


    print("> Linking POIs to streets")

    # Find nearest street for each place
    gdf = gpd.GeoDataFrame(places, geometry=gpd.points_from_xy(places.lon, places.lat), crs="EPSG:4326").to_crs(G.graph["crs"])
    places["street_edge"] = ox.nearest_edges(G, gdf.geometry.x, gdf.geometry.y, return_dist=False)


    print("> Building final nodes")
    
    # Combine streets and POIs into final nodes
    nodes = {}

    # Streets
    for sid, u, v, name, center, length in streets[
        ["id", "u", "v", "name", "center", "length"]
    ].itertuples(index=False):

        safe_name = (
            name[0] if isinstance(name, list) and name
            else name if isinstance(name, str)
            else "Calle Sin Nombre"
        )

        nodes[sid] = {
            "id": sid,
            "type": 0,
            "name": safe_name,
            "coords": center,
            "len": float(length or 0),
            "conns": list((adj[u] | adj[v]) - {sid}),
            "popular_times": None,
        }

    # POIs
    next_id = len(nodes)

    for _, r in places.iterrows():
        parent = street_lookup.get(r["street_edge"])
        if parent is None:
            continue

        pid = next_id
        next_id += 1

        popular_times = r["popular_times"]

        nodes[pid] = {
            "id": pid,
            "type": 1,
            "name": r["name"],
            "coords": (r["lat"], r["lon"]),
            "len": 0,
            "conns": [parent],
            "popular_times": popular_times.tolist() if popular_times is not None else None,
        }

        nodes[parent]["conns"].append(pid)

        if popular_times is not None:
            existing = nodes[parent]["popular_times"]
            if existing is None:
                nodes[parent]["popular_times"] = popular_times.tolist()
            else:
                a = np.array(existing)
                if a.shape == popular_times.shape == (7, 24):
                    nodes[parent]["popular_times"] = (a + popular_times).tolist()


    print("> Saving map.json.gz")

    # Export to JSON.gz
    with gzip.open(OUTPUT_FILE, "wt", encoding="utf-8") as f:
        json.dump(list(nodes.values()), f)

    print(f"> Map built with {len(nodes)} nodes")


# RUN MAP BUILDER ONLY IF IT DOESN'T EXIST

if __name__ == "__main__":

    if os.path.exists(OUTPUT_FILE):
        print(f"> '{OUTPUT_FILE}' already exists.")
    else:
        if GOOGLE_API_KEY == "":
            raise ValueError("GOOGLE_API_KEY not set")
        build_map()
