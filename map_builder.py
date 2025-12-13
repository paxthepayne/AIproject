"""
Builds a table of streets of Barcelona with data from OpenStreetMap
Final attributes: ["id", "name", "length", "center", "connections"]
"""

import osmnx as ox

print("Building 'map.json'...")

# Get Data
G = ox.graph_from_place("Barcelona, Spain", network_type='walk', simplify=True)
_, edges = ox.graph_to_gdfs(G)
edges = edges.reset_index()

# Simple columns
edges['id'] = range(len(edges))
edges['name'] = edges['name'].apply(lambda x: x[0] if isinstance(x, list) else x).fillna("Calle sin nombre")
edges['length'] = edges['length'].astype(float)

# Geometry columns
centroids = edges.to_crs(edges.estimate_utm_crs()).centroid.to_crs(edges.crs)
edges['center'] = list(zip(centroids.y, centroids.x))

# Connections
node_map = edges[['id', 'u', 'v']].melt(id_vars='id', value_name='node')
node_index = node_map.groupby('node')['id'].apply(set).to_dict()
edges['connections'] = edges.apply(lambda r: list((node_index.get(r['u'], set()) | node_index.get(r['v'], set())) - {r['id']}), axis=1)

# Export
output_cols = ['id', 'name', 'length', 'center', 'connections']
edges[output_cols].to_json("map.json", orient="records", indent=2)

print(f"> Processed {len(edges)} streets.")