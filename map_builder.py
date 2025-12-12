import osmnx as ox
import json

print("Building the Q-Learning maps 'map_streets.json' and 'map_nodes.json' with data from OpenStreetMap...")

# Get the Data
place_name = "Barcelona, Spain"
G = ox.graph_from_place(place_name, network_type='walk', simplify=True)
_, edges = ox.graph_to_gdfs(G, nodes=True, node_geometry=True)
edges = edges.reset_index()

# Id
edges['id'] = range(len(edges))

# Name
edges['name'] = edges['name'].apply(lambda x: x[0] if isinstance(x, list) else x)
edges['name'] = edges['name'].fillna("Calle sin nombre")

# Connections
edges['connections'] = list(zip(edges['u'], edges['v']))

# Coordinates (Midpoint)
centroids = edges.geometry.to_crs(edges.estimate_utm_crs()).centroid.to_crs(edges.crs)
edges['mid_lat'] = centroids.y
edges['mid_lon'] = centroids.x
edges['coordinates'] = list(zip(edges['mid_lat'], edges['mid_lon']))

# Export Streets Map
final_table = edges[['id', 'name', 'connections', 'coordinates']]
final_table.to_json("map_streets.json", orient="records", indent=2)

# Create Nodes Map (Node ID -> List of Street IDs)
node_map_df = edges[['id', 'u', 'v']].melt(id_vars=['id'], value_name='node_id')
node_to_streets = node_map_df.groupby('node_id')['id'].apply(list).to_dict()
node_to_streets = {int(k): [int(x) for x in v] for k, v in node_to_streets.items()}

# Export Nodes Map
with open("map_nodes.json", "w") as f:
    json.dump(node_to_streets, f)

print("> export complete")