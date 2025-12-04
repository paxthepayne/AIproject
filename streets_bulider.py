import osmnx as ox
import pandas as pd

# 1. Get the Data
place_name = "Barri Gòtic, Barcelona, Spain"
print("Downloading data...")
G = ox.graph_from_place(place_name, network_type='walk', simplify=True)

# 2. Convert Graph to Tables
nodes, edges = ox.graph_to_gdfs(G)

# 3. Prepare Node Coordinates
# OSMnx uses x=Longitude, y=Latitude
# We verify the index is named 'osmid' for merging (it usually is by default)
nodes = nodes[['y', 'x']].rename(columns={'y': 'lat', 'x': 'lon'})

# 4. Prepare Edges
edges_reset = edges.reset_index()

# 5. Merge Start Node (u) Coordinates
df = edges_reset.merge(nodes, left_on='u', right_index=True)
df = df.rename(columns={'lat': 'u_lat', 'lon': 'u_lon'})

# 6. Merge End Node (v) Coordinates
df = df.merge(nodes, left_on='v', right_index=True)
df = df.rename(columns={'lat': 'v_lat', 'lon': 'v_lon'})

# 7. Create the specific columns you requested

# A. CLEAN NAME: Handle cases where name is a list or NaN
df['name'] = df['name'].apply(lambda x: x[0] if isinstance(x, list) else x)
df['name'] = df['name'].fillna("Unnamed Street")

# B. GENERATE ID: Create a unique identifier for the street segment
# We can use the dataframe index, or create a composite "u-v" string
df['id'] = df.index

# C. FORMAT CONNECTIONS: Create the list of lists [[id, lat, lon], [id, lat, lon]]
# Note: User requested [id, lat, lon] order
df['connections'] = df.apply(
    lambda row: [
        [row['u'], row['u_lat'], row['u_lon']], 
        [row['v'], row['v_lat'], row['v_lon']]
    ], axis=1
)

# 8. Final Selection
final_table = df[['name', 'id', 'connections']].copy()

# Display
print(f"Generated {len(final_table)} streets.")
pd.set_option('display.max_colwidth', None) # Show full connection lists
print(final_table)

final_table.to_json("streets_grid.json", orient="records", indent=2)