"""
Main execution script for Smart Crowd Router.
Manages data loading, user interaction, and pathfinding execution.
"""

import pandas as pd
import datetime
import random
import q_learning

print(f"[AI Project - Smart Crowd Router] \nBy ______________________________\n")

# Load Data
streets = pd.read_json("map.json").set_index('id')

places = pd.read_csv("places_database.csv")
places = places.rename(columns={"latitude": "lat", "longitude": "lon"})
places = places.dropna(subset=["lat", "lon", "popular_times"])

pois_list = places[["lat", "lon", "popular_times"]].to_dict(orient="records")

# Environment Setup
now = datetime.datetime.now()
weekday = now.weekday()
hour = now.hour

print(f"[{now.strftime('%A %H:00')} - Sunny]")

# Location Selection
named_streets = streets[streets['name'] != "Calle sin nombre"].index

start = random.choice(named_streets)
print(f"· Location: {streets.at[start, 'name']} (id {start})")

goal = random.choice(named_streets)
print(f"· Destination: {streets.at[goal, 'name']} (id {goal})\n")

# Pathfinding
path = q_learning.train(start, goal, streets, pois_list, weekday=weekday, hour=hour)

# Report Results
total_length = streets.loc[path, "length"].sum()
path_names = streets.loc[path, "name"].tolist()
clean_path = []

for name in path_names:
    if name != "Calle sin nombre" and (not clean_path or clean_path[-1] != name):
        clean_path.append(name)

print(f"\n[Path found] {int(total_length)} meters\n" + " -> ".join(clean_path))

# just for debugging --------------------------------------------------------
streets_df = pd.read_json("map.json").set_index("id")
streets_list = streets_df.reset_index().to_dict(orient="records")

street_poi_map = q_learning.build_street_poi_map(streets_list, pois_list, radius_m=120, sigma_m=60)

print("\n[Debug: crowd per street]")
for sid in path:
    c = q_learning.street_crowd(sid, weekday, hour, street_poi_map, pois_list)  # 0~100
    print(streets_df.at[sid, "name"], "| len:", streets_df.at[sid, "length"], "| crowd:", round(c, 1))

print("\n[Debug: poi count per street on path]")
for sid in path:
    print(streets_df.at[sid, "name"], len(street_poi_map.get(sid, [])))
