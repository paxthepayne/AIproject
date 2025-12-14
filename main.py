"""
Main execution script for Smart Crowd Router.
Manages data loading, user interaction, and pathfinding execution.
"""

import pandas as pd
import datetime
import pickle
import tools

print(f"[AI Project - Smart Crowd Router] By ______________________________\n")

# Load Map Data as DataFrame
print("Loading data...\n")
with open("map.pkl", "rb") as f: map_data = pickle.load(f)
city_map = pd.DataFrame(map_data).set_index('id').rename(columns={'coords': 'coordinates', 'len': 'length', 'conns': 'connections', 'pop_open': 'populartimes_open', 'pop_closed': 'populartimes_closed'})

# Time and Weather
now = datetime.datetime.now()
weather = "Sunny"
print(f"[{now.strftime('%A %H:00')} - {weather}]")

# Location Selection
start_name, start = tools.find_place(city_map, start_id=None, point_type="start")
goal_name, goal = tools.find_place(city_map, start_id=start, point_type="goal")

# Pathfinding
path = tools.train(start, goal, city_map)

# Report Results
total_length = city_map.loc[path, "length"].sum()
path_names = city_map.loc[path, "name"].tolist()

# Clean up path (remove duplicates and unnamed streets for display)
clean_path = []
for name in path_names:
    if name != "Calle sin nombre" and (not clean_path or clean_path[-1] != name):
        clean_path.append(name)

print(f"\n[Path found] ~{int(total_length)} meters")
print(" -> ".join(clean_path))