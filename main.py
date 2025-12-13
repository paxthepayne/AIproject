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

# Environment Setup
now = datetime.datetime.now()
print(f"[{now.strftime('%A %H:00')} - Sunny]")

# Location Selection
named_streets = streets[streets['name'] != "Calle sin nombre"].index

start = random.choice(named_streets)
print(f"· Location: {streets.at[start, 'name']} (id {start})")

goal = random.choice(named_streets)
print(f"· Destination: {streets.at[goal, 'name']} (id {goal})\n")

# Pathfinding
path = q_learning.train(start, goal, streets)

# Report Results
total_length = streets.loc[path, "length"].sum()
path_names = streets.loc[path, "name"].tolist()
clean_path = []

for name in path_names:
    if name != "Calle sin nombre" and (not clean_path or clean_path[-1] != name):
        clean_path.append(name)

print(f"\n[Path found] {int(total_length)} meters\n" + " -> ".join(clean_path))