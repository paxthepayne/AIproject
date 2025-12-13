"""
Class that manages everything
"""
import q_learning
import pandas as pd
import json

# -----------------------------------------------------------------------------
# Load data for Q-learning
# -----------------------------------------------------------------------------

# Maps for agent movement
streets = pd.read_json("map_streets.json").set_index('id') # streets table ["id", "name", "connections", "coordinates"]
with open("map_nodes.json", "r") as f: node_to_streets = json.load(f) # dictionary for fast 'neighbour streets' lookup

# Crowd Maps for reward calculation
# calculate crowd level of every street

# -----------------------------------------------------------------------------
# Interact with the user
# -----------------------------------------------------------------------------
print(f"--- Title of the program ---\n")

# User inputs
# Get coordinates of user and find closest street
start = 0
print(f"Start location: {streets.at[start, "name"]}")
# Get coordinates of destination and find closest street
goal = 500
print(f"Destination: {streets.at[goal, "name"]}\n")

# Pathfinding
path = q_learning.train(start, goal, streets, node_to_streets)

# Report results (path found, walk time, expected crowd exposure, maybe compare with shortest path?)
names = []
for id in path:
    name = streets.at[id, "name"]
    if (not names or names[-1] != name) and name != "Calle sin nombre": names.append(name)
print(f"\nPath found:", " -> ".join(names))