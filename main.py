"""
Main execution script for Smart Crowd Router.
Manages data loading, user interaction, and pathfinding execution.
"""

import pandas as pd
import datetime
import random
import math
import q_learning
import string_matching

print(f"[AI Project - Smart Crowd Router] \nBy ______________________________\n")

# Load Data
streets = pd.read_json("map.json").set_index('id')
places = pd.read_csv("places_database.csv")

# Environment Setup
now = datetime.datetime.now()
print(f"[{now.strftime('%A %H:00')} - Sunny]")


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in meters."""
    R = 6371000  # Earth's radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def find_nearest_street_id(lat, lon, streets_df):
    """Find the nearest street ID to given coordinates."""
    min_distance = float('inf')
    nearest_id = None
    
    for idx, row in streets_df.iterrows():
        center = row['center']
        street_lat, street_lon = center[0], center[1]
        distance = haversine_distance(lat, lon, street_lat, street_lon)
        
        if distance < min_distance:
            min_distance = distance
            nearest_id = idx
    
    return nearest_id


def get_street_id_for_location(name, location_type, streets_df, places_df):
    """
    Get the street ID for a location (either a place or a street).
    
    For streets: find the ID directly by name
    For places: find coordinates and then find nearest street
    """
    if location_type == "street":
        # Find street ID by name
        named_streets = streets_df[streets_df['name'] != "Calle sin nombre"]
        matching = named_streets[named_streets['name'] == name]
        if len(matching) > 0:
            return matching.index[0]
        return None
    
    elif location_type == "place":
        # Find place coordinates
        matching_place = places_df[places_df['name'] == name]
        if len(matching_place) > 0:
            lat = matching_place.iloc[0]['latitude']
            lon = matching_place.iloc[0]['longitude']
            # Find nearest street to this place
            return find_nearest_street_id(lat, lon, streets_df)
        return None
    
    return None


# Prepare name lists
named_streets = streets[streets['name'] != "Calle sin nombre"]
unique_street_names = named_streets['name'].unique().tolist()
unique_place_names = places['name'].unique().tolist()

# Menu
print("\n[Menu]")
print("  1. Generate random places and calculate best path")
print("  2. Enter specific places manually")
choice = input("\nYour choice (1 or 2): ").strip()

if choice == "1":
    # Random selection (from streets only for simplicity)
    start_id = random.choice(named_streets.index)
    goal_id = random.choice(named_streets.index)
    start_name = streets.at[start_id, 'name']
    goal_name = streets.at[goal_id, 'name']
    print(f"\n· Location: {start_name} (id {start_id})")
    print(f"· Destination: {goal_name} (id {goal_id})\n")

elif choice == "2":
    # Manual input with fuzzy matching for both places and streets
    print("\n[Enter a place or street - the system will find the closest match]")
    print(f"  ({len(unique_place_names)} places and {len(unique_street_names)} streets available)\n")
    
    # Get starting point
    start_input = input("Enter starting place or street: ").strip()
    start_name, start_type = string_matching.interactive_location_selection(
        start_input, 
        unique_place_names,
        unique_street_names,
        "starting point"
    )
    
    # Get destination
    goal_input = input("\nEnter destination place or street: ").strip()
    goal_name, goal_type = string_matching.interactive_location_selection(
        goal_input, 
        unique_place_names,
        unique_street_names,
        "ending point"
    )
    
    # Get street IDs for pathfinding
    start_id = get_street_id_for_location(start_name, start_type, streets, places)
    goal_id = get_street_id_for_location(goal_name, goal_type, streets, places)
    
    if start_id is None or goal_id is None:
        print("\nError: Could not find location on the map.")
        exit()
    
    print(f"\n· Location: {start_name} ({start_type}, nearest street id {start_id})")
    print(f"· Destination: {goal_name} ({goal_type}, nearest street id {goal_id})\n")

else:
    print("\nInvalid choice. Please run again and enter 1 or 2.")
    exit()

# Pathfinding
path = q_learning.train(start_id, goal_id, streets)

# Report Results
total_length = streets.loc[path, "length"].sum()
path_names = streets.loc[path, "name"].tolist()
clean_path = []

for name in path_names:
    if name != "Calle sin nombre" and (not clean_path or clean_path[-1] != name):
        clean_path.append(name)

print(f"\n[Path found] {int(total_length)} meters\n" + " -> ".join(clean_path))
