"""
Main execution script for Smart Crowd Router.
Manages data loading, user interaction, and pathfinding execution.
"""

# --- IMPORTS ---

# System
import gzip
import json

# Data Handling
import pandas as pd

# Time
import datetime

# Multiprocessing
from multiprocessing import Pool

# Custom Tools
import tools

# Function to run training in parallel
def run_train(args):
    return tools.train(*args)


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print(f"[AI Project - Smart Crowd Router] By Filippo Pacini, Jacopo Crispolti, Liseth Berdeja, Sieun You\n")

    # Load Map Data as DataFrame
    with gzip.open("map.json.gz", "rt", encoding="utf-8") as f: data = json.load(f)
    df = pd.DataFrame(data).set_index("id").rename(columns={"coords": "coordinates", "len": "length", "conns": "connections"})

    # Separate Q-learning map and popular times
    city_map = df[["type", "name", "coordinates", "length", "connections"]]
    populartimes = df[["popular_times"]]

    # Get current Time and Weather (Sunny/Cloudy/Rainy)
    print("Fetching weather info...")
    current_time = datetime.datetime.now()
    weather = tools.get_weather()
    print(f"> {current_time.strftime('%A %H:%M')}, {weather}\n")

    # Get current crowd levels from populartimes, based on current time and weather
    weekday, hour = current_time.weekday(), current_time.hour
    weather_modifier = 1.3 if weather == "Sunny" else 0.3 if weather == "Rainy" else 0.9
    current_crowds = weather_modifier * populartimes["popular_times"].apply(lambda x: x[weekday][hour] if x is not None else 0.0)

    # Locations Selection
    start_name, start_id = tools.find_place(city_map, start_id=None)
    goal_name, goal_id = tools.find_place(city_map, start_id=start_id)

    # Pathfinding (in parallel)
    with Pool(processes=2) as pool:
        results = pool.map(
            run_train,
            [
                (start_id, goal_id, city_map, current_crowds),
                (start_id, goal_id, city_map, current_crowds, True),  # shortest_path = True
            ],
        )
    (path, q_table), (shortest_path, _) = results

    # Report Results
    path_length = city_map.loc[path, "length"].sum()
    shortest_path_length = city_map.loc[shortest_path, "length"].sum()

    path_total_crowd = sum(
        tools.estimate_crowd(s, current_crowds)
        for s in path
    )
    shortest_path_total_crowd = sum(
        tools.estimate_crowd(s, current_crowds)
        for s in shortest_path
    )

    clean_path = tools.clean(city_map.loc[path, "name"].tolist())
    clean_shortest_path = tools.clean(city_map.loc[shortest_path, "name"].tolist())

    print(f"\n[Suggested path] {int(path_length)} meters | {path_total_crowd*1000/path_length:.2f} average crowd exposure")
    print(" -> ".join(clean_path), "\n")
    print(f"[Shortest path] {int(shortest_path_length)} meters | {shortest_path_total_crowd*1000/shortest_path_length:.2f} average crowd exposure")
    print(" -> ".join(clean_shortest_path), "\n")

    # Optional Navigation Mode
    choice = input("Enter navigation mode? (y/n): ").strip().lower()
    if choice == "y":
        tools.navigation_mode(start_id, goal_id, city_map, current_crowds, q_table)

