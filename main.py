"""
Main execution script for Smart Crowd Router.
Manages data loading, user interaction, and pathfinding execution.
"""

import pandas as pd
import datetime
import pickle
import tools
from multiprocessing import Pool


def run_train(args):
    return tools.train(*args)


if __name__ == "__main__":
    print(f"[AI Project - Smart Crowd Router] By Filippo Pacini, Jacopo Crispolti, Liseth Berdeja, Sieun You\n")

    # Load Map Data as DataFrame
    with open("map.pkl", "rb") as f:
        map_data = pickle.load(f)

    city_map = (
        pd.DataFrame(map_data)
        .set_index("id")
        .rename(
            columns={
                "coords": "coordinates",
                "len": "length",
                "conns": "connections",
                "pop_open": "populartimes_open",
                "pop_closed": "populartimes_closed",
            }
        )
    )

    # Current Time
    current_time = datetime.datetime.now()

    # Weather and Events
    weather = tools.get_barcelona_weather("")
    print(f"[{current_time.strftime('%A %H:%M')} - {weather}]\n")

    # Convert events ONCE to a list of coordinates (critical for multiprocessing)
    events_df = tools.fetch_events(current_time)
    events = events_df["coordinates"].tolist()

    # Location Selection
    start_name, start = tools.find_place(city_map, start_id=None, point_type="start")
    goal_name, goal = tools.find_place(city_map, start_id=start, point_type="goal")

    # Pathfinding (in parallel)
    with Pool(processes=2) as pool:
        results = pool.map(
            run_train,
            [
                (start, goal, city_map, current_time, weather, events),
                (start, goal, city_map, current_time, weather, events, True),  # shortest_path = True
            ],
        )

    path, shortest_path = results

    # Report Results
    path_length = city_map.loc[path, "length"].sum()
    shortest_path_length = city_map.loc[shortest_path, "length"].sum()

    path_names = city_map.loc[path, "name"].tolist()
    shortest_path_names = city_map.loc[shortest_path, "name"].tolist()

    # Crowd exposure (now includes events implicitly)
    path_total_crowd = sum(
        tools.estimate_crowd(city_map, s, current_time, weather, events)
        for s in path
    )

    shortest_path_total_crowd = sum(
        tools.estimate_crowd(city_map, s, current_time, weather, events)
        for s in shortest_path
    )

    # Clean up path (remove duplicates and unnamed streets for display)
    clean_path = tools.clean(path_names)
    clean_shortest_path = tools.clean(shortest_path_names)

    print(f"\n[Path found] {int(path_length)} meters | {path_total_crowd} total crowd")
    print(" -> ".join(clean_path))
    print(
        f"[Shortest path] {int(shortest_path_length)} meters | {shortest_path_total_crowd} total crowd\n"
        + " -> ".join(clean_shortest_path)
        + "\n"
    )
