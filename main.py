"""
Main execution script for Smart Crowd Router.
Manages data loading, user interaction, and pathfinding execution.
"""

import pandas as pd
import datetime
import pickle
import tools
from multiprocessing import Pool, cpu_count
    
def run_train(args):
    return tools.train(*args)

if __name__ == "__main__":
    print(f"[AI Project - Smart Crowd Router] By Filippo Pacini, Jacopo Crispolti, Liseth Berdeja, Sieun You\n")

    # Load Map Data as DataFrame
    with open("map.pkl", "rb") as f: map_data = pickle.load(f)
    city_map = pd.DataFrame(map_data).set_index('id').rename(columns={'coords': 'coordinates', 'len': 'length', 'conns': 'connections', 'pop_open': 'populartimes_open', 'pop_closed': 'populartimes_closed'})

    # Time and Weather
    current_time = datetime.datetime.now()
    weather = "Sunny"
    print(f"[{current_time.strftime('%A %I:%M')} - {weather}]\n")

    # Location Selection
    start_name, start = tools.find_place(city_map, start_id=None, point_type="start")
    goal_name, goal = tools.find_place(city_map, start_id=start, point_type="goal")

    # Pathfinding (in parallel)
    with Pool(processes=2) as pool:
        results = pool.map(
            run_train,
            [
                (start, goal, city_map, current_time),
                (start, goal, city_map, current_time, True) # shortest_path = True
            ]
        )
    path, shortest_path = results

    # Report Results
    path_length = city_map.loc[path, "length"].sum()
    delta_length = path_length - city_map.loc[shortest_path, "length"].sum()
    path_names = city_map.loc[path, "name"].tolist()
    shortest_path_names = city_map.loc[shortest_path, "name"].tolist()

    # Clean up path (remove duplicates and unnamed streets for display)
    clean_path = tools.clean(path_names)
    clean_shortest_path = tools.clean(shortest_path_names)


    print(f"\n[Path found] {int(path_length)} meters {("("+str(int(delta_length))+" more than shortest)") if int(delta_length)>0 else ""}")
    print(" -> ".join(clean_path))
    print("[Shortest path]\n" + " -> ".join(clean_shortest_path) + "\n")