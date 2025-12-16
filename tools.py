import math
import random
import pandas as pd
from difflib import SequenceMatcher
import re
import datetime
import unicodedata
import requests
import matplotlib.pyplot as plt



# ==========================================
# WEATHER & EVENTS
# ==========================================

def get_barcelona_weather(api_key):
    lat, lon = 41.3851, 2.1734

    url = "https://weather.googleapis.com/v1/currentConditions:lookup"
    params = {
        "key": api_key,
        "location.latitude": lat,
        "location.longitude": lon
    }

    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()

        wtype = data["weatherCondition"]["type"]

        if wtype == "CLEAR":
            return "Sunny"
        if wtype in ["CLOUDY", "MOSTLY_CLOUDY", "PARTLY_CLOUDY"]:
            return "Cloudy"
        if wtype in ["RAIN", "LIGHT_RAIN", "HEAVY_RAIN", "SHOWERS", "THUNDERSTORMS"]:
            return "Rainy"

        return "Cloudy"
    except Exception:
        return "Cloudy"


BCN_AGENDA_URL = (
    "https://opendata-ajuntament.barcelona.cat/data/api/action/"
    "datastore_search?resource_id=877ccf66-9106-4ae2-be51-95a9f6469e4c&limit=15000"
)


def fetch_events(day):
    """
    Fetch events active on a given calendar day.

    Parameters
    ----------
    day : datetime.date | datetime.datetime | str (YYYY-MM-DD)

    Returns
    -------
    pd.DataFrame with at least:
        - coordinates : (lat, lon)
    """

    if isinstance(day, str):
        day = datetime.date.fromisoformat(day)
    elif isinstance(day, datetime.datetime):
        day = day.date()

    resp = requests.get(BCN_AGENDA_URL)
    records = resp.json()["result"]["records"]
    df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame(columns=["coordinates"])

    lat_col = 'geo_epgs_4326_lat' if 'geo_epgs_4326_lat' in df.columns else 'lat'
    lon_col = 'geo_epgs_4326_lon' if 'geo_epgs_4326_lon' in df.columns else 'lon'

    df['lat'] = pd.to_numeric(df[lat_col], errors='coerce')
    df['lon'] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])

    df['start_dt'] = pd.to_datetime(df['start_date'], errors='coerce').dt.date

    if 'end_date' in df.columns:
        df['end_dt'] = pd.to_datetime(df['end_date'], errors='coerce').dt.date
        df['end_dt'] = df['end_dt'].fillna(df['start_dt'])
    else:
        df['end_dt'] = df['start_dt']

    df = df[
        (df['start_dt'] <= day) &
        (df['end_dt'] >= day)
    ]

    df = df.copy()
    df['coordinates'] = list(zip(df['lat'], df['lon']))

    return df[['coordinates']]


# ==========================================
# 1. GEOMETRY & SEARCH TOOLS
# ==========================================

def distance(start_coordinates, end_coordinates, type="manhattan"):
    lat1, lon1 = start_coordinates
    lat2, lon2 = end_coordinates

    avg_lat_rad = math.radians((lat1 + lat2) / 2)
    lat_scale = 111132
    lon_scale = 111319 * math.cos(avg_lat_rad)

    dy_meters = abs(lat1 - lat2) * lat_scale
    dx_meters = abs(lon1 - lon2) * lon_scale

    if type == "manhattan":
        dist = dy_meters + dx_meters
    elif type == "euclidean":
        dist = math.sqrt(dy_meters ** 2 + dx_meters ** 2)

    return dist


def find_place(streets_df, start_id=None, point_type="start"):
    """
    Optimized find_place with fast global search.
    """
    # 1. Prepare Search Space
    search_df = streets_df

    # If we have a start location, filter aggressively by distance first
    if start_id is not None:
        try:
            start_coords = streets_df.at[start_id, 'coordinates']
            # Fast Pre-filter using Lat/Lon box (avoid calculating dist for everything)
            lat, lon = start_coords
            # ~2.5km box approximation
            search_df = streets_df[
                (streets_df['coordinates'].str[0].between(lat - 0.025, lat + 0.025)) &
                (streets_df['coordinates'].str[1].between(lon - 0.035, lon + 0.035))
            ].copy()

            # Precise Distance Calculation on the smaller subset
            search_df['dist_temp'] = search_df['coordinates'].apply(
                lambda x: distance(start_coords, x, type="euclidean")
            )
            search_df = search_df[search_df['dist_temp'] <= 2000]
        except Exception:
            pass  # Fallback to global search if start_id is invalid

    while True:
        prompt = f"Enter {'destination (<2km distance)' if start_id else 'your location'}"
        query = input(f"{prompt}: ").strip()
        if not query:
            continue

        candidates = []

        # 2. FAST SEARCH: Vectorized substring match
        matches = search_df[search_df['name'].str.contains(query, case=False, na=False, regex=False)]

        # If we found matches, score them
        if not matches.empty:
            for nid, row in matches.iterrows():
                name = row['name']
                # Base score for substring match
                score = 1.0 if query.lower() == name.lower() else 0.8

                # Bonus for Places
                if row['type'] in [1, 'place']:
                    score += 0.2

                # Distance (if available)
                dist = row['dist_temp'] if 'dist_temp' in row else 0
                candidates.append({'name': name, 'id': nid, 'score': score, 'dist': dist})

        # 3. SLOW SEARCH (Fallback): Fuzzy matching
        elif len(candidates) == 0:
            print("Searching places...")

            # Optimization: If searching globally, check PLACES first
            if start_id is None:
                fuzzy_pool = search_df[search_df['type'].isin([1, 'place'])]
            else:
                fuzzy_pool = search_df

            for nid, row in fuzzy_pool.iterrows():
                name = str(row['name'])
                if name == "nan" or name == "Calle Sin Nombre":
                    continue

                ratio = SequenceMatcher(None, query.lower(), name.lower()).ratio()
                if ratio > 0.65:
                    score = ratio + (0.2 if row['type'] in [1, 'place'] else 0)
                    dist = row['dist_temp'] if 'dist_temp' in row else 0
                    candidates.append({'name': name, 'id': nid, 'score': score, 'dist': dist})

        # 4. Deduplication logic
        unique_candidates = {}
        for c in candidates:
            name = c['name']
            if name not in unique_candidates:
                unique_candidates[name] = c
            else:
                existing = unique_candidates[name]
                # Prioritize: Closer Distance > Higher Score
                is_closer = (start_id is not None and c['dist'] < existing['dist'])
                is_better_score = (start_id is None and c['score'] > existing['score'])

                if is_closer or is_better_score:
                    unique_candidates[name] = c

        final_list = list(unique_candidates.values())
        final_list.sort(key=lambda x: x['score'], reverse=True)
        top_matches = final_list[:5]

        if not top_matches:
            print(f"No matches, try again.")
            continue

        ntype = "Place" if streets_df.at[top_matches[0]['id'], 'type'] == 1 else "Street"
        print(f"> Best match: {top_matches[0]['name']} [{ntype}]\n")

        return top_matches[0]['name'], top_matches[0]['id']


def normalize_street_name(name):
    if not isinstance(name, str):
        return name

    # Lowercase
    name = name.lower()

    # Remove accents
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))

    # Remove common fillers
    name = re.sub(r'\b(de|del|la|el|plaça|plaza)\b', '', name)

    # Collapse spaces
    name = re.sub(r'\s+', ' ', name).strip()

    return name


def clean(path_names):
    clean_path = []
    seen = set()
    for name in path_names:
        if name == "Calle Sin Nombre":
            continue
        norm = normalize_street_name(name)
        if norm not in seen:
            clean_path.append(name)
            seen.add(norm)
    return clean_path

def plot_paths(city_map, path, shortest_path):
    # Extract coordinates
    path_coords = city_map.loc[path, "coordinates"].tolist()
    sp_coords = city_map.loc[shortest_path, "coordinates"].tolist()

    # Split lat / lon
    path_lats, path_lons = zip(*path_coords)
    sp_lats, sp_lons = zip(*sp_coords)

    plt.figure(figsize=(10, 10))

    # Crowd-aware path
    plt.plot(
        path_lons,
        path_lats,
        marker="o",
        linewidth=2,
        label="Crowd-aware path"
    )

    # Shortest path
    plt.plot(
        sp_lons,
        sp_lats,
        marker="o",
        linewidth=2,
        linestyle="--",
        label="Shortest path"
    )

    # Start & goal
    plt.scatter(
        [path_lons[0], path_lons[-1]],
        [path_lats[0], path_lats[-1]],
        s=100,
        zorder=5,
        label="Start / Goal"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Smart Crowd Router – Path Comparison")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


# ==========================================
# 2. CROWD & Q-LEARNING AGENT
# ==========================================

def estimate_crowd(streets, state, time, weather, events):
    """
    Estimate crowd level for a given street node.
    Includes:
    - Google popular times
    - Weather impact
    - Nearby events
    """
    populartimes_open = streets.at[state, "populartimes_open"]
    populartimes_closed = streets.at[state, "populartimes_closed"]

    crowd = 0.0

    # Weather multiplier
    weather_multiplier = 1.0
    if weather == "Sunny":
        weather_multiplier = 1.3
    elif weather == "Cloudy":
        weather_multiplier = 0.9
    elif weather == "Rainy":
        weather_multiplier = 0.2

    if populartimes_open is not None:
        crowd += weather_multiplier * populartimes_open[time.weekday(), time.hour]
    if populartimes_closed is not None:
        closed_weather_multiplier = 1 + 0.5 * (weather_multiplier - 1)
        crowd += closed_weather_multiplier * populartimes_closed[time.weekday(), time.hour]

    # Event influence (bounded)
    #for ev_coords in events:
    #    d = max(distance(streets.at[state, "coordinates"], ev_coords), 1)
    #    if d < 500:
    #        crowd *= 0.05/d

    return crowd


def calculate_reward(state, next_state, goal, streets, time, weather, events, shortest_path):
    """
    Reward function for Q-learning agent.
    """
    # Basic reward structure
    if next_state == goal:
        reward = 10000
    else:
        reward = -1

    # Distance calculations
    dist_current = distance(
        streets.at[state, "coordinates"],
        streets.at[goal, "coordinates"]
    )
    dist_next = distance(
        streets.at[next_state, "coordinates"],
        streets.at[goal, "coordinates"]
    )

    # Bonus for moving in the right direction
    if dist_next < dist_current:
        reward += 1

    if shortest_path:
        edge_len = streets.at[next_state, "length"]
        reward -= edge_len / 50
    else:
        crowd = estimate_crowd(streets, next_state, time, weather, events)
        reward -= crowd

    return reward


def choose_action(state, epsilon, Q, goal, streets, time, weather, events, shortest_path):
    # Get connections
    next_states = streets.at[state, "connections"]

    # Ensure state and actions exists in Q-table
    if state not in Q:
        Q[state] = {action: 0.0 for action in next_states}
    for action in next_states:
        if action not in Q[state]:
            Q[state][action] = 0.0

    # Epsilon-Greedy Logic
    if random.random() < epsilon:
        next_state = random.choice(next_states)
    else:
        max_q = max(Q[state][a] for a in next_states)
        best = [a for a in next_states if Q[state][a] == max_q]
        next_state = random.choice(best)

    reward = calculate_reward(
        state, next_state, goal,
        streets, time, weather, events, shortest_path
    )

    return next_state, reward


def train(
    start,
    goal,
    streets,
    time,
    weather,
    events,
    shortest_path=False,
    parameters=[0.5, 0.999, 1.0, 1.0, 0.997],
    episodes=5000,
    min_delta=0.01,
    patience=5
):
    Q = {}
    alpha, a_decay, gamma, epsilon, e_decay = parameters

    start_name = streets.at[start, "name"]
    goal_name = streets.at[goal, "name"]
    if not shortest_path:
        print(f"[Q-Learning] from '{start_name}' to '{goal_name}'")

    stable_episodes = 0  # Counter for convergence check

    for episode in range(episodes):
        path = [start]
        state = start
        steps = 0
        max_change = 0  # Track max Q-value change this episode

        # Calculate current decays
        curr_alpha = max(alpha * (a_decay ** episode), 0.01)
        curr_epsilon = max(epsilon * (e_decay ** episode), 0.01)

        while state != goal and steps < 5000:

            next_state, reward = choose_action(
                state, curr_epsilon, Q,
                goal, streets, time, weather, events, shortest_path
            )

            # Ensure next state exists in Q
            if next_state not in Q:
                Q[next_state] = {}

            # Q-Learning Formula
            max_next_q = max(Q[next_state].values()) if Q[next_state] else 0.0
            current_q = Q[state].get(next_state, 0.0)

            new_q = current_q + curr_alpha * (reward + gamma * max_next_q - current_q)
            Q[state][next_state] = new_q

            # Track change for convergence check
            diff = abs(new_q - current_q)
            if diff > max_change:
                max_change = diff

            # Move agent
            path.append(next_state)
            state = next_state
            steps += 1

        # Check for convergence
        if max_change < min_delta:
            stable_episodes += 1
        else:
            stable_episodes = 0

        # Stop early if converged
        if stable_episodes >= patience:
            if not shortest_path:
                print(f"-> Values converged at episode {episode} (max Δ = {min_delta}, patience = {patience})")
            break

        # Log progress periodically
        if episode % 200 == 100 and episode != 0:
            if not shortest_path:
                print(f"· Episode {episode}: {steps} steps, Δ = {max_change:.4f}")

    return path
