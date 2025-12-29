"""
Tools Module

Contains utility functions for:
- Weather fetching (Open-Meteo)
- Geometry and distance calculations
- Text normalization and fuzzy search
- Pathfinding algorithms (A*, Q-Learning)
- Interactive Navigation Mode
"""

# --- IMPORTS ---

# System & Math
import math
import random
import heapq
import re
import unicodedata

# Data Handling
import pandas as pd
import requests

# Text Processing
from unidecode import unidecode
from rapidfuzz import process, fuzz


# --- WEATHER ---

def get_weather():
    """
    Fetches current weather condition for Barcelona (Lat 41.3851, Lon 2.1734)
    using the Open-Meteo API.
    Returns: 'Sunny', 'Cloudy', or 'Rainy'.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": 41.3851,
        "longitude": 2.1734,
        "current_weather": "true"
    }

    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()

        # WMO Weather interpretation codes
        code = data["current_weather"]["weathercode"]

        if code == 0:
            return "Sunny"
        if code in [1, 2, 3]:
            return "Cloudy"
        if code in [51, 53, 55, 61, 63, 65, 80, 81, 82, 95]: # Rain/Thunder/Drizzle
            return "Rainy"
        
        return "Cloudy" # Default fallback
        
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return "Cloudy"


# --- DISTANCE CALCULATIONS ---

def distance(start_coordinates, end_coordinates, type="manhattan"):
    """
    Calculates distance between two (lat, lon) tuples.
    Approximates meters using fixed latitude scaling for Barcelona.
    """
    lat1, lon1 = start_coordinates
    lat2, lon2 = end_coordinates

    # Convert to meters (Approximate conversion for BCN latitude)
    avg_lat_rad = math.radians((lat1 + lat2) / 2)
    lat_scale = 111132
    lon_scale = 111319 * math.cos(avg_lat_rad)

    dy_meters = abs(lat1 - lat2) * lat_scale
    dx_meters = abs(lon1 - lon2) * lon_scale

    if type == "manhattan":
        dist = dy_meters + dx_meters
    elif type == "euclidean":
        dist = math.sqrt(dy_meters ** 2 + dx_meters ** 2)
    else:
        dist = 0

    return dist


# --- SEARCH & AUTOCOMPLETE ---

def find_place(streets_df, start_id=None):
    """
    Interactive place search using vectorization and fuzzy matching.
    If 'start_id' is provided, restricts search to a 3km radius (local destination).
    """
    # Pre-process names if not already done
    if 'name_clean' not in streets_df.columns:
        streets_df['name_clean'] = streets_df['name'].apply(lambda x: unidecode(str(x)).lower())

    search_df = streets_df

    # 1. Radius Filter (if start_id provided)
    if start_id is not None:
        try:
            start_coords = streets_df.at[start_id, 'coordinates']
            lat, lon = start_coords
            
            # Rough bounding box filter (faster than distance calc)
            mask = (
                streets_df['coordinates'].str[0].between(lat - 0.025, lat + 0.025) &
                streets_df['coordinates'].str[1].between(lon - 0.035, lon + 0.035)
            )
            search_df = streets_df[mask].copy()

            # Precise distance filter
            search_df['dist_temp'] = search_df['coordinates'].apply(
                lambda x: distance(start_coords, x, type="euclidean")
            )
            search_df = search_df[search_df['dist_temp'] <= 2000]
            
        except KeyError:
            print("Start ID invalid, switching to global search.")
            search_df = streets_df 

    # 2. Interactive Loop
    while True:
        prompt = f"Enter {'destination (<3km)' if start_id else 'location'}"
        user_input = input(f"{prompt}: ").strip()
        if not user_input:
            continue

        query_clean = unidecode(user_input).lower()

        # Exact substring match
        mask = search_df['name_clean'].str.contains(query_clean, regex=False)
        candidates = search_df[mask].copy()
        candidates['score'] = 100 
        
        # Fuzzy Fallback if few results
        if len(candidates) < 5:
            choices = search_df[~mask]['name_clean'].to_dict()
            
            fuzzy_matches = process.extract(
                query_clean, 
                choices, 
                scorer=fuzz.partial_ratio, 
                limit=5,
                score_cutoff=75
            )
            
            for match_str, score, idx in fuzzy_matches:
                row = search_df.loc[idx].copy()
                row['score'] = score - 10 # Penalize fuzzy matches slightly
                candidates = pd.concat([candidates, row.to_frame().T])

        if candidates.empty:
            print("No matches found. Try again.")
            continue

        # Boost score for 'Places' (POIs) over 'Streets'
        candidates['score'] += candidates['type'].apply(lambda x: 10 if x in [1, 'place'] else 0)
        
        # Sort by Score (desc) then Distance (asc)
        sort_cols = ['score']
        ascending = [False]
        if 'dist_temp' in candidates.columns:
            sort_cols.append('dist_temp')
            ascending.append(True)
            
        candidates = candidates.sort_values(by=sort_cols, ascending=ascending).head(1)
        
        best_match = candidates.iloc[0]
        ntype = "Place" if best_match['type'] == 1 else "Street"
        print(f"> Best match: {best_match['name']} [{ntype}]\n")
        
        return best_match['name'], best_match.name 


# --- STRING CLEANING ---

def normalize_street_name(name):
    """Normalizes street names by removing prefixes and accents."""
    if not isinstance(name, str):
        return name

    name = name.lower()
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    
    # Remove common prefixes
    name = re.sub(r'\b(de|del|la|el|plaça|plaza)\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    return name

def clean(path_names):
    """
    Cleans a list of street names to remove duplicates and normalize text.
    Used for displaying the final path to the user.
    """
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


# --- A* ALGORITHM (Shortest Path) ---

def a_star(start, goal, streets):
    """
    Calculates the shortest path using A* algorithm.
    - Heuristic: Euclidean distance to goal.
    - Cost: Physical length of street segments.
    """
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    
    goal_coords = streets.at[goal, "coordinates"]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1] # Return reversed path

        current_g = g_score[current]

        # Explore neighbors
        for neighbor in streets.at[current, "connections"]:
            weight = streets.at[neighbor, "length"]
            tentative_g = current_g + weight

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = distance(streets.at[neighbor, "coordinates"], goal_coords, type="euclidean")
                f = tentative_g + h
                heapq.heappush(open_set, (f, neighbor))

    return []


# --- Q-LEARNING AGENT (Crowd Optimized) ---

def estimate_crowd(street_id, crowds):
    """Helper to safely get crowd density for a street."""
    if pd.isna(crowds[street_id]):
        return 0.0
    return crowds[street_id]

def calculate_reward(state, next_state, goal, streets, crowds):
    """
    Reward function logic:
    - High positive reward for reaching the goal.
    - Small negative reward for every step (to encourage speed).
    - Positive reward for moving geographically closer to goal.
    - Negative penalty proportional to crowd density.
    """
    if next_state == goal:
        reward = 10000
    else:
        reward = -1

    dist_current = distance(streets.at[state, "coordinates"], streets.at[goal, "coordinates"])
    dist_next = distance(streets.at[next_state, "coordinates"], streets.at[goal, "coordinates"])

    # Directional Bonus
    if dist_next < dist_current:
        reward += 1

    # Crowd Penalty
    crowd = estimate_crowd(next_state, crowds)
    reward -= crowd

    return reward

def choose_action(state, epsilon, Q, goal, streets, crowds):
    """Selects next state using Epsilon-Greedy strategy."""
    next_states = streets.at[state, "connections"]

    if not next_states:
        return goal, -1000 # Dead end penalty
    
    # Initialize Q-values for new states
    if state not in Q:
        Q[state] = {action: 0.0 for action in next_states}
    for action in next_states:
        if action not in Q[state]:
            Q[state][action] = 0.0

    # Exploration vs Exploitation
    if random.random() < epsilon:
        next_state = random.choice(next_states)
    else:
        max_q = max(Q[state][a] for a in next_states)
        best = [a for a in next_states if Q[state][a] == max_q]
        next_state = random.choice(best)

    reward = calculate_reward(state, next_state, goal, streets, crowds)

    return next_state, reward

def train(start, goal, streets, crowds,
          parameters=[0.5, 0.9992, 1.0, 1.0, 0.999], 
          episodes=5000, min_delta=0.01, patience=5):
    """
    Trains the Q-Learning agent to find a crowd-optimized path.
    Stops early if Q-values converge.
    """
    Q = {}
    alpha, a_decay, gamma, epsilon, e_decay = parameters

    start_name = streets.at[start, "name"]
    goal_name = streets.at[goal, "name"]
    
    print(f"[Q-Learning] from '{start_name}' to '{goal_name}'")

    stable_episodes = 0 

    for episode in range(episodes):
        path = [start]
        state = start
        steps = 0
        max_change = 0 

        # Decay learning rate and exploration rate
        curr_alpha = max(alpha * (a_decay ** episode), 0.01)
        curr_epsilon = max(epsilon * (e_decay ** episode), 0.01)

        while state != goal and steps < 3000:

            next_state, reward = choose_action(state, curr_epsilon, Q, goal, streets, crowds)

            if next_state not in Q:
                Q[next_state] = {}

            # Bellman Equation Update
            max_next_q = max(Q[next_state].values()) if Q[next_state] else 0.0
            current_q = Q[state].get(next_state, 0.0)
            
            new_q = current_q + curr_alpha * (reward + gamma * max_next_q - current_q)
            Q[state][next_state] = new_q

            # Track convergence
            diff = abs(new_q - current_q)
            if diff > max_change:
                max_change = diff

            path.append(next_state)
            state = next_state
            steps += 1

        # Check convergence criteria
        if max_change < min_delta:
            stable_episodes += 1
        else:
            stable_episodes = 0

        if stable_episodes >= patience:
            print(f"-> Values converged at episode {episode} (max Δ = {min_delta}, patience = {patience})")
            break

        if episode in [10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000]:
            print(f"· Episode {episode}: {steps} steps, Δ = {max_change:.4f}")

    return path, Q


# --- SMART NAVIGATION MODE ---

def navigation_mode(start, goal, streets, crowds, Q):
    """
    Interactive turn-by-turn navigation simulation.
    Aggregates small segments into instructions like "Continue for X meters".
    """
    print("\n[Smart Navigation Mode]\n")
    
    curr = start
    goal_coords = streets.at[goal, "coordinates"]
    
    visited = {start}
    path = [start]
    last_rejects = set()
    
    walked_dist = 0
    since_decision_dist = 0
    
    STOPWORDS = {
        "carrer", "de", "del", "d", "la", "el", "els", "les", "los", "las", 
        "plaça", "placa", "plaza", "passeig", "avinguda", "av", "rambla", 
        "calle", "gran", "via", "travessera", "sant", "santa", "passatge"
    }

    def get_tok(n):
        """Extracts significant tokens from a street name."""
        name_str = str(n).lower()
        if "sin nombre" in name_str: 
            return ({"unnamed"}, "Unnamed Street")
        
        # Remove punctuation and split
        clean_text = re.sub(r'[^\w\s]', '', unidecode(name_str))
        tokens = [t for t in clean_text.split() if t not in STOPWORDS]
        display_name = " ".join(t.title() for t in tokens) or n
        
        return (set(tokens), display_name)

    cur_toks, cur_disp = get_tok(streets.at[start, "name"])

    while curr != goal:
        cur_node_name = streets.at[curr, "name"]
        cur_xy = streets.at[curr, "coordinates"]
        dist_to_goal = distance(cur_xy, goal_coords, type="euclidean")
        
        # Clear rejection memory if we've walked far enough
        if since_decision_dist > 100: 
            last_rejects.clear() 

        # 1. Gather & Deduplicate Options (Connections)
        opts_map = {}
        connections = [x for x in streets.at[curr, "connections"] if x not in visited]
        
        for n in connections:
            n_name = streets.at[n, "name"]
            n_xy = streets.at[n, "coordinates"]
            d_goal = distance(n_xy, goal_coords, type="euclidean")
            toks, disp = get_tok(n_name)
            
            # Determine progress relative to goal
            diff = d_goal - dist_to_goal
            if diff < -5: prog = "closer"
            elif diff > 5: prog = "further"
            else: prog = "steady"
            
            # Check if this is a continuation of current street
            is_cont = bool(toks & cur_toks)
            
            opt = {
                "id": n, 
                "name": n_name, 
                "display": disp, 
                "tokens": toks,
                "is_cont": is_cont, 
                "q": Q.get(curr, {}).get(n, -float("inf")),
                "len": streets.at[n, "length"], 
                "d_goal": d_goal, 
                "prog": prog, 
                "diff": diff
            }
            
            # Store unique options by name tokens (keep the one closer to goal)
            key = frozenset(toks)
            if key not in opts_map or d_goal < opts_map[key]["d_goal"]: 
                opts_map[key] = opt

        raw_options = list(opts_map.values())
        
        if not raw_options:
            status = "Dead end." if not streets.at[curr, "connections"] else "Stuck. Backtracking..."
            print(status)
            break

        # 2. Filter & Sort Options
        # Prefer paths that don't move away, unless it's the only way or a continuation
        final_options = [o for o in raw_options if o["prog"] != "further" or o["is_cont"]]
        if not final_options:
            final_options = raw_options
            
        # Sort by: Getting closer to goal (diff asc), then Q-value (desc)
        final_options.sort(key=lambda x: (x["diff"], -x["q"]))
        best = final_options[0]

        # 3. Auto-walk Logic (Skip interaction if obvious)
        
        # If best option looks like a continuation but was previously rejected, switch logic
        cont_opt = next((o for o in final_options if o["is_cont"]), None)
        if cont_opt and best["tokens"] != cont_opt["tokens"] and (best["tokens"] & last_rejects): 
            best = cont_opt
        
        # Check if all other options were explicitly rejected recently
        others_rejected = all(not (o["tokens"] & last_rejects) for o in final_options if o is not best)
        
        should_auto_walk = (best["id"] != goal) and (
            len(final_options) == 1 or 
            (best["is_cont"] and since_decision_dist < 100) or 
            (others_rejected and since_decision_dist < 100)
        )

        if should_auto_walk:
            # Update display name if we are just starting a segment
            if walked_dist == 0: 
                cur_disp, cur_toks = best["display"], best["tokens"]
                
            walked_dist += best["len"]
            since_decision_dist += best["len"]
            
            visited.add(best["id"])
            path.append(best["id"])
            curr = best["id"]
            continue

        # 4. Display & Input (Intersection reached)
        if walked_dist: 
            print(f"Go straight on {cur_disp} for {int(walked_dist)}m")
        
        walked_dist = 0
        since_decision_dist = 0 
        
        if best["id"] == goal: 
            print(f"Arrived at {best['name']}!")
            return

        print(f"Intersection at {cur_node_name}")
        
        for i, o in enumerate(final_options, 1):
            verb = "continue" if o["is_cont"] else "turn onto"
            print(f"  {i}. {verb} {o['display']} ({o['prog']}, {int(o['d_goal'])} m to goal, <10 crowd)")

        while True:
            ch = input("> ").strip().lower()
            if ch == 'q': 
                return
            
            if ch.isdigit() and 0 <= (idx := int(ch)-1) < len(final_options):
                sel = final_options[idx]
                
                # Update rejection memory
                if not sel["is_cont"]: 
                    last_rejects = set()
                else: 
                    # If continuing, remember we rejected the turns
                    last_rejects = {t for o in final_options if o is not sel for t in o["tokens"]}
                
                cur_disp, cur_toks = sel["display"], sel["tokens"]
                visited.add(sel["id"])
                path.append(sel["id"])
                curr = sel["id"]
                since_decision_dist = 0
                break
            
            print("Invalid selection.")