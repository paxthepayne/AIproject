# --- IMPORTS ---

import math
import random
import pandas as pd
from unidecode import unidecode
from rapidfuzz import process, fuzz
import re
import unicodedata
import requests
import heapq


# --- WEATHER ---

def get_weather():
    # Open-Meteo endpoint
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": 41.3851,
        "longitude": 2.1734,
        "current_weather": "true"
    }

    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()

        code = data["current_weather"]["weathercode"]

        if code == 0:
            return "Sunny"
        if code in [1, 2, 3]:
            return "Cloudy"
        if code in [51, 53, 55, 61, 63, 65, 80, 81, 82, 95]: # Rain/Thunder codes
            return "Rainy"
        
        return "Cloudy" # Default fallback
        
    except Exception as e:
        print(f"Error: {e}")
        return "Cloudy"


# --- DISTANCE ---

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


# --- PLACE NAME AUTOCOMPLETE ---

def find_place(streets_df, start_id=None):
    """
    Optimized place search using vectorization and smart fuzzy matching.
    """
    if 'name_clean' not in streets_df.columns:
        streets_df['name_clean'] = streets_df['name'].apply(lambda x: unidecode(str(x)).lower())

    search_df = streets_df

    if start_id is not None:
        try:
            start_coords = streets_df.at[start_id, 'coordinates']
            lat, lon = start_coords
            
            mask = (
                streets_df['coordinates'].str[0].between(lat - 0.025, lat + 0.025) &
                streets_df['coordinates'].str[1].between(lon - 0.035, lon + 0.035)
            )
            search_df = streets_df[mask].copy()

            search_df['dist_temp'] = search_df['coordinates'].apply(
                lambda x: distance(start_coords, x, type="euclidean")
            )
            search_df = search_df[search_df['dist_temp'] <= 2000]
            
        except KeyError:
            print("Start ID invalid, switching to global search.")
            search_df = streets_df 

    while True:
        prompt = f"Enter {'destination (<3km)' if start_id else 'location'}"
        user_input = input(f"{prompt}: ").strip()
        if not user_input:
            continue

        query_clean = unidecode(user_input).lower()

        mask = search_df['name_clean'].str.contains(query_clean, regex=False)
        candidates = search_df[mask].copy()
        
        candidates['score'] = 100 
        
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
                row['score'] = score - 10 
                candidates = pd.concat([candidates, row.to_frame().T])

        if candidates.empty:
            print("No matches found. Try again.")
            continue

        candidates['score'] += candidates['type'].apply(lambda x: 10 if x in [1, 'place'] else 0)
        
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


# --- PATH CLEANING ---

def normalize_street_name(name):
    if not isinstance(name, str):
        return name

    name = name.lower()
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r'\b(de|del|la|el|plaça|plaza)\b', '', name)
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


# --- A* ALGORITHM (Shortest Path) ---

def a_star(start, goal, streets):
    """
    Calculates the shortest path using A* algorithm.
    Heuristic: Euclidean distance to goal.
    Cost: Length of street segments.
    """
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    goal_coords = streets.at[goal, "coordinates"]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            # Return reversed path and empty dict (to match structure)
            return path[::-1]

        current_g = g_score[current]

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


# --- Q-LEARNING AGENT ---

def estimate_crowd(street_id, crowds):
    if pd.isna(crowds[street_id]):
        return 0.0
    return crowds[street_id]

def calculate_reward(state, next_state, goal, streets, crowds):
    """
    Reward function for Q-learning agent (Crowd Optimized).
    """
    if next_state == goal:
        reward = 10000
    else:
        reward = -1

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

    # Penalty based on crowd
    crowd = estimate_crowd(next_state, crowds)
    reward -= crowd

    return reward

def choose_action(state, epsilon, Q, goal, streets, crowds):
    next_states = streets.at[state, "connections"]

    if not next_states:
        return goal, -1000
    
    if state not in Q:
        Q[state] = {action: 0.0 for action in next_states}
    for action in next_states:
        if action not in Q[state]:
            Q[state][action] = 0.0

    # Epsilon-Greedy
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

        curr_alpha = max(alpha * (a_decay ** episode), 0.01)
        curr_epsilon = max(epsilon * (e_decay ** episode), 0.01)

        while state != goal and steps < 3000:

            next_state, reward = choose_action(
                state, curr_epsilon, Q,
                goal, streets, crowds
            )

            if next_state not in Q:
                Q[next_state] = {}

            max_next_q = max(Q[next_state].values()) if Q[next_state] else 0.0
            current_q = Q[state].get(next_state, 0.0)

            new_q = current_q + curr_alpha * (reward + gamma * max_next_q - current_q)
            Q[state][next_state] = new_q

            diff = abs(new_q - current_q)
            if diff > max_change:
                max_change = diff

            path.append(next_state)
            state = next_state
            steps += 1

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


# --- NAVIGATION ---

def navigation_mode(start, goal, streets, crowds, Q):
    print("\n[Smart Navigation Mode]\n")
    
    curr, goal_coords = start, streets.at[goal, "coordinates"]
    visited, path, last_rejects = {start}, [start], set()
    walked = since_dec = 0
    
    STOPWORDS = {"carrer", "de", "del", "d", "la", "el", "els", "les", "los", "las", "plaça", "placa", "plaza", "passeig", "avinguda", "av", "rambla", "calle", "gran", "via", "travessera", "sant", "santa", "passatge"}

    def get_tok(n):
        if "sin nombre" in str(n).lower(): return ({"unnamed"}, "Unnamed Street")
        clean = [t for t in re.sub(r'[^\w\s]', '', unidecode(str(n)).lower()).split() if t not in STOPWORDS]
        return (set(clean), " ".join(t.title() for t in clean) or n)

    cur_toks, cur_disp = get_tok(streets.at[start, "name"])

    while curr != goal:
        cur_node, cur_xy = streets.at[curr, "name"], streets.at[curr, "coordinates"]
        dist_g = distance(cur_xy, goal_coords, type="euclidean")
        
        if since_dec > 100: last_rejects.clear() 

        # 1. Gather & Deduplicate Options
        opts_map = {}
        for n in [x for x in streets.at[curr, "connections"] if x not in visited]:
            n_nm, n_xy = streets.at[n, "name"], streets.at[n, "coordinates"]
            d_goal = distance(n_xy, goal_coords, type="euclidean")
            toks, disp = get_tok(n_nm)
            
            diff = d_goal - dist_g
            prog = "closer" if diff < -5 else ("further" if diff > 5 else "steady")
            
            opt = {
                "id": n, "name": n_nm, "display": disp, "tokens": toks,
                "is_cont": bool(toks & cur_toks), "q": Q.get(curr, {}).get(n, -float("inf")),
                "len": streets.at[n, "length"], "d_goal": d_goal, "prog": prog, "diff": diff
            }
            key = frozenset(toks)
            if key not in opts_map or d_goal < opts_map[key]["d_goal"]: opts_map[key] = opt

        raw = list(opts_map.values())
        if not raw: print("Dead end." if not streets.at[curr, "connections"] else "Stuck. Backtracking..."); break

        # 2. Filter & Sort
        final = [o for o in raw if o["prog"] != "further" or o["is_cont"]] or raw
        final.sort(key=lambda x: (x["diff"], -x["q"]))
        best = final[0]

        # 3. Auto-walk Logic
        cont_opt = next((o for o in final if o["is_cont"]), None)
        if cont_opt and best["tokens"] != cont_opt["tokens"] and (best["tokens"] & last_rejects): best = cont_opt
        
        rej_others = all(not (o["tokens"] & last_rejects) for o in final if o is not best)
        
        auto = (best["id"] != goal) and (
            len(final) == 1 or 
            (best["is_cont"] and since_dec < 100) or 
            (rej_others and since_dec < 100)
        )

        if auto:
            if walked == 0: cur_disp, cur_toks = best["display"], best["tokens"]
            walked += best["len"]; since_dec += best["len"]
            visited.add(best["id"]); path.append(best["id"])
            curr = best["id"]
            continue

        # 4. Display & Input
        if walked: print(f"Go straight on {cur_disp} for {int(walked)}m")
        walked = since_dec = 0 
        
        if best["id"] == goal: print(f"Arrived at {best['name']}!"); return
        print(f"Intersection at {cur_node}")
        
        for i, o in enumerate(final, 1):
            c_val = estimate_crowd(o["id"], crowds)
            verb = "continue" if o["is_cont"] else "turn onto"
            # Updated Output Format
            print(f"  {i}. {verb} {o['display']} ({o['prog']}, {int(o['d_goal'])} m to goal, <10 crowd)")

        while True:
            ch = input("> ").strip().lower()
            if ch == 'q': return
            if ch.isdigit() and 0 <= (idx := int(ch)-1) < len(final):
                sel = final[idx]
                if not sel["is_cont"]: last_rejects = set()
                else: last_rejects = {t for o in final if o is not sel for t in o["tokens"]}
                
                cur_disp, cur_toks = sel["display"], sel["tokens"]
                visited.add(sel["id"]); path.append(sel["id"]); curr = sel["id"]; since_dec = 0
                break
            print("Invalid.")