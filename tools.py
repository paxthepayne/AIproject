# --- IMPORTS ---

import math
import random
import pandas as pd
import pandas as pd
from unidecode import unidecode
from rapidfuzz import process, fuzz
import re
import unicodedata
import requests
from collections import defaultdict


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
    # --- 1. PRE-COMPUTATION (Do this once) ---
    # Create a normalized column for fast vector search (lowercase + no accents)
    # We use unidecode to turn "Família" -> "familia"
    if 'name_clean' not in streets_df.columns:
        streets_df['name_clean'] = streets_df['name'].apply(lambda x: unidecode(str(x)).lower())

    search_df = streets_df

    # --- 2. GEOGRAPHIC FILTER (If start location exists) ---
    if start_id is not None:
        try:
            start_coords = streets_df.at[start_id, 'coordinates']
            lat, lon = start_coords
            
            # Fast Box Filter (Vectorized)
            # 0.025 deg lat ~= 2.7km. This is much faster than calculating distance for all.
            mask = (
                streets_df['coordinates'].str[0].between(lat - 0.025, lat + 0.025) &
                streets_df['coordinates'].str[1].between(lon - 0.035, lon + 0.035)
            )
            search_df = streets_df[mask].copy()

            # Precise Distance Calculation on small subset
            # (Assuming you have a 'distance' function defined elsewhere)
            search_df['dist_temp'] = search_df['coordinates'].apply(
                lambda x: distance(start_coords, x, type="euclidean")
            )
            search_df = search_df[search_df['dist_temp'] <= 2000]
            
        except KeyError:
            print("Start ID invalid, switching to global search.")
            search_df = streets_df # Fallback

    # --- 3. SEARCH LOOP ---
    while True:
        prompt = f"Enter {'destination (<3km)' if start_id else 'location'}"
        user_input = input(f"{prompt}: ").strip()
        if not user_input:
            continue

        # Normalize Input (remove accents, lowercase)
        query_clean = unidecode(user_input).lower()

        # A. FAST EXACT MATCH (Vectorized)
        # matches any part of string: "sagrada" in "basilica de la sagrada familia"
        mask = search_df['name_clean'].str.contains(query_clean, regex=False)
        candidates = search_df[mask].copy()
        
        candidates['score'] = 100 # Base score for exact substring match
        
        # B. FUZZY MATCH (Fallback or Enhancement)
        # If we have too few matches, or to find non-exact matches (typos)
        if len(candidates) < 5:
            # Create a dict {index: name_clean} for RapidFuzz
            choices = search_df[~mask]['name_clean'].to_dict()
            
            # extract returns: (match_string, score, index)
            # scorer=fuzz.partial_ratio is best for substrings! 
            # It finds "sagrada familia" inside "basilica..." with score 100.
            fuzzy_matches = process.extract(
                query_clean, 
                choices, 
                scorer=fuzz.partial_ratio, 
                limit=5,
                score_cutoff=75
            )
            
            # Append fuzzy results
            for match_str, score, idx in fuzzy_matches:
                row = search_df.loc[idx].copy()
                row['score'] = score - 10 # Slightly penalize fuzzy matches vs exact
                # Convert Series to DataFrame (transposed) and append
                candidates = pd.concat([candidates, row.to_frame().T])

        # --- 4. SCORING & DEDUPLICATION ---
        if candidates.empty:
            print("No matches found. Try again.")
            continue

        # Add Bonus Points
        # 1. Places get a bonus
        candidates['score'] += candidates['type'].apply(lambda x: 10 if x in [1, 'place'] else 0)
        
        # 2. Distance penalty (if distance exists)
        if 'dist_temp' in candidates.columns:
            # Penalize 1 point per 100m? Or just sort by distance secondary.
            pass

        # Sort: Score Descending, then Distance Ascending
        sort_cols = ['score']
        ascending = [False]
        if 'dist_temp' in candidates.columns:
            sort_cols.append('dist_temp')
            ascending.append(True)
            
        candidates = candidates.sort_values(by=sort_cols, ascending=ascending).head(1)
        
        best_match = candidates.iloc[0]
        ntype = "Place" if best_match['type'] == 1 else "Street"
        print(f"> Best match: {best_match['name']} [{ntype}]\n")
        
        return best_match['name'], best_match.name # name is the index


# --- PATH CLEANING ---

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


# --- Q-LEARNING AGENT ---

def estimate_crowd(street_id, crowds):
    """
    Estimate crowd level for a street segment.
    """
    if pd.isna(crowds[street_id]):
        return 0.0
    return crowds[street_id]

def calculate_reward(state, next_state, goal, streets, crowds, shortest_path):
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

    # Penalty based on crowd or length
    if shortest_path:
        edge_len = streets.at[next_state, "length"]
        reward -= edge_len/50
    else:
        crowd = estimate_crowd(next_state, crowds)
        reward -= crowd

    return reward

def choose_action(state, epsilon, Q, goal, streets, crowds, shortest_path):
    # Get connections
    next_states = streets.at[state, "connections"]

    if not next_states:
        return goal, -1000
    
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
        streets, crowds, shortest_path
    )

    return next_state, reward

def train(start, goal, streets, crowds, shortest_path=False, 
          parameters=[0.5, 0.9992, 1.0, 1.0, 0.999], 
          episodes=5000, min_delta=0.01, patience=5):
    
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

        while state != goal and steps < 3000:

            next_state, reward = choose_action(
                state, curr_epsilon, Q,
                goal, streets, crowds, shortest_path
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
        if episode in [10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000]:
            if not shortest_path:
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