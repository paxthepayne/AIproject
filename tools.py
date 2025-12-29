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
        prompt = f"Enter {'destination (<2km)' if start_id else 'location'}"
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
        reward -= edge_len / 50
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
          parameters=[0.5, 0.999, 1.0, 1.0, 0.997], 
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


# --- HELPER: BEARING ---

def get_cardinal_direction(start_coords, end_coords):
    """Calculates direction (N, NE, E...) between two lat/lon points."""
    lat1, lon1 = math.radians(start_coords[0]), math.radians(start_coords[1])
    lat2, lon2 = math.radians(end_coords[0]), math.radians(end_coords[1])

    d_lon = lon2 - lon1
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(d_lon))

    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ix = round(compass_bearing / 45)
    return dirs[ix % 8]


# --- NAVIGATION MODE ---

def navigation_mode(start, goal, streets, crowds, Q):
    """
    Fixed Navigation Mode:
    - ALWAYS shows the option to 'Continue' on your current street.
    - Calculates 'Turn' vs 'Continue' based on where you came from.
    - Suppresses Nagging for 100m.
    - Prevents Loops.
    """
    
    current = start
    prev = None
    goal_coords = streets.at[goal, "coordinates"]
    
    visited = {start}
    path_stack = [start]
    
    walked_dist = 0
    
    # Track the street we are CURRENTLY walking on
    current_street_display = streets.at[start, "name"]
    
    # ANTI-NAG MEMORY
    last_rejected_tokens = set()
    dist_since_decision = 0 

    print("\n[👇 Smart Navigation Mode]")
    print("Agent suppresses repeated questions for 100m, then re-confirms.\n")

    # --- HELPER: TOKENS ---
    def get_tokens(name):
        """Returns (set_of_clean_words, display_string)"""
        if not isinstance(name, str): return (set(), "")
        
        if "sin nombre" in name.lower():
            return ({"unnamed"}, "Unnamed Street")

        clean = unidecode(str(name)).lower()
        clean = re.sub(r'[^\w\s]', '', clean)
        tokens = clean.split()
        
        stopwords = {
            "carrer", "de", "del", "d", "la", "el", "els", "les", "los", "las",
            "plaça", "placa", "plaza", "passeig", "avinguda", "av", "rambla", 
            "calle", "gran", "via", "travessera", "sant", "santa", "passatge"
        }
        
        meaningful = [t for t in tokens if t not in stopwords]
        display = " ".join(t.title() for t in tokens if t in meaningful)
        if not display: display = name
        
        return (set(meaningful), display)

    # Initialize tokens for the starting street
    current_street_tokens, current_street_display = get_tokens(streets.at[start, "name"])

    while current != goal:
        current_node_name = streets.at[current, "name"]
        current_coords = streets.at[current, "coordinates"]
        
        curr_dist_goal = distance(current_coords, goal_coords, type="euclidean")
        
        # 1. CHECK SUPPRESSION LIMIT
        if dist_since_decision > 100:
            last_rejected_tokens.clear()
            dist_since_decision = 0
        
        neighbors = streets.at[current, "connections"]
        if not neighbors:
            print("🚫 Dead end.")
            break

        # --- 2. GATHER OPTIONS ---
        raw_options = []
        for n in neighbors:
            if n in visited: continue
                
            n_name = streets.at[n, "name"]
            n_coords = streets.at[n, "coordinates"]
            q_val = Q.get(current, {}).get(n, -float("inf"))
            d_goal = distance(n_coords, goal_coords, type="euclidean")
            bearing = get_cardinal_direction(current_coords, n_coords)

            diff = d_goal - curr_dist_goal
            if diff < -5:   prog = "↓ closer"
            elif diff > 5:  prog = "↑ further"
            else:           prog = "= steady"

            tokens_set, display_str = get_tokens(n_name)
            
            # CRITICAL FIX: Determine "Continue" based on stored current street,
            # NOT the node name (which might be the cross-street).
            is_continue = bool(tokens_set & current_street_tokens)

            raw_options.append({
                "id": n,
                "name": n_name,
                "display": display_str,
                "tokens": tokens_set,
                "is_continue": is_continue, # Flag for filtering
                "q": q_val,
                "dir": bearing,
                "len": streets.at[n, "length"],
                "dist_to_goal": d_goal,
                "progress": prog,
                "diff": diff
            })

        if not raw_options:
            print("⚠️ Stuck. Backtracking...")
            break

        # --- 3. DEDUPLICATE ---
        best_per_street = {}
        for opt in raw_options:
            key = frozenset(opt["tokens"])
            if key not in best_per_street:
                best_per_street[key] = opt
            else:
                if opt["dist_to_goal"] < best_per_street[key]["dist_to_goal"]:
                    best_per_street[key] = opt
        
        deduped_options = list(best_per_street.values())

        # --- 4. FILTER ---
        # Show options if:
        # A) They are "Closer" or "Steady"
        # B) OR they are the "Continue" option (Never hide the current street!)
        
        good_options = []
        for o in deduped_options:
            if o["progress"] in ["↓ closer", "= steady"] or o["is_continue"]:
                good_options.append(o)
                
        final_options = good_options if good_options else deduped_options
        final_options.sort(key=lambda x: (x["diff"], -x["q"]))

        best = final_options[0]

        # --- 5. INTELLIGENT AUTO-WALK ---
        
        # Identify if we have a valid Continue option
        continue_option = next((o for o in final_options if o["is_continue"]), None)
        
        # Override Best if we recently rejected the turn and can continue
        if continue_option and (best["tokens"] != continue_option["tokens"]):
            if best["tokens"] & last_rejected_tokens:
                best = continue_option

        # Auto-walk conditions
        # 1. Only 1 option
        # 2. Staying on same street
        # 3. Turning, but we rejected other options recently
        
        rejected_others = True
        for o in final_options:
            if o is not best:
                if not (o["tokens"] & last_rejected_tokens):
                    rejected_others = False
                    break
        
        is_same_street = best["is_continue"]
        
        should_auto_walk = (best["id"] != goal) and (
            len(final_options) == 1 or is_same_street or (rejected_others and dist_since_decision < 100)
        )

        if should_auto_walk:
            # First move update
            if walked_dist == 0:
                # If we just switched streets (e.g. at a turn), update name
                # But if we are continuing, keep the name
                current_street_display = best["display"]
                current_street_tokens = best["tokens"]

            walked_dist += best["len"]
            dist_since_decision += best["len"]
            
            prev = current
            current = best["id"]
            visited.add(current)
            path_stack.append(current)
            continue

        # --- 6. DISPLAY ---
        
        if walked_dist > 0:
            steps = int(walked_dist / 0.8)
            print(f"🚶 Go straight on {current_street_display} for {int(walked_dist)}m ({steps} steps)")
            print(f"   ↓")
        
        walked_dist = 0
        dist_since_decision = 0 

        if best["id"] == goal:
            print(f"🏁 Arrived at {best['name']}!")
            return

        print(f"📍 Intersection at {current_node_name}")
        
        for i, opt in enumerate(final_options, 1):
            steps_tot = int(opt["dist_to_goal"] / 0.8)
            crowd = estimate_crowd(opt["id"], crowds)
            c_str = "<10" if crowd < 10 else str(int(crowd))
            star = "★" if i == 1 else " "
            
            # Verb logic is now robust
            verb = "continue" if opt["is_continue"] else "turn onto"
            
            print(f"  {i}. {star} [{opt['dir']}] {verb} {opt['display']} "
                  f"({opt['progress']}, {steps_tot} steps left, crowd: {c_str})")

        # --- 7. INPUT ---
        while True:
            choice = input("> ").strip().lower()
            if choice == 'q': return
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(final_options):
                    selected = final_options[idx]
                    
                    # Memory Update
                    if not selected["is_continue"]:
                        last_rejected_tokens = set()
                    else:
                        new_rejects = set()
                        for i, o in enumerate(final_options):
                            if i != idx:
                                new_rejects.update(o["tokens"])
                        last_rejected_tokens = new_rejects

                    dist_since_decision = 0
                    
                    # Update "Current Street" tracking
                    current_street_display = selected["display"]
                    current_street_tokens = selected["tokens"]
                    
                    prev = current
                    current = selected["id"]
                    visited.add(current)
                    path_stack.append(current)
                    break
                else:
                    print("Invalid.")
            else:
                print("Enter number or 'q'.")