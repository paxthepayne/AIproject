"""
Q-Learning Logic Class.
Contains the reward calculation, action selection, and main training loop for the pathfinding agent.
"""

import random
import math, json, random
from collections import defaultdict
import pandas as pd

DAYMAP = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
INV_DAYMAP = {v:k for k,v in DAYMAP.items()}

# meters
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

# popular_times(str) -> (weekday, hour) crowd(0~100)
def poi_crowd_at(popular_times_str, weekday, hour):
    pt = json.loads(popular_times_str)
    day_name = INV_DAYMAP[weekday]
    for d in pt:
        if d["name"] == day_name:
            return float(d["data"][hour])
    return 0.0

# make cache for not to be slow 
def build_street_poi_map(streets_list, pois_list, radius_m=200, sigma_m=100):
    mapping = {}
    for s in streets_list:
        sid = s["id"]
        lat, lon = s["center"]
        lst = []
        for i, p in enumerate(pois_list):
            d = haversine_m(lat, lon, p["lat"], p["lon"])
            if d <= radius_m:
                w = math.exp(-(d*d)/(2*sigma_m*sigma_m))
                lst.append((i, w))
        mapping[sid] = lst
    return mapping

def street_crowd(street_id, weekday, hour, street_poi_map, pois_list):
    pairs = street_poi_map.get(street_id, [])
    if not pairs:
        return 0.0
    num = den = 0.0
    for poi_idx, w in pairs:
        c = poi_crowd_at(pois_list[poi_idx]["popular_times"], weekday, hour)  # 0~100
        num += w * c
        den += w
    return (num / den) if den > 0 else 0.0

def calculate_reward(state, next_state, goal, streets_by_id,
                     street_poi_map, pois_list,
                     weekday=6, hour=14,
                     crowd_w=10.0, length_w=1.0):
    if next_state == goal:
        return 10000

    reward = -1

    # penalty (for length/move)
    reward -= length_w * streets_by_id[next_state]["length"]

    # crowd penalty (0~100 -> 0~1 normalization)
    crowd = street_crowd(next_state, weekday, hour, street_poi_map, pois_list)
    reward -= crowd_w * (crowd / 100.0)

    return reward

def choose_action(state, epsilon, Q, goal, streets_by_id,
                  street_poi_map, pois_list,
                  weekday=6, hour=14):
    next_states = streets_by_id[state]["connections"]

    if state not in Q:
        Q[state] = {action: 0.0 for action in next_states}
    for action in next_states:
        if action not in Q[state]:
            Q[state][action] = 0.0

    if random.random() < epsilon:
        next_state = random.choice(next_states)
    else:
        max_q = max(Q[state][a] for a in next_states)
        best = [a for a in next_states if Q[state][a] == max_q]
        next_state = random.choice(best)

    r = calculate_reward(state, next_state, goal, streets_by_id,
                         street_poi_map, pois_list,
                         weekday=weekday, hour=hour)
    return next_state, r

def train(start, goal, streets, pois_list, parameters=[0.7, 0.999, 0.99, 1.0, 0.995], weekday=6, hour=14, episodes=2000, min_delta=0.1, patience=3):
    if isinstance(streets, pd.DataFrame):
        streets_by_id = streets.to_dict(orient="index")
        streets = streets.reset_index().to_dict("records")
    else:
        streets_by_id = {s["id"]: s for s in streets}
        streets = streets
    
    street_poi_map = build_street_poi_map(streets, pois_list, radius_m=120, sigma_m=60)
    
    Q = {}
    alpha, a_decay, gamma, epsilon, e_decay = parameters
    
    start_name = streets_by_id[start]["name"]
    goal_name  = streets_by_id[goal]["name"]
    print(f"[Q-Learning] from '{start_name}' to '{goal_name}'")
    
    stable_episodes = 0 # Counter for convergence check

    for episode in range(episodes):
        path = [start]
        state = start
        steps = 0
        max_change = 0 # Track max Q-value change this episode
        
        # Calculate current decays
        curr_alpha = max(alpha * (a_decay ** episode), 0.01)
        curr_epsilon = max(epsilon * (e_decay ** episode), 0.01)
        
        while state != goal and steps < 5000:

            next_state, reward = choose_action(state, curr_epsilon, Q, goal, streets_by_id, street_poi_map, pois_list, weekday=weekday, hour=hour)
            
            # Ensure next state exists in Q
            if next_state not in Q:
                Q[next_state] = {}

            # Q-Learning Formula
            # Q(s,a) = Q(s,a) + alpha * (R + gamma * max(Q(s',a')) - Q(s,a))
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
        if max_change < min_delta: stable_episodes += 1
        else: stable_episodes = 0
        
        # Stop early if converged
        if stable_episodes >= patience:
            print(f"-> Values converged at episode {episode} (max Δ = {min_delta}, patience = {patience})")
            break

        # Log progress periodically
        if episode % 200 == 100 and episode != 0:
            print(f"· Episode {episode}: {steps} steps, Δ = {max_change:.4f}")
             
    return path