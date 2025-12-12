import pandas as pd
import random
import json

# Load streets map
streets = pd.read_json("map_streets.json").set_index('id')

# Load nodes map
with open("map_nodes.json", "r") as f:
    node_to_streets = json.load(f)

# Setup
START = 1
GOAL = 50000
# Updated Parameters
PARAMETERS = [
    0.7,    # alpha (learning rate) - reduced for stability
    0.999,  # alpha decay - slower decay
    0.99,   # gamma (discount) - INCREASED to see further ahead
    1.0,    # epsilon (initial exploration)
    0.995   # epsilon decay
]

def calculate_reward(state, next_state):

    if next_state == GOAL:
        reward = 10000
    else:
        reward = -1 
    
    # OPTIONAL: Simple Potential-Based Shaping (Heuristic)
    # This helps guide the agent without getting it stuck in local optima
    # (Requires calculating distance for state and next_state)
    lat1, lon1 = streets.at[state, "coordinates"]
    lat2, lon2 = streets.at[next_state, "coordinates"]
    goal_lat, goal_lon = streets.at[GOAL, "coordinates"]
    dist_current = abs(lat1 - goal_lat) + abs(lon1 - goal_lon)
    dist_next = abs(lat2 - goal_lat) + abs(lon2 - goal_lon)
    
    if dist_next < dist_current:
        reward += 1.0  # Small bonus for moving in the right direction
    
    return reward

def choose_action(state, epsilon, Q):
    # Get connections
    u_node, v_node = streets.at[state, "connections"]
    neighbors_u = node_to_streets.get(str(u_node), [])
    neighbors_v = node_to_streets.get(str(v_node), [])

    # List of possible next streets
    next_states = list(set(neighbors_u + neighbors_v) - {state})

    # Ensure state, actions exist in Q
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

    return next_state, calculate_reward(state, next_state)

def train(start, goal, parameters, episodes=1000, pathfind=False):
    Q = {}
    alpha, a_decay, gamma, epsilon, e_decay = parameters
    
    print(f"Training from street {start} to {goal}...")

    for episode in range(episodes):
        path = [start]
        state = start
        steps = 0
        
        # Calculate current decays
        curr_alpha = max(alpha * (a_decay ** episode), 0.01)
        curr_epsilon = max(epsilon * (e_decay ** episode), 0.01)
        if episode == episodes-1: curr_epsilon = 0
        while state != goal and steps < 3000:

            next_state, reward = choose_action(state, curr_epsilon, Q)
            
            # Ensure next state exists in Q
            if next_state not in Q:
                Q[next_state] = {}

            # Q-Learning
            max_next_q = max(Q[next_state].values()) if Q[next_state] else 0.0
            current_q = Q[state].get(next_state, 0.0)
            Q[state][next_state] = current_q + curr_alpha * (reward + gamma * max_next_q - current_q)

            # Progress
            if episode == episodes-1: path.append(next_state)
            state = next_state
            steps += 1

        if episode % 20 == 0:
            print(f"Episode {episode}: {steps} steps. Epsilon: {curr_epsilon:.4f}")
            
    if pathfind: 
        return path


# --- TEST ---
path = train(START, GOAL, PARAMETERS, pathfind=True)
print(f"Path found ({len(path)} steps): {path}")