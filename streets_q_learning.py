import pandas as pd
import random

# Load the JSON file
streets = pd.read_json("streets_grid.json")

# Create a dictionary: { node_id: [list_of_street_ids] }
node_to_streets = {}
for street_id, row in streets.iterrows():
    # Get the two ends of the street
    u_id = row['connections'][0][0]
    v_id = row['connections'][1][0]
    # Add this street to the list for both nodes
    node_to_streets.setdefault(u_id, []).append(street_id)
    node_to_streets.setdefault(v_id, []).append(street_id)

# Setup
START = 1
GOAL = 100

def calculate_reward(state):
    # Get street coordinates
    our_lat = sum(streets.at[state, "connections"][i][1] for i in [0, 1]) / 2
    our_lon = sum(streets.at[state, "connections"][i][2] for i in [0, 1]) / 2

    # Get GOAL coordinates
    end_lat = sum(streets.at[GOAL, "connections"][i][1] for i in [0, 1]) / 2
    end_lon = sum(streets.at[GOAL, "connections"][i][2] for i in [0, 1]) / 2

    return -(abs(our_lat - end_lat) + abs(our_lon - end_lon))*1000


def choose_action(state, epsilon, Q):
    # Get streets connected to same intersections
    next_states = {street for i in [0, 1] for street in node_to_streets.get(streets.at[state, "connections"][i][0], [])}
    next_states.remove(state)
    next_states = list(next_states)

    if state not in Q:
        Q[state] = {}
    for action in next_states:
        if action not in Q[state]:
            Q[state][action] = 0.0
    
    # Explore
    if random.random() < epsilon: 
        next_state = random.choice(next_states)
    # Exploit
    else:
        max_q = max(Q[state][a] for a in next_states)
        best = [a for a in next_states if Q[state][a] == max_q]
        next_state = random.choice(best)

    return next_state, calculate_reward(next_state)

def train(start, goal, parameters, episodes=100):
    Q = {}
    alpha, a_decay, gamma, epsilon, e_decay = parameters

    for episode in range(episodes):
        state = start
        steps = 0
        curr_alpha = alpha * (a_decay ** episode)
        curr_epsilon = epsilon * (e_decay ** episode)
        while state != goal:
            next_state, reward = choose_action(state, curr_epsilon, Q)

            if next_state not in Q: Q[next_state] = {}
            
            # Update Q
            max_next_q = max(Q[next_state].values()) if Q[next_state] else 0
            Q[state][next_state] += curr_alpha * (reward + gamma * max_next_q - Q[state][next_state])

            state = next_state
            steps += 1
        if episode % 10 == 0: print(f"Episode {episode} finished in {steps} steps.")
            


train(START, GOAL, [1, 1, 0.7, 1, 0.99])