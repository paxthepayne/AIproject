import random

def calculate_reward(state, next_state, goal, streets):
    # Basic reward
    if next_state == goal: reward = 10000
    else: reward = -1 
    
    # Distance calculations
    lat1, lon1 = streets.at[state, "coordinates"]
    lat2, lon2 = streets.at[next_state, "coordinates"]
    goal_lat, goal_lon = streets.at[goal, "coordinates"]
    dist_current = abs(lat1 - goal_lat) + abs(lon1 - goal_lon)
    dist_next = abs(lat2 - goal_lat) + abs(lon2 - goal_lon)
    
    # Bonus for moving in the right direction
    if dist_next < dist_current: reward += 1  
    
    return reward

def choose_action(state, epsilon, Q, goal, streets, node_to_streets):
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

    return next_state, calculate_reward(state, next_state, goal, streets)

def train(start, goal, streets, node_to_streets, parameters=[0.7, 0.999, 0.99, 1.0, 0.995], episodes=3000, min_delta=0.1, patience=3):
    Q = {}
    alpha, a_decay, gamma, epsilon, e_decay = parameters
    
    print(f"[Q-Learning] from '{streets.at[start, "name"]}' to '{streets.at[goal, "name"]}'")
    
    stable_episodes = 0 # Counter for convergence check

    for episode in range(episodes):
        path = [start]
        state = start
        steps = 0
        max_change = 0 # Track max Q-value change this episode
        
        # Calculate current decays
        curr_alpha = max(alpha * (a_decay ** episode), 0.01)
        curr_epsilon = max(epsilon * (e_decay ** episode), 0.01)
        
        while state != goal and steps < 3000:

            next_state, reward = choose_action(state, curr_epsilon, Q, goal, streets, node_to_streets)
            
            # Ensure next state exists in Q
            if next_state not in Q:
                Q[next_state] = {}

            # Q-Learning
            max_next_q = max(Q[next_state].values()) if Q[next_state] else 0.0
            current_q = Q[state].get(next_state, 0.0)
            
            # Calculate new value
            new_q = current_q + curr_alpha * (reward + gamma * max_next_q - current_q)
            Q[state][next_state] = new_q
            
            # Track change for convergence
            diff = abs(new_q - current_q)
            if diff > max_change: max_change = diff

            # Progress
            path.append(next_state)
            state = next_state
            steps += 1

        # Check for convergence
        if max_change < min_delta: stable_episodes += 1
        else: stable_episodes = 0
        
        if stable_episodes >= patience:
            print(f"-> Values converged at episode {episode} (max Δ = {min_delta}, patience = {patience})")
            break

        if episode % 200 == 100 and episode != 0:
            print(f"· Episode {episode}: {steps} steps, Δ = {max_change:.4f}")
             
    return path