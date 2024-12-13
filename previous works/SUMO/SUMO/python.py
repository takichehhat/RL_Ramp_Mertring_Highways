import random
import traci
import numpy as np
import sys

# Define constants
EPISODES = 1000
CYCLE_DURATION = 60  # seconds
LANE_LENGTH = 500  # meters
GREEN_DURATION = 30  # seconds

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_PROB = 0.2

# Initialize Q-table
num_states = 2  # highway and ramp states
num_actions = 11  # number of green light proportions
q_table = np.zeros((num_states, num_actions))

# Read the configuration file name from the command line
if len(sys.argv) != 2:
    print("Usage: python python.py ./config.sumocfg")
    sys.exit(1)

config_file = sys.argv[1]

# Initialize SUMO
sumo_binary = "sumo"
sumo_cmd = [sumo_binary, "-c", config_file]

# Function to get the current state
def get_state():
    # List of edges representing the highway and ramp
    highway_edges = [":J15_0", ":J15_3", "E10", "E8", "E9"]
    ramp_edge_id = "E9"

    # Calculate total vehicle count on the highway
    total_vehicle_count = sum(traci.edge.getLastStepVehicleNumber(edge_id) for edge_id in highway_edges)

    # Calculate density for each edge
    edge_densities = {edge_id: traci.edge.getLastStepVehicleNumber(edge_id) / LANE_LENGTH for edge_id in highway_edges}

    # Optionally, print individual densities for debugging
    for edge_id, density in edge_densities.items():
        print(f"Density on edge {edge_id}: {density}")

    # Calculate overall highway density
    highway_density = total_vehicle_count / (LANE_LENGTH * len(highway_edges))

    # Extract density for the ramp
    ramp_density = traci.edge.getLastStepVehicleNumber(ramp_edge_id) / LANE_LENGTH

    return highway_density, ramp_density

# Function to choose an action using epsilon-greedy policy
def choose_action(state):
    if random.uniform(0, 1) < EXPLORATION_PROB:
        return random.randint(0, num_actions - 1)  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

# Function to update Q-value based on the Bellman equation
def update_q_value(state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state])
    current_q = q_table[state][action]
    next_q = q_table[next_state][best_next_action]
    new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_q - current_q)
    q_table[state][action] = new_q

# Function to calculate reward based on the simulation state
def calculate_reward():
    highway_flow = traci.edge.getLastStepMeanSpeed(":J15_0") * traci.edge.getLastStepVehicleNumber(":J15_0")
    ramp_flow = traci.edge.getLastStepMeanSpeed("E9") * traci.edge.getLastStepVehicleNumber("E9")
    return highway_flow + ramp_flow

# Run the simulation
for episode in range(EPISODES):
    traci.start(sumo_cmd)

    for step in range(CYCLE_DURATION):
        traci.simulationStep()
        state = get_state()
        action = choose_action(state)

        # Apply action (change traffic light duration)
        traci.trafficlight.setProgram("traffic_light", f"program_{action}")

        next_state = get_state()
        reward = calculate_reward()

        update_q_value(state, action, reward, next_state)

    traci.close()

# Save the Q-table or use it for testing
np.save("q_table.npy", q_table)
