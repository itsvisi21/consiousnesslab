import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define initial conditions for Lagrange points and figure-eight orbit (simplified example)
def initialize_known_solutions(solution_type="lagrange"):
    """
    Initialize positions and velocities for known solutions like Lagrange points or figure-eight orbits.
    """
    if solution_type == "lagrange":
        # Simplified Lagrange point setup (equilateral triangle configuration)
        positions = np.array([
            [1.0, 0.0, 0.0],  # Body 1
            [-0.5, np.sqrt(3)/2, 0.0],  # Body 2
            [-0.5, -np.sqrt(3)/2, 0.0]  # Body 3
        ])
        velocities = np.zeros_like(positions)  # Static equilibrium
    elif solution_type == "figure_eight":
        # Approximate positions and velocities for the figure-eight solution (simplified example)
        positions = np.array([
            [0.97000436, -0.24308753, 0.0],  # Body 1
            [-0.97000436, 0.24308753, 0.0],  # Body 2
            [0.0, 0.0, 0.0]  # Body 3
        ])
        velocities = np.array([
            [0.466203685, 0.43236573, 0.0],  # Body 1
            [0.466203685, 0.43236573, 0.0],  # Body 2
            [-0.93240737, -0.86473146, 0.0]  # Body 3
        ])
    else:
        raise ValueError("Unknown solution type. Choose 'lagrange' or 'figure_eight'.")
    return positions, velocities

# Simplified functions to compute Hamiltonian and simulate evolution
def compute_relativistic_hamiltonian(positions, velocities, masses, time_elapsed=1.0):
    """
    Placeholder for computing a Hamiltonian matrix for gravitational systems.
    """
    n = len(masses)
    hamiltonian = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = np.linalg.norm(positions[i] - positions[j])
                hamiltonian[i, j] = -1.0 / distance  # Simplified gravity-like term
    return hamiltonian

def multi_time_step_evolution_rescaled(hamiltonian, initial_state, time_steps, delta_t):
    """
    Simulate quantum state evolution with Hamiltonian over multiple time steps.
    """
    states = [initial_state]
    current_state = initial_state
    for _ in range(time_steps):
        current_state = np.dot(hamiltonian, current_state)
        current_state /= np.linalg.norm(current_state)  # Normalize to prevent overflow
        states.append(current_state)
    return states

def encode_to_quantum_state(positions, velocities):
    """
    Encode positions and velocities into a normalized quantum state.
    """
    quantum_state = np.random.random(len(positions)) + 1j * np.random.random(len(positions))
    quantum_state /= np.linalg.norm(quantum_state)
    return quantum_state

# Initialize configurations for Lagrange points and figure-eight orbit
positions_lagrange, velocities_lagrange = initialize_known_solutions("lagrange")
positions_figure_eight, velocities_figure_eight = initialize_known_solutions("figure_eight")

quantum_state_lagrange = encode_to_quantum_state(positions_lagrange, velocities_lagrange)
quantum_state_figure_eight = encode_to_quantum_state(positions_figure_eight, velocities_figure_eight)

time_steps = 50
delta_t = 1.0

# Lagrange Points Simulation
refined_hamiltonian_lagrange = compute_relativistic_hamiltonian(
    positions_lagrange, velocities_lagrange, [1e24, 1e24, 1e24]
)
evolved_lagrange_states = multi_time_step_evolution_rescaled(
    refined_hamiltonian_lagrange, quantum_state_lagrange, time_steps, delta_t
)

# Figure-Eight Simulation
refined_hamiltonian_figure_eight = compute_relativistic_hamiltonian(
    positions_figure_eight, velocities_figure_eight, [1e24, 1e24, 1e24]
)
evolved_figure_eight_states = multi_time_step_evolution_rescaled(
    refined_hamiltonian_figure_eight, quantum_state_figure_eight, time_steps, delta_t
)

# Visualize Quantum Dynamics for Lagrange and Figure-Eight Solutions
def visualize_solution_trajectories(states, title):
    """
    Visualize trajectories for the given quantum states over time.
    """
    spatial_grid = np.linspace(-2, 2, len(states[0]))  # Adjust grid based on positions
    for time_step, state in enumerate(states):
        positions = spatial_grid * np.abs(state) / np.max(np.abs(state))  # Scale positions by amplitudes
        plt.scatter(positions, [time_step] * len(positions), label=f"Time Step {time_step}" if time_step < 5 else "", alpha=0.6)
    plt.title(f"{title} Quantum State Trajectories")
    plt.xlabel("Position")
    plt.ylabel("Time Step")
    plt.legend()
    plt.grid()
    plt.show()

# visualize_solution_trajectories(evolved_lagrange_states, "Lagrange Points")
# visualize_solution_trajectories(evolved_figure_eight_states, "Figure-Eight Orbit")

# Define classical trajectories for Lagrange Points and Figure-Eight Orbits (simplified)
def generate_classical_trajectories(solution_type, time_steps, delta_t):
    """
    Generate classical trajectories for Lagrange points and figure-eight orbits.
    """
    if solution_type == "lagrange":
        # Static classical positions for Lagrange points
        positions = np.array([
            [1.0, 0.0, 0.0],  # Body 1
            [-0.5, np.sqrt(3)/2, 0.0],  # Body 2
            [-0.5, -np.sqrt(3)/2, 0.0]  # Body 3
        ])
        trajectories = [positions for _ in range(time_steps)]
    elif solution_type == "figure_eight":
        # Approximate periodic motion for figure-eight orbits (simplified)
        trajectories = []
        for t in range(time_steps):
            phase = 2 * np.pi * t / time_steps
            positions = np.array([
                [np.cos(phase), np.sin(phase), 0.0],  # Body 1
                [np.cos(phase + 2 * np.pi / 3), np.sin(phase + 2 * np.pi / 3), 0.0],  # Body 2
                [np.cos(phase + 4 * np.pi / 3), np.sin(phase + 4 * np.pi / 3), 0.0]  # Body 3
            ])
            trajectories.append(positions)
    else:
        raise ValueError("Unknown solution type. Choose 'lagrange' or 'figure_eight'.")
    return np.array(trajectories)

# Generate classical trajectories
classical_lagrange_trajectories = generate_classical_trajectories("lagrange", time_steps, delta_t)
classical_figure_eight_trajectories = generate_classical_trajectories("figure_eight", time_steps, delta_t)

# Compare quantum and classical trajectories
def compare_quantum_classical(quantum_states, classical_trajectories, title, spatial_grid):
    """
    Compare quantum and classical trajectories for visualization.
    """
    # Map quantum states to positions
    quantum_positions = []
    for state in quantum_states:
        probabilities = np.abs(state) ** 2
        probabilities /= np.sum(probabilities)  # Normalize
        positions = spatial_grid * probabilities
        quantum_positions.append(positions)
    quantum_positions = np.array(quantum_positions)
    
    # Plot comparison
#     for dim, label in enumerate(["x", "y", "z"]):
#         plt.figure(figsize=(10, 6))
#         for body in range(classical_trajectories.shape[1]):
#             # Classical trajectory for the body
#             classical = classical_trajectories[:, body, dim]
#             plt.plot(range(len(classical)), classical, label=f"Classical Body {body + 1} ({label})", linestyle="--")
            
#             # Quantum trajectory for the body
#             quantum = quantum_positions[:, body]
#             plt.plot(range(len(quantum)), quantum, label=f"Quantum Body {body + 1} ({label})")
        
#         plt.title(f"{title} - {label.upper()} Dimension")
#         plt.xlabel("Time Step")
#         plt.ylabel(f"{label}-Position")
#         plt.legend()
#         plt.grid()
#         plt.show()

# # Define spatial grid for quantum positions
# spatial_grid = np.linspace(-2, 2, 3)

# # Compare Lagrange Points
# compare_quantum_classical(evolved_lagrange_states, classical_lagrange_trajectories, "Lagrange Points", spatial_grid)

# # Compare Figure-Eight Orbits
# compare_quantum_classical(evolved_figure_eight_states, classical_figure_eight_trajectories, "Figure-Eight Orbits", spatial_grid)



# Define initial conditions for Lagrange points and figure-eight orbit
def initialize_known_solutions(solution_type="lagrange"):
    if solution_type == "lagrange":
        positions = np.array([
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3)/2, 0.0],
            [-0.5, -np.sqrt(3)/2, 0.0]
        ])
        velocities = np.zeros_like(positions)
    elif solution_type == "figure_eight":
        positions = np.array([
            [0.97000436, -0.24308753, 0.0],
            [-0.97000436, 0.24308753, 0.0],
            [0.0, 0.0, 0.0]
        ])
        velocities = np.array([
            [0.466203685, 0.43236573, 0.0],
            [0.466203685, 0.43236573, 0.0],
            [-0.93240737, -0.86473146, 0.0]
        ])
    else:
        raise ValueError("Unknown solution type. Choose 'lagrange' or 'figure_eight'.")
    return positions, velocities

# Compute Hamiltonian
def compute_relativistic_hamiltonian(positions, velocities, masses, time_elapsed=1.0):
    n = len(masses)
    hamiltonian = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = np.linalg.norm(positions[i] - positions[j])
                hamiltonian[i, j] = -1.0 / distance
    return hamiltonian

# Multi-time-step evolution
def multi_time_step_evolution_rescaled(hamiltonian, initial_state, time_steps, delta_t):
    states = [initial_state]
    current_state = initial_state
    for _ in range(time_steps):
        current_state = np.dot(hamiltonian, current_state)
        current_state /= np.linalg.norm(current_state)
        states.append(current_state)
    return states

# Encode into quantum states
def encode_to_quantum_state(positions, velocities):
    quantum_state = np.random.random(len(positions)) + 1j * np.random.random(len(positions))
    quantum_state /= np.linalg.norm(quantum_state)
    return quantum_state

# Generate classical trajectories
def generate_classical_trajectories(solution_type, time_steps, delta_t):
    if solution_type == "lagrange":
        positions = np.array([
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3)/2, 0.0],
            [-0.5, -np.sqrt(3)/2, 0.0]
        ])
        trajectories = [positions for _ in range(time_steps)]
    elif solution_type == "figure_eight":
        trajectories = []
        for t in range(time_steps):
            phase = 2 * np.pi * t / time_steps
            positions = np.array([
                [np.cos(phase), np.sin(phase), 0.0],
                [np.cos(phase + 2 * np.pi / 3), np.sin(phase + 2 * np.pi / 3), 0.0],
                [np.cos(phase + 4 * np.pi / 3), np.sin(phase + 4 * np.pi / 3), 0.0]
            ])
            trajectories.append(positions)
    else:
        raise ValueError("Unknown solution type. Choose 'lagrange' or 'figure_eight'.")
    return np.array(trajectories)

# Deviation analysis
def compute_deviation(quantum_states, classical_trajectories, spatial_grid):
    quantum_positions = []
    for state in quantum_states:
        probabilities = np.abs(state) ** 2
        probabilities /= np.sum(probabilities)
        positions = spatial_grid * probabilities
        quantum_positions.append(positions)
    quantum_positions = np.array(quantum_positions)
    
    mse = []
    for dim in range(3):
        dim_mse = []
        for body in range(classical_trajectories.shape[1]):
            classical = classical_trajectories[:, body, dim]
            quantum = quantum_positions[:, body]
            dim_mse.append(np.mean((classical - quantum) ** 2))
        mse.append(dim_mse)
    return np.array(mse)

# Main workflow
time_steps = 50
delta_t = 1.0
spatial_grid = np.linspace(-2, 2, 3)

# Initialize configurations
positions_lagrange, velocities_lagrange = initialize_known_solutions("lagrange")
positions_figure_eight, velocities_figure_eight = initialize_known_solutions("figure_eight")

quantum_state_lagrange = encode_to_quantum_state(positions_lagrange, velocities_lagrange)
quantum_state_figure_eight = encode_to_quantum_state(positions_figure_eight, velocities_figure_eight)

refined_hamiltonian_lagrange = compute_relativistic_hamiltonian(
    positions_lagrange, velocities_lagrange, [1e24, 1e24, 1e24]
)
evolved_lagrange_states = multi_time_step_evolution_rescaled(
    refined_hamiltonian_lagrange, quantum_state_lagrange, time_steps, delta_t
)

refined_hamiltonian_figure_eight = compute_relativistic_hamiltonian(
    positions_figure_eight, velocities_figure_eight, [1e24, 1e24, 1e24]
)
evolved_figure_eight_states = multi_time_step_evolution_rescaled(
    refined_hamiltonian_figure_eight, quantum_state_figure_eight, time_steps, delta_t
)

classical_lagrange_trajectories = generate_classical_trajectories("lagrange", time_steps, delta_t)
classical_figure_eight_trajectories = generate_classical_trajectories("figure_eight", time_steps, delta_t)

# Align states and trajectories
min_time_steps = min(len(evolved_lagrange_states), classical_lagrange_trajectories.shape[0])
aligned_lagrange_states = evolved_lagrange_states[:min_time_steps]
aligned_lagrange_trajectories = classical_lagrange_trajectories[:min_time_steps]

aligned_figure_eight_states = evolved_figure_eight_states[:min_time_steps]
aligned_figure_eight_trajectories = classical_figure_eight_trajectories[:min_time_steps]

# Compute deviations
deviation_lagrange = compute_deviation(aligned_lagrange_states, aligned_lagrange_trajectories, spatial_grid)
deviation_figure_eight = compute_deviation(aligned_figure_eight_states, aligned_figure_eight_trajectories, spatial_grid)

def display_dataframe_to_user(name, dataframe):
    """
    Display a DataFrame with a given name for analysis.

    Args:
        name (str): The name/title of the DataFrame analysis.
        dataframe (dict or pd.DataFrame): The data to display.

    Returns:
        None
    """
    print(f"\n=== {name} ===\n")
    
    # Check if input is a dictionary and convert to DataFrame
    if isinstance(dataframe, dict):
        dataframe = pd.DataFrame(dataframe)

    # Display the DataFrame
    print(dataframe.to_string(index=False))  # Display DataFrame as a formatted string

# Define initial conditions for Lagrange points and figure-eight orbit
def initialize_known_solutions(solution_type="lagrange"):
    if solution_type == "lagrange":
        positions = np.array([
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3)/2, 0.0],
            [-0.5, -np.sqrt(3)/2, 0.0]
        ])
        velocities = np.zeros_like(positions)
    elif solution_type == "figure_eight":
        positions = np.array([
            [0.97000436, -0.24308753, 0.0],
            [-0.97000436, 0.24308753, 0.0],
            [0.0, 0.0, 0.0]
        ])
        velocities = np.array([
            [0.466203685, 0.43236573, 0.0],
            [0.466203685, 0.43236573, 0.0],
            [-0.93240737, -0.86473146, 0.0]
        ])
    else:
        raise ValueError("Unknown solution type. Choose 'lagrange' or 'figure_eight'.")
    return positions, velocities

# Compute Hamiltonian
def compute_relativistic_hamiltonian(positions, velocities, masses, time_elapsed=1.0):
    n = len(masses)
    hamiltonian = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = np.linalg.norm(positions[i] - positions[j])
                hamiltonian[i, j] = -1.0 / distance
    return hamiltonian

# Multi-time-step evolution
def multi_time_step_evolution_rescaled(hamiltonian, initial_state, time_steps, delta_t):
    states = [initial_state]
    current_state = initial_state
    for _ in range(time_steps):
        current_state = np.dot(hamiltonian, current_state)
        current_state /= np.linalg.norm(current_state)
        states.append(current_state)
    return states

# Encode into quantum states
def encode_to_quantum_state(positions, velocities):
    quantum_state = np.random.random(len(positions)) + 1j * np.random.random(len(positions))
    quantum_state /= np.linalg.norm(quantum_state)
    return quantum_state

# Generate classical trajectories
def generate_classical_trajectories(solution_type, time_steps, delta_t):
    if solution_type == "lagrange":
        positions = np.array([
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3)/2, 0.0],
            [-0.5, -np.sqrt(3)/2, 0.0]
        ])
        trajectories = [positions for _ in range(time_steps)]
    elif solution_type == "figure_eight":
        trajectories = []
        for t in range(time_steps):
            phase = 2 * np.pi * t / time_steps
            positions = np.array([
                [np.cos(phase), np.sin(phase), 0.0],
                [np.cos(phase + 2 * np.pi / 3), np.sin(phase + 2 * np.pi / 3), 0.0],
                [np.cos(phase + 4 * np.pi / 3), np.sin(phase + 4 * np.pi / 3), 0.0]
            ])
            trajectories.append(positions)
    else:
        raise ValueError("Unknown solution type. Choose 'lagrange' or 'figure_eight'.")
    return np.array(trajectories)

# Refined quantum mapping with interpolation
from scipy.interpolate import interp1d

def refined_quantum_mapping_with_interpolation(quantum_states, spatial_grid):
    refined_positions = []
    original_grid = np.linspace(-2, 2, len(quantum_states[0]))

    for state in quantum_states:
        probabilities = np.abs(state) ** 2
        probabilities /= np.sum(probabilities)

        interpolation_function = interp1d(original_grid, probabilities, kind='linear', fill_value="extrapolate")
        refined_probabilities = interpolation_function(spatial_grid)

        refined_probabilities /= np.sum(refined_probabilities)
        positions = spatial_grid * refined_probabilities
        refined_positions.append(positions)

    return np.array(refined_positions)

# Deviation analysis
def compute_deviation_with_refined_mapping(refined_positions, classical_trajectories):
    mse = []
    for dim in range(3):
        dim_mse = []
        for body in range(classical_trajectories.shape[1]):
            classical = classical_trajectories[:, body, dim]
            quantum = refined_positions[:, body]
            dim_mse.append(np.mean((classical - quantum) ** 2))
        mse.append(dim_mse)
    return np.array(mse)

# Main workflow
time_steps = 50
delta_t = 1.0
spatial_grid = np.linspace(-2, 2, 3)
high_resolution_grid = np.linspace(-10, 10, 100)

# Initialize configurations
positions_lagrange, velocities_lagrange = initialize_known_solutions("lagrange")
positions_figure_eight, velocities_figure_eight = initialize_known_solutions("figure_eight")

quantum_state_lagrange = encode_to_quantum_state(positions_lagrange, velocities_lagrange)
quantum_state_figure_eight = encode_to_quantum_state(positions_figure_eight, velocities_figure_eight)

refined_hamiltonian_lagrange = compute_relativistic_hamiltonian(
    positions_lagrange, velocities_lagrange, [1e24, 1e24, 1e24]
)
evolved_lagrange_states = multi_time_step_evolution_rescaled(
    refined_hamiltonian_lagrange, quantum_state_lagrange, time_steps, delta_t
)

refined_hamiltonian_figure_eight = compute_relativistic_hamiltonian(
    positions_figure_eight, velocities_figure_eight, [1e24, 1e24, 1e24]
)
evolved_figure_eight_states = multi_time_step_evolution_rescaled(
    refined_hamiltonian_figure_eight, quantum_state_figure_eight, time_steps, delta_t
)

classical_lagrange_trajectories = generate_classical_trajectories("lagrange", time_steps, delta_t)
classical_figure_eight_trajectories = generate_classical_trajectories("figure_eight", time_steps, delta_t)

# Align states and trajectories
min_time_steps = min(len(evolved_lagrange_states), classical_lagrange_trajectories.shape[0])
aligned_lagrange_states = evolved_lagrange_states[:min_time_steps]
aligned_lagrange_trajectories = classical_lagrange_trajectories[:min_time_steps]

aligned_figure_eight_states = evolved_figure_eight_states[:min_time_steps]
aligned_figure_eight_trajectories = classical_figure_eight_trajectories[:min_time_steps]

# Apply refined mapping
refined_lagrange_positions = refined_quantum_mapping_with_interpolation(aligned_lagrange_states, high_resolution_grid)
refined_figure_eight_positions = refined_quantum_mapping_with_interpolation(aligned_figure_eight_states, high_resolution_grid)

# Compute deviations
refined_deviation_lagrange = compute_deviation_with_refined_mapping(refined_lagrange_positions, aligned_lagrange_trajectories)
refined_deviation_figure_eight = compute_deviation_with_refined_mapping(refined_figure_eight_positions, aligned_figure_eight_trajectories)

# Display refined deviations
refined_deviation_data = {
    "Dimension": ["x", "y", "z"],
    "Refined Lagrange Body 1": refined_deviation_lagrange[:, 0],
    "Refined Lagrange Body 2": refined_deviation_lagrange[:, 1],
    "Refined Lagrange Body 3": refined_deviation_lagrange[:, 2],
    "Refined Figure-Eight Body 1": refined_deviation_figure_eight[:, 0],
    "Refined Figure-Eight Body 2": refined_deviation_figure_eight[:, 1],
    "Refined Figure-Eight Body 3": refined_deviation_figure_eight[:, 2],
}
refined_deviation_df = pd.DataFrame(refined_deviation_data)
display_dataframe_to_user("Refined Quantum-Classical Trajectory Deviations with Interpolation", refined_deviation_df)


