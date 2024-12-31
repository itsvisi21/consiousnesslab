import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant
masses = [5.0e24, 5.0e24, 5.0e24]  # Masses of the three bodies (kg)
time_step = 1  # Time step for evolution (arbitrary units)

# Initial conditions for three bodies (positions and velocities in 3D)
initial_positions = np.array([
    [1.2, -0.5, 0.1],   # Body 1
    [-0.8, 0.3, 1.0],   # Body 2
    [0.4, -1.2, 0.0]    # Body 3
])
initial_velocities = np.array([
    [-0.8, 0.0, 0.6],   # Body 1
    [0.2, -1.0, -0.4],  # Body 2
    [0.5, 1.2, -0.3]    # Body 3
])

# Function to map continuous values to Qutrit states
def encode_to_qutrit(value, epsilon=0.5):
    if value < -epsilon:
        return -1
    elif -epsilon <= value <= epsilon:
        return 0
    else:
        return 1

# Encode positions and velocities into Qutrits
epsilon = 0.5
encoded_positions = np.vectorize(encode_to_qutrit)(initial_positions, epsilon)

# Quantum Fourier Transform (QFT) function for a single Qutrit
def qft(qutrit, N=3):
    return [np.exp(-2j * np.pi * k * qutrit / N) / np.sqrt(N) for k in range(N)]

# Apply QFT to encoded positions
qft_positions = [qft(q) for q in encoded_positions.flatten()]

# Simplify the Hamiltonian for diagonal approximation
def simplified_hamiltonian(masses):
    return np.diag([-G * m for m in masses])

# Compute simplified Hamiltonian
simplified_hamiltonian_matrix = simplified_hamiltonian(masses)

# Simplify quantum state to match three bodies
quantum_state_final = np.array([np.mean(qft) for qft in qft_positions[:3]])

# Function to evolve the system
def evolve_with_simplified_hamiltonian(hamiltonian, state, time_step):
    unitary = np.eye(len(state)) - 1j * np.diag(hamiltonian) * time_step
    return np.dot(unitary, state)

# Perform quantum evolution
evolved_state_final = evolve_with_simplified_hamiltonian(simplified_hamiltonian_matrix, quantum_state_final, time_step)

# Classical function to compute derivatives (Newtonian motion)
def three_body_derivatives(t, y, masses):
    positions = y[:9].reshape((3, 3))  # First 9 values are positions
    velocities = y[9:].reshape((3, 3))  # Last 9 values are velocities
    derivatives = np.zeros_like(y)
    G = 6.67430e-11  # Gravitational constant

    for i in range(3):  # For each body
        force = np.zeros(3)
        for j in range(3):  # Interactions with other bodies
            if i != j:
                distance = np.linalg.norm(positions[i] - positions[j])
                force += -G * masses[j] * (positions[i] - positions[j]) / (distance ** 3)
        derivatives[9 + 3 * i:12 + 3 * i] = force  # Update accelerations
    derivatives[:9] = velocities.flatten()  # Velocities are derivatives of positions
    return derivatives

# Solve for one time step using classical solver
y0 = np.hstack((initial_positions.flatten(), initial_velocities.flatten()))
time_span = (0, 1)  # Simulate from t=0 to t=1 (arbitrary units)
result = solve_ivp(three_body_derivatives, time_span, y0, args=(masses,))

# Extract classical results
final_positions = result.y[:9, -1].reshape((3, 3))

# Results comparison
results_final = {
    "Simplified Hamiltonian (Diagonal)": simplified_hamiltonian_matrix.tolist(),
    "Initial Simplified Quantum State": quantum_state_final.tolist(),
    "Evolved Simplified Quantum State": evolved_state_final.tolist(),
    "Classical Final Positions": final_positions.tolist(),
}

def display_dataframe_to_user(name, dataframe):
    """
    Display a DataFrame with a given name for analysis.

    Args:
        name (str): The name/title of the DataFrame analysis.
        dataframe (pd.DataFrame): The DataFrame to display.

    Returns:
        None
    """
    print(f"\n=== {name} ===\n")
    print(dataframe.to_string(index=False))  # Display DataFrame as a formatted string

# Example DataFrame
optimal_thresholds = pd.DataFrame({
    "Noise Level": [0.1, 0.2, 0.3, 0.4],
    "Accuracy (%)": [92.5, 88.1, 85.7, 80.2],
    "Threshold": [0.15, 0.25, 0.35, 0.45]
})

# Reformat results for display
results_formatted = {
    "Metric": [
        "Simplified Hamiltonian (Diagonal)",
        "Initial Simplified Quantum State",
        "Evolved Simplified Quantum State",
        "Classical Final Positions",
    ],
    "Values": [
        simplified_hamiltonian_matrix.tolist(),
        quantum_state_final.tolist(),
        evolved_state_final.tolist(),
        final_positions.tolist(),
    ]
}

# Convert to a DataFrame for structured display
results_df = pd.DataFrame(results_formatted)

# Display final results using the function
# display_dataframe_to_user("Quantum-Classical Comparison Results", results_df)

# Function to compute the improved Hamiltonian with off-diagonal interaction terms
def compute_improved_hamiltonian(positions, masses):
    n = len(masses)
    hamiltonian = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = np.linalg.norm(positions[i] - positions[j])
                hamiltonian[i, j] = -G * masses[i] * masses[j] / (distance ** 3)
    return hamiltonian

# Compute the improved Hamiltonian using initial positions
improved_hamiltonian_matrix = compute_improved_hamiltonian(initial_positions, masses)

# Function to evolve the system using the improved Hamiltonian
def evolve_with_improved_hamiltonian(hamiltonian, state, time_step):
    unitary = np.eye(len(state), dtype=np.complex128) - 1j * hamiltonian * time_step  # First-order approximation
    return np.dot(unitary, state)

# Reinitialize the quantum state for three bodies
quantum_state_improved = np.array([np.mean(qft) for qft in qft_positions[:3]])

# Perform quantum evolution with the improved Hamiltonian
evolved_state_improved = evolve_with_improved_hamiltonian(improved_hamiltonian_matrix, quantum_state_improved, time_step)

refined_results_df = pd.DataFrame({
    "Metric": [
        "Improved Hamiltonian Matrix",
        "Initial Quantum State (Improved)",
        "Evolved Quantum State (Improved)",
        "Classical Final Positions"
    ],
    "Values": [
        improved_hamiltonian_matrix.tolist(),
        quantum_state_improved.tolist(),
        evolved_state_improved.tolist(),
        final_positions.tolist()
    ]
})

# Display the refined results as a DataFrame for analysis
# display_dataframe_to_user("Refined Quantum Simulation Results", refined_results_df)

# Function to compute the relativistic Hamiltonian with variable masses
def compute_relativistic_hamiltonian(positions, velocities, masses, time, c=3e8):
    n = len(masses)
    hamiltonian = np.zeros((n, n), dtype=np.complex128)
    
    # Update masses dynamically (example: mass loss over time)
    mass_loss_rate = 1e20  # Mass loss rate (arbitrary example, kg/s)
    updated_masses = [m - mass_loss_rate * time for m in masses]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = np.linalg.norm(positions[i] - positions[j])
                relative_velocity = np.linalg.norm(velocities[i] - velocities[j])
                # Relativistic corrections in potential energy
                hamiltonian[i, j] = -G * updated_masses[i] * updated_masses[j] / distance * (
                    1 + (relative_velocity ** 2) / (c ** 2) + (G * masses[i]) / (c ** 2 * distance)
                )
    return hamiltonian

# Example parameters for time and velocities
time_elapsed = 1  # 1 second
initial_velocities = np.array([
    [-0.8, 0.0, 0.6],   # Body 1
    [0.2, -1.0, -0.4],  # Body 2
    [0.5, 1.2, -0.3]    # Body 3
])

# Compute the refined Hamiltonian
refined_hamiltonian_matrix = compute_relativistic_hamiltonian(
    initial_positions, initial_velocities, masses, time_elapsed
)

# Display the refined Hamiltonian for analysis
# print(refined_hamiltonian_matrix.tolist())

# Function to map quantum amplitudes to probabilistic positions
def map_quantum_amplitudes_to_positions(quantum_state):
    # Calculate probabilities from quantum amplitudes
    probabilities = np.abs(quantum_state) ** 2
    probabilities /= np.sum(probabilities)  # Normalize probabilities
    
    # Define a discrete spatial grid for sampling (arbitrary example)
    spatial_grid = np.linspace(-10, 10, len(probabilities))
    
    # Sample positions based on the probability distribution
    sampled_positions = np.random.choice(spatial_grid, size=len(probabilities), p=probabilities)
    
    return sampled_positions

# Example: Use the evolved quantum state to extract positions
sampled_positions = map_quantum_amplitudes_to_positions(evolved_state_improved)

# Combine results for comparison
# Reorganize results into a structured DataFrame for clarity
trajectory_results_df = pd.DataFrame({
    "Metric": ["Sampled Quantum Positions", "Classical Final Positions"],
    "Values": [sampled_positions.tolist(), final_positions.tolist()]
})

# Display the trajectory mapping results
# display_dataframe_to_user("Quantum Trajectory Mapping Results", trajectory_results_df)

# Extend to N-body system (e.g., N=5 bodies)
N = 5

# Generate random initial positions and velocities for N bodies
np.random.seed(42)  # For reproducibility
initial_positions_N = np.random.uniform(-10, 10, (N, 3))  # Random positions in 3D space
initial_velocities_N = np.random.uniform(-1, 1, (N, 3))  # Random velocities in 3D space
masses_N = np.random.uniform(1e24, 1e26, N)  # Random masses between 1e24 and 1e26 kg

# Compute the Hamiltonian for N-body system with relativistic corrections
refined_hamiltonian_N = compute_relativistic_hamiltonian(
    initial_positions_N, initial_velocities_N, masses_N, time_elapsed
)

# Initialize quantum state for N bodies
quantum_state_N = np.random.random(N) + 1j * np.random.random(N)  # Random complex amplitudes
quantum_state_N /= np.linalg.norm(quantum_state_N)  # Normalize the quantum state

# Evolve the quantum system with the refined Hamiltonian for N bodies
evolved_state_N = evolve_with_improved_hamiltonian(refined_hamiltonian_N, quantum_state_N, time_step)

# Reorganize the results into a structured DataFrame for better clarity and display
multi_body_results_df = pd.DataFrame({
    "Metric": [
        "Initial Positions (N-Bodies)",
        "Initial Velocities (N-Bodies)",
        "Refined Hamiltonian (N-Bodies)",
        "Initial Quantum State (N-Bodies)",
        "Evolved Quantum State (N-Bodies)"
    ],
    "Values": [
        initial_positions_N.tolist(),
        initial_velocities_N.tolist(),
        refined_hamiltonian_N.tolist(),
        quantum_state_N.tolist(),
        evolved_state_N.tolist()
    ]
})

# Parameters for multi-time-step simulation
time_steps = 10  # Number of steps
delta_t = 1  # Time step size (arbitrary units)

# Display the multi-body simulation results in a structured format
# display_dataframe_to_user("Multi-Body Quantum Simulation Results", multi_body_results_df)

import numpy as np

# Quantum Evolution Functions
def evolve_with_improved_hamiltonian(hamiltonian, state, delta_t):
    """
    Evolve the quantum state using the improved Hamiltonian with a first-order approximation.
    """
    unitary = np.eye(len(state), dtype=np.complex128) - 1j * hamiltonian * delta_t
    return np.dot(unitary, state)

def multi_time_step_evolution_rescaled(hamiltonian, initial_state, time_steps, delta_t):
    """
    Perform quantum evolution over multiple time steps with amplitude rescaling.
    """
    states = [initial_state]  # Store states at each time step
    current_state = initial_state

    for _ in range(time_steps):
        current_state = evolve_with_improved_hamiltonian(hamiltonian, current_state, delta_t)
        current_state /= np.linalg.norm(current_state)  # Rescale for stability
        states.append(current_state)

    return states

# Trajectory Extraction Functions
def extract_trajectories_with_rescaling(states, spatial_grid):
    """
    Extract probabilistic positions from quantum states over time using amplitude probabilities.
    """
    trajectories = []
    for state in states:
        probabilities = np.abs(state) ** 2
        probabilities /= np.sum(probabilities)  # Normalize safely
        sampled_positions = np.random.choice(spatial_grid, size=len(probabilities), p=probabilities)
        trajectories.append(sampled_positions.tolist())
    return trajectories

# Example Multi-Time-Step Simulation for N-Bodies
N = 5  # Number of bodies
time_steps = 10  # Number of time steps
delta_t = 1.0  # Time step size (arbitrary units)

# Example Initial Parameters
np.random.seed(42)  # For reproducibility
initial_positions_N = np.random.uniform(-10, 10, (N, 3))  # Random positions
initial_velocities_N = np.random.uniform(-1, 1, (N, 3))  # Random velocities
masses_N = np.random.uniform(1e24, 1e26, N)  # Random masses

# Example Hamiltonian (Relativistic and Inter-Body Interactions)
def compute_relativistic_hamiltonian(positions, velocities, masses, time, c=3e8):
    n = len(masses)
    hamiltonian = np.zeros((n, n), dtype=np.complex128)
    mass_loss_rate = 1e20  # Example mass loss rate (kg/s)
    updated_masses = [m - mass_loss_rate * time for m in masses]

    for i in range(n):
        for j in range(n):
            if i != j:
                distance = np.linalg.norm(positions[i] - positions[j])
                relative_velocity = np.linalg.norm(velocities[i] - velocities[j])
                hamiltonian[i, j] = -G * updated_masses[i] * updated_masses[j] / distance * (
                    1 + (relative_velocity ** 2) / (c ** 2) + (G * masses[i]) / (c ** 2 * distance)
                )
    return hamiltonian

# Compute Hamiltonian
time_elapsed = 1  # Example elapsed time
G = 6.67430e-11  # Gravitational constant
refined_hamiltonian_N = compute_relativistic_hamiltonian(
    initial_positions_N, initial_velocities_N, masses_N, time_elapsed
)

# Initialize Quantum State
quantum_state_N = np.random.random(N) + 1j * np.random.random(N)  # Random complex amplitudes
quantum_state_N /= np.linalg.norm(quantum_state_N)  # Normalize the initial state

# Perform Multi-Time-Step Evolution
evolved_states_over_time = multi_time_step_evolution_rescaled(
    refined_hamiltonian_N, quantum_state_N, time_steps, delta_t
)

# Define Spatial Grid for Trajectory Mapping
spatial_grid = np.linspace(-10, 10, N)

# Extract Quantum Trajectories
quantum_trajectories = extract_trajectories_with_rescaling(evolved_states_over_time, spatial_grid)

# Organize Results for Analysis
# Convert dictionary to a DataFrame for structured display
multi_time_step_results_df = pd.DataFrame({
    "Metric": ["Quantum Trajectories Over Time", "Initial Positions (N-Bodies)", "Initial Velocities (N-Bodies)", "Refined Hamiltonian (N-Bodies)"],
    "Values": [quantum_trajectories, initial_positions_N.tolist(), initial_velocities_N.tolist(), refined_hamiltonian_N.tolist()]
})

# display_dataframe_to_user("Multi-Time-Step Quantum Trajectories", multi_time_step_results_df)

# Compare quantum trajectories with classical results
def compare_quantum_and_classical(quantum_trajectories, classical_positions):
    """
    Overlay quantum and classical trajectories for comparison.
    """
    time_steps = len(quantum_trajectories)
    body_count = len(classical_positions)

    plt.figure(figsize=(10, 6))
    for body in range(body_count):
        # Quantum Trajectories for a body
        quantum_trajectory = [step[body] for step in quantum_trajectories]

        # Classical Position (constant for simplicity in this example)
        classical_position = [classical_positions[body][0]] * time_steps

        # Plot Quantum Trajectory
        plt.plot(range(time_steps), quantum_trajectory, label=f"Quantum Body {body + 1}")

        # Plot Classical Trajectory
        plt.plot(range(time_steps), classical_position, '--', label=f"Classical Body {body + 1}")

    plt.title("Quantum vs. Classical Trajectories Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Position")
    plt.legend()
    plt.grid()
    plt.show()

# Visualize quantum trajectories
def visualize_quantum_dynamics(quantum_trajectories):
    """
    Plot the quantum trajectories over time to observe dynamic behavior.
    """
    time_steps = len(quantum_trajectories)
    body_count = len(quantum_trajectories[0])

    plt.figure(figsize=(10, 6))
    for body in range(body_count):
        trajectory = [step[body] for step in quantum_trajectories]
        plt.plot(range(time_steps), trajectory, label=f"Quantum Body {body + 1}")

    plt.title("Quantum Trajectories Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Position")
    plt.legend()
    plt.grid()
    plt.show()

# Recompute the quantum trajectories using the available functions and data

# Perform multi-time-step evolution with rescaling
evolved_states_over_time_rescaled = multi_time_step_evolution_rescaled(
    refined_hamiltonian_N, quantum_state_N, time_steps, delta_t
)

# Extract quantum trajectories from the rescaled states
quantum_trajectories_rescaled = extract_trajectories_with_rescaling(
    evolved_states_over_time_rescaled, spatial_grid
)

# Define classical positions for comparison
classical_positions = final_positions.tolist()

# Compare and visualize
# compare_quantum_and_classical(quantum_trajectories_rescaled, classical_positions)
# visualize_quantum_dynamics(quantum_trajectories_rescaled)

# Refine quantum-to-classical mapping by incorporating phase information
def refined_trajectory_mapping(states, spatial_grid, smoothing_factor=0.1):
    """
    Refine trajectory mapping by incorporating phase dynamics and smoothing.
    """
    trajectories = []
    for state in states:
        probabilities = np.abs(state) ** 2
        probabilities /= np.sum(probabilities)  # Normalize probabilities
        
        # Incorporate phase information for directional adjustment
        phases = np.angle(state)  # Extract phase from the complex state
        directional_influence = np.sign(phases) * smoothing_factor
        
        # Compute positions with phase adjustment
        sampled_positions = np.random.choice(spatial_grid, size=len(probabilities), p=probabilities)
        adjusted_positions = sampled_positions + directional_influence  # Adjust by phase
        trajectories.append(adjusted_positions.tolist())
    
    return trajectories

# Refined multi-time-step evolution with adjusted mapping
refined_quantum_trajectories = refined_trajectory_mapping(
    evolved_states_over_time_rescaled, spatial_grid, smoothing_factor=0.2
)

# Compare and visualize refined results
# compare_quantum_and_classical(refined_quantum_trajectories, classical_positions)
# visualize_quantum_dynamics(refined_quantum_trajectories)

# Refine the implementation to handle 3D positions for comparison
def iterative_refinement_mapping_3d(states, spatial_grid, classical_positions, iterations=10, smoothing_factor=0.1):
    """
    Use iterative refinement to map quantum states to 3D positions and compare with classical trajectories.

    Parameters:
    - states: list of numpy.ndarray
        Quantum states over time steps.
    - spatial_grid: numpy.ndarray
        Discrete spatial positions for mapping.
    - classical_positions: list of list
        Classical positions to compare against (3D coordinates per body).
    - iterations: int
        Number of iterations for refinement.
    - smoothing_factor: float
        Weight for phase-based positional adjustments.

    Returns:
    - refined_trajectories: list of list
        Refined quantum trajectories over time for each spatial dimension.
    - deviation_log: list of float
        Log of deviations between quantum and classical trajectories.
    """
    refined_trajectories = []
    deviation_log = []

    for time_step, state in enumerate(states):
        probabilities = np.abs(state) ** 2
        probabilities /= np.sum(probabilities)  # Normalize probabilities

        # Initialize refined positions for 3D space
        refined_positions = np.random.choice(spatial_grid, size=(len(probabilities), 3))
        for _ in range(iterations):
            # Compare with classical positions for the current time step
            classical_positions_step = classical_positions[min(time_step, len(classical_positions) - 1)]
            deviations = refined_positions - classical_positions_step
            deviation_norm = np.linalg.norm(deviations, axis=1).mean()
            deviation_log.append(deviation_norm)

            # Adjust positions based on phase and deviations (dimension-wise)
            phases = np.angle(state)
            for dim in range(3):  # For x, y, z dimensions
                refined_positions[:, dim] -= smoothing_factor * np.sign(phases) * deviations[:, dim]

        refined_trajectories.append(refined_positions.tolist())

    return refined_trajectories, deviation_log

# Perform iterative refinement mapping for 3D positions
refined_quantum_trajectories_3d, deviation_log_3d = iterative_refinement_mapping_3d(
    evolved_states_over_time_rescaled, spatial_grid, classical_positions, iterations=10, smoothing_factor=0.2
)

# Visualize refined quantum dynamics for 3D trajectories
# for dim, label in enumerate(['x', 'y', 'z']):
#     dim_trajectories = [[pos[dim] for pos in step] for step in refined_quantum_trajectories_3d]
#     plt.figure(figsize=(8, 5))
#     for body in range(len(dim_trajectories[0])):
#         trajectory = [step[body] for step in dim_trajectories]
#         plt.plot(range(len(dim_trajectories)), trajectory, label=f"Body {body + 1}")
#     plt.title(f"Refined Quantum Dynamics in {label}-dimension")
#     plt.xlabel("Time Step")
#     plt.ylabel(f"{label}-Position")
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Plot deviation log
# plt.figure(figsize=(8, 5))
# plt.plot(range(len(deviation_log_3d)), deviation_log_3d, label="Deviation Norm")
# plt.title("Deviation Log During Iterative Refinement (3D)")
# plt.xlabel("Iteration")
# plt.ylabel("Deviation Norm")
# plt.grid()
# plt.legend()
# plt.show()
# Extend the simulation duration and refine trajectories for long-term convergence
extended_time_steps = 50  # Increase the number of time steps for long-term simulation

# Perform extended multi-time-step evolution
evolved_states_extended = multi_time_step_evolution_rescaled(
    refined_hamiltonian_N, quantum_state_N, extended_time_steps, delta_t
)

# Refine trajectories for the extended simulation
refined_quantum_trajectories_extended, deviation_log_extended = iterative_refinement_mapping_3d(
    evolved_states_extended, spatial_grid, classical_positions, iterations=10, smoothing_factor=0.2
)

# Visualize refined quantum dynamics for extended simulation
for dim, label in enumerate(['x', 'y', 'z']):
    dim_trajectories_extended = [[pos[dim] for pos in step] for step in refined_quantum_trajectories_extended]
    plt.figure(figsize=(10, 6))
    for body in range(len(dim_trajectories_extended[0])):
        trajectory = [step[body] for step in dim_trajectories_extended]
        plt.plot(range(len(dim_trajectories_extended)), trajectory, label=f"Body {body + 1}")
    plt.title(f"Extended Refined Quantum Dynamics in {label}-dimension")
    plt.xlabel("Time Step")
    plt.ylabel(f"{label}-Position")
    plt.legend()
    plt.grid()
    plt.show()

# Plot deviation log for extended simulation
plt.figure(figsize=(10, 6))
plt.plot(range(len(deviation_log_extended)), deviation_log_extended, label="Deviation Norm (Extended)")
plt.title("Deviation Log During Extended Iterative Refinement (3D)")
plt.xlabel("Iteration")
plt.ylabel("Deviation Norm")
plt.grid()
plt.legend()
plt.show()

# Introduce parallel processing for faster refinement on higher-dimensional systems
from joblib import Parallel, delayed

def parallel_refinement(state, spatial_grid, classical_positions_step, iterations=10, smoothing_factor=0.1):
    probabilities = np.abs(state) ** 2
    probabilities /= np.sum(probabilities)  # Normalize probabilities

    refined_positions = np.random.choice(spatial_grid, size=(len(probabilities), 3))
    for _ in range(iterations):
        deviations = refined_positions - classical_positions_step
        phases = np.angle(state)
        for dim in range(3):  # For x, y, z dimensions
            refined_positions[:, dim] -= smoothing_factor * np.sign(phases) * deviations[:, dim]
    return refined_positions

# Apply parallel refinement for each time step
refined_quantum_trajectories_parallel = Parallel(n_jobs=-1)(
    delayed(parallel_refinement)(state, spatial_grid, classical_positions[min(time_step, len(classical_positions) - 1)])
    for time_step, state in enumerate(evolved_states_extended)
)

# Visualize parallel refinement results (optional)
for dim, label in enumerate(['x', 'y', 'z']):
    dim_trajectories_parallel = [[pos[dim] for pos in step] for step in refined_quantum_trajectories_parallel]
    plt.figure(figsize=(10, 6))
    for body in range(len(dim_trajectories_parallel[0])):
        trajectory = [step[body] for step in dim_trajectories_parallel]
        plt.plot(range(len(dim_trajectories_parallel)), trajectory, label=f"Body {body + 1}")
    plt.title(f"Parallel Refined Quantum Dynamics in {label}-dimension")
    plt.xlabel("Time Step")
    plt.ylabel(f"{label}-Position")
    plt.legend()
    plt.grid()
    plt.show()

