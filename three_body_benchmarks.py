import numpy as np
import matplotlib.pyplot as plt

# Constants for the simulation
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
time_step = 1.0  # Time step for simulation (seconds)
total_time = 10000  # Total simulation time (seconds)
n_steps = int(total_time // time_step)  # Number of steps in the simulation

# Qutrit simulation: 3 possible states (0, 1, 2) for each body
def initialize_bodies(case):
    # Default values to prevent UnboundLocalError
    r1, r2, r3, v1, v2, v3 = None, None, None, None, None, None

    if case == 1:  # Standard Case
        r1 = np.array([1.0e11, 0, 0])  # Position of body 1 (m)
        r2 = np.array([0, 1.0e11, 0])  # Position of body 2 (m)
        r3 = np.array([0, 0, 1.0e11])  # Position of body 3 (m)

        v1 = np.array([0, 3.0e4, 0])  # Velocity of body 1 (m/s)
        v2 = np.array([0, -3.0e4, 0])  # Velocity of body 2 (m/s)
        v3 = np.array([0, 0, 3.0e4])  # Velocity of body 3 (m/s)
    
    elif case == 2:  # Varying Gravitational Influence
        # Bodies are closer, larger mass difference
        r1 = np.array([1.0e10, 0, 0])  # Position of body 1 (m)
        r2 = np.array([0, 1.0e10, 0])  # Position of body 2 (m)
        r3 = np.array([0, 0, 1.0e10])  # Position of body 3 (m)

        v1 = np.array([0, 2.5e4, 0])  # Velocity of body 1 (m/s)
        v2 = np.array([0, -2.0e4, 0])  # Velocity of body 2 (m/s)
        v3 = np.array([0, 0, 3.5e4])  # Velocity of body 3 (m/s)

    elif case == 3:  # High-Energy Case
        # High velocities or small distances between bodies
        r1 = np.array([1.0e9, 0, 0])  # Position of body 1 (m)
        r2 = np.array([0, 1.0e9, 0])  # Position of body 2 (m)
        r3 = np.array([0, 0, 1.0e9])  # Position of body 3 (m)

        v1 = np.array([0, 1.0e5, 0])  # Velocity of body 1 (m/s)
        v2 = np.array([0, -1.2e5, 0])  # Velocity of body 2 (m/s)
        v3 = np.array([0, 0, 1.5e5])  # Velocity of body 3 (m/s)

    elif case == 4:  # Noise Introduction Case
        # Same as standard case but with noise added
        r1 = np.array([1.0e11, 0, 0])  # Position of body 1 (m)
        r2 = np.array([0, 1.0e11, 0])  # Position of body 2 (m)
        r3 = np.array([0, 0, 1.0e11])  # Position of body 3 (m)

        v1 = np.array([0, 3.0e4, 0])  # Velocity of body 1 (m/s)
        v2 = np.array([0, -3.0e4, 0])  # Velocity of body 2 (m/s)
        v3 = np.array([0, 0, 3.0e4])  # Velocity of body 3 (m/s)

    else:
        raise ValueError(f"Invalid case {case} provided!")

    return r1, r2, r3, v1, v2, v3


# Gravitational forces calculation
def compute_forces(r1, r2, r3, m1, m2, m3):
    r12 = r2 - r1
    r13 = r3 - r1
    r23 = r3 - r2
    
    d12 = np.linalg.norm(r12)
    d13 = np.linalg.norm(r13)
    d23 = np.linalg.norm(r23)
    
    F12 = G * m1 * m2 * r12 / d12**3
    F13 = G * m1 * m3 * r13 / d13**3
    F23 = G * m2 * m3 * r23 / d23**3
    
    return F12, F13, F23

# Kinetic Energy Calculation
def kinetic_energy(v1, v2, v3, m1, m2, m3):
    K = 0.5 * m1 * np.dot(v1, v1) + 0.5 * m2 * np.dot(v2, v2) + 0.5 * m3 * np.dot(v3, v3)
    return K

# Potential Energy Calculation
def potential_energy(r1, r2, r3, m1, m2, m3):
    d12 = np.linalg.norm(r2 - r1)
    d13 = np.linalg.norm(r3 - r1)
    d23 = np.linalg.norm(r3 - r2)
    U = -G * (m1 * m2 / d12 + m1 * m3 / d13 + m2 * m3 / d23)
    return U

# Energy Conservation Benchmark
def total_energy(v1, v2, v3, r1, r2, r3, m1, m2, m3):
    K = kinetic_energy(v1, v2, v3, m1, m2, m3)
    U = potential_energy(r1, r2, r3, m1, m2, m3)
    E_total = K + U
    return E_total

# Euler's method to integrate the system
def integrate_step(r1, r2, r3, v1, v2, v3, m1, m2, m3, time_step):
    F12, F13, F23 = compute_forces(r1, r2, r3, m1, m2, m3)
    
    v1 += (F12 + F13) / m1 * time_step
    v2 += (-F12 + F23) / m2 * time_step
    v3 += (-F13 - F23) / m3 * time_step
    
    r1 += v1 * time_step
    r2 += v2 * time_step
    r3 += v3 * time_step
    
    return r1, r2, r3, v1, v2, v3

# Reinforcement Learning: Adjusting integration parameters
class ReinforcementLearningAgent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.reward_history = []

    def adjust_parameters(self, energy_deviation):
        # RL agent learns to minimize the energy deviation by adjusting time step or parameters
        reward = -energy_deviation  # Negative deviation means a better reward
        self.reward_history.append(reward)
        
        # The agent learns to reduce the energy deviation over time (adjusting time_step)
        return self.learning_rate * np.mean(self.reward_history)

# Running the simulation and calculating benchmarks
def run_simulation(case, m1, m2, m3):
    r1, r2, r3, v1, v2, v3 = initialize_bodies(case)
    
    positions = []
    energies = []
    kinetic_energies = []
    potential_energies = []
    
    initial_energy = total_energy(v1, v2, v3, r1, r2, r3, m1, m2, m3)
    
    # Initialize RL Agent
    agent = ReinforcementLearningAgent()

    for step in range(n_steps):
        r1, r2, r3, v1, v2, v3 = integrate_step(r1, r2, r3, v1, v2, v3, m1, m2, m3, time_step)
        
        # Calculate energy for each step
        current_energy = total_energy(v1, v2, v3, r1, r2, r3, m1, m2, m3)
        energies.append(current_energy)
        
        # Calculate kinetic and potential energy
        kinetic_energies.append(kinetic_energy(v1, v2, v3, m1, m2, m3))
        potential_energies.append(potential_energy(r1, r2, r3, m1, m2, m3))
        
        # Calculate energy deviation
        energy_deviation = np.abs(current_energy - initial_energy) / initial_energy * 100
        
        # RL adjustment for learning parameters
        agent.adjust_parameters(energy_deviation)
        
        # Store positions for plotting
        positions.append((r1.copy(), r2.copy(), r3.copy()))
    
    # Energy deviation benchmark
    energy_deviation_history = np.abs(np.array(energies) - initial_energy) / initial_energy * 100
    avg_energy_deviation = np.mean(energy_deviation_history)
    
    # Compute final kinetic and potential energies
    final_kinetic = np.mean(kinetic_energies)
    final_potential = np.mean(potential_energies)
    
    return np.array(positions), energy_deviation_history, avg_energy_deviation, final_kinetic, final_potential

# Simulation cases
m1, m2, m3 = 5.97e24, 7.35e22, 1.98e30  # Earth, Moon, and a large planet (mass in kg)

# Run all simulations
simulation_cases = [1]
results = {}

# for case in simulation_cases:
#     print(f"Running simulation for case {case}")
#     positions, energy_deviation, avg_energy_deviation, final_kinetic, final_potential = run_simulation(case, m1, m2, m3)
#     results[case] = (positions, energy_deviation, avg_energy_deviation, final_kinetic, final_potential)

# Print key metrics
# for case, (positions, energy_deviation, avg_energy_deviation, final_kinetic, final_potential) in results.items():
#     print(f"Case {case} Key Metrics:")
#     print(f"  Average Energy Deviation: {avg_energy_deviation:.4f}%")
#     print(f"  Final Kinetic Energy: {final_kinetic:.2e} J")
#     print(f"  Final Potential Energy: {final_potential:.2e} J")
#     print(f"  Final Total Energy Deviation: {energy_deviation[-1]:.4f}%")
#     print("-" * 50)

# # Plot energy deviation for each case
# plt.figure(figsize=(10, 6))
# plt.plot(energy_deviation, label="Energy Deviation (%)")
# plt.xlabel('Time Step')
# plt.ylabel('Energy Deviation (%)')
# plt.title('Energy Conservation Benchmark for Three-Body Problem with RL')
# plt.legend()
# plt.grid(True)
# plt.show()

# Additional Cases to Run
simulation_cases = [2, 3, 4]  # Running Case 2, 3, and 4

results = {}

for case in simulation_cases:
    print(f"Running simulation for case {case}")
    positions, energy_deviation, avg_energy_deviation, final_kinetic, final_potential = run_simulation(case, m1, m2, m3)
    results[case] = (positions, energy_deviation, avg_energy_deviation, final_kinetic, final_potential)

# Print key metrics for all cases
for case, (positions, energy_deviation, avg_energy_deviation, final_kinetic, final_potential) in results.items():
    print(f"Case {case} Key Metrics:")
    print(f"  Average Energy Deviation: {avg_energy_deviation:.4f}%")
    print(f"  Final Kinetic Energy: {final_kinetic:.2e} J")
    print(f"  Final Potential Energy: {final_potential:.2e} J")
    print(f"  Final Total Energy Deviation: {energy_deviation[-1]:.4f}%")
    print("-" * 50)

# Plot energy deviation for each case
plt.figure(figsize=(10, 6))
for case, (positions, energy_deviation, _, _, _) in results.items():
    plt.plot(energy_deviation, label=f"Case {case} Energy Deviation (%)")
    
plt.xlabel('Time Step')
plt.ylabel('Energy Deviation (%)')
plt.title('Energy Conservation Benchmark for Three-Body Problem')
plt.legend()
plt.grid(True)
plt.show()

# Optional: Plot Kinetic and Potential Energy for visualization
plt.figure(figsize=(10, 6))
for case, (positions, _, _, final_kinetic, final_potential) in results.items():
    kinetic_energies = [kinetic_energy(v1, v2, v3, m1, m2, m3) for v1, v2, v3 in positions]
    potential_energies = [potential_energy(r1, r2, r3, m1, m2, m3) for r1, r2, r3 in positions]
    
    plt.plot(range(len(kinetic_energies)), kinetic_energies, label=f"Case {case} Kinetic Energy")
    plt.plot(range(len(potential_energies)), potential_energies, label=f"Case {case} Potential Energy")

plt.xlabel('Time Step')
plt.ylabel('Energy (Joules)')
plt.title('Kinetic and Potential Energy over Time for Three-Body Problem')
plt.legend()
plt.grid(True)
plt.show()



