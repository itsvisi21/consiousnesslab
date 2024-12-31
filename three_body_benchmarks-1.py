import numpy as np
import matplotlib.pyplot as plt

# Constants for the simulation
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
time_step = 1.0  # Time step for simulation (seconds)
total_time = 10000  # Total simulation time (seconds)
n_steps = int(total_time // time_step)  # Number of steps in the simulation

# Qutrit simulation: 3 possible states (0, 1, 2) for each body
def initialize_bodies(case):
    r1, r2, r3, v1, v2, v3 = None, None, None, None, None, None
    if case == 1:  # Standard Case
        r1 = np.array([1.0e11, 0, 0])  # Position of body 1 (m)
        r2 = np.array([0, 1.0e11, 0])  # Position of body 2 (m)
        r3 = np.array([0, 0, 1.0e11])  # Position of body 3 (m)
        v1 = np.array([0, 3.0e4, 0])  # Velocity of body 1 (m/s)
        v2 = np.array([0, -3.0e4, 0])  # Velocity of body 2 (m/s)
        v3 = np.array([0, 0, 3.0e4])  # Velocity of body 3 (m)
    elif case == 2:  # Varying Gravitational Influence
        r1 = np.array([1.0e10, 0, 0])  # Position of body 1 (m)
        r2 = np.array([0, 1.0e10, 0])  # Position of body 2 (m)
        r3 = np.array([0, 0, 1.0e10])  # Position of body 3 (m)
        v1 = np.array([0, 2.5e4, 0])  # Velocity of body 1 (m/s)
        v2 = np.array([0, -2.0e4, 0])  # Velocity of body 2 (m/s)
        v3 = np.array([0, 0, 3.5e4])  # Velocity of body 3 (m/s)
    elif case == 3:  # High-Energy Case
        r1 = np.array([1.0e9, 0, 0])  # Position of body 1 (m)
        r2 = np.array([0, 1.0e9, 0])  # Position of body 2 (m)
        r3 = np.array([0, 0, 1.0e9])  # Position of body 3 (m)
        v1 = np.array([0, 1.0e5, 0])  # Velocity of body 1 (m/s)
        v2 = np.array([0, -1.2e5, 0])  # Velocity of body 2 (m/s)
        v3 = np.array([0, 0, 1.5e5])  # Velocity of body 3 (m/s)
    elif case == 4:  # Noise Introduction Case
        r1 = np.array([1.0e11, 0, 0])  # Position of body 1 (m)
        r2 = np.array([0, 1.0e11, 0])  # Position of body 2 (m)
        r3 = np.array([0, 0, 1.0e11])  # Position of body 3 (m)
        v1 = np.array([0, 3.0e4, 0])  # Velocity of body 1 (m/s)
        v2 = np.array([0, -3.0e4, 0])  # Velocity of body 2 (m/s)
        v3 = np.array([0, 0, 3.0e4])  # Velocity of body 3 (m)
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

# Runge-Kutta (4th Order) integration method
def runge_kutta_step(r1, r2, r3, v1, v2, v3, m1, m2, m3, time_step):
    F12, F13, F23 = compute_forces(r1, r2, r3, m1, m2, m3)
    
    k1v1 = F12 + F13 / m1
    k1v2 = (-F12 + F23) / m2
    k1v3 = (-F13 - F23) / m3
    
    k2v1 = F12 + F13 / m1
    k2v2 = (-F12 + F23) / m2
    k2v3 = (-F13 - F23) / m3
    
    # Implementing Runge-Kutta steps
    return r1, r2, r3, v1, v2, v3

# Introduce noise to the system
def add_noise(r1, r2, r3, v1, v2, v3, noise_level):
    """Introduce noise into the system."""
    noise_r = np.random.normal(0, noise_level, 3)  # Noise for positions
    noise_v = np.random.normal(0, noise_level, 3)  # Noise for velocities
    r1 += noise_r
    r2 += noise_r
    r3 += noise_r
    v1 += noise_v
    v2 += noise_v
    v3 += noise_v
    return r1, r2, r3, v1, v2, v3

# Running the simulation and calculating benchmarks
def run_simulation(case, m1, m2, m3, noise_level=0.0):
    r1, r2, r3, v1, v2, v3 = initialize_bodies(case)
    
    # Add noise if specified
    r1, r2, r3, v1, v2, v3 = add_noise(r1, r2, r3, v1, v2, v3, noise_level)
    
    positions = []
    energies = []
    kinetic_energies = []
    potential_energies = []
    
    initial_energy = total_energy(v1, v2, v3, r1, r2, r3, m1, m2, m3)
    
    for step in range(n_steps):
        r1, r2, r3, v1, v2, v3 = runge_kutta_step(r1, r2, r3, v1, v2, v3, m1, m2, m3, time_step)
        
        current_energy = total_energy(v1, v2, v3, r1, r2, r3, m1, m2, m3)
        energies.append(current_energy)
        
        kinetic_energies.append(kinetic_energy(v1, v2, v3, m1, m2, m3))
        potential_energies.append(potential_energy(r1, r2, r3, m1, m2, m3))
        
        energy_deviation = np.abs(current_energy - initial_energy) / initial_energy * 100
        
        positions.append((r1.copy(), r2.copy(), r3.copy()))
    
    energy_deviation_history = np.abs(np.array(energies) - initial_energy) / initial_energy * 100
    avg_energy_deviation = np.mean(energy_deviation_history)
    
    final_kinetic = np.mean(kinetic_energies)
    final_potential = np.mean(potential_energies)
    
    return np.array(positions), energy_deviation_history, avg_energy_deviation, final_kinetic, final_potential

# Simulation cases
m1, m2, m3 = 5.97e24, 7.35e22, 1.98e30  # Earth, Moon, and a large planet (mass in kg)

simulation_cases = [1, 2, 3, 4]
results = {}

# Running the simulation for different cases and noise levels
# for case in simulation_cases:
#     for noise in [0.05, 0.10, 0.20, 0.30]:
#         print(f"Running simulation for case {case} with noise level {noise}")
#         positions, energy_deviation, avg_energy_deviation, final_kinetic, final_potential = run_simulation(case, m1, m2, m3, noise)
#         results[(case, noise)] = (positions, energy_deviation, avg_energy_deviation, final_kinetic, final_potential)
        
#         print(f"  Average Energy Deviation: {avg_energy_deviation:.4f}%")
#         print(f"  Final Kinetic Energy: {final_kinetic:.2e} J")
#         print(f"  Final Potential Energy: {final_potential:.2e} J")
#         print(f"  Final Total Energy Deviation: {energy_deviation[-1]:.4f}%")
#         print("-" * 50)

# Plot energy deviation for each case and noise level
# plt.figure(figsize=(10, 6))
# for (case, noise), (_, energy_deviation, _, _, _) in results.items():
#     plt.plot(energy_deviation, label=f"Case {case} Noise {noise} Energy Deviation")
    
# plt.xlabel('Time Step')
# plt.ylabel('Energy Deviation (%)')
# plt.title('Energy Deviation with Increasing Noise Levels')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Optional: Plot Kinetic and Potential Energy for visualization
# plt.figure(figsize=(10, 6))
# for (case, noise), (_, _, _, final_kinetic, final_potential) in results.items():
#     kinetic_energies = [kinetic_energy(v1, v2, v3, m1, m2, m3) for v1, v2, v3 in positions]
#     potential_energies = [potential_energy(r1, r2, r3, m1, m2, m3) for r1, r2, r3 in positions]
    
#     plt.plot(range(len(kinetic_energies)), kinetic_energies, label=f"Case {case} Kinetic Energy")
#     plt.plot(range(len(potential_energies)), potential_energies, label=f"Case {case} Potential Energy")

# plt.xlabel('Time Step')
# plt.ylabel('Energy (Joules)')
# plt.title('Kinetic and Potential Energy over Time for Three-Body Problem')
# plt.legend()
# plt.grid(True)
# plt.show()

# Plot Kinetic and Potential Energy for each case and noise level
# plt.figure(figsize=(10, 6))
# for (case, noise), (positions, _, _, _, _) in results.items():
#     kinetic_energies = [kinetic_energy(v1, v2, v3, m1, m2, m3) for v1, v2, v3 in positions]
#     potential_energies = [potential_energy(r1, r2, r3, m1, m2, m3) for r1, r2, r3 in positions]
    
#     plt.plot(range(len(kinetic_energies)), kinetic_energies, label=f"Case {case} Noise {noise} Kinetic Energy")
#     plt.plot(range(len(potential_energies)), potential_energies, label=f"Case {case} Noise {noise} Potential Energy")

# plt.xlabel('Time Step')
# plt.ylabel('Energy (Joules)')
# plt.title('Kinetic and Potential Energy over Time for Three-Body Problem')
# plt.legend()
# plt.grid(True)
# plt.show()

# Increased noise levels for testing robustness
noise_levels = [0.5, 1.0, 2.0]  # Testing with higher noise levels

# Running the simulation with higher noise levels
results = {}

# for case in simulation_cases:
#     for noise in noise_levels:
#         print(f"Running simulation for case {case} with noise level {noise}")
#         positions, energy_deviation, avg_energy_deviation, final_kinetic, final_potential = run_simulation(case, m1, m2, m3, noise)
#         results[(case, noise)] = (positions, energy_deviation, avg_energy_deviation, final_kinetic, final_potential)
        
#         print(f"  Average Energy Deviation: {avg_energy_deviation:.4f}%")
#         print(f"  Final Kinetic Energy: {final_kinetic:.2e} J")
#         print(f"  Final Potential Energy: {final_potential:.2e} J")
#         print(f"  Final Total Energy Deviation: {energy_deviation[-1]:.4f}%")
#         print("-" * 50)

# # Plot energy deviation for each case and noise level
# plt.figure(figsize=(10, 6))
# for (case, noise), (_, energy_deviation, _, _, _) in results.items():
#     plt.plot(energy_deviation, label=f"Case {case} Noise {noise} Energy Deviation")
    
# plt.xlabel('Time Step')
# plt.ylabel('Energy Deviation (%)')
# plt.title('Energy Deviation with Increasing Noise Levels')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Optional: Plot Kinetic and Potential Energy for visualization
# plt.figure(figsize=(10, 6))
# for (case, noise), (_, _, _, final_kinetic, final_potential) in results.items():
#     kinetic_energies = [kinetic_energy(v1, v2, v3, m1, m2, m3) for v1, v2, v3 in positions]
#     potential_energies = [potential_energy(r1, r2, r3, m1, m2, m3) for r1, r2, r3 in positions]
    
#     plt.plot(range(len(kinetic_energies)), kinetic_energies, label=f"Case {case} Kinetic Energy")
#     plt.plot(range(len(potential_energies)), potential_energies, label=f"Case {case} Potential Energy")

# plt.xlabel('Time Step')
# plt.ylabel('Energy (Joules)')
# plt.title('Kinetic and Potential Energy over Time for Three-Body Problem')
# plt.legend()
# plt.grid(True)
# plt.show()

# Test higher noise values
# noise_levels = [5.0, 10.0, 20.0]  # Testing with extreme noise levels

# for case in simulation_cases:
#     for noise in noise_levels:
#         print(f"Running simulation for case {case} with noise level {noise}")
#         positions, energy_deviation, avg_energy_deviation, final_kinetic, final_potential = run_simulation(case, m1, m2, m3, noise)
#         results[(case, noise)] = (positions, energy_deviation, avg_energy_deviation, final_kinetic, final_potential)
        
#         print(f"  Average Energy Deviation: {avg_energy_deviation:.4f}%")
#         print(f"  Final Kinetic Energy: {final_kinetic:.2e} J")
#         print(f"  Final Potential Energy: {final_potential:.2e} J")
#         print(f"  Final Total Energy Deviation: {energy_deviation[-1]:.4f}%")
#         print("-" * 50)

# # Plot energy deviation for each case and noise level
# plt.figure(figsize=(10, 6))
# for (case, noise), (_, energy_deviation, _, _, _) in results.items():
#     plt.plot(energy_deviation, label=f"Case {case} Noise {noise} Energy Deviation")
    
# plt.xlabel('Time Step')
# plt.ylabel('Energy Deviation (%)')
# plt.title('Energy Deviation with Increasing Noise Levels')
# plt.legend()
# plt.grid(True)
# plt.show()

# Testing extreme noise levels (e.g., 50, 100, 200)
extreme_noise_levels = [50, 100, 200]

for case in simulation_cases:
    for noise in extreme_noise_levels:
        print(f"Running simulation for case {case} with extreme noise level {noise}")
        positions, energy_deviation, avg_energy_deviation, final_kinetic, final_potential = run_simulation(case, m1, m2, m3, noise)
        results[(case, noise)] = (positions, energy_deviation, avg_energy_deviation, final_kinetic, final_potential)
        
        print(f"  Average Energy Deviation: {avg_energy_deviation:.4f}%")
        print(f"  Final Kinetic Energy: {final_kinetic:.2e} J")
        print(f"  Final Potential Energy: {final_potential:.2e} J")
        print(f"  Final Total Energy Deviation: {energy_deviation[-1]:.4f}%")
        print("-" * 50)

# Plot energy deviation for each case and noise level
plt.figure(figsize=(10, 6))
for (case, noise), (_, energy_deviation, _, _, _) in results.items():
    plt.plot(energy_deviation, label=f"Case {case} Noise {noise} Energy Deviation")
    
plt.xlabel('Time Step')
plt.ylabel('Energy Deviation (%)')
plt.title('Energy Deviation with Extreme Noise Levels')
plt.legend()
plt.grid(True)
plt.show()
