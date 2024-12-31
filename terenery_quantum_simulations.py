import numpy as np
import matplotlib.pyplot as plt

# Constants for the simulation
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
time_step = 1.0  # Time step for simulation (seconds)
total_time = 10000  # Total simulation time (seconds)
n_steps = int(total_time // time_step)  # Number of steps in the simulation

# Initial positions and velocities of the bodies (in SI units)
# [x, y, z] positions and [vx, vy, vz] velocities

def initialize_bodies(case):
    if case == 1:  # Standard Case
        # Standard initial conditions: bodies far apart, moderate velocities
        r1 = np.array([1.0e11, 0, 0])  # Position of body 1 (m)
        r2 = np.array([0, 1.0e11, 0])  # Position of body 2 (m)
        r3 = np.array([0, 0, 1.0e11])  # Position of body 3 (m)

        v1 = np.array([0, 3.0e4, 0])  # Velocity of body 1 (m/s)
        v2 = np.array([0, -3.0e4, 0])  # Velocity of body 2 (m/s)
        v3 = np.array([0, 0, 3.0e4])  # Velocity of body 3 (m/s)
    
    elif case == 2:  # Varying Gravitational Influence
        # Bodies are closer together, larger mass difference
        r1 = np.array([1.0e10, 0, 0])  # Position of body 1 (m)
        r2 = np.array([0, 1.0e10, 0])  # Position of body 2 (m)
        r3 = np.array([0, 0, 1.0e10])  # Position of body 3 (m)

        v1 = np.array([0, 2.5e4, 0])  # Velocity of body 1 (m/s)
        v2 = np.array([0, -2.0e4, 0])  # Velocity of body 2 (m/s)
        v3 = np.array([0, 0, 3.5e4])  # Velocity of body 3 (m/s)
    
    elif case == 3:  # High-Energy Case
        # Extremely fast orbits
        r1 = np.array([1.0e9, 0, 0])  # Position of body 1 (m)
        r2 = np.array([0, 1.0e9, 0])  # Position of body 2 (m)
        r3 = np.array([0, 0, 1.0e9])  # Position of body 3 (m)

        v1 = np.array([0, 1.0e5, 0])  # Velocity of body 1 (m/s)
        v2 = np.array([0, -1.2e5, 0])  # Velocity of body 2 (m/s)
        v3 = np.array([0, 0, 1.5e5])  # Velocity of body 3 (m/s)
    
    elif case == 4:  # Noise Introduction Case
        # Same as standard case but with noise added later
        r1 = np.array([1.0e11, 0, 0])  # Position of body 1 (m)
        r2 = np.array([0, 1.0e11, 0])  # Position of body 2 (m)
        r3 = np.array([0, 0, 1.0e11])  # Position of body 3 (m)

        v1 = np.array([0, 3.0e4, 0])  # Velocity of body 1 (m/s)
        v2 = np.array([0, -3.0e4, 0])  # Velocity of body 2 (m/s)
        v3 = np.array([0, 0, 3.0e4])  # Velocity of body 3 (m/s)
    
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

# Noise addition for Case 4
def add_noise(r1, r2, r3, v1, v2, v3, noise_level=0.05):
    # Random noise based on percentage of the velocity and position
    noise_r = np.random.normal(0, noise_level, 3)  # Random noise for positions
    noise_v = np.random.normal(0, noise_level, 3)  # Random noise for velocities
    r1 += noise_r
    r2 += noise_r
    r3 += noise_r
    v1 += noise_v
    v2 += noise_v
    v3 += noise_v
    return r1, r2, r3, v1, v2, v3

# Running the simulation for multiple cases
def run_simulation(case, m1, m2, m3):
    r1, r2, r3, v1, v2, v3 = initialize_bodies(case)
    
    positions = []
    
    for step in range(n_steps):
        if case == 4:  # Case 4: Noise Introduction
            r1, r2, r3, v1, v2, v3 = add_noise(r1, r2, r3, v1, v2, v3)
        
        r1, r2, r3, v1, v2, v3 = integrate_step(r1, r2, r3, v1, v2, v3, m1, m2, m3, time_step)
        
        # Store positions for plotting
        positions.append((r1.copy(), r2.copy(), r3.copy()))
    
    return np.array(positions)

# Simulation cases
m1, m2, m3 = 5.97e24, 7.35e22, 1.98e30  # Earth, Moon, and a large planet (mass in kg)

# Run all simulations
simulation_cases = [1, 2, 3, 4]
results = {}

for case in simulation_cases:
    print(f"Running simulation for case {case}")
    positions = run_simulation(case, m1, m2, m3)
    results[case] = positions

# Plotting results for each case
fig = plt.figure(figsize=(14, 8))

for case, positions in results.items():
    ax = fig.add_subplot(2, 2, case, projection='3d')
    ax.plot(positions[:, 0, 0], positions[:, 0, 1], positions[:, 0, 2], label=f'Body 1 - Case {case}')
    ax.plot(positions[:, 1, 0], positions[:, 1, 1], positions[:, 1, 2], label=f'Body 2 - Case {case}')
    ax.plot(positions[:, 2, 0], positions[:, 2, 1], positions[:, 2, 2], label=f'Body 3 - Case {case}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Three-Body Problem - Case {case}')
    ax.legend()

plt.tight_layout()
plt.show()
