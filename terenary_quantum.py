import numpy as np
import random
import matplotlib.pyplot as plt

# Constants for the simulation
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
m1 = 5.97e24     # Mass of body 1 (kg)
m2 = 7.35e22     # Mass of body 2 (kg)
m3 = 1.98e30     # Mass of body 3 (kg)
time_step = 1.0  # Time step for simulation (seconds)
total_time = 10000  # Total simulation time (seconds)
n_steps = total_time // time_step  # Number of steps in the simulation

# Initial positions and velocities of the bodies (in SI units)
# [x, y, z] positions and [vx, vy, vz] velocities
r1 = np.array([1.0e11, 0, 0])  # Position of body 1 (m)
r2 = np.array([0, 1.0e11, 0])  # Position of body 2 (m)
r3 = np.array([0, 0, 1.0e11])  # Position of body 3 (m)

v1 = np.array([0, 3.0e4, 0])  # Velocity of body 1 (m/s)
v2 = np.array([0, -3.0e4, 0])  # Velocity of body 2 (m/s)
v3 = np.array([0, 0, 3.0e4])  # Velocity of body 3 (m/s)

# Store positions for plotting
positions = []

# Classical numerical method (Euler's method)
def compute_forces(r1, r2, r3, m1, m2, m3):
    # Gravitational forces between the bodies
    r12 = r2 - r1
    r13 = r3 - r1
    r23 = r3 - r2
    
    d12 = np.linalg.norm(r12)
    d13 = np.linalg.norm(r13)
    d23 = np.linalg.norm(r23)
    
    # Force calculations based on Newton's Law of Gravitation
    F12 = G * m1 * m2 * r12 / d12**3
    F13 = G * m1 * m3 * r13 / d13**3
    F23 = G * m2 * m3 * r23 / d23**3
    
    return F12, F13, F23

def integrate_step(r1, r2, r3, v1, v2, v3, m1, m2, m3, time_step):
    # Compute forces
    F12, F13, F23 = compute_forces(r1, r2, r3, m1, m2, m3)
    
    # Update velocities
    v1 += (F12 + F13) / m1 * time_step
    v2 += (-F12 + F23) / m2 * time_step
    v3 += (-F13 - F23) / m3 * time_step
    
    # Update positions
    r1 += v1 * time_step
    r2 += v2 * time_step
    r3 += v3 * time_step
    
    return r1, r2, r3, v1, v2, v3

# Correct the n_steps calculation to ensure it's an integer
n_steps = int(total_time // time_step)  # Ensure it's an integer

# Simulate the three-body problem using numerical integration (Euler method)
for step in range(n_steps):
    r1, r2, r3, v1, v2, v3 = integrate_step(r1, r2, r3, v1, v2, v3, m1, m2, m3, time_step)
    
    # Store positions for plotting
    positions.append((r1.copy(), r2.copy(), r3.copy()))


# Convert positions to numpy arrays for easier manipulation
positions = np.array(positions)

# Plot the trajectory of the three bodies
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0, 0], positions[:, 0, 1], positions[:, 0, 2], label='Body 1')
ax.plot(positions[:, 1, 0], positions[:, 1, 1], positions[:, 1, 2], label='Body 2')
ax.plot(positions[:, 2, 0], positions[:, 2, 1], positions[:, 2, 2], label='Body 3')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Three-Body Problem Simulation')
ax.legend()
plt.show()
