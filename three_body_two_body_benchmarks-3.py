import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# Constants in AU, Solar Masses, and Years
G = 39.47841760435743  # Gravitational constant in (AU^3 / (Solar Mass * Year^2))

# Initial Conditions for Lagrange points and figure-eight orbits (using AU and Solar Masses)
def initialize_known_solutions(solution_type="lagrange"):
    # Assume masses of the bodies for simplicity (in solar masses)
    m1, m2, m3 = 1.0, 1.0, 1.0  # Example for equal masses
    
    if solution_type == "lagrange":
        positions = np.array([ [1.0, 0.0], [-0.5, np.sqrt(3)/2], [-0.5, -np.sqrt(3)/2] ])  # in AU
        velocities = np.array([ [0.0, 0.0], [0.0, 0.0], [0.0, 0.0] ])  # in AU/year (static equilibrium)
    elif solution_type == "figure_eight":
        positions = np.array([ [0.97000436, -0.24308753], [-0.97000436, 0.24308753], [0.0, 0.0] ])  # in AU
        velocities = np.array([ [0.466203685, 0.43236573], [0.466203685, 0.43236573], [-0.93240737, -0.86473146] ])  # in AU/year
    else:
        raise ValueError("Unknown solution type. Choose 'lagrange' or 'figure_eight'.")
    
    return positions, velocities, m1, m2, m3

# Kinetic Energy Calculation for each body
def compute_kinetic_energy(velocities, masses):
    return 0.5 * masses * np.linalg.norm(velocities, axis=1)**2  # Return an array for each body

# Potential Energy Calculation
def compute_potential_energy(positions, masses):
    n = len(masses)
    potential_energy = 0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])  # Distance in AU
            potential_energy -= G * masses[i] * masses[j] / r  # Gravitational potential energy in AU^3/(Solar Mass * Year^2)
    return potential_energy

# Total Energy Calculation (Kinetic + Potential)
def compute_total_energy(velocities, positions, masses):
    KE = compute_kinetic_energy(velocities, masses)
    PE = compute_potential_energy(positions, masses)
    return np.sum(KE) + PE  # Sum of kinetic energy for all bodies and potential energy

# Update positions and velocities using simple Euler method for integration
def update_system(positions, velocities, masses, dt):
    n = len(masses)
    new_positions = positions + velocities * dt  # Update positions using velocity
    new_velocities = np.copy(velocities)
    
    for i in range(n):
        force = np.array([0.0, 0.0])
        for j in range(n):
            if i != j:
                r_ij = positions[j] - positions[i]
                r_mag = np.linalg.norm(r_ij)
                force += G * masses[i] * masses[j] * r_ij / r_mag**3
        # Update velocities based on force (F = ma => a = F/m)
        new_velocities[i] += force / masses[i] * dt
    
    return new_positions, new_velocities

# Simulation Setup
time_steps = 1000
dt = 0.1  # Time step in years
spatial_grid = np.linspace(-2, 2, 3)

# Initialize positions, velocities, and masses for three bodies
positions, velocities, m1, m2, m3 = initialize_known_solutions("lagrange")
masses = np.array([m1, m2, m3])

# Prepare lists for storing output
time_data = []
positions_data = []
velocities_data = []
kinetic_energy_data = []
potential_energy_data = []
total_energy_data = []

# Time integration loop
for t in range(time_steps):
    # Store data for each time step
    KE = compute_kinetic_energy(velocities, masses)
    PE = compute_potential_energy(positions, masses)
    TE = compute_total_energy(velocities, positions, masses)
    
    time_data.append(t * dt)
    positions_data.append(positions)
    velocities_data.append(velocities)
    kinetic_energy_data.append(KE)
    potential_energy_data.append(PE)
    total_energy_data.append(TE)
    
    # Update system (positions and velocities)
    positions, velocities = update_system(positions, velocities, masses, dt)

# Convert collected data into a DataFrame for easy viewing
output_data = {
    "Time Step": time_data,
    "Body 1 Position": [p[0] for p in positions_data],
    "Body 2 Position": [p[1] for p in positions_data],
    "Body 1 Velocity": [v[0] for v in velocities_data],
    "Body 2 Velocity": [v[1] for v in velocities_data],
    "Kinetic Energy 1": [KE[0] for KE in kinetic_energy_data],
    "Kinetic Energy 2": [KE[1] for KE in kinetic_energy_data],
    "Kinetic Energy 3": [KE[2] for KE in kinetic_energy_data],
    "Potential Energy": potential_energy_data,
    "Total Energy": total_energy_data,
}

output_df = pd.DataFrame(output_data)
csv_file_path = "quantum_classical_comparison.csv"
output_df.to_csv(csv_file_path, index=False)  # index=False to omit the row indices
# Display the results
# display_dataframe_to_user("Refined Quantum-Classical Trajectory Deviations with Interpolation", output_df)
# Display graphs
plt.figure(figsize=(10, 6))

# Plot Positions over Time
plt.subplot(2, 2, 1)
plt.plot(time_data, [pos[0] for pos in positions_data], label='Body 1 Position')
plt.plot(time_data, [pos[1] for pos in positions_data], label='Body 2 Position')
plt.plot(time_data, [pos[2] for pos in positions_data], label='Body 3 Position')
plt.xlabel('Time (Years)')
plt.ylabel('Position (AU)')
plt.title('Positions of Bodies Over Time')
plt.legend()
plt.grid()

# Plot Velocities over Time
plt.subplot(2, 2, 2)
plt.plot(time_data, [vel[0] for vel in velocities_data], label='Body 1 Velocity')
plt.plot(time_data, [vel[1] for vel in velocities_data], label='Body 2 Velocity')
plt.plot(time_data, [vel[2] for vel in velocities_data], label='Body 3 Velocity')
plt.xlabel('Time (Years)')
plt.ylabel('Velocity (AU/Year)')
plt.title('Velocities of Bodies Over Time')
plt.legend()
plt.grid()

# Plot Kinetic Energy over Time
plt.subplot(2, 2, 3)
plt.plot(time_data, [KE[0] for KE in kinetic_energy_data], label='Kinetic Energy Body 1')
plt.plot(time_data, [KE[1] for KE in kinetic_energy_data], label='Kinetic Energy Body 2')
plt.plot(time_data, [KE[2] for KE in kinetic_energy_data], label='Kinetic Energy Body 3')
plt.xlabel('Time (Years)')
plt.ylabel('Kinetic Energy (AU^3/Solar Mass/Year^2)')
plt.title('Kinetic Energy of Bodies Over Time')
plt.legend()
plt.grid()

# Plot Total Energy over Time
plt.subplot(2, 2, 4)
plt.plot(time_data, total_energy_data, label='Total Energy', color='black')
plt.xlabel('Time (Years)')
plt.ylabel('Total Energy (AU^3/Solar Mass/Year^2)')
plt.title('Total Energy of System Over Time')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()