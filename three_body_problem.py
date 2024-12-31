import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import hbar, pi, G

# Define the gravitational potential between two bodies
def gravitational_potential(r, m1, m2):
    """Calculate the gravitational potential between two point masses m1 and m2."""
    return -G * m1 * m2 / r

# Define the Gaussian wave packet
def gaussian_wave_packet(x, x0, sigma, n):
    """Generate a Gaussian wave packet centered at x0 with width sigma and a phase factor."""
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * n * pi * (x - x0))

# Normalize wavefunction to prevent overflow
def normalize_wavefunction(Psi):
    """Normalize the wavefunction to unit norm."""
    norm = np.linalg.norm(Psi)
    if norm != 0:
        Psi /= norm
    return Psi

# Limit interaction potential to avoid overflow
def ternary_interaction(Psi_i, Psi_j, alpha, limit=1e3):
    """Interaction potential modulated by quantum states, capped at a limit."""
    interaction = np.real(np.conj(Psi_i) * Psi_j)
    # Cap the value to avoid overflow
    return np.clip(alpha * interaction, -limit, limit)

# Kinetic energy operator (finite difference)
def kinetic_operator(Psi, dx):
    """Calculate the kinetic energy term using finite differences"""
    Psi_xx = np.roll(Psi, -1) - 2 * Psi + np.roll(Psi, 1)
    return -0.5 * hbar**2 / (2 * dx**2) * Psi_xx

# Full Hamiltonian (including gravitational and ternary interaction)
def hamiltonian(Psi1, Psi2, Psi3, x_grid, alpha):
    """Construct the Hamiltonian for the three-body system"""
    # Gravitational interactions (assuming point masses for simplicity)
    r12 = np.abs(x_grid - np.roll(x_grid, 1))  # Distance between body 1 and 2
    r13 = np.abs(x_grid - np.roll(x_grid, 2))  # Distance between body 1 and 3
    r23 = np.abs(np.roll(x_grid, 1) - np.roll(x_grid, 2))  # Distance between body 2 and 3

    V_grav = (gravitational_potential(r12, 1, 1) + gravitational_potential(r13, 1, 1) +
              gravitational_potential(r23, 1, 1))
    
    # Ternary interactions
    V_ternary = ternary_interaction(Psi1, Psi2, alpha) + ternary_interaction(Psi1, Psi3, alpha) + ternary_interaction(Psi2, Psi3, alpha)
    
    # Total Hamiltonian
    H = kinetic_operator(Psi1, dx) + kinetic_operator(Psi2, dx) + kinetic_operator(Psi3, dx) + V_grav + V_ternary
    return H

# Time evolution using the finite difference method
# Time evolution using the finite difference method
def evolve_wavefunction(Psi1, Psi2, Psi3, time_steps, dt, alpha, x_grid):
    """Evolve the wavefunctions over time and collect data"""
    
    # Data storage
    probability_densities = []  # Store |Psi(x)|^2
    gravitational_potentials = []  # Store gravitational potential at each step
    interaction_terms = []  # Store ternary interaction terms at each step
    
    for t in range(time_steps):
        H = hamiltonian(Psi1, Psi2, Psi3, x_grid, alpha)
        
        # Update wavefunctions
        Psi1 -= 1j * H * dt / hbar
        Psi2 -= 1j * H * dt / hbar
        Psi3 -= 1j * H * dt / hbar

        # Normalize the wavefunctions after each step
        Psi1 = normalize_wavefunction(Psi1)
        Psi2 = normalize_wavefunction(Psi2)
        Psi3 = normalize_wavefunction(Psi3)
        
        # Store data for analysis
        probability_densities.append((np.abs(Psi1)**2, np.abs(Psi2)**2, np.abs(Psi3)**2))
        
        # Compute gravitational potential at this step
        r12 = np.abs(x_grid - np.roll(x_grid, 1))
        r13 = np.abs(x_grid - np.roll(x_grid, 2))
        r23 = np.abs(np.roll(x_grid, 1) - np.roll(x_grid, 2))
        V_grav = gravitational_potential(r12, 1, 1) + gravitational_potential(r13, 1, 1) + gravitational_potential(r23, 1, 1)
        gravitational_potentials.append(V_grav)
        
        # Compute ternary interaction terms
        V_ternary = ternary_interaction(Psi1, Psi2, alpha) + ternary_interaction(Psi1, Psi3, alpha) + ternary_interaction(Psi2, Psi3, alpha)
        interaction_terms.append(V_ternary)
        
        if t % 100 == 0:  # Plot every 100 steps
            plot_wavefunctions(Psi1, Psi2, Psi3, t)
    
    # Convert data to numpy arrays for further analysis
    probability_densities = np.array(probability_densities)
    gravitational_potentials = np.array(gravitational_potentials)
    interaction_terms = np.array(interaction_terms)
    
    return Psi1, Psi2, Psi3, probability_densities, gravitational_potentials, interaction_terms


# Function to plot wavefunctions
def plot_wavefunctions(Psi1, Psi2, Psi3, t):
    """Visualize the wavefunctions of the 3 bodies"""
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, np.abs(Psi1)**2, label="Body 1 (State 0)")
    plt.plot(x_grid, np.abs(Psi2)**2, label="Body 2 (State 1)")
    plt.plot(x_grid, np.abs(Psi3)**2, label="Body 3 (State 2)")
    plt.xlabel('Position')
    plt.ylabel('Probability Density')
    plt.title(f"Wavefunctions at Time Step {t}")
    plt.legend()
    plt.show()

# Parameters
alpha = 0.1  # Ternary interaction strength

# Grid setup
x_min, x_max, dx = 0, 10, 0.1  # Spatial grid
x_grid = np.arange(x_min, x_max, dx)

# Initial positions and widths for the wave packets
x0_1, x0_2, x0_3 = 2, 5, 8  # Initial positions
sigma = 0.5  # Width of the Gaussian wave packets

# Initial wavefunctions for three bodies
# Psi_1 = gaussian_wave_packet(x_grid, x0_1, sigma, 0)
# Psi_2 = gaussian_wave_packet(x_grid, x0_2, sigma, 1)
# Psi_3 = gaussian_wave_packet(x_grid, x0_3, sigma, 2)

# Run the simulation
# time_steps = 1000
# dt = 0.01
# Psi1, Psi2, Psi3 = evolve_wavefunction(Psi_1, Psi_2, Psi_3, time_steps, dt, alpha)
# Parameters
alpha = 0.1  # Ternary interaction strength

# Grid setup
x_min, x_max, dx = 0, 10, 0.1  # Spatial grid
x_grid = np.arange(x_min, x_max, dx)

# Initial positions and widths for the wave packets
x0_1, x0_2, x0_3 = 2, 5, 8  # Initial positions
sigma = 0.5  # Width of the Gaussian wave packets

# Initial wavefunctions for three bodies
Psi_1 = gaussian_wave_packet(x_grid, x0_1, sigma, 0)
Psi_2 = gaussian_wave_packet(x_grid, x0_2, sigma, 1)
Psi_3 = gaussian_wave_packet(x_grid, x0_3, sigma, 2)

# Run the simulation
time_steps = 1000
dt = 0.01
Psi1, Psi2, Psi3, probability_densities, gravitational_potentials, interaction_terms = evolve_wavefunction(Psi_1, Psi_2, Psi_3, time_steps, dt, alpha, x_grid)

# Optionally, you can save the data to a file for later analysis
np.savez("three_body_simulation_data.npz", probability_densities=probability_densities, 
         gravitational_potentials=gravitational_potentials, interaction_terms=interaction_terms) 

data = np.load("three_body_simulation_data.npz")
probability_densities = data["probability_densities"]
gravitational_potentials = data["gravitational_potentials"]
interaction_terms = data["interaction_terms"]

# Save Probability Densities to CSV
# Here, we assume that probability_densities is a 3D array: (time_steps, bodies, positions)
# We flatten the second and third dimensions and save as a 2D array (time_step, body_position)
prob_density_reshaped = probability_densities.reshape(probability_densities.shape[0], -1)
np.savetxt("probability_densities.csv", prob_density_reshaped, delimiter=",", header="TimeStep, Body1_Pos1, Body1_Pos2, ..., Body3_PosN", comments="")

# Save Gravitational Potentials to CSV
# Assuming gravitational_potentials is a 2D array (time_steps, positions)
np.savetxt("gravitational_potentials.csv", gravitational_potentials, delimiter=",", header="TimeStep, Position1, Position2, ..., PositionN", comments="")

# Save Interaction Terms to CSV
# Assuming interaction_terms is a 2D array (time_steps, positions)
np.savetxt("interaction_terms.csv", interaction_terms, delimiter=",", header="TimeStep, Interaction1, Interaction2, ..., InteractionN", comments="")

# Optionally, use pandas to create DataFrames for more flexibility in the CSV structure
# Example with pandas (optional)
df_prob_density = pd.DataFrame(prob_density_reshaped)
df_prob_density.to_csv("probability_densities_pandas.csv", index=False)

df_gravitational_potentials = pd.DataFrame(gravitational_potentials)
df_gravitational_potentials.to_csv("gravitational_potentials_pandas.csv", index=False)

df_interaction_terms = pd.DataFrame(interaction_terms)
df_interaction_terms.to_csv("interaction_terms_pandas.csv", index=False)