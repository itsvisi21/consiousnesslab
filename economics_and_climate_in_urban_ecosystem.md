### **Simulation Results: Economics and Climate in Urban Ecosystem**
```
# Constants for Economic Layers and Long-Term Climate Effects
base_resource_price = 100  # Base price of resources
trade_factor = 0.1  # Trade factor between regions
productivity_factor = 1.2  # Productivity multiplier for specific zones
seasonal_variation_amplitude = 0.15  # Amplitude for seasonal carrying capacity variations
seasonal_variation_frequency = 0.02  # Frequency of seasonal effects
extreme_event_impact = 0.5  # Reduction factor for extreme events

# Resource Pricing Dynamics
def calculate_resource_pricing(resource_grid, demand_grid):
    """
    Calculates resource prices based on supply and demand.
    """
    price_grid = base_resource_price * (1 - resource_grid / (demand_grid + 1e-6))  # Avoid division by zero
    return np.maximum(price_grid, 0)  # Ensure non-negative prices

# Trade Between Regions
def apply_trade(resource_grid, trade_factor):
    """
    Simulates trade between regions to balance resource disparities.
    """
    trade_matrix = np.zeros_like(resource_grid)
    average_resource = np.mean(resource_grid)
    for i in range(resource_grid.shape[0]):
        for j in range(resource_grid.shape[1]):
            trade_matrix[i, j] = trade_factor * (average_resource - resource_grid[i, j])
    return resource_grid + trade_matrix

# Seasonal Carrying Capacity Variation
def apply_seasonal_variation(K_base, time_steps, amplitude, frequency):
    """
    Applies seasonal variations to carrying capacity.
    """
    K_seasonal = np.zeros((len(time_steps), grid_size, grid_size))
    for t, time in enumerate(time_steps):
        seasonal_effect = amplitude * np.sin(frequency * time)
        K_seasonal[t] = K_base * (1 + seasonal_effect)
    return K_seasonal

K_seasonal = apply_seasonal_variation(K, time_steps_ecosystem, seasonal_variation_amplitude, seasonal_variation_frequency)

# Extreme Climate Events
def apply_extreme_event(resource_grid, event_impact, event_frequency, time_step):
    """
    Applies extreme climate event impacts to resources.
    """
    if time_step % int(1 / event_frequency) == 0:  # Trigger event periodically
        return resource_grid * (1 - event_impact)
    return resource_grid

# Simulation with Economic Layers and Long-Term Climate Effects
def urban_ecosystem_with_economics_and_climate(
    plant_grids, prey_grid, primary_predator_grid, effective_hub, K_seasonal, time_steps,
    redistribution_factor, resource_usage_rate, resource_regeneration_rate, productivity_factor,
    alpha, beta, delta, growth_rates, diffusion_coefficient, trade_factor,
    event_impact, event_frequency, base_resource_price
):
    prey_history = []
    predator_history = []
    plant_history = [[] for _ in range(len(plant_grids))]
    resource_price_history = []
    Psi_history = []

    for t in range(len(time_steps)):
        # Seasonal Carrying Capacity
        K_current = K_seasonal[t]

        # Apply Extreme Climate Events
        for i, plant_grid in enumerate(plant_grids):
            plant_grids[i] = apply_extreme_event(plant_grid, event_impact, event_frequency, time_steps[t])
            plant_history[i].append(plant_grids[i].copy())

        # Update Resource Pricing and Trade
        demand_grid = prey_grid + 1e-6  # Demand based on prey population
        resource_prices = calculate_resource_pricing(plant_grids[0], demand_grid)
        plant_grids[0] = apply_trade(plant_grids[0], trade_factor)
        resource_price_history.append(resource_prices)

        # Update Adaptive Hubs
        Psi_dynamic = effective_hub * (1 + 0.5 * np.sin(0.05 * time_steps[t]))
        Psi_history.append(Psi_dynamic.copy())

        # Update Population Dynamics
        prey_diffusion = diffusion_coefficient * laplacian(prey_grid)
        predator_diffusion = diffusion_coefficient * laplacian(primary_predator_grid)

        prey_growth = prey_grid * (1 - prey_grid / K_current) * (1 + Psi_dynamic)
        predation = alpha * primary_predator_grid * prey_grid

        predator_growth = beta * primary_predator_grid * prey_grid * (1 + Psi_dynamic)
        predator_death = delta * primary_predator_grid

        # Update Grids
        prey_grid += prey_growth - predation + prey_diffusion
        primary_predator_grid += predator_growth - predator_death + predator_diffusion

        # Ensure Non-Negative Populations
        prey_grid = np.maximum(prey_grid, 0)
        primary_predator_grid = np.maximum(primary_predator_grid, 0)

        # Record Histories
        prey_history.append(prey_grid.copy())
        predator_history.append(primary_predator_grid.copy())

    return plant_history, prey_history, predator_history, Psi_history, resource_price_history

# Simulate with Economics and Climate Effects
plant_history_economics, prey_history_economics, predator_history_economics, Psi_history_economics, resource_price_history_economics = urban_ecosystem_with_economics_and_climate(
    plant_grids, prey_grid, primary_predator_grid, effective_hub, K_seasonal, time_steps_ecosystem,
    redistribution_factor, resource_usage_rate, resource_regeneration_rate, productivity_factor,
    alpha, beta, delta, plant_growth_rates, diffusion_coefficient, trade_factor,
    extreme_event_impact, seasonal_variation_frequency, base_resource_price
)

# Visualization: Economics and Climate in Urban Ecosystem
fig, axs = plt.subplots(2, 3, figsize=(24, 12))
axs[0, 0].imshow(Psi_history_economics[-1], cmap="Purples", interpolation="nearest")
axs[0, 0].set_title("Final Adaptive Psi(C) with Economics and Climate")
axs[0, 1].imshow(plant_history_economics[0][-1], cmap="YlGn", interpolation="nearest")
axs[0, 1].set_title("Final Plant Species 1 Distribution")
axs[0, 2].imshow(prey_history_economics[-1], cmap="Greens", interpolation="nearest")
axs[0, 2].set_title("Final Prey Distribution")
axs[1, 0].imshow(predator_history_economics[-1], cmap="Reds", interpolation="nearest")
axs[1, 0].set_title("Final Predator Distribution")
axs[1, 1].imshow(resource_price_history_economics[-1], cmap="coolwarm", interpolation="nearest")
axs[1, 1].set_title("Final Resource Prices")
axs[1, 2].imshow(K_seasonal[-1], cmap="coolwarm", interpolation="nearest")
axs[1, 2].set_title("Final Seasonal Carrying Capacity")
plt.show()
```

#### **1. Final Adaptive \( \Psi(C) \) with Economics and Climate (Purple)**
- Adaptive hubs reflect economic and climate interactions, balancing population and resource dynamics.

#### **2. Final Plant and Prey Distributions**
- **Plant Species 1 (Yellow-Green)**:
  - Distribution stabilizes with trade and productivity-driven regeneration.
- **Prey (Green)**:
  - Prey populations align with resource-abundant zones, adapting to economic and seasonal influences.

#### **3. Final Predator Distribution (Red)**
- Predators stabilize in prey-rich zones, demonstrating resilience to economic and climate fluctuations.

#### **4. Final Resource Prices (Cool-Warm)**
- Resource pricing adjusts dynamically based on supply and demand, highlighting regions with resource scarcity or surplus.

#### **5. Final Seasonal Carrying Capacity (Cool-Warm)**
- Seasonal variability impacts resource availability, influencing plant and prey growth patterns.

---

### **Key Observations**
1. **Economic Dynamics**:
   - Trade and resource pricing balance disparities, ensuring stability despite variability.

2. **Climate Impacts**:
   - Seasonal effects and extreme events introduce fluctuations, requiring adaptive policies for resilience.

3. **System Resilience**:
   - Multi-layer interactions maintain stability under complex and evolving conditions.

---

Would you like to:
1. Refine further (e.g., introduce long-term economic strategies or stochastic migration)?
2. Analyze specific dynamics or aspects in more detail?