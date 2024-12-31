import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters for simulations
total_days = 365 * 5  # 5 years
daily_ticket_sales = 10000  # Initial daily ticket sales
daily_growth_rate = 1.02  # 2% daily growth
total_pool = 16000000  # Initial pool size (Qube)
reward_scaling = [
    (15000000, 0.6),  # >15M Qube: 60% scaling
    (10000000, 0.4),  # 10M-15M Qube: 40% scaling
    (5000000, 0.2),   # 5M-10M Qube: 20% scaling
    (0, 0.1),         # <5M Qube: 10% scaling
]
daily_reward_cap = 0.001  # Cap as 0.1% of remaining pool
penalty_rate = 0.1  # Penalty rate for early withdrawals

# Initialize variables
remaining_pool = total_pool
daily_rewards = []
daily_prices = []
daily_penalties = []
prices = [1.0]  # Start at $1.0 per Qube

# for day in range(total_days):
#     # Adjust ticket sales
#     if day > 0:
#         daily_ticket_sales = min(daily_ticket_sales * daily_growth_rate, 100000)  # Cap ticket sales at 100,000

#     # Calculate scaling factor
#     scaling_factor = next((factor for threshold, factor in reward_scaling if remaining_pool > threshold), 0.0)

#     # Calculate daily reward
#     daily_reward = scaling_factor * min(daily_ticket_sales, remaining_pool * daily_reward_cap)

#     # Apply penalty contributions
#     penalty_contribution = penalty_rate * daily_reward
#     remaining_pool += penalty_contribution

#     # Update pool
#     remaining_pool -= daily_reward
#     remaining_pool = max(0, remaining_pool)  # Ensure pool doesn't go negative

#     # Update price based on pool scarcity
#     price = prices[-1] * (1 + (1 - remaining_pool / total_pool))
#     prices.append(price)

#     # Record daily stats
#     daily_rewards.append(daily_reward)
#     daily_prices.append(price)
#     daily_penalties.append(penalty_contribution)

# # Create a DataFrame for results
# simulation_results = pd.DataFrame({
#     'Day': np.arange(1, total_days + 1),
#     'Remaining Pool (Qube)': np.linspace(total_pool, remaining_pool, total_days),
#     'Daily Ticket Sales': np.full(total_days, daily_ticket_sales),
#     'Daily Rewards Paid': daily_rewards,
#     'Qube Price ($)': daily_prices,
#     'Penalty Contributions': daily_penalties
# })

# # Save the results
# simulation_results.to_csv("qube_simulation_results.csv", index=False)

# # Visualization
# plt.figure(figsize=(10, 6))
# plt.plot(simulation_results['Day'], simulation_results['Qube Price ($)'], label='Qube Price ($)')
# plt.xlabel('Day')
# plt.ylabel('Qube Price ($)')
# plt.title('Qube Price Dynamics Over 5 Years')
# plt.legend()
# plt.grid()
# plt.savefig('qube_price_projection.png')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(simulation_results['Day'], simulation_results['Remaining Pool (Qube)'], label='Remaining Pool (Qube)')
# plt.xlabel('Day')
# plt.ylabel('Remaining Pool (Qube)')
# plt.title('Reward Pool Sustainability Over 5 Years')
# plt.legend()
# plt.grid()
# plt.savefig('reward_pool_sustainability.png')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(simulation_results['Day'], simulation_results['Penalty Contributions'], label='Penalty Contributions')
# plt.xlabel('Day')
# plt.ylabel('Penalty Contributions (Qube)')
# plt.title('Cumulative Penalty Contributions Over 5 Years')
# plt.legend()
# plt.grid()
# plt.savefig('penalty_impact.png')
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(simulation_results['Day'], simulation_results['Daily Rewards Paid'], label='Daily Rewards Paid (Qube)')
# plt.xlabel('Day')
# plt.ylabel('Daily Rewards Paid (Qube)')
# plt.title('Reward Distribution Over 5 Years')
# plt.legend()
# plt.grid()
# plt.savefig('reward_distribution.png')
# plt.show()

# Simulation parameters
qube_prices = np.linspace(0.1, 5, 100)  # Simulating QubeCoin prices from $0.1 to $5
ticket_price_usd = 1.0  # Fixed ticket price in USD
daily_ticket_sales = 100000  # Daily tickets sold

# Calculate QubeCoin contributions
qube_contributions_per_ticket = ticket_price_usd / qube_prices
daily_qube_contributions = daily_ticket_sales * qube_contributions_per_ticket


# Save the simulation results to a file for documentation
file_path = "./figures/QubeCoin_Contributions_Simulation.png"

# Save the figure
plt.figure(figsize=(10, 6))

# Qube contributions per ticket
plt.subplot(2, 1, 1)
plt.plot(qube_prices, qube_contributions_per_ticket, label="QubeCoins Per Ticket")
plt.title("Impact of Fixed $1 Ticket Pricing on QubeCoin Contributions")
plt.xlabel("QubeCoin Price ($)")
plt.ylabel("QubeCoins Per Ticket")
plt.grid(True)
plt.legend()

# Daily Qube contributions
plt.subplot(2, 1, 2)
plt.plot(qube_prices, daily_qube_contributions, label="Daily Qube Contributions", color="orange")
plt.xlabel("QubeCoin Price ($)")
plt.ylabel("Daily Qube Contributions")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(file_path)

