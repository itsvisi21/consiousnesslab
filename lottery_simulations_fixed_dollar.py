import numpy as np
import matplotlib.pyplot as plt

# Parameters for simulation
days = 365  # Simulate for one year
initial_qube_price = 1.0  # USD
initial_reward_pool = 16000000  # Initial Qube reward pool
daily_ticket_sales = 100000  # Tickets sold daily
ticket_price_usd = 1.0  # Fixed ticket price in USD
reward_scaling_thresholds = [(15e6, 0.6), (10e6, 0.4), (5e6, 0.2), (0, 0.05)]  # Reward scaling factors
penalty_contribution_factor = 0.1  # 10% of daily rewards
reward_pool_depletion_rate = 0.0001  # 0.01% daily cap on remaining pool

# Variables to track daily dynamics
qube_price = initial_qube_price
reward_pool = initial_reward_pool
qube_contributions = []
daily_rewards_paid = []
daily_penalties = []
qube_price_trend = []

# Simulation loop
for day in range(days):
    # Calculate daily Qube contribution per ticket based on current price
    qube_per_ticket = ticket_price_usd / qube_price
    daily_qube_contribution = daily_ticket_sales * qube_per_ticket

    # Determine reward scaling based on remaining pool
    scaling_factor = next(scale for threshold, scale in reward_scaling_thresholds if reward_pool > threshold)
    daily_rewards = reward_pool * reward_pool_depletion_rate * scaling_factor

    # Calculate penalties
    penalty_contribution = daily_rewards * penalty_contribution_factor

    # Update pool and Qube price
    reward_pool = reward_pool - daily_rewards + daily_qube_contribution + penalty_contribution
    qube_price = qube_price * (1 + 0.0005)  # Simulating a 0.05% daily price increase

    # Track results
    qube_contributions.append(daily_qube_contribution)
    daily_rewards_paid.append(daily_rewards)
    daily_penalties.append(penalty_contribution)
    qube_price_trend.append(qube_price)

# Assuming daily_rewards_paid is a list already populated
reward_pool_balance = [initial_reward_pool]  # Start with the initial reward pool

# Calculate the reward pool balance day by day
for i in range(1, len(daily_rewards_paid)):
    new_balance = reward_pool_balance[i - 1] - daily_rewards_paid[i] + qube_contributions[i] + daily_penalties[i]
    reward_pool_balance.append(new_balance)


# Visualizing Results
# Save the generated images
output_directory = "./figures/"
file_names = ["qube_price_trend.png", "daily_qube_contributions.png", "daily_rewards_paid.png", "reward_pool_balance.png"]

# Redo plots to save them individually
plt.figure(figsize=(6, 4))

# Qube price trend
plt.plot(qube_price_trend, label="Qube Price ($)")
plt.title("Qube Price Trend")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.legend()
plt.savefig(output_directory + file_names[0])
plt.clf()

# Daily Qube contributions
plt.plot(qube_contributions, label="Daily Qube Contributions")
plt.title("Daily Qube Contributions")
plt.xlabel("Days")
plt.ylabel("Qube")
plt.legend()
plt.savefig(output_directory + file_names[1])
plt.clf()

# Daily rewards paid
plt.plot(daily_rewards_paid, label="Daily Rewards Paid")
plt.title("Daily Rewards Paid")
plt.xlabel("Days")
plt.ylabel("Qube")
plt.legend()
plt.savefig(output_directory + file_names[2])
plt.clf()

# Reward pool balance
plt.plot(reward_pool_balance, label="Reward Pool Balance")
plt.title("Reward Pool Balance Over Time")
plt.xlabel("Days")
plt.ylabel("Reward Pool (Qube)")
plt.legend()
plt.savefig(output_directory + file_names[3])
plt.clf()

file_paths = [output_directory + name for name in file_names]
file_paths
