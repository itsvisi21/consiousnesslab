# Since the code environment was reset, I'll reinitialize the process.

# Re-import necessary libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# Function to plot graphs for the simulation results
def plot_simulation_results(results_df):
    metrics = [
        "Total Losers Partially Compensated",
        "Total Losers Fully Compensated",
        "Total Losers Not Compensated",
        "Total Profits Distributed to Players",
        "Total Table (House) Profit",
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, len(metrics) * 4))
    for i, metric in enumerate(metrics):
        axes[i].plot(results_df["Games Level"], results_df[metric], marker='o')
        axes[i].set_title(metric, fontsize=14)
        axes[i].set_xlabel("Games Level", fontsize=12)
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].grid(True)
    plt.tight_layout()
    plt.show()

# Function to display DataFrame
def display_dataframe_to_user(name, dataframe):
    print(f"\n=== {name} ===\n")
    print(dataframe.to_string(index=False))

# Assuming the simulation data was stored previously, we will mock or generate example data for the ML process

# Mock data generation based on the prior simulation structure
np.random.seed(42)
mock_data = {
    "Daily Ticket Sales": np.random.uniform(10000, 100000, 1000),
    "Scaled_Daily_Rewards": np.random.uniform(0.2, 1.5, 1000),
    "Contribution_Ratio": np.random.uniform(0.1, 0.8, 1000),
    "Remaining Pool (Qube)": np.random.uniform(500000, 10000000, 1000),
    "Exhausted Pool": np.random.choice([False, False, False, True], 1000)  # Mostly non-exhausted for analysis
}

# Create DataFrame
df = pd.DataFrame(mock_data)

# Filter non-exhausted pool rows
df = df[df["Exhausted Pool"] == False]

# Prepare features (X) and target (Y)
X = df[["Daily Ticket Sales", "Scaled_Daily_Rewards", "Contribution_Ratio"]]
y = df["Remaining Pool (Qube)"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest Regressor model
model = RandomForestRegressor(random_state=42)

# Define parameter grid for optimization
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Perform Grid Search for hyperparameter tuning
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', verbose=1, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # Retrieve the best model and evaluate its performance
# best_model = grid_search.best_estimator_
# r2_score_train = best_model.score(X_train, y_train)
# r2_score_test = best_model.score(X_test, y_test)

# # Feature importance analysis to determine optimized parameters
# feature_importance = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': best_model.feature_importances_
# }).sort_values(by='Importance', ascending=False)

# # Output model performance and feature importance
# {
#     "Best Parameters": grid_search.best_params_,
#     "Training R2 Score": r2_score_train,
#     "Testing R2 Score": r2_score_test,
#     "Feature Importance": feature_importance.to_dict(orient='records')
# }

# # Display the results for the redesigned model simulation
# display_dataframe_to_user(name="Redesigned Model Simulation with Scaling and Volatility", dataframe=feature_importance)

# Redesigned Model Parameters
# optimized_parameters = {
#     "Initial Daily Ticket Sales": 10_000,  # Start with low participation
#     "Max Daily Ticket Sales": 100_000,     # Gradually scale to high participation
#     "Daily Growth Rate": 1.02,             # 2% growth in ticket sales per day
#     "Activity-Based Percentage": 0.015,    # 1.5% contribution from daily ticket sales
#     "Base Transaction Fees": 500,          # Initial transaction fees
#     "Transaction Fee Growth": 10,          # Incremental transaction fee per 1,000 tickets
#     "Base Marketplace Fees": 1_000,        # Initial weekly marketplace fees
#     "Marketplace Fee Growth": 0.05,        # 5% of weekly sales growth contribution
#     "External Funding Contribution": 100_000,  # Increased monthly external funding injection
# }

# # Aggressive scaling thresholds for rewards
# optimized_scaling_thresholds = [
#     {"Remaining Pool Threshold": 15_000_000, "Scaling Factor": 0.7},  # 70% baseline rewards
#     {"Remaining Pool Threshold": 5_000_000, "Scaling Factor": 0.4},   # 40% baseline rewards
#     {"Remaining Pool Threshold": 1_000_000, "Scaling Factor": 0.1},   # 10% baseline rewards
# ]

# # Simulation function
# def simulate_optimized_model(parameters, total_pool, node_distribution, all_nodes, decades, scaling_thresholds):
#     results = []
#     remaining_pool = total_pool
#     daily_ticket_sales = parameters["Initial Daily Ticket Sales"]
#     total_days = sum(decades) * 365  # Number of simulation days

#     for day in range(1, total_days + 1):
#         # Adjust daily ticket sales with growth rate
#         if daily_ticket_sales < parameters["Max Daily Ticket Sales"]:
#             daily_ticket_sales *= parameters["Daily Growth Rate"]

#         # Add funding contributions
#         transaction_fee = parameters["Base Transaction Fees"] + (parameters["Transaction Fee Growth"] * (daily_ticket_sales // 1_000))
#         marketplace_fee = parameters["Base Marketplace Fees"] + (parameters["Marketplace Fee Growth"] * daily_ticket_sales)
#         activity_contribution = daily_ticket_sales * parameters["Activity-Based Percentage"]
#         remaining_pool += transaction_fee + (marketplace_fee / 7) + activity_contribution

#         # Inject external funding monthly
#         if day % 30 == 0:
#             remaining_pool += parameters["External Funding Contribution"]

#         # Determine scaling factor based on remaining pool thresholds
#         scaling_factor = next(
#             (threshold["Scaling Factor"]
#              for threshold in scaling_thresholds
#              if remaining_pool >= threshold["Remaining Pool Threshold"]),
#             0  # Fallback to 0 if no thresholds are met
#         )

#         # Stop simulation if pool is depleted
#         if scaling_factor == 0:
#             results.append({
#                 "Day": day,
#                 "Remaining Pool (Qube)": 0,
#                 "Daily Ticket Sales": daily_ticket_sales,
#                 "Exhausted Pool": True
#             })
#             break

#         # Scale down rewards dynamically
#         scaled_nodes = [
#             {"Stake": node["Stake"], "Reward/Game": node["Reward/Game"] * scaling_factor}
#             for node in all_nodes
#         ]

#         # Calculate total rewards for the day and deduct from remaining pool
#         daily_rewards = sum(
#             node["Reward/Game"] * 10 * node_distribution[node["Stake"]]
#             for node in scaled_nodes if node["Stake"] in node_distribution
#         )
#         remaining_pool -= daily_rewards

#         # Record daily results
#         results.append({
#             "Day": day,
#             "Remaining Pool (Qube)": max(0, remaining_pool),
#             "Daily Ticket Sales": daily_ticket_sales,
#             "Daily Rewards Paid Out (Qube)": daily_rewards,
#             "Exhausted Pool": remaining_pool <= 0
#         })

#     return pd.DataFrame(results)

# # Example usage
# # Define mock values for `total_pool`, `node_distribution`, and `all_nodes`
# total_pool_cap = 16_000_000  # Starting Qube pool
# node_distribution = {10_000: 100, 5_000: 200, 1_000: 500}  # Mock node distribution
# all_nodes = [{"Stake": 10_000, "Reward/Game": 7.2},
#              {"Stake": 5_000, "Reward/Game": 3.6},
#              {"Stake": 1_000, "Reward/Game": 1.8}]  # Reward settings
# decades = [10]  # Simulate over 10 years

# # Run the optimized simulation
# optimized_results = simulate_optimized_model(optimized_parameters, total_pool_cap, node_distribution, all_nodes, decades, optimized_scaling_thresholds)

# # Save results to CSV for your review
# optimized_results.to_csv("optimized_simulation_results.csv", index=False)

# # # Display the results for the redesigned model simulation
# display_dataframe_to_user(name="Redesigned Model Simulation with Scaling and Volatility", dataframe=optimized_results)

# Simulation function
# def simulate_optimized_model(parameters, total_pool, node_distribution, all_nodes, decades, scaling_thresholds):
#     results = []
#     remaining_pool = total_pool
#     daily_ticket_sales = parameters["Initial Daily Ticket Sales"]
#     total_days = sum(decades) * 365  # Number of simulation days

#     for day in range(1, total_days + 1):
#         # Adjust daily ticket sales with growth rate
#         if daily_ticket_sales < parameters["Max Daily Ticket Sales"]:
#             daily_ticket_sales *= parameters["Daily Growth Rate"]

#         # Add funding contributions
#         transaction_fee = parameters["Base Transaction Fees"] + (parameters["Transaction Fee Growth"] * (daily_ticket_sales // 1_000))
#         marketplace_fee = parameters["Base Marketplace Fees"] + (parameters["Marketplace Fee Growth"] * daily_ticket_sales)
#         activity_contribution = daily_ticket_sales * parameters["Activity-Based Percentage"]
#         remaining_pool += transaction_fee + (marketplace_fee / 7) + activity_contribution

#         # Inject external funding monthly
#         if day % 30 == 0:
#             remaining_pool += parameters["External Funding Contribution"]

#         # Determine scaling factor based on remaining pool thresholds
#         scaling_factor = next(
#             (threshold["Scaling Factor"]
#              for threshold in scaling_thresholds
#              if remaining_pool >= threshold["Remaining Pool Threshold"]),
#             0  # Fallback to 0 if no thresholds are met
#         )

#         # Stop simulation if pool is depleted
#         if scaling_factor == 0:
#             results.append({
#                 "Day": day,
#                 "Remaining Pool (Qube)": 0,
#                 "Daily Ticket Sales": daily_ticket_sales,
#                 "Exhausted Pool": True
#             })
#             break

#         # Scale down rewards dynamically
#         scaled_nodes = [
#             {"Stake": node["Stake"], "Reward/Game": node["Reward/Game"] * scaling_factor}
#             for node in all_nodes
#         ]

#         # Calculate total rewards for the day and deduct from remaining pool
#         daily_rewards = sum(
#             node["Reward/Game"] * 10 * node_distribution[node["Stake"]]
#             for node in scaled_nodes if node["Stake"] in node_distribution
#         )
#         remaining_pool -= daily_rewards

#         # Record daily results
#         results.append({
#             "Day": day,
#             "Remaining Pool (Qube)": max(0, remaining_pool),
#             "Daily Ticket Sales": daily_ticket_sales,
#             "Daily Rewards Paid Out (Qube)": daily_rewards,
#             "Exhausted Pool": remaining_pool <= 0
#         })

#     return pd.DataFrame(results)

# Define parameters for extreme ticket sales scenario
extreme_sales_parameters = {
    "Initial Daily Ticket Sales": 100_000,  # Sudden spike in ticket sales
    "Max Daily Ticket Sales": 1_000_000,    # Increased maximum ticket sales capacity
    "Daily Growth Rate": 1.02,              # 2% growth in ticket sales per day
    "Activity-Based Percentage": 0.015,     # 1.5% contribution from daily ticket sales
    "Base Transaction Fees": 500,           # Initial transaction fees
    "Transaction Fee Growth": 10,           # Incremental transaction fee per 1,000 tickets
    "Base Marketplace Fees": 1_000,         # Initial weekly marketplace fees
    "Marketplace Fee Growth": 0.05,         # 5% of weekly sales growth contribution
    "External Funding Contribution": 100_000,  # Increased monthly external funding injection
}

# Aggressive scaling thresholds for rewards
optimized_scaling_thresholds = [
    {"Remaining Pool Threshold": 15_000_000, "Scaling Factor": 0.7},  # 70% baseline rewards
    {"Remaining Pool Threshold": 5_000_000, "Scaling Factor": 0.4},   # 40% baseline rewards
    {"Remaining Pool Threshold": 1_000_000, "Scaling Factor": 0.1},   # 10% baseline rewards
]

# Mock node distribution and all_nodes for simulation
total_pool_cap = 16_000_000  # Starting Qube pool
node_distribution = {10_000: 100, 5_000: 200, 1_000: 500}  # Mock node distribution
all_nodes = [
    {"Stake": 10_000, "Reward/Game": 7.2},
    {"Stake": 5_000, "Reward/Game": 3.6},
    {"Stake": 1_000, "Reward/Game": 1.8}
]

# Simulate over a decade (10 years)
decades = [10]

# # Run the simulation
# extreme_sales_results = simulate_optimized_model(
#     extreme_sales_parameters,
#     total_pool_cap,
#     node_distribution,
#     all_nodes,
#     decades,
#     optimized_scaling_thresholds
# )

# # Save results to CSV for further analysis
# extreme_sales_results.to_csv("Extreme_Ticket_Sales_Impact_Simulation.csv", index=False)

# # Optional: Display a portion of the results
# print(extreme_sales_results.head())


# Adjust the parameters to tighten scaling and introduce a cap on daily rewards

# Updated scaling thresholds for tighter control
tightened_scaling_thresholds = [
    {"Remaining Pool Threshold": 15_000_000, "Scaling Factor": 0.6},  # Reduced to 60% baseline rewards
    {"Remaining Pool Threshold": 10_000_000, "Scaling Factor": 0.4},  # 40% baseline rewards
    {"Remaining Pool Threshold": 5_000_000, "Scaling Factor": 0.2},   # 20% baseline rewards
    {"Remaining Pool Threshold": 1_000_000, "Scaling Factor": 0.05},  # 5% baseline rewards
]

# Define a dynamic cap on daily rewards as a percentage of the remaining pool
daily_reward_cap_percentage = 0.001  # Cap rewards to 0.1% of the remaining pool

# Modify simulation function to include a daily reward cap
def simulate_with_tightened_scaling(parameters, total_pool, node_distribution, all_nodes, decades, scaling_thresholds, reward_cap_percentage):
    results = []
    remaining_pool = total_pool
    daily_ticket_sales = parameters["Initial Daily Ticket Sales"]
    total_days = sum(decades) * 365  # Number of simulation days

    for day in range(1, total_days + 1):
        # Adjust daily ticket sales with growth rate
        if daily_ticket_sales < parameters["Max Daily Ticket Sales"]:
            daily_ticket_sales *= parameters["Daily Growth Rate"]

        # Add funding contributions
        transaction_fee = parameters["Base Transaction Fees"] + (parameters["Transaction Fee Growth"] * (daily_ticket_sales // 1_000))
        marketplace_fee = parameters["Base Marketplace Fees"] + (parameters["Marketplace Fee Growth"] * daily_ticket_sales)
        activity_contribution = daily_ticket_sales * parameters["Activity-Based Percentage"]
        remaining_pool += transaction_fee + (marketplace_fee / 7) + activity_contribution

        # Inject external funding monthly
        if day % 30 == 0:
            remaining_pool += parameters["External Funding Contribution"]

        # Determine scaling factor based on remaining pool thresholds
        scaling_factor = next(
            (threshold["Scaling Factor"]
             for threshold in scaling_thresholds
             if remaining_pool >= threshold["Remaining Pool Threshold"]),
            0  # Fallback to 0 if no thresholds are met
        )

        # Stop simulation if pool is depleted
        if scaling_factor == 0:
            results.append({
                "Day": day,
                "Remaining Pool (Qube)": 0,
                "Daily Ticket Sales": daily_ticket_sales,
                "Exhausted Pool": True
            })
            break

        # Scale down rewards dynamically
        scaled_nodes = [
            {"Stake": node["Stake"], "Reward/Game": node["Reward/Game"] * scaling_factor}
            for node in all_nodes
        ]

        # Calculate total rewards for the day
        daily_rewards = sum(
            node["Reward/Game"] * 10 * node_distribution[node["Stake"]]
            for node in scaled_nodes if node["Stake"] in node_distribution
        )

        # Apply daily reward cap
        daily_reward_cap = reward_cap_percentage * remaining_pool
        if daily_rewards > daily_reward_cap:
            daily_rewards = daily_reward_cap

        # Deduct rewards from remaining pool
        remaining_pool -= daily_rewards

        # Record daily results
        results.append({
            "Day": day,
            "Remaining Pool (Qube)": max(0, remaining_pool),
            "Daily Ticket Sales": daily_ticket_sales,
            "Daily Rewards Paid Out (Qube)": daily_rewards,
            "Exhausted Pool": remaining_pool <= 0
        })

    return pd.DataFrame(results)

# Run the simulation with tightened scaling and daily reward cap
tightened_scaling_results = simulate_with_tightened_scaling(
    extreme_sales_parameters,
    total_pool_cap,
    node_distribution,
    all_nodes,
    decades,
    tightened_scaling_thresholds,
    daily_reward_cap_percentage
)

# Save results for review
tightened_scaling_results.to_csv("Tightened_Scaling_And_Reward_Cap_Simulation.csv", index=False)

# Display results to the user
print(tightened_scaling_results.head())
