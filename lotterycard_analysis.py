# # Reinitializing after reset
# from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


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

# # Generate a dataset for simulation with varying parameters
# bet_sizes = np.linspace(10, 1000, 100)  # Simulate bet sizes from $10 to $1000
# house_fees = np.linspace(0.01, 0.05, 10)  # House fee fractions from 1% to 5%
# epsilons = np.linspace(0.01, 0.1, 10)  # Consolation reward growth from 1% to 10%
# betas = np.linspace(0.1, 0.5, 10)  # Immediate compensation fractions from 10% to 50%
# g_cs = np.linspace(1.0, 2.0, 10)  # Deferred compensation growth rates
# n_gs = np.arange(3, 10)  # Number of games for jackpot growth

# # Create a dataset for ML
# data = []
# for b in bet_sizes:
#     for hf in house_fees:
#         for eps in epsilons:
#             for beta in betas:
#                 for g_c in g_cs:
#                     for n_g in n_gs:
#                         # Simulate results based on current parameters
#                         total_bet = 10 * b
#                         house_fee = hf * total_bet
#                         reward_pool = total_bet - house_fee
#                         consolation_prize_per_cc = b * (1 + eps)
#                         consolation_prize_pool = 9 * consolation_prize_per_cc
#                         jackpot_pool = reward_pool - consolation_prize_pool
#                         immediate_compensation = beta * b
#                         deferred_compensation = max((jackpot_pool - immediate_compensation) * g_c, 0)
#                         cumulative_jackpot = immediate_compensation + (deferred_compensation * n_g)
#                         net_gain_cja = cumulative_jackpot - b
                        
#                         # Store data
#                         data.append([b, hf, eps, beta, g_c, n_g, net_gain_cja])

# # Convert to numpy array
# data = np.array(data)

# # Prepare features (parameters) and target (net gain for CJA)
# X = data[:, :-1]  # Features: bet size, house fee, epsilon, beta, g_c, n_g
# y = data[:, -1]   # Target: net gain for CJA

# # Train a linear regression model
# model = LinearRegression()
# model.fit(X, y)

# # Find optimal parameters for positive CJA
# optimal_params = None
# for b in bet_sizes:
#     for hf in house_fees:
#         for eps in epsilons:
#             for beta in betas:
#                 for g_c in g_cs:
#                     for n_g in n_gs:
#                         # Predict net gain for CJA
#                         predicted_gain = model.predict([[b, hf, eps, beta, g_c, n_g]])[0]
#                         if predicted_gain > 0:  # Ensure positive net gain
#                             optimal_params = [b, hf, eps, beta, g_c, n_g]
#                             break
#                 if optimal_params:
#                     break
#             if optimal_params:
#                 break
#         if optimal_params:
#             break
#     if optimal_params:
#         break

# # Extract optimal parameters
# optimal_bet, optimal_house_fee, optimal_epsilon, optimal_beta, optimal_g_c, optimal_n_g = optimal_params

# # Prepare a summary
# optimal_result = {
#     "Optimal Bet Size": optimal_bet,
#     "House Fee Fraction": optimal_house_fee,
#     "Epsilon (CC Reward Growth)": optimal_epsilon,
#     "Immediate Compensation Fraction (Beta)": optimal_beta,
#     "Deferred Compensation Growth Rate (g_c)": optimal_g_c,
#     "Number of Games (n_g)": optimal_n_g,
# }

# # Convert to DataFrame and display
# optimal_result_df = pd.DataFrame([optimal_result])
# display_dataframe_to_user(name="Optimal Parameters for Positive CJA Net Gain", dataframe=optimal_result_df)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

# # Generate data for machine learning to find optimal parameters
# def generate_sustainability_data():
#     bet_sizes = np.linspace(1, 100, 100)  # Simulate bet sizes from $1 to $100
#     house_fees = np.linspace(0.01, 0.05, 10)  # House fee fractions from 1% to 5%
#     epsilons = np.linspace(0.01, 0.1, 10)  # Consolation reward growth from 1% to 10%
#     betas = np.linspace(0.1, 0.5, 10)  # Immediate compensation fractions from 10% to 50%
#     g_cs = np.linspace(1.0, 2.5, 10)  # Deferred compensation growth rates
#     n_gs = np.arange(2, 10)  # Number of games for jackpot growth

#     data = []
#     for b in bet_sizes:
#         for hf in house_fees:
#             for eps in epsilons:
#                 for beta in betas:
#                     for g_c in g_cs:
#                         for n_g in n_gs:
#                             # Calculate sustainability factor
#                             total_bet = 10 * b
#                             house_fee = hf * total_bet
#                             reward_pool = total_bet - house_fee
#                             consolation_prize_pool = 9 * (b * (1 + eps))
#                             jackpot_pool = reward_pool - consolation_prize_pool
#                             immediate_compensation = beta * b
#                             deferred_compensation_per_game = max((jackpot_pool - immediate_compensation) * g_c, 0)
#                             total_required_deferred_compensation = deferred_compensation_per_game * n_g
#                             sustainability_factor = jackpot_pool / total_required_deferred_compensation if total_required_deferred_compensation > 0 else 0
#                             data.append([b, hf, eps, beta, g_c, n_g, sustainability_factor])
#     return np.array(data)

# # Generate data
# data = generate_sustainability_data()

# # Prepare features and target
# X = data[:, :-1]  # Features: bet size, house fee, epsilon, beta, g_c, n_g
# y = data[:, -1]   # Target: sustainability factor

# # Train a RandomForestRegressor to predict sustainability factor
# model = RandomForestRegressor()
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [5, 10, 20],
# }
# grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
# grid_search.fit(X, y)

# # Find minimal bet and parameters for sustainability factor >= 1
# optimal_params = None
# for row in data:
#     if row[-1] >= 1:  # Sustainability factor >= 1
#         optimal_params = row
#         break

# # Prepare results
# if optimal_params is not None:
#     result = {
#         "Minimal Bet Size": optimal_params[0],
#         "House Fee Fraction": optimal_params[1],
#         "Epsilon (CC Reward Growth)": optimal_params[2],
#         "Immediate Compensation Fraction (Beta)": optimal_params[3],
#         "Deferred Compensation Growth Rate (g_c)": optimal_params[4],
#         "Number of Games (n_g)": int(optimal_params[5]),
#         "Sustainability Factor": optimal_params[6],
#     }
#     optimal_params_df = pd.DataFrame([result])
#     display_dataframe_to_user(name="Optimal Parameters for Sustainability Factor >= 1", dataframe=optimal_params_df)
# else:
#     print("No parameters found with a sustainability factor >= 1.")

# # Generate corrected sustainability data with constraints for training
# def generate_corrected_sustainability_data():
#     bet_sizes = np.linspace(1, 50, 20)  # Reduced bet sizes
#     house_fees = np.linspace(0.01, 0.03, 5)  # Narrower range for house fee fractions
#     epsilons = np.linspace(0.01, 0.05, 5)  # Reduced epsilon range
#     betas = np.linspace(0.1, 0.3, 5)  # Narrower beta range
#     g_cs = np.linspace(1.0, 2.0, 5)  # Reduced growth rates
#     n_gs = np.arange(2, 6)  # Fewer games for jackpot growth

#     data = []
#     for b in bet_sizes:
#         for hf in house_fees:
#             for eps in epsilons:
#                 for beta in betas:
#                     for g_c in g_cs:
#                         for n_g in n_gs:
#                             # Calculate sustainability factor
#                             total_bet = 10 * b
#                             house_fee = hf * total_bet
#                             reward_pool = total_bet - house_fee
#                             consolation_prize_pool = 9 * (b * (1 + eps))
#                             jackpot_pool = reward_pool - consolation_prize_pool
#                             immediate_compensation = beta * b
#                             deferred_compensation_per_game = max((jackpot_pool - immediate_compensation) * g_c, 0)
#                             total_required_deferred_compensation = deferred_compensation_per_game * n_g
#                             sustainability_factor = (
#                                 jackpot_pool / total_required_deferred_compensation
#                                 if total_required_deferred_compensation > 0
#                                 else 0
#                             )
#                             if sustainability_factor >= 1:  # Enforce the constraint
#                                 data.append([b, hf, eps, beta, g_c, n_g, sustainability_factor])
#     return np.array(data)

# # Generate corrected data
# corrected_data = generate_corrected_sustainability_data()

# # Prepare features and target
# X = corrected_data[:, :-1]  # Features: bet size, house fee, epsilon, beta, g_c, n_g
# y = corrected_data[:, -1]   # Target: sustainability factor

# # Train a RandomForestRegressor to predict sustainability factor
# model = RandomForestRegressor()
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [5, 10, 20],
# }
# # Prepare features and target
# X_corrected = corrected_data[:, :-1]  # Features
# y_corrected = corrected_data[:, -1]   # Target

# # Train a RandomForestRegressor to predict sustainability factor with constraints
# model_corrected = RandomForestRegressor()
# grid_search_corrected = GridSearchCV(model_corrected, param_grid, cv=3, scoring='neg_mean_squared_error')
# grid_search_corrected.fit(X_corrected, y_corrected)

# # Find minimal bet and parameters for sustainability factor >= 1
# optimal_params_corrected = None
# for row in corrected_data:
#     if row[-1] >= 1:  # Sustainability factor >= 1
#         optimal_params_corrected = row
#         break

# # Prepare results
# if optimal_params_corrected is not None:
#     result_corrected = {
#         "Minimal Bet Size": optimal_params_corrected[0],
#         "House Fee Fraction": optimal_params_corrected[1],
#         "Epsilon (CC Reward Growth)": optimal_params_corrected[2],
#         "Immediate Compensation Fraction (Beta)": optimal_params_corrected[3],
#         "Deferred Compensation Growth Rate (g_c)": optimal_params_corrected[4],
#         "Number of Games (n_g)": int(optimal_params_corrected[5]),
#         "Sustainability Factor": optimal_params_corrected[6],
#     }
#     optimal_params_corrected_df = pd.DataFrame([result_corrected])
#     display_dataframe_to_user(
#         name="Corrected Optimal Parameters for Sustainability Factor >= 1", dataframe=optimal_params_corrected_df
#     )
# else:
#     print("No parameters found with a sustainability factor >= 1.")

# Generate constrained sustainability data with improved pool tracking
def generate_refined_sustainability_data():
    bet_sizes = np.linspace(1, 50, 20)  # Reduced bet sizes
    house_fees = np.linspace(0.01, 0.03, 5)  # Narrower range for house fee fractions
    epsilons = np.linspace(0.01, 0.05, 5)  # Reduced epsilon range
    betas = np.linspace(0.1, 0.3, 5)  # Narrower beta range
    g_cs = np.linspace(1.0, 2.0, 5)  # Reduced growth rates
    n_gs = np.arange(2, 6)  # Fewer games for jackpot growth

    data = []
    for b in bet_sizes:
        for hf in house_fees:
            for eps in epsilons:
                for beta in betas:
                    for g_c in g_cs:
                        for n_g in n_gs:
                            # Step 1: Calculate initial conditions
                            total_bet = 10 * b
                            house_fee = hf * total_bet
                            reward_pool = total_bet - house_fee
                            consolation_prize_pool = 9 * (b * (1 + eps))
                            jackpot_pool = reward_pool - consolation_prize_pool

                            # Step 2: Validate pool size
                            if jackpot_pool <= 0:
                                continue

                            # Step 3: Immediate compensation
                            immediate_compensation = beta * b
                            if immediate_compensation > jackpot_pool:
                                continue

                            # Step 4: Track dynamic deferred compensation
                            remaining_pool = jackpot_pool - immediate_compensation
                            deferred_compensation = 0
                            for _ in range(n_g):
                                current_deferred = max(remaining_pool * g_c, 0)
                                if current_deferred > remaining_pool:
                                    current_deferred = remaining_pool
                                deferred_compensation += current_deferred
                                remaining_pool -= current_deferred

                            # Step 5: Calculate sustainability factor
                            if deferred_compensation + immediate_compensation < jackpot_pool:
                                continue

                            sustainability_factor = (
                                jackpot_pool / (deferred_compensation + immediate_compensation)
                            )

                            if sustainability_factor >= 1:
                                data.append([b, hf, eps, beta, g_c, n_g, sustainability_factor])
    return np.array(data)

# Generate refined data
refined_data = generate_refined_sustainability_data()

# Prepare features and target
X = refined_data[:, :-1]  # Features: bet size, house fee, epsilon, beta, g_c, n_g
y = refined_data[:, -1]   # Target: sustainability factor

# Train a RandomForestRegressor to predict sustainability factor
model = RandomForestRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
}

# Prepare features and target
X_refined = refined_data[:, :-1]  # Features
y_refined = refined_data[:, -1]   # Target

# Train RandomForestRegressor with refined data
model_refined = RandomForestRegressor()
grid_search_refined = GridSearchCV(model_refined, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search_refined.fit(X_refined, y_refined)

# Find minimal bet and parameters for SF >= 1
optimal_params_refined = None
for row in refined_data:
    if row[-1] >= 1:
        optimal_params_refined = row
        break

# Prepare results
if optimal_params_refined is not None:
    result_refined = {
        "Minimal Bet Size": optimal_params_refined[0],
        "House Fee Fraction": optimal_params_refined[1],
        "Epsilon (CC Reward Growth)": optimal_params_refined[2],
        "Immediate Compensation Fraction (Beta)": optimal_params_refined[3],
        "Deferred Compensation Growth Rate (g_c)": optimal_params_refined[4],
        "Number of Games (n_g)": int(optimal_params_refined[5]),
        "Sustainability Factor": optimal_params_refined[6],
    }
    optimal_params_refined_df = pd.DataFrame([result_refined])
    display_dataframe_to_user(
        name="Refined Optimal Parameters for Sustainability Factor >= 1", dataframe=optimal_params_refined_df
    )
else:
    print("No parameters found with a sustainability factor >= 1.")
