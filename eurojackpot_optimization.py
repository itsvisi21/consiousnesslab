import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import optuna
import json
import concurrent.futures
import threading
import logging

# Set up logging for monitoring
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Semaphore to limit the number of parallel threads
SEMAPHORE_LIMIT = 5
semaphore = threading.Semaphore(SEMAPHORE_LIMIT)

# Define a function to normalize values
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Define a function to compute a custom score for predictions
def compute_score(prediction, weights, R_min, R_max, B_min, B_max, P_max):
    R_norm = normalize(prediction["Randomness"], R_min, R_max)
    B_norm = normalize(prediction["Balance"], B_min, B_max)
    P_norm = prediction["Proximity"] / P_max
    N_score = 1 if prediction["Is Novel"] else 0
    return (weights["w_R"] * R_norm + weights["w_B"] * B_norm + 
            weights["w_P"] * P_norm + weights["w_N"] * N_score)

# Define a function to print the results
def print_pretty_output(best_params, ranked_predictions):
    print("\n### Best Parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    print("\n### Ranked Predictions:")
    for i, prediction in enumerate(ranked_predictions, start=1):
        print(f"\n{i}. Prediction (Score: {prediction['Score']:.3f})")
        print(f"   Main Numbers: {list(map(int, prediction['Main Numbers']))}")
        print(f"   Euro Numbers: {list(map(int, prediction['Euro Numbers']))}")
        print(f"   Proximity: {prediction['Proximity']}")
        print(f"   Randomness: {prediction['Randomness']:.2f}")
        print(f"   Balance: {prediction['Balance']:.2f}")
        print(f"   Is Novel: {'Yes' if prediction['Is Novel'] else 'No'}")

# Load the dataset
with open('./eurojackpot/eurojackpot_complete.json', 'r') as file:
    compact_data = json.load(file)

# Prepare the dataset
eurojackpot_data = {
    "Date": [entry.split(",")[0] for entry in compact_data],
    "Main_Numbers": [list(map(int, entry.split(",")[1:6])) for entry in compact_data],
    "Euro_Numbers": [list(map(int, entry.split(",")[6:8])) for entry in compact_data],
}
eurojackpot_df = pd.DataFrame(eurojackpot_data)

# Separate main numbers into individual columns
numbers_df = pd.DataFrame(eurojackpot_df['Main_Numbers'].tolist(), columns=[f"Number_{i}" for i in range(1, 6)])
eurojackpot_df = pd.concat([eurojackpot_df.drop(columns=['Main_Numbers']), numbers_df], axis=1)

# Feature engineering: Calculate frequencies, gaps, randomness, balance
def add_features(df):
    frequencies = {num: np.count_nonzero(df[[f"Number_{i+1}" for i in range(5)]].values.flatten() == num) for num in range(1, 51)}
    df["Hot_Count"] = df.apply(lambda row: sum(1 for num in row[[f"Number_{i+1}" for i in range(5)]] if frequencies[num] > 5), axis=1)
    df["Cold_Count"] = df.apply(lambda row: sum(1 for num in row[[f"Number_{i+1}" for i in range(5)]] if frequencies[num] <= 5), axis=1)
    df["Sequential_Gap"] = df.apply(lambda row: np.mean(np.diff(sorted(row[[f"Number_{i+1}" for i in range(5)]]))), axis=1)
    df[["Randomness", "Balance"]] = df.apply(lambda row: pd.Series([np.std(row[[f"Number_{i+1}" for i in range(5)]]), np.mean(row[[f"Number_{i+1}" for i in range(5)]])]), axis=1)
    return df

eurojackpot_df = add_features(eurojackpot_df)

# Prepare training and testing datasets
X = eurojackpot_df.drop(columns=["Date", "Euro_Numbers"])
y = eurojackpot_df["Balance"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting Model
balance_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42)
balance_model.fit(X_train, y_train)

# Evaluate Gradient Boosting Model
balance_predictions = balance_model.predict(X_test)
balance_mse = mean_squared_error(y_test, balance_predictions)

# Run Optuna optimization
def custom_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
    }
    model = XGBRegressor(objective='reg:squarederror', random_state=42, **params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score_deviation = np.mean(np.abs(predictions - 1))
    return score_deviation

study = optuna.create_study(direction="minimize")
study.optimize(custom_objective, n_trials=50)

# Get the best parameters
best_params = study.best_params

# Final model training and prediction
final_model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
final_model.fit(X_train, y_train)
final_predictions = final_model.predict(X_test)
final_score_deviation = np.mean(np.abs(final_predictions - 1))

# Calculate proximity and generate predictions
def calculate_proximity(predicted_main, predicted_euro, last_known_main, last_known_euro):
    main_overlap = len(set(predicted_main) & set(last_known_main))
    euro_overlap = len(set(predicted_euro) & set(last_known_euro))
    return main_overlap + euro_overlap

last_known_row = eurojackpot_df[eurojackpot_df["Date"] == "2024-12-27"].iloc[0]
last_known_main = last_known_row[[f"Number_{i}" for i in range(1, 6)]].tolist()
last_known_euro = last_known_row["Euro_Numbers"]

# Validate the predicted draw against historical data
def validate_prediction(predicted_main, predicted_euro, historical_data):
    """
    Check if the predicted draw matches any historical draws.
    """
    main_match = historical_data.apply(
        lambda row: set(predicted_main) == set(row[[f"Number_{i}" for i in range(1, 6)]]), axis=1
    )
    euro_match = historical_data.apply(
        lambda row: set(predicted_euro) == set(row["Euro_Numbers"]), axis=1
    )
    matches = main_match & euro_match
    return matches.any()

# Function to calculate randomness and balance metrics for a prediction
def calculate_randomness_and_balance(predicted_main):
    """
    Calculate randomness (standard deviation) and balance (mean) for a set of main numbers.
    """
    randomness = np.std(predicted_main)  # Standard deviation as randomness factor
    balance = np.mean(predicted_main)   # Mean as balance factor
    return randomness, balance

# Function to simulate a single prediction (parallelized)
def simulate_single_prediction(eurojackpot_df, last_known_main, last_known_euro, weights, R_min, R_max, B_min, B_max, P_max, proximity_target, score_min):
    with semaphore:
        while True:
            predicted_main = sorted(np.random.choice(range(1, 51), 5, replace=False))
            predicted_euro = sorted(np.random.choice(range(1, 11), 2, replace=False))
            proximity = calculate_proximity(predicted_main, predicted_euro, last_known_main, last_known_euro)
            if proximity == proximity_target:
                randomness, balance = calculate_randomness_and_balance(predicted_main)
                prediction = {
                    "Main Numbers": predicted_main,
                    "Euro Numbers": predicted_euro,
                    "Proximity": proximity,
                    "Randomness": randomness,
                    "Balance": balance,
                    "Is Novel": not validate_prediction(predicted_main, predicted_euro, eurojackpot_df),
                }
                predicted_score = compute_score(prediction, weights, R_min, R_max, B_min, B_max, P_max)
                if predicted_score >= score_min:
                    prediction["Score"] = predicted_score
                    logging.info(f"Generated prediction with score: {predicted_score:.3f}")
                    return prediction

# Parallel prediction generation
def generate_predictions_parallel(eurojackpot_df, last_known_main, last_known_euro, num_simulations=10, proximity_target=1, score_min=0.75):
    predictions = []
    weights = {"w_R": 0.3, "w_B": 0.3, "w_P": 0.2, "w_N": 0.2}
    R_min, R_max = eurojackpot_df["Randomness"].min(), eurojackpot_df["Randomness"].max()
    B_min, B_max = eurojackpot_df["Balance"].min(), eurojackpot_df["Balance"].max()
    P_max = 2

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                simulate_single_prediction,
                eurojackpot_df,
                last_known_main,
                last_known_euro,
                weights,
                R_min,
                R_max,
                B_min,
                B_max,
                P_max,
                proximity_target,
                score_min,
            )
            for _ in range(num_simulations)
        ]
        for future in concurrent.futures.as_completed(futures):
            predictions.append(future.result())

    return predictions


# Call the parallelized prediction generator
ranked_predictions = generate_predictions_parallel(
    eurojackpot_df,
    last_known_main,
    last_known_euro,
    num_simulations=10,
    proximity_target=1,
    score_min=0.75,
)

# Rank predictions and print
ranked_predictions = sorted(ranked_predictions, key=lambda x: x["Score"], reverse=True)
print_pretty_output(best_params, ranked_predictions)

# Final output for predictions
if ranked_predictions:
    print(f"\nMost Likely Prediction (Score: {ranked_predictions[0]['Score']:.3f}):")
    print(f"Main Numbers: {list(map(int, ranked_predictions[0]['Main Numbers']))}")
    print(f"Euro Numbers: {list(map(int, ranked_predictions[0]['Euro Numbers']))}")
else:
    print("No valid predictions found.")

