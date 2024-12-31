import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import json
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
    }
    model = XGBRegressor(objective='reg:squarederror', random_state=42, **params)
    model.fit(X_train_b, y_train_b)
    predictions = model.predict(X_test_b)
    return mean_squared_error(y_test_b, predictions)

def calculate_proximity(predicted_main, predicted_euro, last_known_main, last_known_euro):
    """
    Calculate proximity as the count of matching numbers between the predicted draw and the last known draw.
    """
    main_overlap = len(set(predicted_main) & set(last_known_main))
    euro_overlap = len(set(predicted_euro) & set(last_known_euro))
    return {
        "Main Numbers Overlap": main_overlap,
        "Euro Numbers Overlap": euro_overlap,
        "Total Overlap": main_overlap + euro_overlap
    }

# Calculate features for the dataset
def calculate_features(main_numbers):
    randomness = np.std(main_numbers)  # Standard deviation as randomness factor
    balance = np.mean(main_numbers)   # Mean as balance factor
    return randomness, balance

def calculate_number_frequencies(historical_data, num_range=50):
    # Flatten all main numbers into a single list
    all_numbers = historical_data[[f"Number_{i+1}" for i in range(5)]].values.flatten()
    # Count frequency of each number in the range 1 to num_range
    frequencies = {num: np.count_nonzero(all_numbers == num) for num in range(1, num_range + 1)}
    return frequencies

# Add frequency-based features to the dataset
def add_frequency_features(historical_data, frequencies, hot_threshold=5):
    hot_counts = []
    cold_counts = []

    for _, row in historical_data.iterrows():
        main_numbers = row[[f"Number_{i+1}" for i in range(5)]]
        hot_count = sum(1 for num in main_numbers if frequencies[num] > hot_threshold)
        cold_count = sum(1 for num in main_numbers if frequencies[num] <= hot_threshold)
        hot_counts.append(hot_count)
        cold_counts.append(cold_count)
    
    historical_data["Hot_Count"] = hot_counts
    historical_data["Cold_Count"] = cold_counts
    return historical_data

# Calculate sequential gaps as a feature
def add_sequential_gaps(historical_data):
    sequential_gaps = []
    for _, row in historical_data.iterrows():
        main_numbers = sorted(row[[f"Number_{i+1}" for i in range(5)]]);
        gap = np.mean(np.diff(main_numbers))
        sequential_gaps.append(gap)
    historical_data["Sequential_Gap"] = sequential_gaps
    return historical_data

# Generate Next Predicted Draw
def generate_next_draw(best_params, dataset):
    model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    model.fit(dataset.drop(columns=["Date", "Euro_Numbers", "Randomness", "Balance"]), dataset["Balance"])
    predictions = model.predict(dataset.drop(columns=["Date", "Euro_Numbers", "Randomness", "Balance"]))
    next_draw = dataset.iloc[np.argmax(predictions)][[f"Number_{i}" for i in range(1, 6)]]
    euro_numbers = [np.random.randint(1, 11) for _ in range(2)]  # Randomized Euro numbers
    return next_draw.tolist(), euro_numbers

# Load the saved compact format JSON data
with open('./eurojackpot/eurojackpot_complete.json', 'r') as file:
    compact_data = json.load(file)

# Convert the compact format to the structured format
eurojackpot_data = {
    "Date": [entry.split(",")[0] for entry in compact_data],
    "Main_Numbers": [list(map(int, entry.split(",")[1:6])) for entry in compact_data],
    "Euro_Numbers": [list(map(int, entry.split(",")[6:8])) for entry in compact_data],
}

# Convert to DataFrame and flatten main numbers into individual columns
eurojackpot_df = pd.DataFrame(eurojackpot_data)
numbers_df = pd.DataFrame(eurojackpot_df['Main_Numbers'].tolist(), columns=[f"Number_{i}" for i in range(1, 6)])
eurojackpot_df = pd.concat([eurojackpot_df.drop(columns=['Main_Numbers']), numbers_df], axis=1)

frequencies = calculate_number_frequencies(eurojackpot_df)
eurojackpot_df = add_frequency_features(eurojackpot_df, frequencies)
eurojackpot_df = add_sequential_gaps(eurojackpot_df)

eurojackpot_df[['Randomness', 'Balance']] = eurojackpot_df.apply(
    lambda row: pd.Series(calculate_features([row[f"Number_{i+1}"] for i in range(5)])),
    axis=1
)

X = eurojackpot_df.drop(columns=["Date", "Euro_Numbers", "Hot_Count", "Cold_Count", "Sequential_Gap"])
X["Hot_Count"] = eurojackpot_df["Hot_Count"]
X["Cold_Count"] = eurojackpot_df["Cold_Count"]
X["Sequential_Gap"] = eurojackpot_df["Sequential_Gap"]

# Prepare training and testing datasets with new features
randomness_y = eurojackpot_df["Randomness"]
balance_y = eurojackpot_df["Balance"]

X = eurojackpot_df.drop(columns=["Date", "Euro_Numbers", "Randomness", "Balance"])
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, balance_y, test_size=0.2, random_state=42)

# Step 4: Train Gradient Boosting Model
balance_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42)
balance_model.fit(X_train_b, y_train_b)

# Evaluate Gradient Boosting Model
balance_predictions_full = balance_model.predict(X_test_b)
balance_mse_full = mean_squared_error(y_test_b, balance_predictions_full)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best parameters:", study.best_params)

next_main_numbers, next_euro_numbers = generate_next_draw(study.best_params, eurojackpot_df)

# print("Next Predicted Draw:")
# print("Main Numbers:", [int(num) for num in next_main_numbers])
# print("Euro Numbers:", next_euro_numbers)

# # Output the performance comparison
# print({
#     "Gradient Boosting Balance Prediction Error (MSE)": balance_mse_full
# })

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

# Function to calculate proximity trends between consecutive draws
def calculate_proximity_trends(historical_data):
    proximities = []
    for i in range(1, len(historical_data)):
        previous_row = historical_data.iloc[i - 1]
        current_row = historical_data.iloc[i]
        
        previous_main = previous_row[[f"Number_{i}" for i in range(1, 6)]].tolist()
        previous_euro = previous_row["Euro_Numbers"]
        
        current_main = current_row[[f"Number_{i}" for i in range(1, 6)]].tolist()
        current_euro = current_row["Euro_Numbers"]
        
        proximity = calculate_proximity(current_main, current_euro, previous_main, previous_euro)
        proximities.append(proximity["Total Overlap"])
    
    return proximities

# Calculate proximity trends
proximity_trends = calculate_proximity_trends(eurojackpot_df)

# Output proximity trend analysis
proximity_trend_analysis = {
    "Proximity Trends": proximity_trends,
    "Average Proximity": np.mean(proximity_trends),
    "Maximum Proximity": np.max(proximity_trends),
    "Minimum Proximity": np.min(proximity_trends)
}

# print(proximity_trend_analysis)


# Validate the predicted draw
predicted_main_numbers = [int(num) for num in next_main_numbers]
predicted_euro_numbers = next_euro_numbers
validation_result = validate_prediction(predicted_main_numbers, predicted_euro_numbers, eurojackpot_df)

# Output validation results
# print({
#     "Validation Result": validation_result,
#     "Predicted Main Numbers": predicted_main_numbers,
#     "Predicted Euro Numbers": predicted_euro_numbers
# })

# Adjust to ensure the last known draw corresponds to the specific date "Friday 27th December 2024"

# Filter the dataset for the specified date
last_known_row = eurojackpot_df[eurojackpot_df["Date"] == "2024-12-27"].iloc[0]

# Extract last known draw details
last_known_main = last_known_row[[f"Number_{i}" for i in range(1, 6)]].tolist()
last_known_euro = last_known_row["Euro_Numbers"]

# Recalculate proximity based on the specified last known draw
proximity_result = calculate_proximity(predicted_main_numbers, predicted_euro_numbers, last_known_main, last_known_euro)

# Output the updated proximity results
# print({
#     "Last Known Main Numbers": last_known_main,
#     "Last Known Euro Numbers": last_known_euro,
#     "Predicted Main Numbers": predicted_main_numbers,
#     "Predicted Euro Numbers": predicted_euro_numbers,
#     "Proximity Result": proximity_result
# })
# Function to generate proximity-driven predictions
def generate_proximity_driven_predictions(historical_data, num_simulations=10, proximity_target=1):
    """
    Generate predictions that aim to align with historical proximity trends.
    """
    predictions = []
    for _ in range(num_simulations):
        while True:
            # Generate random main and Euro numbers within the valid ranges
            predicted_main = sorted(np.random.choice(range(1, 51), 5, replace=False))
            predicted_euro = sorted(np.random.choice(range(1, 11), 2, replace=False))
            
            # Calculate proximity to the last known draw
            last_known_main = historical_data.iloc[-1][[f"Number_{i}" for i in range(1, 6)]].tolist()
            last_known_euro = historical_data.iloc[-1]["Euro_Numbers"]
            proximity = calculate_proximity(predicted_main, predicted_euro, last_known_main, last_known_euro)
            
            # Accept the prediction if proximity matches the target
            if proximity["Total Overlap"] == proximity_target:
                predictions.append({
                    "Main Numbers": predicted_main,
                    "Euro Numbers": predicted_euro,
                    "Proximity": proximity["Total Overlap"]
                })
                break
    return predictions

# Simulate proximity-driven predictions
proximity_target = 1  # Aligning with the average proximity
simulated_predictions = generate_proximity_driven_predictions(eurojackpot_df, num_simulations=5, proximity_target=proximity_target)

# Output simulated predictions
# print(simulated_predictions)

# Function to calculate randomness and balance metrics for a prediction
def calculate_randomness_and_balance(predicted_main):
    """
    Calculate randomness (standard deviation) and balance (mean) for a set of main numbers.
    """
    randomness = np.std(predicted_main)
    balance = np.mean(predicted_main)
    return {
        "Randomness": randomness,
        "Balance": balance
    }

# Analyze randomness and balance for each prediction
analysis_results = []
for prediction in simulated_predictions:
    metrics = calculate_randomness_and_balance(prediction["Main Numbers"])
    analysis_results.append({
        "Main Numbers": prediction["Main Numbers"],
        "Euro Numbers": prediction["Euro Numbers"],
        "Proximity": prediction["Proximity"],
        "Randomness": metrics["Randomness"],
        "Balance": metrics["Balance"]
    })

# Output the analysis results
# print(analysis_results)
# Analyze the historical randomness and balance ranges for selection
historical_randomness = eurojackpot_df.apply(
    lambda row: np.std(row[[f"Number_{i}" for i in range(1, 6)]]), axis=1
)
historical_balance = eurojackpot_df.apply(
    lambda row: np.mean(row[[f"Number_{i}" for i in range(1, 6)]]), axis=1
)

# Define historical ranges
historical_randomness_range = (historical_randomness.min(), historical_randomness.max())
historical_balance_range = (historical_balance.min(), historical_balance.max())

# Filter predictions that fall within the historical ranges
filtered_predictions = [
    prediction for prediction in analysis_results
    if (
        historical_randomness_range[0] <= prediction["Randomness"] <= historical_randomness_range[1] and
        historical_balance_range[0] <= prediction["Balance"] <= historical_balance_range[1]
    )
]

# Output the most plausible prediction
filtered_predictions if filtered_predictions else "No predictions fell within historical ranges."

# print(filtered_predictions)
# Validate the filtered predictions for novelty against historical data
novelty_results = []
for prediction in filtered_predictions:
    is_novel = not validate_prediction(
        prediction["Main Numbers"], prediction["Euro Numbers"], eurojackpot_df
    )
    novelty_results.append({
        "Main Numbers": prediction["Main Numbers"],
        "Euro Numbers": prediction["Euro Numbers"],
        "Proximity": prediction["Proximity"],
        "Randomness": prediction["Randomness"],
        "Balance": prediction["Balance"],
        "Is Novel": is_novel
    })

# Best parameters from Optuna optimization
best_params = study.best_params

# Print the best parameters from the optimization
print("Best Parameters from Optimization:")
print("{")
for key, value in best_params.items():
    print(f'    "{key}": {value},')
print("}")

# Print the novelty validation results
print("\nNovelty Validation Results:")
print("[")
for result in novelty_results:
    print("    {")
    print(f'        "Main Numbers": {[int(num) for num in result["Main Numbers"]]},')
    print(f'        "Euro Numbers": {[int(num) for num in result["Euro Numbers"]]},')
    print(f'        "Proximity": {result["Proximity"]},')
    print(f'        "Randomness": {result["Randomness"]},')
    print(f'        "Balance": {result["Balance"]},')
    print(f'        "Is Novel": {str(result["Is Novel"]).lower()}')
    print("    },")
print("]")
