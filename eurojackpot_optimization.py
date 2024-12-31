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

def print_pretty_output(best_params, ranked_predictions):
    print("\n### Best Parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    print("\n### Ranked Predictions:")
    for i, prediction in enumerate(ranked_predictions, start=1):
        if "Score" not in prediction:
            print(f"\n{i}. Prediction is missing a 'Score': {prediction}")
            continue
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


# Now call generate_predictions()
# simulated_predictions = generate_predictions_parallel(eurojackpot_df, num_simulations=10, proximity_target=1,score_min=0.75)

# # Rank the predictions by their score
# ranked_predictions = sorted(simulated_predictions, key=lambda x: x["Score"], reverse=True)

# # Print the ranked predictions
# print_pretty_output(best_params, ranked_predictions)

# # Final output for predictions
# if len(ranked_predictions) > 0:
#     print(f"\nMost Likely Prediction (Score: {ranked_predictions[0]['Score']:.3f}):")
#     print(f"Main Numbers: {list(map(int, ranked_predictions[0]['Main Numbers']))}")
#     print(f"Euro Numbers: {list(map(int, ranked_predictions[0]['Euro Numbers']))}")
# else:
#     print("No valid predictions found.")

import random

# Initialize parameters
POPULATION_SIZE = 100
GENERATIONS = 25
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

def is_valid_prediction(prediction):
    # Validate main numbers: 5 unique numbers between 1 and 50
    if len(prediction["Main Numbers"]) != 5 or not all(1 <= num <= 50 for num in prediction["Main Numbers"]):
        return False

    # Validate euro numbers: 2 unique numbers between 1 and 12
    if len(prediction["Euro Numbers"]) != 2 or not all(1 <= num <= 12 for num in prediction["Euro Numbers"]):
        return False

    # Ensure uniqueness of numbers
    if len(set(prediction["Main Numbers"])) != 5 or len(set(prediction["Euro Numbers"])) != 2:
        return False

    return True


# Fitness function to evaluate predictions
def fitness_function(prediction, weights, R_min, R_max, B_min, B_max, P_max):
    randomness, balance = prediction["Randomness"], prediction["Balance"]
    proximity = prediction["Proximity"]
    novelty = 1 if prediction["Is Novel"] else 0
    score = compute_score(
        {"Randomness": randomness, "Balance": balance, "Proximity": proximity, "Is Novel": novelty},
        weights,
        R_min,
        R_max,
        B_min,
        B_max,
        P_max,
    )
    prediction["Score"] = score  # Ensure the score is saved in the prediction
    return score

# Generate a random prediction
def generate_random_prediction(weights, R_min, R_max, B_min, B_max, P_max):
    while True:
        main_numbers = sorted(np.random.choice(range(1, 51), 5, replace=False))
        euro_numbers = sorted(np.random.choice(range(1, 13), 2, replace=False))  # Update range to 1-12
        randomness, balance = calculate_randomness_and_balance(main_numbers)
        proximity = calculate_proximity(main_numbers, euro_numbers, last_known_main, last_known_euro)
        is_novel = not validate_prediction(main_numbers, euro_numbers, eurojackpot_df)
        score = compute_score(
            {"Randomness": randomness, "Balance": balance, "Proximity": proximity, "Is Novel": is_novel},
            weights,
            R_min,
            R_max,
            B_min,
            B_max,
            P_max,
        )
        prediction = {
            "Main Numbers": main_numbers,
            "Euro Numbers": euro_numbers,
            "Randomness": randomness,
            "Balance": balance,
            "Proximity": proximity,
            "Is Novel": is_novel,
            "Score": score,
        }
        if is_valid_prediction(prediction):  # Validate prediction
            return prediction


# Generate initial population
def generate_initial_population( weights, R_min, R_max, B_min, B_max, P_max):
    return [generate_random_prediction( weights, R_min, R_max, B_min, B_max, P_max) for _ in range(POPULATION_SIZE)]

# Selection: Tournament method
def select_parents(population, fitness_scores):
    selected = random.choices(population, weights=fitness_scores, k=2)
    return selected

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        while True:
            crossover_point = random.randint(1, 4)  # Between main numbers indices
            child_main = sorted(parent1["Main Numbers"][:crossover_point] + parent2["Main Numbers"][crossover_point:])
            child_main = sorted(set(child_main))[:5]  # Ensure uniqueness and size

            child_euro = sorted(random.choice([parent1["Euro Numbers"], parent2["Euro Numbers"]]))
            if len(child_euro) < 2:
                child_euro = sorted(np.random.choice(range(1, 13), 2, replace=False))

            randomness, balance = calculate_randomness_and_balance(child_main)
            proximity = calculate_proximity(child_main, child_euro, last_known_main, last_known_euro)
            is_novel = not validate_prediction(child_main, child_euro, eurojackpot_df)
            prediction = {
                "Main Numbers": child_main,
                "Euro Numbers": child_euro,
                "Randomness": randomness,
                "Balance": balance,
                "Proximity": proximity,
                "Is Novel": is_novel,
            }
            if is_valid_prediction(prediction):  # Validate offspring
                return prediction
    return parent1  # No crossover


def mutate(prediction, weights, R_min, R_max, B_min, B_max, P_max):
    if random.random() < MUTATION_RATE:
        while True:
            mutated_main = prediction["Main Numbers"][:]
            if len(mutated_main) < 5:
                mutated_main = sorted(np.random.choice(range(1, 51), 5, replace=False))
            idx = random.randint(0, 4)
            mutated_main[idx] = random.randint(1, 50)
            mutated_main = sorted(set(mutated_main))[:5]

            mutated_euro = prediction["Euro Numbers"][:]
            if len(mutated_euro) < 2:
                mutated_euro = sorted(np.random.choice(range(1, 13), 2, replace=False))  # Update range to 1-12
            idx = random.randint(0, len(mutated_euro) - 1)
            mutated_euro[idx] = random.randint(1, 12)
            mutated_euro = sorted(set(mutated_euro))[:2]

            randomness, balance = calculate_randomness_and_balance(mutated_main)
            proximity = calculate_proximity(mutated_main, mutated_euro, last_known_main, last_known_euro)
            is_novel = not validate_prediction(mutated_main, mutated_euro, eurojackpot_df)
            score = compute_score(
                {"Randomness": randomness, "Balance": balance, "Proximity": proximity, "Is Novel": is_novel},
                weights,
                R_min,
                R_max,
                B_min,
                B_max,
                P_max,
            )
            prediction = {
                "Main Numbers": mutated_main,
                "Euro Numbers": mutated_euro,
                "Randomness": randomness,
                "Balance": balance,
                "Proximity": proximity,
                "Is Novel": is_novel,
                "Score": score,
            }
            if is_valid_prediction(prediction):  # Validate mutation
                return prediction
    return prediction


def validate_population(population, weights, R_min, R_max, B_min, B_max, P_max):
    valid_population = []
    for prediction in population:
        if is_valid_prediction(prediction):
            if "Score" not in prediction:
                prediction["Score"] = compute_score(
                    prediction,
                    weights,
                    R_min,
                    R_max,
                    B_min,
                    B_max,
                    P_max,
                )
            valid_population.append(prediction)
    return valid_population


# Genetic Algorithm
def genetic_algorithm():
    weights = {"w_R": 0.3, "w_B": 0.3, "w_P": 0.2, "w_N": 0.2}
    R_min, R_max = eurojackpot_df["Randomness"].min(), eurojackpot_df["Randomness"].max()
    B_min, B_max = eurojackpot_df["Balance"].min(), eurojackpot_df["Balance"].max()
    P_max = 2
    population = generate_initial_population(weights, R_min, R_max, B_min, B_max, P_max)

    for generation in range(GENERATIONS):
        fitness_scores = [
            fitness_function(prediction, weights, R_min, R_max, B_min, B_max, P_max)
            for prediction in population
        ]
        next_generation = []

        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population, fitness_scores)
            child = crossover(parent1, parent2)
            child = mutate(child, weights, R_min, R_max, B_min, B_max, P_max)
            next_generation.append(child)

        population = next_generation
        population = validate_population(population, weights, R_min, R_max, B_min, B_max, P_max)  # Validate
        logging.info(f"Generation {generation + 1}: Best Score: {max(fitness_scores):.3f}")

    ranked_population = sorted(
        population,
        key=lambda x: x["Score"],
        reverse=True,
    )
    return ranked_population


# Run the genetic algorithm
ranked_predictions = genetic_algorithm()

# Print results
print_pretty_output({}, ranked_predictions)
