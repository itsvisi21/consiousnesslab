# Re-import necessary libraries and redefine functions for a fresh environment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random

# Define Qutrit States and Bases
qutrit_states = [-1, 0, 1]
bases = ["computational", "hadamard"]

# Generate random qutrit states and bases
def generate_qutrits_and_bases(length):
    states = [np.random.choice(qutrit_states) for _ in range(length)]
    chosen_bases = [np.random.choice(bases) for _ in range(length)]
    return states, chosen_bases

# Simulate sender preparation
def sender_prepare(length):
    states, chosen_bases = generate_qutrits_and_bases(length)
    return states, chosen_bases

# Simulate receiver measurement
def receiver_measure(length):
    _, chosen_bases = generate_qutrits_and_bases(length)
    return chosen_bases

# Basis reconciliation
def basis_reconciliation(sender_bases, receiver_bases, sender_states):
    matched_indices = [
        i for i, (sb, rb) in enumerate(zip(sender_bases, receiver_bases)) if sb == rb
    ]
    matched_states = [sender_states[i] for i in matched_indices if sender_states[i] is not None]
    return matched_states, matched_indices

# Error correction with None handling
def error_correction_with_none_handling(matched_states):
    corrected_key = []
    parity_errors = 0

    for i in range(0, len(matched_states), 3):  # Group states in blocks of 3 for parity checks
        block = matched_states[i:i + 3]

        # Skip blocks with None values
        if any(state is None for state in block):
            corrected_key.extend([state for state in block if state is not None])
            continue

        # Calculate parity (sum modulo 3)
        parity = sum(block) % 3

        # If parity is non-zero, correct the block
        if parity != 0:
            block[-1] = (block[-1] - parity) % 3  # Adjust last trit to correct parity
            if block[-1] == 2:  # Handle wrap-around for modular arithmetic
                block[-1] = -1
            parity_errors += 1

        corrected_key.extend(block)

    return corrected_key, parity_errors

# Privacy amplification using a hash function
def privacy_amplification(corrected_key, desired_key_length):
    corrected_key_string = ''.join(map(str, corrected_key))
    hashed_key = hashlib.sha256(corrected_key_string.encode()).hexdigest()
    truncated_key = hashed_key[:desired_key_length // 4]  # 4 bits per hex digit
    return truncated_key

# Apply physical noise
def apply_physical_noise(states, P_loss=0.1, P_depolarization=0.1):
    noisy_states = []
    for state in states:
        if np.random.rand() < P_loss:
            noisy_states.append(None)  # Representing photon loss
        elif np.random.rand() < P_depolarization:
            noisy_states.append(np.random.choice(qutrit_states))  # Randomized state due to depolarization
        else:
            noisy_states.append(state)  # No noise
    return noisy_states

# Simulate with noise
def simulate_with_noise(length, P_loss=0.1, P_depolarization=0.1, P_error=0.1):
    sender_states, sender_bases = sender_prepare(length)
    noisy_states = apply_physical_noise(sender_states, P_loss, P_depolarization)
    receiver_bases = [
        np.random.choice(bases) if np.random.rand() < P_error else rb
        for rb in receiver_measure(length)
    ]
    matched_states, matched_indices = basis_reconciliation(sender_bases, receiver_bases, noisy_states)
    return {
        "Original States": sender_states,
        "Noisy States": noisy_states,
        "Matched States": matched_states,
        "Matched Indices": matched_indices,
        "Key Length": len(matched_states),
    }

# Calculate security metrics
def calculate_security_metrics(simulation_results, corrected_key, final_key_length):
    original_states = simulation_results["Original States"]
    noisy_states = simulation_results["Noisy States"]
    matched_states = simulation_results["Matched States"]
    
    # Key Loss Rate
    lost_states = noisy_states.count(None)
    KLR = (lost_states / len(original_states)) * 100
    
    # Error Rate
    erroneous_states = sum(
        1 for orig, matched in zip(original_states, matched_states)
        if orig != matched and orig is not None and matched is not None
    )
    ER = (erroneous_states / len(matched_states)) * 100 if matched_states else 0
    
    # Privacy Amplification Efficiency
    PAE = final_key_length / len(matched_states) if matched_states else 0
    
    return {
        "Key Loss Rate (KLR)": KLR,
        "Error Rate (ER)": ER,
        "Privacy Amplification Efficiency (PAE)": PAE,
    }

# Run simulations with noise levels for Qutrit BB84
def run_simulations_with_noise_updated(dataset_size, noise_levels):
    results = []
    for P_loss, P_depolarization, P_error in noise_levels:
        # Simulate the protocol with noise
        simulation_results = simulate_with_noise(
            dataset_size, P_loss=P_loss, P_depolarization=P_depolarization, P_error=P_error
        )
        
        # Perform error correction
        corrected_key, parity_errors = error_correction_with_none_handling(simulation_results["Matched States"])
        
        # Apply privacy amplification
        final_key_length = 128  # Fixed length for secure key
        secure_key = privacy_amplification(corrected_key, final_key_length)
        
        # Calculate security metrics
        metrics = calculate_security_metrics(simulation_results, corrected_key, final_key_length)
        
        # Record results
        results.append({
            "P_loss": P_loss,
            "P_depolarization": P_depolarization,
            "P_error": P_error,
            "Key Length After Matching": len(simulation_results["Matched States"]),
            "Parity Errors Corrected": parity_errors,
            **metrics,
            "Secure Key": secure_key
        })
    return results

# Define noise levels for the simulation
noise_levels = [
    (0.05, 0.05, 0.05),
    (0.10, 0.10, 0.05),
    (0.10, 0.10, 0.10),
    (0.20, 0.10, 0.10),
    (0.20, 0.20, 0.20),
]

# Run the simulation with a dataset size of 1000
simulations_with_noise_results_updated = run_simulations_with_noise_updated(1000, noise_levels)

# Convert results to a DataFrame for visualization
results_df_updated = pd.DataFrame(simulations_with_noise_results_updated)
# print(results_df_updated)

# Plotting trends for P_loss, P_depolarization, and P_error against Key Length and Efficiency

# Extract relevant data
P_loss_values = results_df_updated["P_loss"]
P_depolarization_values = results_df_updated["P_depolarization"]
P_error_values = results_df_updated["P_error"]
Key_Length_values = results_df_updated["Key Length After Matching"]
PAE_values = results_df_updated["Privacy Amplification Efficiency (PAE)"]
ER_values = results_df_updated["Error Rate (ER)"]

# # Plot Key Length vs Noise Parameters
# plt.figure(figsize=(10, 6))
# plt.plot(P_loss_values, Key_Length_values, marker='o', label="Key Length vs P_loss")
# plt.plot(P_depolarization_values, Key_Length_values, marker='s', label="Key Length vs P_depolarization")
# plt.plot(P_error_values, Key_Length_values, marker='^', label="Key Length vs P_error")
# plt.xlabel("Noise Parameters")
# plt.ylabel("Key Length After Matching")
# plt.title("Key Length vs Noise Parameters")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot Privacy Amplification Efficiency vs Noise Parameters
# plt.figure(figsize=(10, 6))
# plt.plot(P_loss_values, PAE_values, marker='o', label="PAE vs P_loss")
# plt.plot(P_depolarization_values, PAE_values, marker='s', label="PAE vs P_depolarization")
# plt.plot(P_error_values, PAE_values, marker='^', label="PAE vs P_error")
# plt.xlabel("Noise Parameters")
# plt.ylabel("Privacy Amplification Efficiency (PAE)")
# plt.title("Privacy Amplification Efficiency vs Noise Parameters")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot Error Rate vs Noise Parameters
# plt.figure(figsize=(10, 6))
# plt.plot(P_loss_values, ER_values, marker='o', label="Error Rate vs P_loss")
# plt.plot(P_depolarization_values, ER_values, marker='s', label="Error Rate vs P_depolarization")
# plt.plot(P_error_values, ER_values, marker='^', label="Error Rate vs P_error")
# plt.xlabel("Noise Parameters")
# plt.ylabel("Error Rate (%)")
# plt.title("Error Rate vs Noise Parameters")
# plt.legend()
# plt.grid(True)
# plt.show()

# Analyze results to determine optimal thresholds
def analyze_optimal_thresholds(results_df):
    # Group results by noise levels
    grouped_results = results_df.groupby(["P_loss", "P_depolarization", "P_error"])
    
    # Calculate mean Key Length and PAE for each noise configuration
    analysis = grouped_results[
        ["Key Length After Matching", "Privacy Amplification Efficiency (PAE)"]
    ].mean().reset_index()
    
    # Sort by descending Key Length and PAE
    sorted_analysis = analysis.sort_values(
        by=["Key Length After Matching", "Privacy Amplification Efficiency (PAE)"], 
        ascending=[False, False]
    )
    
    return sorted_analysis

# Perform the analysis
optimal_thresholds = analyze_optimal_thresholds(results_df_updated)

# Display the results for review
def display_dataframe_to_user(name, dataframe):
    """
    Display a DataFrame with a given name for analysis.

    Args:
        name (str): The name/title of the DataFrame analysis.
        dataframe (pd.DataFrame): The DataFrame to display.

    Returns:
        None
    """
    # print(f"\n=== {name} ===\n")
    # print(dataframe.to_string(index=False))  # Display DataFrame as a formatted string

# Example DataFrame
optimal_thresholds = pd.DataFrame({
    "Noise Level": [0.1, 0.2, 0.3, 0.4],
    "Accuracy (%)": [92.5, 88.1, 85.7, 80.2],
    "Threshold": [0.15, 0.25, 0.35, 0.45]
})

# Validate the Qutrit BB84 protocol with real-world noise thresholds
def validate_real_world_performance(dataset_size, P_loss_threshold=0.15, P_depolarization_threshold=0.15, P_error_threshold=0.15):
    # Simulate the protocol with thresholds
    simulation_results = simulate_with_noise(
        dataset_size,
        P_loss=P_loss_threshold,
        P_depolarization=P_depolarization_threshold,
        P_error=P_error_threshold,
    )
    
    # Perform error correction
    corrected_key, parity_errors = error_correction_with_none_handling(simulation_results["Matched States"])
    
    # Apply privacy amplification
    final_key_length = 128  # Fixed length for secure key
    secure_key = privacy_amplification(corrected_key, final_key_length)
    
    # Calculate security metrics
    metrics = calculate_security_metrics(simulation_results, corrected_key, final_key_length)
    
    # Compile validation results
    validation_results = {
        "Dataset Size": dataset_size,
        "P_loss Threshold": P_loss_threshold,
        "P_depolarization Threshold": P_depolarization_threshold,
        "P_error Threshold": P_error_threshold,
        "Key Length After Matching": len(simulation_results["Matched States"]),
        "Parity Errors Corrected": parity_errors,
        **metrics,
        "Secure Key": secure_key,
    }
    return validation_results

# Run validation for a dataset size of 1000 with optimal noise thresholds
validation_results = validate_real_world_performance(1000)

# Display the validation results
# display_dataframe_to_user(name="Qutrit BB84 Protocol Validation Results", dataframe=pd.DataFrame([validation_results]))


# Noise Monitoring System
class NoiseMonitor:
    def __init__(self, loss_threshold=0.15, depolarization_threshold=0.15, error_threshold=0.15):
        self.loss_threshold = loss_threshold
        self.depolarization_threshold = depolarization_threshold
        self.error_threshold = error_threshold

    def monitor_noise(self, P_loss, P_depolarization, P_error):
        """Monitor noise levels and trigger alerts if thresholds are exceeded."""
        alerts = []
        if P_loss > self.loss_threshold:
            alerts.append(f"P_loss exceeded: {P_loss:.2f} > {self.loss_threshold:.2f}")
        if P_depolarization > self.depolarization_threshold:
            alerts.append(f"P_depolarization exceeded: {P_depolarization:.2f} > {self.depolarization_threshold:.2f}")
        if P_error > self.error_threshold:
            alerts.append(f"P_error exceeded: {P_error:.2f} > {self.error_threshold:.2f}")
        
        return {
            "P_loss": P_loss,
            "P_depolarization": P_depolarization,
            "P_error": P_error,
            "alerts": alerts
        }

# Adaptive Error Correction Adjustment
class AdaptiveErrorCorrection:
    def __init__(self):
        self.mode = "Low Noise"  # Default mode

    def adjust_correction(self, P_loss, P_depolarization, P_error):
        """Adjust error correction intensity based on noise levels."""
        if max(P_loss, P_depolarization, P_error) <= 0.1:
            self.mode = "Low Noise"
        elif max(P_loss, P_depolarization, P_error) <= 0.2:
            self.mode = "Moderate Noise"
        else:
            self.mode = "High Noise"
        return self.mode

# Simulated Noise Monitoring and Adaptive Adjustment
def simulate_noise_monitoring(dataset_size, P_loss, P_depolarization, P_error):
    # Initialize monitoring and adaptive systems
    noise_monitor = NoiseMonitor()
    adaptive_correction = AdaptiveErrorCorrection()
    
    # Monitor noise levels
    noise_status = noise_monitor.monitor_noise(P_loss, P_depolarization, P_error)
    
    # Adjust error correction based on observed noise
    correction_mode = adaptive_correction.adjust_correction(P_loss, P_depolarization, P_error)
    
    return {
        "Dataset Size": dataset_size,
        "Noise Levels": noise_status,
        "Correction Mode": correction_mode
    }

# Test the adaptive system with varying noise levels
test_results = [
    simulate_noise_monitoring(1000, P_loss, P_depolarization, P_error)
    for P_loss, P_depolarization, P_error in [(0.05, 0.05, 0.05), (0.10, 0.15, 0.10), (0.20, 0.20, 0.25)]
]


results_df = pd.DataFrame(test_results)
# display_dataframe_to_user(name="Adaptive Noise Monitoring Results", dataframe=results_df)

# Simulation of Noise Monitoring System Integrated with QKD Protocol

# Placeholder functions for error correction techniques (Hamming, Reed-Solomon, LDPC)
def hamming_correction(matched_states):
    # Simulate basic Hamming correction (for demonstration purposes)
    corrected = [state for state in matched_states if state is not None]  # Simple filtering
    errors_corrected = len(matched_states) - len(corrected)
    return corrected, errors_corrected

def reed_solomon_correction(matched_states):
    # Simulate basic Reed-Solomon correction (for demonstration purposes)
    corrected = [state for state in matched_states if state is not None]  # Simple filtering
    errors_corrected = len(matched_states) - len(corrected)
    return corrected, errors_corrected

def ldpc_correction(matched_states):
    # Simulate basic LDPC correction (for demonstration purposes)
    corrected = [state for state in matched_states if state is not None]  # Simple filtering
    errors_corrected = len(matched_states) - len(corrected)
    return corrected, errors_corrected

# Integrated QKD Protocol Simulation with Noise Monitoring and Error Correction
def simulate_qkd_with_error_correction(dataset_size, P_loss, P_depolarization, P_error):
    # Monitor noise levels
    noise_monitor = NoiseMonitor()
    noise_status = noise_monitor.monitor_noise(P_loss, P_depolarization, P_error)
    
    # Determine correction mode based on noise levels
    adaptive_correction = AdaptiveErrorCorrection()
    correction_mode = adaptive_correction.adjust_correction(P_loss, P_depolarization, P_error)
    
    # Simulate QKD protocol with noise
    simulation_results = simulate_with_noise(dataset_size, P_loss, P_depolarization, P_error)
    matched_states = simulation_results["Matched States"]
    
    # Apply error correction based on mode
    if correction_mode == "Low Noise":
        corrected_key, errors_corrected = hamming_correction(matched_states)
    elif correction_mode == "Moderate Noise":
        corrected_key, errors_corrected = reed_solomon_correction(matched_states)
    else:  # High Noise
        corrected_key, errors_corrected = ldpc_correction(matched_states)
    
    # Apply privacy amplification
    final_key_length = 128  # Fixed length for secure key
    secure_key = privacy_amplification(corrected_key, final_key_length)
    
    # Calculate security metrics
    metrics = calculate_security_metrics(simulation_results, corrected_key, final_key_length)
    
    return {
        "Dataset Size": dataset_size,
        "P_loss": P_loss,
        "P_depolarization": P_depolarization,
        "P_error": P_error,
        "Correction Mode": correction_mode,
        "Errors Corrected": errors_corrected,
        "Key Length After Matching": len(corrected_key),
        "Final Secure Key": secure_key,
        **metrics
    }

# Test the integrated system with different noise levels
test_noise_levels = [
    (0.05, 0.05, 0.05),  # Low Noise
    (0.10, 0.15, 0.10),  # Moderate Noise
    (0.20, 0.20, 0.25)   # High Noise
]

# Run simulations for a dataset of 1000 trits
qkd_results = [
    simulate_qkd_with_error_correction(1000, P_loss, P_depolarization, P_error)
    for P_loss, P_depolarization, P_error in test_noise_levels
]

# Convert results to a DataFrame for analysis
qkd_results_df = pd.DataFrame(qkd_results)

# Display results for user review
# display_dataframe_to_user(name="Integrated QKD Protocol Simulation Results", dataframe=qkd_results_df)

# Implementing real-time feedback for switching between correction modes

class RealTimeErrorCorrection:
    def __init__(self):
        self.correction_mode = "Low Noise"

    def switch_correction_mode(self, P_loss, P_depolarization, P_error):
        """Determine correction mode based on observed noise levels."""
        if max(P_loss, P_depolarization, P_error) <= 0.1:
            self.correction_mode = "Low Noise"
        elif max(P_loss, P_depolarization, P_error) <= 0.2:
            self.correction_mode = "Moderate Noise"
        else:
            self.correction_mode = "High Noise"
        return self.correction_mode

    def apply_error_correction(self, matched_states):
        """Apply the appropriate error correction based on the mode."""
        if self.correction_mode == "Low Noise":
            corrected_key, errors_corrected = hamming_correction(matched_states)
        elif self.correction_mode == "Moderate Noise":
            corrected_key, errors_corrected = reed_solomon_correction(matched_states)
        else:  # High Noise
            corrected_key, errors_corrected = ldpc_correction(matched_states)
        return corrected_key, errors_corrected


# Simulate real-time feedback system with correction mode switching
def simulate_real_time_feedback(dataset_size, noise_levels):
    # Initialize noise monitoring and real-time correction systems
    noise_monitor = NoiseMonitor()
    real_time_correction = RealTimeErrorCorrection()

    # Simulate results for each noise level
    results = []
    for P_loss, P_depolarization, P_error in noise_levels:
        # Monitor noise levels
        noise_status = noise_monitor.monitor_noise(P_loss, P_depolarization, P_error)
        
        # Determine and switch correction mode
        correction_mode = real_time_correction.switch_correction_mode(
            P_loss, P_depolarization, P_error
        )
        
        # Simulate QKD protocol with noise
        simulation_results = simulate_with_noise(dataset_size, P_loss, P_depolarization, P_error)
        matched_states = simulation_results["Matched States"]
        
        # Apply error correction
        corrected_key, errors_corrected = real_time_correction.apply_error_correction(matched_states)
        
        # Apply privacy amplification
        final_key_length = 128  # Fixed length for secure key
        secure_key = privacy_amplification(corrected_key, final_key_length)
        
        # Calculate security metrics
        metrics = calculate_security_metrics(simulation_results, corrected_key, final_key_length)
        
        # Compile results
        results.append({
            "P_loss": P_loss,
            "P_depolarization": P_depolarization,
            "P_error": P_error,
            "Correction Mode": correction_mode,
            "Errors Corrected": errors_corrected,
            "Key Length After Matching": len(corrected_key),
            "Final Secure Key": secure_key,
            **metrics
        })
    
    return results


# Test the real-time feedback system with noise levels
test_noise_levels = [
    (0.05, 0.05, 0.05),  # Low Noise
    (0.10, 0.15, 0.10),  # Moderate Noise
    (0.20, 0.20, 0.25)   # High Noise
]

# Run the simulation
real_time_results = simulate_real_time_feedback(1000, test_noise_levels)

# Convert results to a DataFrame for analysis
real_time_results_df = pd.DataFrame(real_time_results)

# Display results for user review
# display_dataframe_to_user(name="Real-Time Feedback QKD Simulation Results", dataframe=real_time_results_df)

# Predictive Algorithm for Preemptive Mode Switching in QKD Protocol

class PredictiveCorrection:
    def __init__(self):
        self.previous_noise = {"P_loss": 0, "P_depolarization": 0, "P_error": 0}
        self.correction_mode = "Low Noise"

    def predict_noise_trend(self, current_noise):
        """Predict noise trends based on current and previous noise levels."""
        trend = {
            "P_loss": current_noise["P_loss"] - self.previous_noise["P_loss"],
            "P_depolarization": current_noise["P_depolarization"] - self.previous_noise["P_depolarization"],
            "P_error": current_noise["P_error"] - self.previous_noise["P_error"]
        }
        self.previous_noise = current_noise
        return trend

    def preemptive_switch_mode(self, current_noise):
        """Switch correction mode based on predicted trends and current noise levels."""
        predicted_trend = self.predict_noise_trend(current_noise)
        
        # Decision thresholds based on predicted trends and current noise
        if max(current_noise.values()) + max(predicted_trend.values()) <= 0.1:
            self.correction_mode = "Low Noise"
        elif max(current_noise.values()) + max(predicted_trend.values()) <= 0.2:
            self.correction_mode = "Moderate Noise"
        else:
            self.correction_mode = "High Noise"
        
        return self.correction_mode


# Test Predictive Algorithm Integration with QKD Protocol
def simulate_predictive_qkd(dataset_size, noise_levels):
    # Initialize predictive correction system
    predictive_correction = PredictiveCorrection()
    
    results = []
    for P_loss, P_depolarization, P_error in noise_levels:
        current_noise = {"P_loss": P_loss, "P_depolarization": P_depolarization, "P_error": P_error}
        
        # Preemptively switch mode based on predicted trends
        correction_mode = predictive_correction.preemptive_switch_mode(current_noise)
        
        # Simulate QKD protocol with noise
        simulation_results = simulate_with_noise(dataset_size, P_loss, P_depolarization, P_error)
        matched_states = simulation_results["Matched States"]
        
        # Apply appropriate error correction
        if correction_mode == "Low Noise":
            corrected_key, errors_corrected = hamming_correction(matched_states)
        elif correction_mode == "Moderate Noise":
            corrected_key, errors_corrected = reed_solomon_correction(matched_states)
        else:  # High Noise
            corrected_key, errors_corrected = ldpc_correction(matched_states)
        
        # Apply privacy amplification
        final_key_length = 128  # Fixed secure key length
        secure_key = privacy_amplification(corrected_key, final_key_length)
        
        # Calculate metrics
        metrics = calculate_security_metrics(simulation_results, corrected_key, final_key_length)
        
        # Compile results
        results.append({
            "P_loss": P_loss,
            "P_depolarization": P_depolarization,
            "P_error": P_error,
            "Predicted Correction Mode": correction_mode,
            "Errors Corrected": errors_corrected,
            "Key Length After Matching": len(corrected_key),
            "Final Secure Key": secure_key,
            **metrics
        })
    
    return results


# Define noise levels for predictive testing
test_noise_levels_predictive = [
    (0.05, 0.05, 0.05),  # Stable Low Noise
    (0.10, 0.12, 0.08),  # Increasing Moderate Noise
    (0.15, 0.18, 0.20),  # Rising Noise Near High
    (0.20, 0.25, 0.30)   # High Noise
]

# Simulate the predictive system with dataset size 1000
predictive_qkd_results = simulate_predictive_qkd(1000, test_noise_levels_predictive)

# Convert to DataFrame for analysis
predictive_results_df = pd.DataFrame(predictive_qkd_results)

# Display results for user review
# display_dataframe_to_user(name="Predictive QKD Simulation Results", dataframe=predictive_results_df)

# Redefining the simulate_noise_data function
def simulate_noise_data(samples=1000):
    """
    Generate synthetic noise data for testing.
    """
    np.random.seed(42)
    data = {
        "P_loss": np.random.uniform(0.01, 0.30, size=samples),
        "P_depolarization": np.random.uniform(0.01, 0.30, size=samples),
        "P_error": np.random.uniform(0.01, 0.30, size=samples)
    }
    df = pd.DataFrame(data)
    
    # Assign correction modes based on noise thresholds
    conditions = [
        (df["P_loss"] <= 0.1) & (df["P_depolarization"] <= 0.1) & (df["P_error"] <= 0.1),
        (df["P_loss"] <= 0.2) & (df["P_depolarization"] <= 0.2) & (df["P_error"] <= 0.2),
    ]
    choices = ["Low Noise", "Moderate Noise"]
    df["Correction_Mode"] = np.select(conditions, choices, default="High Noise")
    return df

# Reattempting dataset generation and optimization

# Generate the original noise dataset
original_noise_data = simulate_noise_data(1000)

# Extract Low Noise data
low_noise_data = original_noise_data[
    (original_noise_data["P_loss"] <= 0.1) &
    (original_noise_data["P_depolarization"] <= 0.1) &
    (original_noise_data["P_error"] <= 0.1)
]

# Augment Low Noise data for optimization
target_proportion = 0.3  # Desired proportion of Low Noise scenarios
current_size = len(low_noise_data)
target_size = int(len(original_noise_data) * target_proportion)
augment_size = target_size - current_size

# Resample and optimize dataset
if augment_size > 0:
    augmented_low_noise_data = low_noise_data.sample(n=augment_size, replace=True, random_state=42)
    optimized_noise_data = pd.concat([original_noise_data, augmented_low_noise_data], ignore_index=True)
else:
    optimized_noise_data = original_noise_data

# Verify proportions of correction modes
optimized_mode_counts = optimized_noise_data["Correction_Mode"].value_counts(normalize=True)

# Display results
optimized_mode_counts


# Optimizing Dataset for Low Noise Scenarios

def augment_low_noise_data(original_data, target_proportion=0.3):
    """
    Augment the dataset by increasing the representation of Low Noise scenarios.
    """
    # Extract Low Noise data
    low_noise_data = original_data[
        (original_data["P_loss"] <= 0.1) &
        (original_data["P_depolarization"] <= 0.1) &
        (original_data["P_error"] <= 0.1)
    ]

    # Calculate the current and target sizes for Low Noise data
    current_proportion = len(low_noise_data) / len(original_data)
    target_size = int(len(original_data) * target_proportion)
    augment_size = max(0, target_size - len(low_noise_data))

    # Augment by resampling Low Noise data
    augmented_low_noise_data = low_noise_data.sample(n=augment_size, replace=True, random_state=42)
    optimized_data = pd.concat([original_data, augmented_low_noise_data], ignore_index=True)

    return optimized_data

# Original noise dataset
original_noise_data = simulate_noise_data(1000)

# Augment dataset for Low Noise scenarios
optimized_noise_data = augment_low_noise_data(original_noise_data)

# Verify proportions of each correction mode
mode_counts = optimized_noise_data["Correction_Mode"].value_counts(normalize=True)

# Prepare optimized dataset for re-training
X_optimized = optimized_noise_data[["P_loss", "P_depolarization", "P_error"]]
y_optimized = optimized_noise_data["Correction_Mode"]

# Split optimized data
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X_optimized, y_optimized, test_size=0.2, random_state=42)

# Retrain Random Forest model with optimized data
rf_model_optimized = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_optimized.fit(X_train_opt, y_train_opt)

# Validate optimized model
y_pred_opt = rf_model_optimized.predict(X_test_opt)
accuracy_opt = accuracy_score(y_test_opt, y_pred_opt)
report_opt = classification_report(y_test_opt, y_pred_opt)

# Display optimization results
{
    "Original Proportions": original_noise_data["Correction_Mode"].value_counts(normalize=True).to_dict(),
    "Optimized Proportions": mode_counts.to_dict(),
    "Optimized Model Accuracy": f"{accuracy_opt * 100:.2f}%",
    "Optimized Classification Report": report_opt
}


# Step 1: Data Simulation
def simulate_noise_data(samples=1000):
    np.random.seed(42)
    data = {
        "P_loss": np.random.uniform(0.01, 0.30, size=samples),
        "P_depolarization": np.random.uniform(0.01, 0.30, size=samples),
        "P_error": np.random.uniform(0.01, 0.30, size=samples)
    }
    df = pd.DataFrame(data)
    
    # Assign correction modes based on noise thresholds
    conditions = [
        (df["P_loss"] <= 0.1) & (df["P_depolarization"] <= 0.1) & (df["P_error"] <= 0.1),
        (df["P_loss"] <= 0.2) & (df["P_depolarization"] <= 0.2) & (df["P_error"] <= 0.2),
    ]
    choices = ["Low Noise", "Moderate Noise"]
    df["Correction_Mode"] = np.select(conditions, choices, default="High Noise")
    return df

# Generate synthetic data
noise_data = simulate_noise_data(1000)

# Step 2: Model Training
# Prepare features and labels
X = noise_data[["P_loss", "P_depolarization", "P_error"]]
y = noise_data["Correction_Mode"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 3: Model Validation
# Predict on test data
y_pred = rf_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display results
# print(f"Model Accuracy: {accuracy * 100:.2f}%")
# print("\nClassification Report:\n", report)

# Integration with QKD Protocol: Predict correction mode for new noise levels
new_noise_levels = pd.DataFrame([
    {"P_loss": 0.05, "P_depolarization": 0.05, "P_error": 0.05},  # Low Noise
    {"P_loss": 0.12, "P_depolarization": 0.15, "P_error": 0.10},  # Moderate Noise
    {"P_loss": 0.25, "P_depolarization": 0.22, "P_error": 0.28},  # High Noise
])
predicted_modes = rf_model.predict(new_noise_levels)

# Display predictions
# predicted_modes

# Simplify dataset augmentation for Low Noise scenarios

# Extract Low Noise data from the original dataset
low_noise_data = original_noise_data[
    (original_noise_data["P_loss"] <= 0.1) &
    (original_noise_data["P_depolarization"] <= 0.1) &
    (original_noise_data["P_error"] <= 0.1)
]

# Augment by resampling to achieve a higher proportion of Low Noise data
target_proportion = 0.3  # Desired proportion of Low Noise scenarios
current_size = len(low_noise_data)
target_size = int(len(original_noise_data) * target_proportion)
augment_size = target_size - current_size

# Resample Low Noise data
if augment_size > 0:
    augmented_low_noise_data = low_noise_data.sample(n=augment_size, replace=True, random_state=42)
    optimized_noise_data = pd.concat([original_noise_data, augmented_low_noise_data], ignore_index=True)
else:
    optimized_noise_data = original_noise_data

# Check the distribution of correction modes in the optimized dataset
optimized_mode_counts = optimized_noise_data["Correction_Mode"].value_counts(normalize=True)

# Display updated proportions for review
optimized_mode_counts

# Environment Constants
ACTIONS = ["Low Noise", "Moderate Noise", "High Noise"]
STATE_SIZE = 3  # P_loss, P_depolarization, P_error
ACTION_SIZE = len(ACTIONS)
DISCOUNT = 0.95  # Discount factor for future rewards
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000
BATCH_SIZE = 32
EPISODES = 500

# Define the Q-Network
def build_q_network(state_size, action_size):
    model = Sequential([
        Dense(24, input_dim=state_size, activation='relu'),
        Dense(24, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

# Replay Memory
class ReplayMemory:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

# RL Agent
class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = build_q_network(state_size, action_size)
        self.target_network = build_q_network(state_size, action_size)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def update_target_network(self):
        """Synchronize weights from q_network to target_network."""
        self.target_network.set_weights(self.q_network.get_weights())

    def update_epsilon(self):
        """Decay epsilon to reduce exploration over time."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    def train(self):
        if len(self.memory.memory) < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states).reshape(BATCH_SIZE, -1)  # Correct reshaping
        next_states = np.array(next_states).reshape(BATCH_SIZE, -1)  # Correct reshaping

        target_qs = self.q_network.predict(states, verbose=0)
        next_qs = self.target_network.predict(next_states, verbose=0)

        for i in range(BATCH_SIZE):
            target = rewards[i]
            if not dones[i]:
                target += DISCOUNT * np.max(next_qs[i])
            target_qs[i][actions[i]] = target

        self.q_network.fit(states, target_qs, epochs=1, verbose=0, batch_size=BATCH_SIZE)


# Simulate Dynamic Noise Environment
def simulate_dynamic_noise():
    return np.random.uniform(0.01, 0.30, STATE_SIZE)

# Training Loop
agent = QLearningAgent(STATE_SIZE, ACTION_SIZE)
for episode in range(EPISODES):
    state = simulate_dynamic_noise().reshape(1, -1)
    total_reward = 0

    for step in range(200):  # Limit steps per episode
        action = agent.act(state)
        next_state = simulate_dynamic_noise().reshape(1, -1)
        reward = -np.sum(state) if ACTIONS[action] == "Low Noise" and np.max(state) > 0.1 else np.sum(state)
        done = step == 199
        agent.memory.add((state, action, reward, next_state, done))
        agent.train()
        state = next_state
        total_reward += reward
        if done:
            break

    # Update target network every 10 episodes
    if episode % 10 == 0:
        agent.update_target_network()

    agent.update_epsilon()

    if episode % 50 == 0:
        print(f"Episode {episode}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

# Save the trained model
agent.q_network.save("qkd_rl_model.h5")

