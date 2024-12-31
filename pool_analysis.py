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

# Simulation function for balanced strategy
def simulate_balanced_strategy_parallel_chunk(chunk_args):
    total_games, pool_contribution_values, pool_thresholds = chunk_args
    total_pool, total_table_profit, total_distributed_to_winners = 0, 0, 0
    total_loser_compensations, total_fully_compensated, total_not_compensated = 0, 0, 0
    losers_who_got_partial_compensated = 0

    for game in range(1, total_games + 1):
        pool_contribution = (
            pool_contribution_values[0]
            if total_pool < pool_thresholds[0]
            else pool_contribution_values[1]
            if total_pool < pool_thresholds[1]
            else pool_contribution_values[2]
        )
        total_pool += pool_contribution
        total_table_profit += 8
        total_distributed_to_winners += 6 * (10 - 1)

        if total_pool >= 50:
            compensation_rate = 1.1 if total_pool > 100000 else 1.0 if total_pool > 50000 else 0.9
            compensated_amount = int(50 * compensation_rate)

            if compensated_amount >= 50:
                total_fully_compensated += 1
            elif compensated_amount > 0:
                losers_who_got_partial_compensated += 1
            else:
                total_not_compensated += 1

            total_loser_compensations += compensated_amount
            total_pool -= compensated_amount

    return {
        "Total Games Simulated": total_games,
        "Total Losers Partially Compensated": losers_who_got_partial_compensated,
        "Total Losers Fully Compensated": total_fully_compensated,
        "Total Losers Not Compensated": total_not_compensated,
        "Total Profits Distributed to Players": total_distributed_to_winners + total_loser_compensations,
        "Total Table (House) Profit": total_table_profit,
    }

# Parallel simulation function for all levels
def run_simulations_parallel(total_games, pool_contribution_values, pool_thresholds, chunks=cpu_count()):
    games_per_chunk = total_games // chunks
    args_list = [(games_per_chunk, pool_contribution_values, pool_thresholds) for _ in range(chunks)]
    with Pool(processes=chunks) as pool:
        results = pool.map(simulate_balanced_strategy_parallel_chunk, args_list)
    return {
        "Total Games Simulated": sum(r["Total Games Simulated"] for r in results),
        "Total Losers Partially Compensated": sum(r["Total Losers Partially Compensated"] for r in results),
        "Total Losers Fully Compensated": sum(r["Total Losers Fully Compensated"] for r in results),
        "Total Losers Not Compensated": sum(r["Total Losers Not Compensated"] for r in results),
        "Total Profits Distributed to Players": sum(r["Total Profits Distributed to Players"] for r in results),
        "Total Table (House) Profit": sum(r["Total Table (House) Profit"] for r in results),
    }

# Simulate all levels
def simulate_all_levels_parallel(game_levels, pool_contribution_values, pool_thresholds):
    results = {}
    for level in game_levels:
        results[level] = run_simulations_parallel(
            total_games=level,
            pool_contribution_values=pool_contribution_values,
            pool_thresholds=pool_thresholds,
        )
    return results

# Adjust simulation function to include dynamic winner payouts
def simulate_balanced_strategy_with_dynamic_winner_payouts(chunk_args):
    total_games, pool_contribution_values, pool_thresholds, winner_payout_range = chunk_args
    total_pool = 0
    total_table_profit = 0
    total_distributed_to_winners = 0
    total_loser_compensations = 0
    losers_who_got_partial_compensated = 0
    losers_who_got_full_compensated = 0
    losers_not_compensated = 0

    for game in range(1, total_games + 1):
        # Determine pool contribution based on current pool size
        if total_pool < pool_thresholds[0]:
            pool_contribution = pool_contribution_values[0]
        elif total_pool < pool_thresholds[1]:
            pool_contribution = pool_contribution_values[1]
        else:
            pool_contribution = pool_contribution_values[2]

        total_pool += pool_contribution
        total_table_profit += 8  # Fixed house fee

        # Dynamic winner payout between $5 and $7
        winner_payout = winner_payout_range[0] + (game % (winner_payout_range[1] - winner_payout_range[0] + 1))
        total_distributed_to_winners += winner_payout * (10 - 1)  # Winner payouts

        # Assign a loser cyclically
        loser = f"player_{game % 10}"

        # Dynamic loser compensation based on pool size
        if total_pool >= 50:
            if total_pool > 100000:  # High pool, more generous compensations
                compensation_rate = 1.1  # Boost compensation by 10%
            elif total_pool > 50000:  # Medium pool, standard compensations
                compensation_rate = 1.0
            else:  # Low pool, scaled-down compensations
                compensation_rate = 0.9

            compensated_amount = int(50 * compensation_rate)
            losers_who_got_partial_compensated += 1
            total_loser_compensations += compensated_amount
            total_pool -= compensated_amount
        else:
            losers_not_compensated += 1

    results = {
        "Total Games Simulated": total_games,
        "Total Losers Partially Compensated": losers_who_got_partial_compensated,
        "Total Losers Fully Compensated": losers_who_got_full_compensated,
        "Total Losers Not Compensated": losers_not_compensated,
        "Total Profits Distributed to Players": total_distributed_to_winners + total_loser_compensations,
        "Total Table (House) Profit": total_table_profit,
    }
    return results

# Parallel simulation function for dynamic winner payouts
def simulate_all_levels_dynamic_payouts(game_levels, pool_contribution_values, pool_thresholds, winner_payout_range):
    results = {}
    for level in game_levels:
        results[level] = run_simulations_parallel(
            total_games=level,
            pool_contribution_values=pool_contribution_values,
            pool_thresholds=pool_thresholds,
            chunks=cpu_count(),
        )
    return results

# Function to simulate dynamic winner payouts for a given range
def simulate_balanced_strategy_with_custom_winner_payouts(chunk_args):
    total_games, pool_contribution_values, pool_thresholds, winner_payout_range = chunk_args
    total_pool = 0
    total_table_profit = 0
    total_distributed_to_winners = 0
    total_loser_compensations = 0
    losers_who_got_partial_compensated = 0
    losers_who_got_full_compensated = 0
    losers_not_compensated = 0

    for game in range(1, total_games + 1):
        # Determine pool contribution based on current pool size
        if total_pool < pool_thresholds[0]:
            pool_contribution = pool_contribution_values[0]
        elif total_pool < pool_thresholds[1]:
            pool_contribution = pool_contribution_values[1]
        else:
            pool_contribution = pool_contribution_values[2]

        total_pool += pool_contribution
        total_table_profit += 8  # Fixed house fee

        # Dynamic winner payout within the custom range
        winner_payout = winner_payout_range[0] + (game % (winner_payout_range[1] - winner_payout_range[0] + 1))
        total_distributed_to_winners += winner_payout * (10 - 1)  # Winner payouts

        # Assign a loser cyclically
        loser = f"player_{game % 10}"

        # Dynamic loser compensation based on pool size
        if total_pool >= 50:
            if total_pool > 100000:  # High pool, more generous compensations
                compensation_rate = 1.1  # Boost compensation by 10%
            elif total_pool > 50000:  # Medium pool, standard compensations
                compensation_rate = 1.0
            else:  # Low pool, scaled-down compensations
                compensation_rate = 0.9

            compensated_amount = int(50 * compensation_rate)
            losers_who_got_partial_compensated += 1
            total_loser_compensations += compensated_amount
            total_pool -= compensated_amount
        else:
            losers_not_compensated += 1

    results = {
        "Total Games Simulated": total_games,
        "Total Losers Partially Compensated": losers_who_got_partial_compensated,
        "Total Losers Fully Compensated": losers_who_got_full_compensated,
        "Total Losers Not Compensated": losers_not_compensated,
        "Total Profits Distributed to Players": total_distributed_to_winners + total_loser_compensations,
        "Total Table (House) Profit": total_table_profit,
    }
    return results

# Function to run simulations for all levels with the custom payout range
def simulate_all_levels_custom_payouts(game_levels, pool_contribution_values, pool_thresholds, winner_payout_range):
    results = {}
    for level in game_levels:
        args = (level, pool_contribution_values, pool_thresholds, winner_payout_range)
        results[level] = simulate_balanced_strategy_with_custom_winner_payouts(args)
    return results

def simulate_balanced_strategy_with_custom_payout_range(chunk_args):
    total_games, pool_contribution_values, pool_thresholds, winner_payout_range = chunk_args
    total_pool = 0
    total_table_profit = 0
    total_distributed_to_winners = 0
    total_loser_compensations = 0
    losers_who_got_partial_compensated = 0
    losers_who_got_full_compensated = 0
    losers_not_compensated = 0

    for game in range(1, total_games + 1):
        # Determine pool contribution based on current pool size
        if total_pool < pool_thresholds[0]:
            pool_contribution = pool_contribution_values[0]
        elif total_pool < pool_thresholds[1]:
            pool_contribution = pool_contribution_values[1]
        else:
            pool_contribution = pool_contribution_values[2]

        total_pool += pool_contribution
        total_table_profit += 8  # Fixed house fee

        # Dynamic winner payout within the custom range
        winner_payout = winner_payout_range[0] + (game % (winner_payout_range[1] - winner_payout_range[0] + 1))
        total_distributed_to_winners += winner_payout * (10 - 1)  # Winner payouts

        # Assign a loser cyclically
        loser = f"player_{game % 10}"

        # Dynamic loser compensation based on pool size
        if total_pool >= 50:
            if total_pool > 100000:  # High pool, more generous compensations
                compensation_rate = 1.1  # Boost compensation by 10%
            elif total_pool > 50000:  # Medium pool, standard compensations
                compensation_rate = 1.0
            else:  # Low pool, scaled-down compensations
                compensation_rate = 0.9

            compensated_amount = int(50 * compensation_rate)
            losers_who_got_partial_compensated += 1
            total_loser_compensations += compensated_amount
            total_pool -= compensated_amount
        else:
            losers_not_compensated += 1

    results = {
        "Total Games Simulated": total_games,
        "Total Losers Partially Compensated": losers_who_got_partial_compensated,
        "Total Losers Fully Compensated": losers_who_got_full_compensated,
        "Total Losers Not Compensated": losers_not_compensated,
        "Total Profits Distributed to Players": total_distributed_to_winners + total_loser_compensations,
        "Total Table (House) Profit": total_table_profit,
    }
    return results

# Parallel simulation function for all levels with wider payout range
def simulate_all_levels_with_custom_range(game_levels, pool_contribution_values, pool_thresholds, winner_payout_range):
    results = {}
    for level in game_levels:
        args = (level, pool_contribution_values, pool_thresholds, winner_payout_range)
        results[level] = simulate_balanced_strategy_with_custom_payout_range(args)
    return results

# Define game levels and run simulations for $3–$9 payout range
game_levels = [10000, 100000, 1000000, 10000000, 100000000]
custom_range_results = simulate_all_levels_with_custom_range(
    game_levels=game_levels,
    pool_contribution_values=[65, 55, 45],
    pool_thresholds=[50000, 100000],
    winner_payout_range=(3, 9),
)

# Main block
if __name__ == '__main__':
    # Define game levels and run simulations
    # game_levels = [10000, 100000, 1000000, 10000000, 100000000]
    # all_levels_results_parallel = simulate_all_levels_parallel(
    #     game_levels=game_levels,
    #     pool_contribution_values=[65, 55, 45],
    #     pool_thresholds=[50000, 100000],
    # )

    # # Convert results to DataFrame
    # all_levels_results_parallel_df = pd.DataFrame.from_dict(all_levels_results_parallel, orient="index")
    # all_levels_results_parallel_df.index.name = "Games Level"
    # all_levels_results_parallel_df.reset_index(inplace=True)

    # # Display results
    # display_dataframe_to_user(name="All Levels Optimized Strategy Results (Parallel)", dataframe=all_levels_results_parallel_df)

    # # Plot results
    # plot_simulation_results(all_levels_results_parallel_df)

    # Run the simulation with dynamic winner payouts
    # game_levels = [10000, 100000, 1000000, 10000000, 100000000]
    # all_levels_results_dynamic_payouts = simulate_all_levels_parallel(
    #     game_levels=game_levels,
    #     pool_contribution_values=[65, 55, 45],
    #     pool_thresholds=[50000, 100000],
    # )

    # # Convert results to DataFrame
    # all_levels_results_dynamic_payouts_df = pd.DataFrame.from_dict(all_levels_results_dynamic_payouts, orient="index")
    # all_levels_results_dynamic_payouts_df.index.name = "Games Level"
    # all_levels_results_dynamic_payouts_df.reset_index(inplace=True)

    # # Display results
    # display_dataframe_to_user(name="Dynamic Winner Payouts Results", dataframe=all_levels_results_dynamic_payouts_df)

    # # Plot results
    # plot_simulation_results(all_levels_results_dynamic_payouts_df)

    # Define game levels and run simulations for $4–$8 payout range
    # game_levels = [10000, 100000, 1000000, 10000000, 100000000]
    # custom_payout_results = simulate_all_levels_custom_payouts(
    #     game_levels=game_levels,
    #     pool_contribution_values=[65, 55, 45],
    #     pool_thresholds=[50000, 100000],
    #     winner_payout_range=(4, 8),
    # )

    # # Convert results to DataFrame
    # custom_payout_results_df = pd.DataFrame.from_dict(custom_payout_results, orient="index")
    # custom_payout_results_df.index.name = "Games Level"
    # custom_payout_results_df.reset_index(inplace=True)

    # # Save results to display or analyze later
    # custom_payout_results_df.to_csv("custom_payout_results.csv", index=False)

    # # # Display results
    # display_dataframe_to_user(name="Dynamic Winner Payouts Results", dataframe=custom_payout_results_df)

    # # # Plot results
    # plot_simulation_results(custom_payout_results_df)

    # Define game levels and run simulations for $3–$9 payout range
    game_levels = [10000, 100000, 1000000, 10000000, 100000000]
    custom_range_results = simulate_all_levels_with_custom_range(
        game_levels=game_levels,
        pool_contribution_values=[65, 55, 45],
        pool_thresholds=[50000, 100000],
        winner_payout_range=(3, 9),
    )

    # Convert results to DataFrame
    custom_range_results_df = pd.DataFrame.from_dict(custom_range_results, orient="index")
    custom_range_results_df.index.name = "Games Level"
    custom_range_results_df.reset_index(inplace=True)

    # Save results for analysis
    custom_range_results_df.to_csv("custom_range_results.csv", index=False)

    # # Display results
    display_dataframe_to_user(name="Dynamic Winner Payouts Results", dataframe=custom_range_results_df)

    # # Plot results
    plot_simulation_results(custom_range_results_df)