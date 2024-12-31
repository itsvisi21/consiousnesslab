
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import json
from datetime import datetime

# Load the first half JSON data
with open('./eurojackpot/eurojackpot_first_half.json', 'r') as file:
    first_half_data = json.load(file)

# Load the recent JSON data
with open('./eurojackpot/eurojackpot_recent.json', 'r') as file:
    recent_data = json.load(file)

# Parse the first half data
parsed_first_half = []
for entry in first_half_data:
    split_data = entry.split(",")
    # Correct the date format with ordinal suffix handling
    date_string = split_data[0].replace("Friday", "").replace("Tuesday", "").replace("th", "").replace("st", "").replace("nd", "").replace("rd", "").strip()
    date = datetime.strptime(date_string, "%d %B %Y")
    main_numbers = list(map(int, split_data[1:6]))
    euro_numbers = list(map(int, split_data[6:]))
    parsed_first_half.append({"Date": date, "Main_Numbers": main_numbers, "Euro_Numbers": euro_numbers})

# Parse the recent data
parsed_recent = []
for entry in recent_data:
    split_data = entry.split(",")
    date = datetime.strptime(split_data[0].strip(), "%Y-%m-%d")
    main_numbers = list(map(int, split_data[1:6]))
    euro_numbers = list(map(int, split_data[6:8]))
    parsed_recent.append({"Date": date, "Main_Numbers": main_numbers, "Euro_Numbers": euro_numbers})

# Combine and sort by dates
combined_data = parsed_first_half + parsed_recent
combined_data_sorted = sorted(combined_data, key=lambda x: x["Date"])

# Convert to final structured format
eurojackpot_data = {
    "Date": [entry["Date"].strftime("%Y-%m-%d") for entry in combined_data_sorted],
    "Main_Numbers": [entry["Main_Numbers"] for entry in combined_data_sorted],
    "Euro_Numbers": [entry["Euro_Numbers"] for entry in combined_data_sorted],
}

compact_format = [
    f"{entry['Date'].strftime('%Y-%m-%d')},{','.join(map(str, entry['Main_Numbers']))},{','.join(map(str, entry['Euro_Numbers']))}"
    for entry in combined_data_sorted
]

# Remove duplicates from the combined data before saving
unique_data = list({entry["Date"]: entry for entry in combined_data_sorted}.values())

# Convert the unique data into the compact format
compact_format_unique = [
    f"{entry['Date'].strftime('%Y-%m-%d')},{','.join(map(str, entry['Main_Numbers']))},{','.join(map(str, entry['Euro_Numbers']))}"
    for entry in unique_data
]


# Save to a JSON file
file_path = './eurojackpot/eurojackpot_complete.json'
with open(file_path, 'w') as file:
    json.dump(compact_format_unique, file)


