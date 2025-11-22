import pandas as pd
import numpy as np
import os
import sys

def calculate_specific_matches(df_all, metrics_cols):
    """
    Splits the dataset and performs constrained comparisons.
    Returns two DataFrames:
    1. Best Matches (Summary for Reporting)
    2. All Distances (Detailed Data for CI Analysis)
    """
    # 1. SPLIT THE DATA
    targets_df = df_all[df_all['Workload'].str.contains("UW")].copy()
    proxies_df_all = df_all[~df_all['Workload'].str.contains("UW")].copy()

    best_matches = []
    all_distances = []

    # 2. ITERATE THROUGH TARGETS (UW)
    for i, target_row in targets_df.iterrows():
        target_name = target_row['Workload']
        target_vector = target_row[metrics_cols].values.astype(float)
        
        # --- FILTER PROXIES BASED ON YOUR RULES ---
        if "Light" in target_name:
            current_proxies = proxies_df_all[proxies_df_all['Workload'].str.contains("PR")]
            category = "PR (Light)"
        elif "Medium" in target_name or "Moderate" in target_name:
            current_proxies = proxies_df_all[proxies_df_all['Workload'].str.contains("ECM")]
            category = "ECM (Moderate)"
        elif "Hard" in target_name or "Heavy" in target_name:
            current_proxies = proxies_df_all[proxies_df_all['Workload'].str.contains("QS")]
            category = "QS (Heavy)"
        else:
            current_proxies = proxies_df_all
            category = "All"

        best_match_name = None
        min_dist = float('inf')

        # 3. CALCULATE DISTANCE FOR EVERY PROXY IN THE GROUP
        for j, proxy_row in current_proxies.iterrows():
            proxy_name = proxy_row['Workload']
            proxy_vector = proxy_row[metrics_cols].values.astype(float)

            # Euclidean Distance Calculation
            distance = np.sqrt(np.sum((target_vector - proxy_vector) ** 2))
            
            # --- OUTPUT 2: Save to Detailed List (Table of Distances) ---
            all_distances.append({
                'UW_Target': target_name,
                'Comparison_Group': category,
                'AW_Candidate': proxy_name,
                'Euclidean_Distance': distance
            })
            
            # --- OUTPUT 1: Check for Best Match ---
            if distance < min_dist:
                min_dist = distance
                best_match_name = proxy_name

        # Save Best Match Result
        best_matches.append({
            'UW_Target': target_name,
            'Comparison_Group': category,
            'Best_Match_In_Group': best_match_name,
            'Euclidean_Distance': round(min_dist, 5)
        })

    return pd.DataFrame(best_matches), pd.DataFrame(all_distances)

# --- Main Execution ---

# 1. Setup Paths
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

input_file = os.path.join(script_dir, 'Step 2 Resources/PR_scaled_metrics.csv')
output_file_best = os.path.join(script_dir, 'Step 3 Resources/PR_specific_matches.csv') # The Summary
output_file_all = os.path.join(script_dir, 'Step 3 Resources/PR_all_distances.csv')     # The Full Table

# 2. Load Data
try:
    df = pd.read_csv(input_file)
    print(f"Data loaded successfully from {input_file}")
except FileNotFoundError:
    print(f"Error: Could not find {input_file}. Run Step 1 first.")
    sys.exit()

metric_cols = [
    'Normalized_CPU_Load', 
    'Normalized_Peak_RAM', 
    'Normalized_Clock_Speed', 
    'Normalized_Battery_Consumption'
]

# 3. Run Calculation
best_matches_df, all_distances_df = calculate_specific_matches(df, metric_cols)

# 4. Display and Save Results
print("\n--- 1. Best Matches (Point Estimate) ---")
print(best_matches_df)
best_matches_df.to_csv(output_file_best, index=False)
print(f"Saved to: {output_file_best}")

print("\n--- 2. All Distances (For CI Analysis) - First 5 Rows ---")
print(all_distances_df.head())
all_distances_df.to_csv(output_file_all, index=False)
print(f"Saved to: {output_file_all}")