import pandas as pd
import numpy as np
import os

def calculate_specific_matches(df_all, metrics_cols):
    """
    Splits the dataset and performs constrained comparisons:
    - UW_Light is compared ONLY against PR workloads.
    - UW_Medium is compared ONLY against ECM workloads.
    - UW_Hard is compared ONLY against QS workloads.
    """
    # 1. SPLIT THE DATA
    targets_df = df_all[df_all['Workload'].str.contains("UW")].copy()
    proxies_df_all = df_all[~df_all['Workload'].str.contains("UW")].copy()

    results = []

    # 2. ITERATE THROUGH TARGETS (UW)
    for i, target_row in targets_df.iterrows():
        target_name = target_row['Workload']
        target_vector = target_row[metrics_cols].values.astype(float)
        
        # --- FILTER PROXIES BASED ON YOUR RULES ---
        if "Light" in target_name:
            # Compare only against PR
            current_proxies = proxies_df_all[proxies_df_all['Workload'].str.contains("PR")]
            category = "PR (Light)"
        elif "Medium" in target_name or "Moderate" in target_name:
            # Compare only against ECM
            current_proxies = proxies_df_all[proxies_df_all['Workload'].str.contains("ECM")]
            category = "ECM (Moderate)"
        elif "Hard" in target_name or "Heavy" in target_name:
            # Compare only against QS
            current_proxies = proxies_df_all[proxies_df_all['Workload'].str.contains("QS")]
            category = "QS (Heavy)"
        else:
            # Fallback if naming is different
            current_proxies = proxies_df_all
            category = "All"

        # 3. FIND BEST MATCH WITHIN THAT SPECIFIC FAMILY
        best_match_name = None
        min_dist = float('inf')

        for j, proxy_row in current_proxies.iterrows():
            proxy_name = proxy_row['Workload']
            proxy_vector = proxy_row[metrics_cols].values.astype(float)

            # Euclidean Distance
            distance = np.sqrt(np.sum((target_vector - proxy_vector) ** 2))
            
            if distance < min_dist:
                min_dist = distance
                best_match_name = proxy_name

        results.append({
            'UW_Target': target_name,
            'Comparison_Group': category,
            'Best_Match_In_Group': best_match_name,
            'Euclidean_Distance': round(min_dist, 5)
        })

    return pd.DataFrame(results)

# --- Main Execution ---

script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, 'PR_raw_metrics_scaled.csv') 
output_file = os.path.join(script_dir, 'PR_specific_matches.csv')

try:
    df = pd.read_csv(input_file)
    print("Data loaded.")
except FileNotFoundError:
    print("Error: Run Step 1 first to generate raw_metrics_scaled.csv")
    exit()

metric_cols = [
    'Normalized_CPU_Load', 
    'Normalized_Peak_RAM', 
    'Normalized_Clock_Speed', 
    'Normalized_Battery_Consumption'
]

match_df = calculate_specific_matches(df, metric_cols)

print("\n--- Specific Hypothesis Testing Results ---")
print(match_df)

match_df.to_csv(output_file, index=False)