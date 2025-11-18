import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import sys
import matplotlib.pyplot as plt

# --- FILE PATH DEFINITIONS ---
INPUT_FILENAME = 'PR_raw_metrics_scaled.csv'
OUTPUT_FILENAME = 'CI_interval.csv'  # <--- NEW OUTPUT FILENAME

# 1. Get the directory where THIS script is located
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# 2. Construct the full, absolute paths
input_path = os.path.join(script_dir, INPUT_FILENAME)
output_path = os.path.join(script_dir, OUTPUT_FILENAME)

# --- 1. LOAD DATA (AW) ---
try:
    df_aw = pd.read_csv(input_path)
    print(f"Successfully loaded AW data from: {input_path}")
except FileNotFoundError:
    print(f"Error: {INPUT_FILENAME} not found. File was expected at: {input_path}")
    sys.exit()

# --- 2. DEFINE UW TRIALS (PLACEHOLDER/SIMULATION) ---
# NOTE: In your real script, load your 5 raw UW trials here.
# For this example, we'll continue with the placeholder logic:
# ... (insert your UW loading/simulation logic here) ...
# Example placeholder:
df_aw_pr = df_aw[df_aw['Workload'].str.contains("PR")].copy()
df_aw_pr['Bit_Size'] = df_aw_pr['Workload'].str.extract('(\d+)').astype(int)
df_aw_pr = df_aw_pr.sort_values('Bit_Size')

# Simulating 5 trials for demonstration (Replace with your actual UW data loading)
target_mean = np.array([0.043, 0.025, 1.000, 0.050])
np.random.seed(42)
uw_trials = [target_mean + np.random.normal(0, 0.02, 4) for _ in range(5)]
uw_trials = np.array(uw_trials)
n_trials = len(uw_trials)

# --- 3. CALCULATE CI (LOGIC REMAINS THE SAME) ---
# ... (insert the CI calculation loop logic using df_aw_pr and uw_trials) ...
results = []
metrics = ['Normalized_CPU_Load', 'Normalized_Peak_RAM', 'Normalized_Clock_Speed', 'Normalized_Battery_Consumption']
for index, row in df_aw_pr.iterrows():
    # Placeholder calculation logic (same as previous turn)
    aw_vector = row[metrics].values.astype(float)
    distances = [np.sqrt(np.sum((aw_vector - trial_vector) ** 2)) for trial_vector in uw_trials]
    mean_dist = np.mean(distances)
    std_err = stats.sem(distances)
    h = std_err * stats.t.ppf((1 + 0.95) / 2., n_trials - 1)
    
    results.append({
        'Workload': row['Workload'],
        'Mean_Distance': mean_dist,
        'CI_Lower': mean_dist - h,
        'CI_Upper': mean_dist + h,
        'Error_Margin': h
    })
df_res = pd.DataFrame(results)

# --- 4. EXPORT TO NEW CSV FILE ---
# Exporting the results to the requested file path
df_res.to_csv(output_path, index=False)