import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import sys

# --- CONFIGURATION ---
# 1. WE NEED THE RAW AW DATA TO GET THE CORRECT SCALING PARAMETERS
AW_RAW_FILE = 'Step 3 Resources/ECM_fastfind_and_uw_raw_metrics.csv' 
# 2. WE NEED THE RAW UW TRIALS
UW_RAW_FILE = 'Step 3 Resources/UW_moderate_raw_trials.csv'

# Output Files
OUTPUT_CI_FILE = 'Step 4 Resources/ECM_CI_Analysis.txt'
OUTPUT_RANGE_FILE = 'Step 4 Resources/ECM_Optimal_Range.txt'
# Define metrics
METRICS_RAW = ['CPU_Load', 'Peak_RAM', 'Clock_Speed', 'Battery_Consumption']
METRICS_NORM = ['Normalized_CPU_Load', 'Normalized_Peak_RAM', 'Normalized_Clock_Speed', 'Normalized_Battery_Consumption']

# --- PATH SETUP ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

aw_path = os.path.join(script_dir, AW_RAW_FILE)
uw_path = os.path.join(script_dir, UW_RAW_FILE)
output_ci_path = os.path.join(script_dir, OUTPUT_CI_FILE)
output_range_path = os.path.join(script_dir, OUTPUT_RANGE_FILE)

# --- 1. LOAD DATA ---
print("--- Loading Raw Data ---")
if not os.path.exists(aw_path) or not os.path.exists(uw_path):
    print(f"Error: Missing input files.")
    print(f"Checking: {aw_path}")
    print(f"Checking: {uw_path}")
    sys.exit()

df_aw_raw = pd.read_csv(aw_path)
df_uw_raw = pd.read_csv(uw_path)

# Filter AW to only include ECM workloads
df_aw_raw = df_aw_raw[df_aw_raw['Workload'].str.contains("ECM")].copy()

print(f"Loaded AW Candidates: {len(df_aw_raw)}")
print(f"Loaded UW Trials: {len(df_uw_raw)}")

# --- 2. PERFORM UNIFIED SCALING (THE FIX) ---
# We calculate the Min and Max from the AW dataset (The Reference Universe)
# And apply those exact same numbers to scale the UW dataset.

print("--- Applying Global Scaling ---")
df_aw_scaled = df_aw_raw.copy()
df_uw_scaled = df_uw_raw.copy()

for m, m_norm in zip(METRICS_RAW, METRICS_NORM):
    # 1. Get the Global Min/Max from the AW data (or combined if you prefer)
    # Usually, AW range covers the whole spectrum, so we use AW as the ruler.
    global_min = df_aw_raw[m].min()
    global_max = df_aw_raw[m].max()
    
    denom = global_max - global_min
    if denom == 0: denom = 1
    
    # 2. Scale AW
    df_aw_scaled[m_norm] = (df_aw_raw[m] - global_min) / denom
    
    # 3. Scale UW using AW's Min/Max (CRITICAL STEP)
    df_uw_scaled[m_norm] = (df_uw_raw[m] - global_min) / denom

# Extract vectors for calculation
uw_vectors = df_uw_scaled[METRICS_NORM].values

# --- 3. CALCULATE CI & DISTANCES ---
print("--- Calculating Confidence Intervals ---")
results = []
n_trials = len(uw_vectors)
confidence_level = 0.95

for index, row in df_aw_scaled.iterrows():
    aw_vector = row[METRICS_NORM].values.astype(float)
    
    # Calculate Euclidean distance to EACH of the 5 UW trials
    distances = [np.linalg.norm(aw_vector - uw_vector) for uw_vector in uw_vectors]
    
    mean_dist = np.mean(distances)
    std_err = stats.sem(distances)
    t_crit = stats.t.ppf((1 + confidence_level) / 2., n_trials - 1)
    margin_of_error = t_crit * std_err
    
    results.append({
        'Workload': row['Workload'],
        'Mean_Distance': mean_dist,
        'CI_Lower': mean_dist - margin_of_error,
        'CI_Upper': mean_dist + margin_of_error,
        'Margin_Error': margin_of_error
    })

df_results = pd.DataFrame(results)

# --- 4. DEFINE OPTIMAL RANGE ---
# 1. Find the Absolute Best Match (Minimum Mean Distance)
best_match = df_results.loc[df_results['Mean_Distance'].idxmin()]
threshold = best_match['CI_Upper'] 

print(f"\n--- Analysis Results (ECM vs UW_Moderate) ---")
print(f"Best Point Estimate:   {best_match['Workload']}")
print(f"Mean Distance:         {best_match['Mean_Distance']:.5f}")
print(f"Equivalence Threshold: {threshold:.5f}")

# 2. Filter for the Range
valid_proxies = df_results[df_results['Mean_Distance'] <= threshold].copy()
valid_proxies_list = valid_proxies['Workload'].tolist()

print(f"\n--- Valid Proxy Range ---")
print(f"The following bit sizes are statistically equivalent (p > 0.05) to the target:")
print(valid_proxies_list)

# --- 5. EXPORT ---
df_results.to_csv(output_ci_path, index=False)
valid_proxies.to_csv(output_range_path, index=False)
print(f"\nSaved analysis to: {output_ci_path}")
print(f"Saved range list to: {output_range_path}")