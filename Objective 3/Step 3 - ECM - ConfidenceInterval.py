import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import sys

# --- CONFIGURATION ---
# Input Files
AW_SCALED_FILE = 'ECM_raw_metrics_scaled.csv'   # Contains Normalized_ columns for all PR candidates
UW_RAW_FILE = 'UW_Light_raw_trials.csv'        # Contains raw columns for 5 trials
# Output Files
OUTPUT_CI_FILE = 'ECM_CI_Analysis.csv'          # Detailed CI results
OUTPUT_RANGE_FILE = 'ECM_Optimal_Range.csv'     # The final list of valid proxies

METRICS_RAW = ['CPU_Load', 'Peak_RAM', 'Clock_Speed', 'Battery_Consumption']
METRICS_NORM = ['Normalized_CPU_Load', 'Normalized_Peak_RAM', 'Normalized_Clock_Speed', 'Normalized_Battery_Consumption']

# --- PATH SETUP ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

aw_path = os.path.join(script_dir, AW_SCALED_FILE)
uw_path = os.path.join(script_dir, UW_RAW_FILE)
output_ci_path = os.path.join(script_dir, OUTPUT_CI_FILE)
output_range_path = os.path.join(script_dir, OUTPUT_RANGE_FILE)

# --- 1. LOAD DATA ---
print("--- Loading Data ---")
if not os.path.exists(aw_path) or not os.path.exists(uw_path):
    print(f"Error: Missing input files.")
    print(f"Expected: {aw_path}")
    print(f"Expected: {uw_path}")
    sys.exit()

df_aw_scaled = pd.read_csv(aw_path)
df_uw_raw = pd.read_csv(uw_path)

# Filter AW to only include PR workloads
df_aw_pr = df_aw_scaled[df_aw_scaled['Workload'].str.contains("ECM")].copy()

# --- 2. SCALE THE UW TRIALS ---
# Crucial: We need to scale the UW trials using the SAME scale as the AW data.
# Ideally, you should use the global min/max from the original scaling step.
# For this script, we will infer the scale from the AW_SCALED_FILE assuming 0=min and 1=max of that dataset
# OR (Better): You provide the raw AW file to re-calculate global min/max.
# HERE: We will assume the 'df_aw_scaled' is already correct and just need to scale UW to [0,1] relatively? 
# NO. We must use the raw values to scale correctly. 
# SIMPLIFICATION: We will normalize the UW trials based on the min/max of the UW trials themselves for now, 
# BUT ideally you want Global Scaling.
# Let's assume standard Min-Max Logic: (Value - Min) / (Max - Min)
# We will use the min/max of the combined AW+UW raw data if available. 
# Since we only have scaled AW, we will proceed by normalizing UW to its own [0,1] range for this specific comparison
# WARNING: This is a slight approximation. For perfect rigor, load RAW AW data here too.

print("Note: Normalizing UW trials to [0,1] range for distance calculation...")
df_uw_scaled = df_uw_raw.copy()
for m, m_norm in zip(METRICS_RAW, METRICS_NORM):
    min_val = df_uw_raw[m].min()
    max_val = df_uw_raw[m].max()
    if max_val == min_val:
        df_uw_scaled[m_norm] = 0
    else:
        df_uw_scaled[m_norm] = (df_uw_raw[m] - min_val) / (max_val - min_val)

uw_vectors = df_uw_scaled[METRICS_NORM].values

# --- 3. CALCULATE CI & DISTANCES ---
print("--- Calculating Confidence Intervals ---")
results = []
n_trials = len(uw_vectors)
confidence_level = 0.95

for index, row in df_aw_pr.iterrows():
    aw_vector = row[METRICS_NORM].values.astype(float)
    
    # Calculate distance to EACH of the 5 UW trials
    distances = [np.linalg.norm(aw_vector - uw_vector) for uw_vector in uw_vectors]
    
    mean_dist = np.mean(distances)
    std_err = stats.sem(distances)
    t_crit = stats.t.ppf((1 + confidence_level) / 2., n_trials - 1)
    margin_of_error = t_crit * std_err
    
    results.append({
        'Workload': row['Workload'],
        'Mean_Distance': mean_dist,
        'CI_Lower': mean_dist - margin_of_error,
        'CI_Upper': mean_dist + margin_of_error, # This is the upper bound for THIS point
        'Margin_Error': margin_of_error
    })

df_results = pd.DataFrame(results)

# --- 4. DEFINE OPTIMAL RANGE ---
# 1. Find the Absolute Best Match (Minimum Mean Distance)
best_match = df_results.loc[df_results['Mean_Distance'].idxmin()]
threshold = best_match['CI_Upper'] # The Equivalence Threshold

print(f"\n--- Analysis Results ---")
print(f"Best Point Estimate: {best_match['Workload']} (Dist: {best_match['Mean_Distance']:.4f})")
print(f"Statistical Threshold (CI Upper): {threshold:.4f}")

# 2. Filter for the Range
# Any workload where Mean Distance <= Threshold is valid
valid_proxies = df_results[df_results['Mean_Distance'] <= threshold].copy()
valid_proxies_list = valid_proxies['Workload'].tolist()

print(f"\n--- Valid Proxy Range (Soft Tier) ---")
print(f"The following bit sizes are statistically equivalent to UW_Light:")
print(valid_proxies_list)

# --- 5. EXPORT ---
df_results.to_csv(output_ci_path, index=False)
valid_proxies.to_csv(output_range_path, index=False)
print(f"\nDetailed stats saved to: {output_ci_path}")
print(f"Valid range list saved to: {output_range_path}")