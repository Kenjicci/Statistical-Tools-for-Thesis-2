import pandas as pd
import numpy as np
import os
import sys
from statsmodels.multivariate.manova import MANOVA
from scipy import stats
import math

# =============================================================================
# CONFIGURATION
# =============================================================================

# 1. Define Input Files (Relative to this script)
AW_FILE = os.path.join('Step 5 Resources', 'soft_points_raw_trial.csv')
UW_FILE = os.path.join('Step 3 Resources', 'UW_light_raw_trial.csv')

# 2. Define Output File
OUTPUT_FILE = 'Step 5 - PRLight_MANOVA_Results.txt'

# 3. Define the Metric Columns to Test
# NOTE: Ensure these match your CSV headers EXACTLY.
# For Moderate tier, we often exclude Battery if it is 0.
CANDIDATE_METRICS = [
    'CPU_Load', 
    'Peak_RAM', 
    'Clock_Speed', 
    'Battery_Consumption'
]

# =============================================================================
# UTILITY FUNCTIONS (Assumption Checking)
# =============================================================================

def check_assumptions(df_comb, group_col, metrics):
    """Runs basic checks for MANOVA assumptions."""
    print("\n--- Assumption Checks ---")
    
    # 1. Check for Zero Variance (Singular Matrix Prevention)
    valid_metrics = []
    for m in metrics:
        variance = df_comb[m].var()
        if variance < 1e-5:
            print(f"WARNING: Metric '{m}' has near-zero variance ({variance:.5f}). Excluding from MANOVA.")
        else:
            valid_metrics.append(m)
            
    if len(valid_metrics) < 2:
        print("CRITICAL ERROR: Fewer than 2 valid metrics remaining. MANOVA requires at least 2 vectors.")
        return None

    # 2. Shapiro-Wilk (Univariate Normality)
    print(f"Checking Normality on {valid_metrics}:")
    for m in valid_metrics:
        stat, p = stats.shapiro(df_comb[m])
        status = "Normal" if p > 0.05 else "Not Normal"
        print(f"  - {m}: p={p:.4f} ({status})")
        
    return valid_metrics

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_manova():
    # 1. Setup Paths
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    aw_path = os.path.join(script_dir, AW_FILE)
    uw_path = os.path.join(script_dir, UW_FILE)
    output_path = os.path.join(script_dir, OUTPUT_FILE)

    print(f"--- Loading Data for Moderate Tier ---")
    if not os.path.exists(aw_path) or not os.path.exists(uw_path):
        print(f"Error: Input files not found.")
        print(f"Checked: {aw_path}")
        print(f"Checked: {uw_path}")
        sys.exit()

    # 2. Load and Label Data
    try:
        df_aw = pd.read_csv(aw_path)
        df_uw = pd.read_csv(uw_path)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        sys.exit()

    # Assign Groups
    df_aw['Group'] = 'AW_ECM'
    df_uw['Group'] = 'UW_Moderate'

    # Combine
    df_comb = pd.concat([df_aw, df_uw], axis=0, ignore_index=True)
    print(f"Combined Data: {len(df_comb)} total rows (Should be 10 if 5+5)")

    # 3. Filter Columns & Check Assumptions
    # We strictly use the raw metrics defined in config
    try:
        # Select only numeric columns for the test
        df_test = df_comb[['Group'] + CANDIDATE_METRICS].copy()
    except KeyError as e:
        print(f"Error: Column names in CSV do not match configuration.")
        print(f"Configured: {CANDIDATE_METRICS}")
        print(f"Found in CSV: {df_comb.columns.tolist()}")
        sys.exit()

    # Run checks and get the final list of valid metrics (dropping constants)
    final_metrics = check_assumptions(df_test, 'Group', CANDIDATE_METRICS)
    
    if not final_metrics:
        sys.exit()

    # 4. Run MANOVA
    print(f"\n--- Running One-Way MANOVA ---")
    print(f"Comparison: AW_ECM vs UW_Moderate")
    print(f"Feature Vector: {final_metrics}")
    
    # Construct Formula: "Metric1 + Metric2 ~ Group"
    # Use Q() to handle potential spaces in column names
    formula = ' + '.join([f'{m}' for m in final_metrics]) + ' ~ Group'
    
    try:
        manova = MANOVA.from_formula(formula, data=df_test)
        manova_results = manova.mv_test()
        
        # Extract Pillai's Trace (Robust Metric)
        # Structure: results['Term']['stat'].loc['Test', 'Pr > F']
        res_table = manova_results.results['Group']['stat']
        pillai_p = res_table.loc["Pillai's trace", "Pr > F"]
        
        print("\n--- MANOVA RESULTS ---")
        print(manova_results)
        
        # 5. Interpretation
        print("\n" + "="*40)
        print("FINAL STATISTICAL DECISION")
        print("="*40)
        print(f"Pillai's Trace p-value: {pillai_p:.5f}")
        
        result_str = ""
        if pillai_p > 0.05:
            result_str += "DECISION: ACCEPT Null Hypothesis (H0)\n"
            result_str += "CONCLUSION: There is NO significant difference between the groups.\n"
            result_str += "VALIDATION: The ECM points are a STATISTICALLY VALID PROXY."
        else:
            result_str += "DECISION: REJECT Null Hypothesis (H0)\n"
            result_str += "CONCLUSION: There IS a significant difference between the groups.\n"
            result_str += "VALIDATION: The ECM points are NOT a valid proxy."
            
        print(result_str)
        
        # Save to file
        with open(output_path, "w") as f:
            f.write(str(manova_results))
            f.write("\n\n" + "="*40 + "\n")
            f.write(result_str)
        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"MANOVA Failed: {e}")
        print("Tip: Check if your data has enough variance or if sample size is too small.")

if __name__ == "__main__":
    run_manova()