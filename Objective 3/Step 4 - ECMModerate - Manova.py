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
AW_FILE = os.path.join('Step 4 Resources', 'ECM_128_raw_trials_reduced.csv')
UW_FILE = os.path.join('Step 4 Resources', 'UW_moderate_raw_trials.csv')

# 2. Define Output File
OUTPUT_FILE = 'Step 4 - ECMModerate_MANOVA_Results.txt'

# 3. Define the Metric Columns to Test
# NOTE: For Light Tier, we exclude Battery and Clock Speed (Constant/Zero Variance)
CANDIDATE_METRICS = [
    'CPU_Load', 
    'Peak_RAM',
    'Battery_Consumption', 
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

    print(f"--- Loading Data for Medium Tier ---")
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
    try:
        # Select only numeric columns for the test
        df_test = df_comb[['Group'] + CANDIDATE_METRICS].copy()
    except KeyError as e:
        print(f"Error: Column names in CSV do not match configuration.")
        print(f"Configured: {CANDIDATE_METRICS}")
        print(f"Found in CSV: {df_comb.columns.tolist()}")
        sys.exit()

    # Run checks and get the final list of valid metrics
    final_metrics = check_assumptions(df_test, 'Group', CANDIDATE_METRICS)
    
    if not final_metrics:
        sys.exit()

    # 4. Run MANOVA
    print(f"\n--- Running One-Way MANOVA ---")
    print(f"Comparison: AW_ECM vs UW_Moderate")
    print(f"Feature Vector: {final_metrics}")
    
    # Construct Formula
    formula = ' + '.join([f'{m}' for m in final_metrics]) + ' ~ Group'
    
    try:
        manova = MANOVA.from_formula(formula, data=df_test)
        manova_results = manova.mv_test()
        
        # --- EXTRACTING SPECIFIC STATISTICS ---
        # Access the results table for the 'Group' term
        res_table = manova_results.results['Group']['stat']
        
        # 1. Test Statistic (Wilk's Lambda Value)
        wilks_stat = res_table.loc["Wilks' lambda", "Value"]
        
        # 2. F-Value (Wilk's Lambda F Value)
        wilks_f = res_table.loc["Wilks' lambda", "F Value"]
        
        # 3. P-Value (Wilk's Lambda Pr > F)
        wilks_p = res_table.loc["Wilks' lambda", "Pr > F"]
        
        # (Optional) Pillai's Trace for robust checking
        pillai_p = res_table.loc["Pillai's trace", "Pr > F"]
        
        print("\n--- MANOVA RESULTS TABLE ---")
        print(manova_results)
        
        # 5. Interpretation & Detailed Output
        print("\n" + "="*50)
        print("     FINAL STATISTICAL REPORT (For Thesis)")
        print("="*50)
        print(f"Test Statistic (Wilk's Λ): {wilks_stat:.5f}")
        print(f"F-value:                   {wilks_f:.5f}")
        # Use scientific notation for p-value if it is extremely small
        print(f"p-value (Pr > F):          {wilks_p:.5e}")
        print("-" * 50)
        
        result_str = ""
        result_str += f"Test Statistic (Wilk's Λ): {wilks_stat:.5f}\n"
        result_str += f"F-value: {wilks_f:.5f}\n"
        result_str += f"p-value: {wilks_p:.5e}\n\n"
        
        if wilks_p > 0.05:
            result_str += "DECISION: ACCEPT Null Hypothesis (H0)\n"
            result_str += "CONCLUSION: No significant difference (Statistically Equivalent).\n"
        else:
            result_str += "DECISION: REJECT Null Hypothesis (H0)\n"
            result_str += "CONCLUSION: Significant difference exists (Not Statistically Equivalent).\n"
            if wilks_p < 0.0001:
                result_str += "(Note: p < 0.0001 indicates an extremely strong statistical difference.)"
            
        print(result_str)
        
        # Save to file
        with open(output_path, "w") as f:
            f.write(str(manova_results))
            f.write("\n\n" + "="*50 + "\n")
            f.write("EXTRACTED METRICS\n")
            f.write("="*50 + "\n")
            f.write(result_str)
        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"MANOVA Failed: {e}")
        print("Tip: Check if your data has enough variance or if sample size is too small.")

if __name__ == "__main__":
    run_manova()