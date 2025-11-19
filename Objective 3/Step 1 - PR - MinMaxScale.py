import pandas as pd
import numpy as np
import os  # <--- Added this to fix path issues

def min_max_scaling(df, metrics):
    """
    Applies Min-Max Scaling to specified columns in a DataFrame across the entire dataset.
    """
    df_normalized = df.copy()
    
    for metric in metrics:
        min_val = df_normalized[metric].min()
        max_val = df_normalized[metric].max()
        
        if max_val == min_val:
            df_normalized[f'Normalized_{metric}'] = 0
        else:
            df_normalized[f'Normalized_{metric}'] = (df_normalized[metric] - min_val) / (max_val - min_val)
            
    return df_normalized

# --- Main Analysis Script ---

# 1. Fix the Path Issue
# Get the directory where THIS script file is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Combine that directory with the filename
input_file = os.path.join(script_dir, 'Step 1 Resources/PR_fastfind_and_uw_raw_metrics.csv')
output_file = os.path.join(script_dir, 'Step 2 Resources/PR_scaled_metrics.csv')

print(f"Looking for file at: {input_file}") # Debug print to confirm location

try:
    df_data = pd.read_csv(input_file)
    print(f"Successfully loaded data from {input_file}")
except FileNotFoundError:
    print(f"Error: raw_metrics.csv not found at {input_file}")
    print("Please check that the CSV is in the 'Objective 3' folder.")
    exit()

# 2. Define the metrics to be scaled
metrics_to_scale = ['CPU_Load', 'Peak_RAM', 'Clock_Speed', 'Battery_Consumption']

# 3. Perform the Min-Max Scaling
df_normalized = min_max_scaling(df_data, metrics_to_scale)

# 4. Display and Save Results
print("\n--- Combined Mean Signatures and Normalized Data ---")
normalized_columns = ['Workload'] + metrics_to_scale + [f'Normalized_{m}' for m in metrics_to_scale]
print(df_normalized[normalized_columns])

# Save to the same directory as the input
df_normalized.to_csv(output_file, index=False)
print(f"\nFile saved successfully: {output_file}")