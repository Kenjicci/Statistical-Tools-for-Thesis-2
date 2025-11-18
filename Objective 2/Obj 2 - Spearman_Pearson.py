import numpy as np
from scipy.stats import pearsonr, spearmanr # Import correlation functions
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Define Data Loader (Modified to look for CPU Load) ---

def load_data(filepath, y_col_suffix):
    """
    Loads CSV data, assuming 'Bit Length' is the X-axis and Y is determined by the suffix.
    """
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        
        x_col = 'Bit Length (x-axis)'
        
        # We need to find the CPU Load data for Objective 2. 
        # Assuming the CSV files contain a column for CPU Load if used for this objective.
        y_col = 'Mean CPU Load (y-axis)' # Assuming CPU Load data is in this column name
        
        if x_col not in df.columns or y_col not in df.columns:
            # Fallback for demonstration: if the file only has Runtime, we cannot run Objective 2 accurately.
            # We'll check the original file's Y column name from Objective 1 for now.
            if 'Mean Runtime (y-axis)' in df.columns:
                print(f"Warning: Using 'Mean Runtime' data from {filepath} as proxy for CPU Load.")
                y_col = 'Mean Runtime (y-axis)'
            else:
                raise ValueError(f"Required columns '{x_col}' or '{y_col}' not found.")


        x_data = df[x_col].values.astype(np.float64)
        y_data = df[y_col].values.astype(np.float64)

        if len(x_data) < 4:
             raise ValueError(f"Insufficient data points ({len(x_data)} found). Must have at least 4 points.")
        
        return x_data, y_data
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return np.array([]), np.array([])

# --- 2. INPUT DATA: Load from CSV Files ---
print("Loading data for Objective 2 correlation analysis...")

# We load data using the same paths as Objective 1
pr_bits, pr_cpu_load = load_data("Laptop 1 - PR - Spearman Pearson Data.csv", "Load")
ecm_bits, ecm_cpu_load = load_data("Laptop 1 - ECM - Spearman Pearson Data.csv", "Load")
qs_bits, qs_cpu_load = load_data("Laptop 1 - QS - Spearman Pearson Data.csv", "Load")

# --- 3. Correlation Execution Function ---

def run_correlation_and_plot(name, x_data, y_data, plot_color='b'):
    """Performs dual correlation (Pearson, Spearman) and creates a separate plot."""
    if len(x_data) < 4:
        print(f"\n--- SKIPPING {name} --- Insufficient data points.")
        return None, None
        
    try:
        # Calculate Pearson Correlation (r)
        r_corr, r_pvalue = pearsonr(x_data, y_data)
        
        # Calculate Spearman Rank Correlation (rho)
        rho_corr, rho_pvalue = spearmanr(x_data, y_data)
        
        # --- Print Results (Console) ---
        print(f"\n--- {name} Correlation Results ---")
        print(f"Variables: Bit Length vs. CPU Load (or Runtime)")
        print(f"Pearson (r): {r_corr:.4f} (p-value: {r_pvalue:.4e})")
        print(f"Spearman (ρ): {rho_corr:.4f} (p-value: {rho_pvalue:.4e})")

        # Determine Success Criterion (Spearman rho > 0.9 and p < 0.05)
        success_status = "PASSED" if (rho_corr > 0.9 and rho_pvalue < 0.05) else "FAILED"
        print(f"H2 Success Status: {success_status}")

        # --- Plotting ---
        plt.figure(figsize=(9, 7))
        plt.scatter(x_data, y_data, color=plot_color, s=70, label='Data Points') 

        # Add title and stats box
        title_str = f"Objective 2: {name} Resource Scaling (Laptop 1)"
        plt.title(title_str, fontsize=14)
        
        stats_text = (
            f"Result: {success_status}\n"
            f"Pearson (r): {r_corr:.4f} (p={r_pvalue:.2e})\n"
            f"Spearman (ρ): {rho_corr:.4f} (p={rho_pvalue:.2e})\n"
            f"Success Criterion: ρ > 0.9 & p < 0.05"
        )
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

        plt.xlabel('Semiprime Bit Length (Input Size)')
        plt.ylabel('Mean CPU Load Percentage (or Runtime)') # Keep general label
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        return r_corr, r_pvalue, rho_corr, rho_pvalue
        
    except Exception as e:
        print(f"\n--- ERROR running correlation for {name} ---: {e}")
        return 0.0, 1.0, 0.0, 1.0

# --- 4. Main Execution and Plotting ---

# These calls will now each create a separate figure object.
r_pr, p_pr, rho_pr, prho_pr = run_correlation_and_plot(
    "Pollard's Rho", pr_bits, pr_cpu_load, 'g'
)

r_ecm, p_ecm, rho_ecm, prho_ecm = run_correlation_and_plot(
    "ECM", ecm_bits, ecm_cpu_load, 'b'
)

r_qs, p_qs, rho_qs, prho_qs = run_correlation_and_plot(
    "Quadratic Sieve", qs_bits, qs_cpu_load, 'r'
)

# This one plt.show() command will display all 3 figures at once.
plt.show()

# --- 5. Output for Thesis Table 5.4 ---
print("\n--- Summary for Thesis Table 5.4 (Laptop 1) ---")
print("| Algorithm | Pearson (r) | Spearman (ρ) | Spearman p-value | H2 Success |")
print("|---|---|---|---|---|")

data = {
    "Pollard's Rho": [r_pr, rho_pr, prho_pr],
    "ECM": [r_ecm, rho_ecm, prho_ecm],
    "Quadratic Sieve": [r_qs, rho_qs, prho_qs],
}

for name, values in data.items():
    r, rho, p_rho = values
    success = "PASSED" if (rho > 0.9 and p_rho < 0.05) else "FAILED"
    print(f"| {name} | {r:.4f} | {rho:.4f} | {p_rho:.4e} | {success} |")