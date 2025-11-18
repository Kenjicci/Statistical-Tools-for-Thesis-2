import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, rankdata
from scipy.interpolate import UnivariateSpline
import os # <--- IMPORTED OS MODULE HERE
import sys # Added for clean exit on FileNotFoundError


# --- Path Utility ---
def get_file_path(filename):
    """Constructs the absolute path to the data file in the script's directory."""
    try:
        # Get the directory of the currently executing script
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback for interactive environments (though correlation usually runs as script)
        script_dir = os.getcwd()

    return os.path.join(script_dir, filename)


# --- Data loader ---
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        x_col = 'Bit Length (x-axis)'
        y_col = 'CPU Load (y-axis)'

        if x_col not in df.columns or y_col not in df.columns:
            if 'Mean Runtime (y-axis)' in df.columns:
                print(f"Warning: Using 'Mean Runtime' data from {filepath} as proxy for CPU Load.")
                y_col = 'Mean Runtime (y-axis)'
            else:
                raise ValueError(f"Required columns '{x_col}' or '{y_col}' not found in {filepath}.")

        x_data = df[x_col].values.astype(np.float64)
        y_data = df[y_col].values.astype(np.float64)

        if len(x_data) < 4:
            raise ValueError(f"Insufficient data points ({len(x_data)} found). Must have at least 4 points.")

        return x_data, y_data

    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return np.array([]), np.array([])

# --- Plot Pearson ---
def plot_pearson(name, x, y, color='tab:blue'):
    r_val, p_val = pearsonr(x, y)
    pearson_success = "PASSED" if (r_val > 0.9 and p_val < 0.05) else "FAILED"

    coeffs = np.polyfit(x, y, 1)
    line_x = np.linspace(np.min(x), np.max(x), 200)
    line_y = np.polyval(coeffs, line_x)

    plt.figure(figsize=(8,6))
    plt.scatter(x, y, s=70, label='Data points', color=color, alpha=0.9)
    plt.plot(line_x, line_y, linestyle='--', linewidth=2, label='Linear fit', color=color)
    plt.title(f"Pearson: {name} (Laptop 2)", fontsize=14)
    stats_text = f"Pearson r = {r_val:.4f}\np-value = {p_val:.2e}\nSuccess: {pearson_success}"
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plt.xlabel('Semiprime Bit Length (Input Size)')
    plt.ylabel('CPU Load (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()

    return r_val, p_val, pearson_success

# --- Plot Spearman ---
def plot_spearman_smooth(name, x, y, color='tab:orange', smoothing_factor=None):
    rho_val, rho_p = spearmanr(x, y)
    spearman_success = "PASSED" if (rho_val > 0.9 and rho_p < 0.05) else "FAILED"

    x_rank = rankdata(x)
    y_rank = rankdata(y)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    xrank_sorted = x_rank[sort_idx]
    yrank_sorted = y_rank[sort_idx]
    y_sorted = y[sort_idx]

    if smoothing_factor is None:
        smoothing_factor = max(1.0, len(x) * 0.5)

    try:
        spline = UnivariateSpline(xrank_sorted, yrank_sorted, s=smoothing_factor)
        dense_xrank = np.linspace(np.min(xrank_sorted), np.max(xrank_sorted), 400)
        dense_yrank = spline(dense_xrank)
        dense_x = np.interp(dense_xrank, xrank_sorted, x_sorted)
        dense_y = np.interp(dense_yrank, yrank_sorted, y_sorted)

        plt.figure(figsize=(8,6))
        plt.scatter(x, y, s=70, label='Data points', color=color, alpha=0.9)
        plt.plot(dense_x, dense_y, linewidth=2.5, label='Smoothed trend', color=color, alpha=0.95)
        plt.title(f"Spearman (LOWESS-like): {name} (Laptop 2)", fontsize=14)
        stats_text = f"Spearman ρ = {rho_val:.4f}\np-value = {rho_p:.2e}\nSuccess: {spearman_success}"
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        plt.xlabel('Semiprime Bit Length (Input Size)')
        plt.ylabel('CPU Load (%)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')
        plt.tight_layout()

    except Exception as e:
        print(f"Warning: Spearman smoothing failed for {name}: {e}")
        plt.figure(figsize=(8,6))
        plt.scatter(x, y, s=70, label='Data points', color=color, alpha=0.9)
        plt.title(f"Spearman (ranks plotted): {name} (Laptop 3)", fontsize=14)
        stats_text = f"Spearman ρ = {rho_val:.4f}\np-value = {rho_p:.2e}\nSuccess: {spearman_success}"
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        plt.xlabel('Semiprime Bit Length (Input Size)')
        plt.ylabel('CPU Load (%)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')
        plt.tight_layout()

    return rho_val, rho_p, spearman_success

# --- Main execution ---
def run_all_and_summary(pr_file, ecm_file, qs_file):
    # Use the utility function to get the absolute paths
    pr_path = get_file_path(pr_file)
    ecm_path = get_file_path(ecm_file)
    qs_path = get_file_path(qs_file)

    pr_x, pr_y = load_data(pr_path)
    ecm_x, ecm_y = load_data(ecm_path)
    qs_x, qs_y = load_data(qs_path)

    summary = {}

    if pr_x.size < 4:
        print(f"Skipping Pollard's Rho due to insufficient data loaded from: {pr_path}")
    else:
        r_pr, p_pr, r_success = plot_pearson("Pollard's Rho", pr_x, pr_y, color='tab:green')
        rho_pr, prho_pr, rho_success = plot_spearman_smooth("Pollard's Rho", pr_x, pr_y, color='tab:olive')
        summary["Pollard's Rho"] = (r_pr, p_pr, r_success, rho_pr, prho_pr, rho_success)

    if ecm_x.size < 4:
        print(f"Skipping ECM due to insufficient data loaded from: {ecm_path}")
    else:
        r_ecm, p_ecm, r_success = plot_pearson("ECM", ecm_x, ecm_y, color='tab:blue')
        rho_ecm, prho_ecm, rho_success = plot_spearman_smooth("ECM", ecm_x, ecm_y, color='tab:cyan')
        summary["ECM"] = (r_ecm, p_ecm, r_success, rho_ecm, prho_ecm, rho_success)

    if qs_x.size < 4:
        print(f"Skipping Quadratic Sieve due to insufficient data loaded from: {qs_path}")
    else:
        r_qs, p_qs, r_success = plot_pearson("Quadratic Sieve", qs_x, qs_y, color='tab:red')
        rho_qs, prho_qs, rho_success = plot_spearman_smooth("Quadratic Sieve", qs_x, qs_y, color='tab:orange')
        summary["Quadratic Sieve"] = (r_qs, p_qs, r_success, rho_qs, prho_qs, rho_success)

    plt.show()

    # --- Summary Table ---
    print("\n--- Summary for Thesis Table 5.4 (Laptop 2) ---")
    print("| Algorithm | Pearson (r) | Pearson p | Pearson Success | Spearman (ρ) | Spearman p | Spearman Success |")
    print("|---|---:|---:|:---:|---:|---:|:---:|")
    for name, vals in summary.items():
        r, p, r_s, rho, prho, rho_s = vals
        print(f"| {name} | {r:.4f} | {p:.2e} | {r_s} | {rho:.4f} | {prho:.2e} | {rho_s} |")

# --- Run ---
if __name__ == "__main__":
    run_all_and_summary(
        "Laptop 2 - PR - SPearson.csv",
        "Laptop 2 - ECM - SPearson.csv",
        "Laptop 2 - QS - SPearson.csv"
    )