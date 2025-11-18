import numpy as np
from scipy.optimize import curve_fit
from scipy import stats  # Import stats for p-value calculation
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import os # <--- REQUIRED to find files correctly

# --- 1. Define Theoretical Regression Functions ---

def func_power_law(x, a, b):
    """
    Theoretical Model for Pollard's Rho (O(sqrt(p)) which scales to N^(1/4)).
    Model: y = a * x^b
    """
    x = np.array(x, dtype=np.float64)
    return a * np.power(x, b)

def func_sub_exponential(x, a, c):
    """
    Theoretical Model for ECM/QS (Simplified L-Notation, L_n[1/2, c]).
    Model: y = a * exp(c * sqrt(ln(x) * ln(ln(x))))
    """
    x = np.array(x, dtype=np.float64)
    # Ensure x > e for log(log(x)) to be defined and real.
    log_x = np.log(x)
    
    # Handle potential issues with small numbers or negative logs
    if np.any(log_x <= 0):
         # If log_x is negative or zero, we return a large number or handle gracefully
         # For fitting, we usually mask these, but here we will just let numpy warn
         pass

    log_log_x = np.log(log_x)
    
    if np.any(np.isnan(log_log_x)):
        # This might happen if x is too small, but for crypto bit lengths it should be fine
        pass 
        
    return a * np.exp(c * np.sqrt(log_x * log_log_x))

def calculate_r_squared(y_true, y_pred):
    """Calculates the R-squared (Goodness-of-Fit) value."""
    return r2_score(y_true, y_pred)

# --- CSV Data Loader Function (Robust Version) ---
def load_data(filepath):
    """Loads CSV data, cleans column names, and returns Bit Length (x) and Mean Runtime (y) arrays."""
    try:
        if not os.path.exists(filepath):
            print(f"Error: File not found at {filepath}")
            return np.array([]), np.array([])

        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        
        x_col = 'Bit Length (x-axis)'
        y_col = 'Mean Runtime (y-axis)'

        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Required columns '{x_col}' or '{y_col}' not found. Found columns: {list(df.columns)}")

        # Drop empty rows
        df = df.dropna(subset=[x_col, y_col])

        x_data = df[x_col].values.astype(np.float64)
        y_data = df[y_col].values.astype(np.float64)

        if len(x_data) < 4: # Need minimum 4 points for a good non-linear fit
             print(f"Warning: Insufficient data points ({len(x_data)}) in {os.path.basename(filepath)}.")
             # We return it anyway, but the regression function will catch it
             return x_data, y_data
        
        return x_data, y_data
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return np.array([]), np.array([])

# --- 2. INPUT DATA: Load from CSV Files ---
print("Loading data from CSV files...")

# --- FIX: Get the directory where THIS script is located ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the full paths to the files
# CHECK: Ensure these filenames match your actual files (e.g., Laptop 2 vs Laptop 3)
file_pr = os.path.join(script_dir, "Laptop 2 - PR - Non Linear Data.csv")
file_ecm = os.path.join(script_dir, "Laptop 2 - ECM - Non Linear Data.csv")
file_qs = os.path.join(script_dir, "Laptop 2 - QS - Non Linear Data.csv")

pr_bits, pr_runtime_mean = load_data(file_pr)
ecm_bits, ecm_runtime_mean = load_data(file_ecm)
qs_bits, qs_runtime_mean = load_data(file_qs)

# --- 3. Regression Execution Function ---

def run_regression_and_plot(name, func, x_data, y_data, plot_style='r-'):
    """Performs curve fit, calculates R^2/p-values, and creates a separate plot."""
    
    # Check for empty or insufficient data
    if x_data is None or len(x_data) < 4:
        print(f"\n--- SKIPPING {name} --- Insufficient data points (needs 4+).")
        return None, None, None
        
    try:
        # Initial guesses help the solver converge
        if func.__name__ == "func_power_law":
            initial_guess = [1.0, 4.0] # Guess for a, b
        else:
            initial_guess = [0.001, 1.0] # Guess for a, c
            
        # Use curve_fit to find the best parameters
        popt, pcov = curve_fit(func, x_data, y_data, p0=initial_guess, maxfev=10000)
        
        # Predict Y values using the fitted model
        y_pred = func(x_data, *popt)
        
        # Calculate R-squared
        r2 = calculate_r_squared(y_data, y_pred)

        # --- Calculate p-values for parameters ---
        perr = np.sqrt(np.diag(pcov)) # Standard Error for each parameter
        dof = len(x_data) - len(popt) # Degrees of freedom
        t_values = popt / perr # t-statistic for each parameter
        p_values = [2 * stats.t.sf(np.abs(t), dof) for t in t_values] # p-values
        
        # Get the p-value for the *scaling* parameter (the 2nd one, 'b' or 'c')
        p_val_scaling = p_values[1]
        param_names = func.__code__.co_varnames[1:func.__code__.co_argcount]
        
        # --- Print Results (Console) ---
        print(f"\n--- {name} Results ---")
        print(f"Data Points: {len(x_data)}")
        print(f"R-squared (R^2): {r2:.4f}")
        print("Parameter Statistics:")
        for i, pname in enumerate(param_names):
            print(f"  Param '{pname}':")
            print(f"    Estimate: {popt[i]:.4e}")
            print(f"    p-value:  {p_values[i]:.4e}")
        
        # --- Plotting ---
        plt.figure(figsize=(9, 7))
        plt.scatter(x_data, y_data, label='Mean Data', color='blue', s=50, zorder=5)
        
        # Create a smooth fitted line
        x_smooth = np.linspace(min(x_data), max(x_data) * 1.05, 100)
        plt.plot(x_smooth, func(x_smooth, *popt), plot_style, label='Fitted Curve')
        
        title_str = (
            f"Objective 1: {name} Scalability (Laptop 2)\n"
            f"R² = {r2:.4f}  |  p-value (scaling param) = {p_val_scaling:.3e}"
        )
        plt.title(title_str, fontsize=14)
        
        # Stats Box
        stats_text = (
            f"Model: {func.__name__}\n"
            f"R-squared: {r2:.4f}\n"
            f"Parameters:\n"
            f"  {param_names[0]}: {popt[0]:.3e}\n"
            f"  {param_names[1]}: {popt[1]:.3e} (p={p_val_scaling:.3e})"
        )
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

        plt.xlabel('Semiprime Bit Length')
        plt.ylabel('Mean Wall-Clock Runtime (s)')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        return r2, popt, p_values
        
    except Exception as e:
        print(f"\n--- ERROR: {name} ---")
        print(f"An error occurred during fitting: {e}")
        return 0.0, None, None


# --- 4. Main Execution and Plotting ---

r2_pr, params_pr, p_values_pr = run_regression_and_plot(
    "Pollard's Rho", func_power_law, pr_bits, pr_runtime_mean, 'g-'
)

r2_ecm, params_ecm, p_values_ecm = run_regression_and_plot(
    "ECM", func_sub_exponential, ecm_bits, ecm_runtime_mean, 'b-'
)

r2_qs, params_qs, p_values_qs = run_regression_and_plot(
    "Quadratic Sieve", func_sub_exponential, qs_bits, qs_runtime_mean, 'r-'
)

# Display all plots
plt.show()

# --- 5. Output for Thesis Table 5.1 ---
print("\n--- Summary for Thesis Table 5.1 (Laptop 2) ---")
print("| Algorithm | Model | R-squared (R^2) | Scaling Param (p-value) | H1 (p<0.05) | Success (R^2>=0.95) |")
print("|---|---|---|---|---|---|")

if r2_pr is not None and p_values_pr is not None:
    param_name = func_power_law.__code__.co_varnames[2] # 'b'
    p_val_str = f"{p_values_pr[1]:.4e}"
    h1_accepted = "Accepted" if p_values_pr[1] < 0.05 else "Rejected"
    success = "PASSED" if (p_values_pr[1] < 0.05 and r2_pr >= 0.95) else "FAILED"
    print(f"| Pollard's Rho | Power Law ('{param_name}') | {r2_pr:.4f} | {p_val_str} | {h1_accepted} | {success} |")

if r2_ecm is not None and p_values_ecm is not None:
    param_name = func_sub_exponential.__code__.co_varnames[2] # 'c'
    p_val_str = f"{p_values_ecm[1]:.4e}"
    h1_accepted = "Accepted" if p_values_ecm[1] < 0.05 else "Rejected"
    success = "PASSED" if (p_values_ecm[1] < 0.05 and r2_ecm >= 0.95) else "FAILED"
    print(f"| ECM | L-Notation ('{param_name}') | {r2_ecm:.4f} | {p_val_str} | {h1_accepted} | {success} |")

if r2_qs is not None and p_values_qs is not None:
    param_name = func_sub_exponential.__code__.co_varnames[2] # 'c'
    p_val_str = f"{p_values_qs[1]:.4e}"
    h1_accepted = "Accepted" if p_values_qs[1] < 0.05 else "Rejected"
    success = "PASSED" if (p_values_qs[1] < 0.05 and r2_qs >= 0.95) else "FAILED"
    print(f"| Quadratic Sieve | L-Notation ('{param_name}') | {r2_qs:.4f} | {p_val_str} | {h1_accepted} | {success} |")