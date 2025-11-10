import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd # <-- Added pandas for CSV handling

# --- 1. Define Theoretical Regression Functions ---

def func_power_law(x, a, b):
    """
    Theoretical Model for Pollard's Rho (O(sqrt(p)) which scales to N^(1/4)).
    Model: y = a * x^b
    """
    x = np.array(x, dtype=np.float64)
    # Use np.power to handle large numbers and general power law fitting
    return a * np.power(x, b)

def func_sub_exponential(x, a, c):
    """
    Theoretical Model for ECM/QS (Simplified L-Notation, L_n[1/2, c]).
    Model: y = a * exp(c * sqrt(ln(x) * ln(ln(x))))
    """
    x = np.array(x, dtype=np.float64)
    # Ensure x > e for log(log(x)) to be defined and real. Bit lengths > 8 satisfy this.
    log_x = np.log(x)
    log_log_x = np.log(log_x)
    # Handle possible non-real result if log_log_x is negative (e.g., if bit length is too small)
    if np.any(np.isnan(log_log_x)):
        raise ValueError("Log of Log results in non-real numbers. Check input bit lengths.")
        
    return a * np.exp(c * np.sqrt(log_x * log_log_x))

def calculate_r_squared(y_true, y_pred):
    """Calculates the R-squared (Goodness-of-Fit) value."""
    return r2_score(y_true, y_pred)

# --- CSV Data Loader Function (Robust Version) ---
def load_data(filepath):
    """Loads CSV data, cleans column names, and returns Bit Length (x) and Mean Runtime (y) arrays."""
    try:
        df = pd.read_csv(filepath)
        
        # CRITICAL FIX: Strip leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Define the expected column names after stripping spaces
        x_col = 'Bit Length (x-axis)'
        y_col = 'Mean Runtime (y-axis)'

        if x_col not in df.columns or y_col not in df.columns:
            # If after stripping, names still don't match, raise a clear error
            raise ValueError(f"Required columns '{x_col}' or '{y_col}' not found. Found columns: {list(df.columns)}")

        x_data = df[x_col].values.astype(np.float64)
        y_data = df[y_col].values.astype(np.float64)

        if len(x_data) < 4: # Need minimum 4 points for a good non-linear fit
             raise ValueError(f"Insufficient data points ({len(x_data)} found). Must have at least 4 significant points.")
        
        return x_data, y_data
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return np.array([]), np.array([])

# --- 2. INPUT DATA: Load from CSV Files ---
print("Loading data from uploaded CSV files...")

# Pollard's Rho Data
pr_bits, pr_runtime_mean = load_data("Laptop 3 - PR - Non Linear Data.csv")

# ECM Data
ecm_bits, ecm_runtime_mean = load_data("Laptop 3 - ECM - Non Linear Data.csv")

# Quadratic Sieve Data
qs_bits, qs_runtime_mean = load_data("Laptop 3 - QS - Non Linear Data.csv")

# --- 3. Regression Execution Function ---

def run_regression_and_plot(name, func, x_data, y_data, subplot_index, plot_style='r-'):
    """Performs the curve fit, calculates R^2, and plots the result."""
    if len(x_data) < 4:
        print(f"\n--- SKIPPING {name} --- Insufficient data points for regression. Need at least 4 points.")
        return None, None
        
    try:
        # Use curve_fit to find the best parameters
        # p0 provides an initial guess, helping the sub-exponential model converge
        initial_guess = [1.0, 0.1] 
        popt, pcov = curve_fit(func, x_data, y_data, p0=initial_guess, maxfev=5000)
        
        # Predict Y values using the fitted model
        y_pred = func(x_data, *popt)
        
        # Calculate R-squared
        r2 = calculate_r_squared(y_data, y_pred)
        
        # --- Print Results ---
        print(f"\n--- {name} Results ---")
        print(f"Data Points: {len(x_data)}")
        print(f"Fitted Parameters: {popt}")
        print(f"R-squared (R^2): {r2:.4f}")

        # --- Plotting ---
        plt.subplot(1, 3, subplot_index)
        plt.scatter(x_data, y_data, label=f'Mean Data (R²={r2:.3f})', color='blue')
        
        # Create a smooth fitted line
        x_smooth = np.linspace(min(x_data), max(x_data) * 1.05, 100) # Extend x-axis slightly
        plt.plot(x_smooth, func(x_smooth, *popt), plot_style, label='Fitted Curve')
        
        plt.title(f'{name} (R²={r2:.4f})')
        plt.xlabel('Semiprime Bit Length')
        plt.ylabel('Mean Wall-Clock Runtime (s)')
        plt.legend(fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Return the R^2 and parameters for use in Table 5.1
        return r2, popt
        
    except RuntimeError as e:
        # This error is usually "Optimal parameters not found: The iteration is not making good progress."
        # It happens when the data is too noisy or the model is too complex for the data.
        print(f"\n--- ERROR: {name} ---")
        print(f"Could not converge on optimal parameters. R-squared will be low or missing.")
        return 0.0, None # Return 0.0 for R^2 if the fit failed
    except Exception as e:
        print(f"\n--- ERROR: {name} ---")
        print(f"An unexpected error occurred during fitting: {e}")
        return 0.0, None


# --- 4. Main Execution and Plotting ---

plt.figure(figsize=(18, 6))
plt.suptitle("Objective 1: Non-Linear Regression for Scalability Validation (Laptop 3)", fontsize=16)

# 1. Pollard's Rho (Power Law)
r2_pr, params_pr = run_regression_and_plot("Pollard's Rho", func_power_law, pr_bits, pr_runtime_mean, 1, 'g-')

# 2. ECM (Sub-exponential L-Notation)
r2_ecm, params_ecm = run_regression_and_plot("ECM", func_sub_exponential, ecm_bits, ecm_runtime_mean, 2, 'b-')

# 3. Quadratic Sieve (Sub-exponential L-Notation)
r2_qs, params_qs = run_regression_and_plot("Quadratic Sieve", func_sub_exponential, qs_bits, qs_runtime_mean, 3, 'r-')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
plt.show()

# --- 5. Output for Thesis Table 5.1 ---
print("\n--- Summary for Thesis Table 5.1 (Laptop 3) ---")
if r2_pr is not None:
    print(f"Pollard's Rho R^2: {r2_pr:.4f}")
if r2_ecm is not None:
    print(f"ECM R^2: {r2_ecm:.4f}")
if r2_qs is not None:
    print(f"Quadratic Sieve R^2: {r2_qs:.4f}")