# 📊 Thesis Analysis: Predictable Scalability of Factorization Algorithms

This document explains the purpose and usage of the Python script used for **Objective 1: Validating the Predictable Scalability of Classical Factorization Algorithms**.

The high R-squared (R²) value produced by this analysis serves as the statistical proof that the increase in semiprime bit size leads to a highly reliable and mathematically predictable increase in Wall-Clock Runtime.

## 1. Required Packages

To run the analysis script, ensure you have the following packages installed using `pip`.

| Package | Role in Analysis | Installation Command |
|---------|------------------|---------------------|
| **pandas** | Data Handling: Imports the CSV files and cleans column headers. | `pip install pandas` |
| **numpy** | Numerical Core: Handles all complex mathematical arrays and operations (logarithms, powers). | `pip install numpy` |
| **scipy** | Statistical Analysis: Contains the crucial `curve_fit` function for Non-Linear Regression. | `pip install scipy` |
| **scikit-learn** | Validation: Calculates the R-squared (R²) value, which is the primary measure of predictability. | `pip install scikit-learn` |
| **matplotlib** | Visualization: Generates the final plots for your thesis's Results chapter. | `pip install matplotlib` |

### Summary Installation Command:
```bash
pip install numpy scipy matplotlib scikit-learn pandas
```

## 2. Data Requirement and Structure

The script is designed to load your three successfully collected data files.

**File Naming:** Files must be in the same folder as the script and named exactly:
- `Laptop 3 - PR - Non Linear Data.csv`
- `Laptop 3 - ECM - Non Linear Data.csv`
- `Laptop 3 - QS - Non Linear Data.csv`

**Column Headers:** The script automatically strips spaces to avoid errors, but the fundamental headers must be present:
- `Bit Length` (x-axis)
- `Mean Runtime` (y-axis)

## 3. How the Code Works

The `main.py` script performs three independent regression analyses, one for each algorithm, based on its theoretical complexity:

| Algorithm | Theoretical Model Used | Purpose |
|-----------|------------------------|---------|
| **Pollard's Rho (PR)** | Power Law Function (y = a · x^b) | This models the O(√p) complexity. |
| **Elliptic Curve Method (ECM)** | Sub-exponential (L-Notation) | This models the complex, sub-exponential growth pattern. |
| **Quadratic Sieve (QS)** | Sub-exponential (L-Notation) | This models the intensive, but predictable, exponential growth. |

### The Workflow:

1. **Loading:** Data is imported, and columns are automatically cleaned.
2. **Fitting:** The `curve_fit` function finds the optimal parameters (a, b, c) to make the theoretical curve match your measured data.
3. **Validation:** The R² value is calculated. Your strong results indicate an extremely high degree of fit between your experimental results and the theoretical complexity models.
4. **Output:** Prints the R-squared values for your thesis's Table 5.1 and displays the three final validation plots.
