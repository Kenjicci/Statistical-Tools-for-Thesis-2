📊 Thesis Analysis: Predictable Scalability of Factorization Algorithms

This document explains the purpose and usage of the regression_analysis.py script, which is used to analyze the collected data for Objective 1 of the thesis.

Project Goal (Objective 1)

The primary objective of this code is to prove that the Wall-Clock Runtime of the three classical integer factorization algorithms (Pollard's Rho, ECM, and Quadratic Sieve) is predictable and scalable.

This is achieved by using Non-Linear Regression to fit the measured runtime data to the theoretical complexity functions of each algorithm, and calculating the R-squared ($R^2$) value as a measure of predictability.

Packages Required

To run the analysis script, you must have the following Python packages installed.

Package

Purpose

Installation Command

pandas

Imports data from CSV files.

pip install pandas

numpy

Handles large numbers and complex math (log, power).

pip install numpy

scipy

Performs the Non-Linear Regression (curve_fit).

pip install scipy

scikit-learn

Calculates the R-squared ($R^2$) metric.

pip install scikit-learn

matplotlib

Generates the final scatter plots and fitted curves.

pip install matplotlib

You can install all of them at once using:

pip install numpy scipy matplotlib scikit-learn pandas


Data Preparation

The script expects three specific CSV files, each containing the averaged runtime data from your 5 trials.

The files must be named exactly:

Laptop 3 - PR - Non Linear Data.csv

Laptop 3 - ECM - Non Linear Data.csv

Laptop 3 - QS - Non Linear Data.csv

Each file must have two columns with the exact headers:

Bit Length (x-axis)

Mean Runtime (y-axis)

How the Code Works (Simple Flow)

The regression_analysis.py script performs three independent analyses:

Algorithm

Model Used

Why This Model?

Key Output

Pollard's Rho

Power Law Function

This models the algorithm's theoretical $O(\sqrt{p})$ time complexity.

A high R² confirms predictable scalability.

ECM / QS

Sub-exponential (L-Notation)

This models the super-polynomial, but sub-exponential complexity of these algorithms.

A high R² proves the experiment matches theory.

Data Loading: pandas reads the CSV files, extracting the Bit Length (X) and Mean Runtime (Y).

Curve Fitting: The scipy.optimize.curve_fit function mathematically adjusts the parameters (like 'a', 'b', 'c') of the theoretical functions until the resulting curve passes as closely as possible through all the measured data points.

Validation: The R-squared ($R^2$) value is calculated. A result $R^2 \ge 0.95$ proves that the measured execution time is highly predictable based on the increase in bit length.

Visualization: matplotlib generates a three-panel plot showing the data points and the smooth, best-fit curve. This figure is used in your Chapter 5 Results section.