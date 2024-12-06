import numpy as np
from sympy import symbols, Eq, solve
from sympy.interactive import printing
from sympy import pi
import pandas as pd
import matplotlib.pyplot as plt

def lambdaSubstrate(T):
    #lambda_substrate_ref= 3.75# TOCS Manual
    lambda_substrate_ref = 3
    lambdaSub=lambda_substrate_ref + 0.0013 * (T - 20)

    return lambdaSub

def fit_dVdf(lnf,v3w):

    X = lnf
    Y = v3w

    # Perform linear regression using NumPy's polyfit
    coefficients = np.polyfit(X, Y, 1)
    slope, intercept = coefficients

    # Print the slope and intercept
    print(f"Slope: {slope}, Intercept: {intercept}")

    # Linear regression
    Y_pred = slope * X + intercept

    # Plot the data and regression line
    plt.scatter(X, Y, label='Experimental Data')
    plt.plot(X, Y_pred, color='red', label='Regression Line')
    plt.xlabel('ln(f(Hz))')
    plt.ylabel('Re(V3w)')
    plt.legend()
    plt.show()

    return slope

def import_Data(path):

    # File import
    with open(path, 'r') as file:
        lines = file.readlines()[0:]

    data_sample = []
    for line in lines:
        # Split
        parts = line.split()

        # Bye first column
        row = parts[2:]
        data_sample.append(row)

    # Column names
    columns = [
        "Frequency", "Re(1ω)", "Im(1ω)", "Re(3ω)", "Im(3ω)", "Phase(1ω)", "Phase(3ω)",
        "Vref", "Current", "Re(ΔT)", "Δ[Re(ΔT)]", "Im(ΔT)", "Δ[Im(ΔT)]", "Power", "Temperature"
    ]

    # DataFrame
    df_sample = pd.DataFrame(data_sample, columns=columns)

    return df_sample