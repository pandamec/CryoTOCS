
import numpy as np
from sympy import symbols, Eq, solve
from sympy.interactive import printing
from sympy import pi
import pandas as pd
import matplotlib.pyplot as plt

## Calibration

# File import
with open('test\Messung051224\GreaseDataVacuum1.txt', 'r') as file:
    lines = file.readlines()[0:]

data = []
for line in lines:
    # Split
    parts = line.split()

    # Bye first column
    row = parts[2:]
    data.append(row)

# Column names
columns = [
    "Frequency", "Re(1ω)", "Im(1ω)", "Re(3ω)", "Im(3ω)", "Phase(1ω)", "Phase(3ω)",
    "Vref", "Current", "Re(ΔT)", "Δ[Re(ΔT)]", "Im(ΔT)", "Δ[Im(ΔT)]", "Power", "Temperature"
]

# DataFrame
df = pd.DataFrame(data, columns=columns)
print(df)

### Korrelationsgleichung fuer die Kalibrierung

# LaTeX rendering
printing.init_printing()

# Define variables
dRdT = symbols('dR/dT')
dVdf= symbols('(dV3w/dlnf)')
dR,dT,L,lambda_grease,lambda_sub,f,P,I = symbols('dR dT L lambda_grease lambda_sub f P I')

equation_dRdT = Eq( ((-4*pi*L)/(P*I))*(lambda_grease+lambda_sub)*(dVdf)- dRdT,0)

print(equation_dRdT)

## dRdT Berechnung (unter Vacuum)

### Parameters fuer die Berechnung

# Geometrie und Bedingungen
T_ref=26                  # Room temperature, 24 04.12.24

# Slope dRe(V3w)/dlnf Berechnung
v3w=df['Re(3ω)'].values
f=df['Frequency'].values
v3w=v3w.astype(float)
f=f.astype(float)

## dRdT Berechnung

L_mess=1000e-6
P_mess= df['Power'].values
P_mess=P_mess.astype(float)

I_mess= df['Current'].values
I_mess=I_mess.astype(float)

### Substrate thermal conductivity

lambda_substrate_ref= 1.2 # Temperature 20C
lambdaSubstrate=lambda lambda_substrate_ref,T: lambda_substrate_ref + 0.0013*(T-20)

### Slope
lnf=np.log(f)

X=lnf
Y=v3w
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

# Sölope 'dV3w/dlnf' calculated for dR/dT calculation

dVdf_mess=slope
print(dVdf_mess)

### dRdT Berechnung

P_mean= np.mean(P_mess)
I_mean= np.mean(I_mess)
lambdaGrease_mess= 0.194
lambdaSubstrate_mess=lambdaSubstrate(lambda_substrate_ref,T_ref)
print(P_mean)
print(I_mean)
print(lambdaGrease_mess)
print(lambdaSubstrate_mess)


substituted_eq = equation_dRdT.subs({P: P_mean, I:I_mean,lambda_grease:lambdaGrease_mess,lambda_sub:lambdaSubstrate_mess,dVdf:dVdf_mess, L:L_mess})

sol = solve(substituted_eq, dRdT)
dRdT_mess=float(sol[0])
print(dRdT_mess)

## Messung mit gleichem Grease

