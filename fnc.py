# %% Cell 1
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

# %% Cell 3
# Solve 1st order ODE
def dvdt(t, v):
    return 2*v**2
v0 = 3 #initial condition

t = np.linspace(0, 1, 10)
sol_m1 = odeint(dvdt, v0=v0, t=t, tfirst=True)
sol_m2 = solve_ivp(dvdt, t_span=(0, max(t)), x0=[v0], t_eval=t)
