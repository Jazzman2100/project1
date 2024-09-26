# %% Cell 1
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
# %% Cell 2
# Simulate a SHO with a damped force
spring_mass = 0.05 #kg
spring_const = 20 #N/m
time_array = [] #second
position_array = [] #m
velocity_array = [] #m/s
acceleration_array = [] #m/s**2
t_end = 15 #second
dt = 0.5 #second
time_int = int(t_end/dt)
# print(time_int)
def damped_SHO(m, k, b, t, x, v, a, t_end, dt):
    while t <= t_end:
        time_array.append(t)
        position_array.append(x)
        velocity_array.append(v)
        acceleration_array.append(a)
        force_spring = -k*x
        force_damping = -b*v
        a = (force_spring+force_damping)/m
        v = v + a*dt
        x = x + v*dt
        t = t + dt
    return time_array, position_array, velocity_array, acceleration_array

print(damped_SHO(0.05, 200, 0.1, 0, 0, 5, 0, 10, 0.1))

plt.plot(time_array, position_array)
plt.title("Position vs Time")
plt.xlabel("time(s)")
plt.ylabel("position(m)")
