#!/usr/bin/env python3
# sho_run.py
from sho.sho_solver import simulate_sho, plot_sho_results
from scipy.integrate import solve_ivp
import numpy as np

def main():
    # Parameters for the simulation
    mass = float(input("Enter the mass of the object (kg): "))
    spring_constant = float(input("Enter the spring constant (N/m): "))
    initial_position = float(input("Enter the initial position (m): "))
    initial_velocity = float(input("Enter the initial velocity (m/s): "))
    time_step = float(input("Enter the time step (s): "))
    total_time = float(input("Enter the total time (s): "))

    # Simulate the SHO using Euler's method
    time, position, velocity = simulate_sho(mass, spring_constant, initial_position, initial_velocity, time_step, total_time)

    # Simulate the SHO using Runge-Kutta method
    time_rk, position_rk, velocity_rk = simulate_sho(mass, spring_constant, initial_position, initial_velocity, time_step, total_time, method='rk')

    # Simulate the SHO using SciPy's solve_ivp method
    def sho_system(t, y):
        position, velocity = y
        dydt = [velocity, -spring_constant / mass * position]
        return dydt

    sol = solve_ivp(sho_system, [0, total_time], [initial_position, initial_velocity], t_eval=np.arange(0, total_time, time_step))

    time_scipy = sol.t
    position_scipy = sol.y[0]
    velocity_scipy = sol.y[1]

    # Now call the plot function with all the results
    plot_sho_results(time, position, velocity, time_rk, position_rk, velocity_rk, time_scipy, position_scipy, velocity_scipy)

if __name__ == "__main__":
    main()
