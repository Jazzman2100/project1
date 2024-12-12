#!/usr/bin/env python3

import numpy as np

def simulate_sho(mass, spring_constant, initial_position, initial_velocity, time_step, total_time, method='euler'):
    """
    Simulates a Simple Harmonic Oscillator (SHO) using the specified numerical method.
    
    Parameters:
    mass: float, mass of the object (kg)
    spring_constant: float, spring constant (N/m)
    initial_position: float, initial position of the object (m)
    initial_velocity: float, initial velocity of the object (m/s)
    time_step: float, time step for the simulation (s)
    total_time: float, total time for the simulation (s)
    method: str, numerical method to use ('euler' or 'rk' for Runge-Kutta)
    
    Returns:
    time: array, array of time values
    position: array, array of positions corresponding to the time
    velocity: array, array of velocities corresponding to the time
    """
    
    # Number of time steps
    num_steps = int(total_time / time_step)
    
    # Initialize arrays to store the results
    time = np.zeros(num_steps)
    position = np.zeros(num_steps)
    velocity = np.zeros(num_steps)
    
    # Initial conditions
    time[0] = 0
    position[0] = initial_position
    velocity[0] = initial_velocity
    
    if method == 'euler':
        # Euler method for SHO
        for i in range(1, num_steps):
            time[i] = time[i - 1] + time_step
            acceleration = -spring_constant / mass * position[i - 1]
            velocity[i] = velocity[i - 1] + acceleration * time_step
            position[i] = position[i - 1] + velocity[i - 1] * time_step
            
    elif method == 'rk':
        # Runge-Kutta method for SHO
        for i in range(1, num_steps):
            time[i] = time[i - 1] + time_step
            
            # Runge-Kutta method to calculate position and velocity
            k1v = -spring_constant / mass * position[i - 1]
            k1x = velocity[i - 1]
            
            k2v = -spring_constant / mass * (position[i - 1] + 0.5 * k1x * time_step)
            k2x = velocity[i - 1] + 0.5 * k1v * time_step
            
            k3v = -spring_constant / mass * (position[i - 1] + 0.5 * k2x * time_step)
            k3x = velocity[i - 1] + 0.5 * k2v * time_step
            
            k4v = -spring_constant / mass * (position[i - 1] + k3x * time_step)
            k4x = velocity[i - 1] + k3v * time_step
            
            # Update position and velocity using Runge-Kutta method
            position[i] = position[i - 1] + (time_step / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
            velocity[i] = velocity[i - 1] + (time_step / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
    
    else:
        raise ValueError("Unknown method: {}. Use 'euler' or 'rk'.".format(method))

    return time, position, velocity


import matplotlib.pyplot as plt

def plot_sho_results(time, position, velocity, time_rk, position_rk, velocity_rk, time_scipy, position_scipy, velocity_scipy):
    """
    Plots the results of the SHO simulation for comparison and saves them as PNG files.
    
    Parameters:
    time, position, velocity: Arrays of time, position, and velocity from Euler's method
    time_rk, position_rk, velocity_rk: Arrays from Runge-Kutta method
    time_scipy, position_scipy, velocity_scipy: Arrays from SciPy's solve_ivp method
    """
    # Plot position vs time
    plt.figure(figsize=(10, 6))
    plt.plot(time, position, label="Euler's Method", color='blue')
    plt.plot(time_rk, position_rk, label="Runge-Kutta Method", color='green')
    plt.plot(time_scipy, position_scipy, label="SciPy (solve_ivp)", color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Simple Harmonic Oscillator - Position vs Time')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file (without showing it)
    plt.savefig('sho_position.png')

    # Plot velocity vs time
    plt.figure(figsize=(10, 6))
    plt.plot(time, velocity, label="Euler's Method", color='blue')
    plt.plot(time_rk, velocity_rk, label="Runge-Kutta Method", color='green')
    plt.plot(time_scipy, velocity_scipy, label="SciPy (solve_ivp)", color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Simple Harmonic Oscillator - Velocity vs Time')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file (without showing it)
    plt.savefig('sho_velocity.png')
