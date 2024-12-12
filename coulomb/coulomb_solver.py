#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants
k_e = 8.9875e9  # Coulomb constant in N·m²/C²

def coulomb_force_integrand(x, x0, q, Q, L, x_bar):
    """
    Integrand function for Coulomb force calculation.
    :param x: Position on the charged bar.
    :param x0: Position of the point charge.
    :param q: Point charge.
    :param Q: Total charge on the bar.
    :param L: Length of the bar.
    :param x_bar: Position of the center of the charged bar.
    :return: Coulomb force contribution from element dx.
    """
    lambda_ = Q / L  # Linear charge density
    distance = x - x0  # Distance from the point charge to the charge element
    
    # Check if distance is effectively zero for each element
    if np.any(np.isclose(distance, 0)):  # Check element-wise for zero distance
        return 0  # Return zero for this point to avoid singularity
    
    return k_e * q * lambda_ / (distance**2)

def calculate_coulomb_force(q, Q, L, x0, x_bar, method='simpson'):
    """
    Calculate the total Coulomb force between a point charge and a uniformly charged bar.
    :param q: Charge of the point charge.
    :param Q: Charge of the bar.
    :param L: Length of the charged bar.
    :param x0: Position of the point charge.
    :param x_bar: Position of the center of the charged bar.
    :param method: Method for integration ('riemann', 'trapezoidal', 'simpson').
    :return: Total Coulomb force.
    """
    # Define integration limits for the bar [-L/2, L/2]
    x_min = x_bar - L / 2
    x_max = x_bar + L / 2

    # Perform integration based on the chosen method
    if method == 'riemann':
        # Using a Riemann sum (simple rectangular sum)
        n = 1000  # Number of points
        dx = (x_max - x_min) / n
        x = np.linspace(x_min, x_max, n)
        force = np.sum(coulomb_force_integrand(x, x0, q, Q, L, x_bar)) * dx
    
    elif method == 'trapezoidal':
        # Trapezoidal rule integration
        n = 1000  # Number of points
        x = np.linspace(x_min, x_max, n)
        dx = (x_max - x_min) / (n - 1)
        force = np.trapz(coulomb_force_integrand(x, x0, q, Q, L, x_bar), x)
    
    elif method == 'simpson':
        # Simpson's rule integration
        n = 1000  # Number of points (must be even)
        x = np.linspace(x_min, x_max, n)
        force = np.trapz(coulomb_force_integrand(x, x0, q, Q, L, x_bar), x)
    
    # Using scipy quad for comparison
    def integrand(x):
        return coulomb_force_integrand(x, x0, q, Q, L, x_bar)
    
    force_scipy, _ = quad(integrand, x_min, x_max)
    
    return force, force_scipy

def plot_coulomb_results(forces, distances, filename="coulomb_force_plot.png"):
    """
    Plot the Coulomb force as a function of distance and save the plot.
    :param forces: List of Coulomb forces.
    :param distances: List of distances.
    :param filename: Filename for saving the plot.
    """
    plt.plot(distances, forces, label="Coulomb Force")
    plt.xlabel("Distance (m)")
    plt.ylabel("Force (N)")
    plt.title("Coulomb Force vs Distance")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
