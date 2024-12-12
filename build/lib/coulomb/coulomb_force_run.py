#!/usr/bin/env python3

import numpy as np
from coulomb.coulomb_solver import calculate_coulomb_force, plot_coulomb_results

def main():
    # Prompt user for inputs
    q = float(input("Enter the charge of the point charge (in Coulombs): "))
    Q = float(input("Enter the total charge on the bar (in Coulombs): "))
    L = float(input("Enter the length of the bar (in meters): "))
    
    # Prompt for the position of the point charge
    x0 = float(input("Enter the position of the point charge along the x-axis (in meters): "))
    
    # Prompt for the position of the charged bar (its center position)
    x_bar = float(input("Enter the center position of the charged bar along the x-axis (in meters): "))
    
    # Prompt for the integration method
    method = input("Choose the integration method ('riemann', 'trapezoidal', 'simpson'): ")
    
    # Calculate the Coulomb force using the chosen method
    force, force_scipy = calculate_coulomb_force(q, Q, L, x0, x_bar, method)
    
    print(f"Calculated Coulomb Force using {method}: {force} N")
    print(f"Calculated Coulomb Force using scipy: {force_scipy} N")
    
    # Generate plot of Coulomb force as a function of distance
    filename = input("Enter the filename to save the plot (e.g., coulomb_force_plot.png): ")
    distances = np.linspace(x_bar - L / 2, x_bar + L / 2, 100)  # Distances across the charged bar
    forces = [calculate_coulomb_force(q, Q, L, x0, x, method)[0] for x in distances]
    plot_coulomb_results(forces, distances, filename)

if __name__ == "__main__":
    main()
