#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="physics_simulations",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
    entry_points={
        'console_scripts': [
            'sho_run = sho.sho_run:main',  # this points to the main function in sho_run.py
            'coulomb_force_run = coulomb.coulomb_force_run:main',  # this points to the main function in coulomb_force_run.py
        ],
    },
)
