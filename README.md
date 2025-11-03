# 🔋 Battery Optimization using PyBaMM & Bayesian Optimization

This project was developed by Amanda Tartarotti Cardozo da Silva and Miriam Romaniuc for the Optimization Techniques course in the 2025/2026 academic year of the Master in Computacional Engineering and Smart Systems at University of Basque Country (EHU/UPV).

This work explores **battery parameter optimization** using the **PyBaMM** framework and **Bayesian Optimization (BO)**.  
The goal is to maximize battery **energy density** while satisfying voltage and manufacturing constraints.

---

## Overview

- The project starts from a base implementation that uses **Scipy's SLSQP optimizer**.  
- A second implementation introduces **Bayesian Optimization (BO)** using the **GPyOpt** library.
- Both methods use **PyBaMM’s DFN model** (`pybamm.lithium_ion.DFN`) for simulation.

## Useful references:
- PyBaMM Cheat Sheet: [https://lazyjobseeker.github.io/en/posts/pybamm-cheat-sheet-1/](https://lazyjobseeker.github.io/en/posts/pybamm-cheat-sheet-1/)
- PyBOP (Bayesian Optimization for Batteries): [https://github.com/pybop-team/PyBOP](https://github.com/pybop-team/PyBOP)
- GPyOpt Documentation: [https://sheffieldml.github.io/GPyOpt/](https://sheffieldml.github.io/GPyOpt/)

---

## Installation

You can install dependencies either via **Conda** or **pip**.

### Option 1: Using Conda

```bash
# Create and activate a new environment
conda create -n battery-opt python=3.10
conda activate battery-opt

# Install core dependencies
conda install -c conda-forge numpy scipy matplotlib pandas paramz
pip install pybamm GPy GPyOpt

```

**********************************************************************************************************************
