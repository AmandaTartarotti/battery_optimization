# 🔋 Battery Optimization using PyBaMM & Bayesian Optimization

This project explores **battery parameter optimization** using the **PyBaMM** framework and **Bayesian Optimization (BO)**.  
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

*****************************************************************************************
-> I tested the initial script, it is working but for some reason the plot is not being displayed fot me. I also found this website that provides some explanation about our topic: https://lazyjobseeker.github.io/en/posts/pybamm-cheat-sheet-1/#pybamm
-> About the BO library, I found this one while searching for options: https://github.com/pybop-team/PyBOP, it is quite new but very foccus on our problem. There is another library that is more 'safe' to work GPyOpt, it is more general-purposed so I did initial implementation with that
