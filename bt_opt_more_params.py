import pybamm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import GPy
import GPyOpt


class BatteryOptimizer:
    def __init__(self):
        """Initialize the battery optimizer with a DFN model"""
        self.model = pybamm.lithium_ion.DFN()

    def simulate_battery(self, params_dict):
        """
        Simulate battery performance with given parameters

        Args:
            params_dict: Dictionary of parameter values to optimize

        Returns:
            dict: Simulation results including energy density and voltage
        """
        # Get default parameters
        params = pybamm.ParameterValues("Chen2020")

        # Update with optimized parameters
        for key, value in params_dict.items():
            params.update({key: value})

        # Create simulation
        sim = pybamm.Simulation(self.model, parameter_values=params)

        try:
            # Solve for 1C discharge

            # solution = sim.solve([0, 3600])  # 1 hour simulation commented this
            experiment = pybamm.Experiment(["Discharge at 1C until 3.0 V"])
            sim = pybamm.Simulation(self.model, parameter_values=params, experiment=experiment)
            solution = sim.solve()

            # Extract results
            time = solution["Time [s]"].data
            voltage = solution["Terminal voltage [V]"].data
            current = solution["Current [A]"].data

            # Calculate energy delivered
            energy = np.trapz(voltage * current, time)

            # Get design parameters for normalization
            positive_electrode_thickness = params_dict.get(
                "Positive electrode thickness [m]",
                7.0e-5
            )
            negative_electrode_thickness = params_dict.get(
                "Negative electrode thickness [m]",
                8.5e-5
            )

            # Simple energy density calculation (Wh/L)
            total_thickness = positive_electrode_thickness + negative_electrode_thickness
            energy_density = (energy / 3600) / (total_thickness * 1000)  # Wh/L

            # Check for minimum voltage constraint
            min_voltage = np.min(voltage)

            return {
                "energy_density": energy_density,
                "min_voltage": min_voltage,
                "voltage_profile": voltage,
                "time": time,
                "success": True
            }

        except Exception as e:
            print(f"Simulation failed: {e}")
            return {
                "energy_density": 0,
                "min_voltage": 0,
                "voltage_profile": None,
                "time": None,
                "success": False
            }

    def objective_function(self, x):
        """
        Objective function for optimization

        Args:
            x: Array of parameters [positive_thickness, negative_thickness]

        Returns:
            float: Negative energy density (to maximize)
        """
        # inside objective_function(x) with 7 dims:
        pos_thk, neg_thk, sep_thk, pos_eps, neg_eps, pos_rad, neg_rad = x

        params_dict = {
            "Positive electrode thickness [m]": pos_thk,
            "Negative electrode thickness [m]": neg_thk,
            "Separator thickness [m]": sep_thk,
            "Positive electrode porosity": pos_eps,
            "Negative electrode porosity": neg_eps,
            "Positive particle radius [m]": pos_rad,
            "Negative particle radius [m]": neg_rad,
        }

        penalty = 0.0
        total_thk = pos_thk + sep_thk + neg_thk
        if total_thk > 2.5e-4:
            penalty += 1e5 * (total_thk - 2.5e-4)

        results = self.simulate_battery(params_dict)
        if not results["success"] or results["min_voltage"] < 3.0:
            return 1e6  # infeasible / failed

        return -(results["energy_density"]) + penalty

    """ no need for these in bayesian  opt
    def constraints(self):

        # Voltage should not drop below 3.0V during discharge
        def voltage_constraint(x):
            params_dict = {
                "Positive electrode thickness [m]": x[0],
                "Negative electrode thickness [m]": x[1]
            }
            results = self.simulate_battery(params_dict)
            return results["min_voltage"] - 3.0  # min_voltage >= 3.0V

        # Total thickness constraint (manufacturing limits)
        def thickness_constraint(x):
            return 20e-5 - (x[0] + x[1])  # total_thickness <= 20μm

        return [
            {'type': 'ineq', 'fun': voltage_constraint},
            {'type': 'ineq', 'fun': thickness_constraint}
        ]

        """

    def optimize_design(self, initial_guess=None):
        """
        Optimize battery electrode thicknesses

        Args:
            initial_guess: Initial guess for [positive_thickness, negative_thickness]

        Returns:
            dict: Optimization results
        """
        if initial_guess is None:
            initial_guess = [7.0e-5, 8.5e-5]  # Default values in meters

        # Bounds for parameters (in meters)
        bounds = [
            (5.0e-5, 12.0e-5),  # Positive electrode thickness
            (6.0e-5, 12.0e-5)  # Negative electrode thickness
        ]

        print("Starting optimization...")
        print(f"Initial guess: Positive={initial_guess[0] * 1e6:.1f}μm, "
              f"Negative={initial_guess[1] * 1e6:.1f}μm")

        # Define the optimization domain in GPyOpt format
        domain = [
            {'name': 'pos_thk', 'type': 'continuous', 'domain': (5e-5, 12e-5)},
            {'name': 'neg_thk', 'type': 'continuous', 'domain': (6e-5, 12e-5)},
            {'name': 'sep_thk', 'type': 'continuous', 'domain': (1e-5, 3e-5)},
            {'name': 'pos_eps', 'type': 'continuous', 'domain': (0.25, 0.40)},
            {'name': 'neg_eps', 'type': 'continuous', 'domain': (0.25, 0.40)},
            {'name': 'pos_rad', 'type': 'continuous', 'domain': (1e-6, 8e-6)},
            {'name': 'neg_rad', 'type': 'continuous', 'domain': (1e-6, 1e-5)},
        ]

        # Wrap the objective function so it matches GPyOpt’s expected input shape
        def f(X):
            # X is a 2D array of shape (n_samples, n_dimensions)
            results = [self.objective_function(x) for x in X]
            return np.array(results).reshape(-1, 1)

        # Create and run the Bayesian optimizer
        kernel = GPy.kern.Matern52(input_dim=len(domain), ARD=True)
        bo = GPyOpt.methods.BayesianOptimization(
            f=f, domain=domain, kernel=kernel,
            acquisition_type='EI', acquisition_jitter=0.01
        )

        bo.run_optimization(max_iter=20, verbosity=True)

        ls = bo.model.model.kern.lengthscale.values
        rel_importance = (1 / ls ** 2) / np.sum(1 / ls ** 2)
        print("Relative importance per var (pos_thk .. neg_rad):", rel_importance)

        # Mimic SciPy’s `result` object so the rest of the code still works
        class BOResult:
            pass

        result = BOResult()
        result.x = bo.x_opt
        result.fun = bo.fx_opt
        result.success = True
        result.message = "Bayesian optimization completed"
        result.nit = len(bo.Y)

        return result

    def plot_results(self, initial_params, optimized_params):
        """Plot comparison between initial and optimized designs"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Voltage comparison ---
        # Simulate initial design (only first 2 matter for visualization)
        initial_results = self.simulate_battery({
            "Positive electrode thickness [m]": initial_params[0],
            "Negative electrode thickness [m]": initial_params[1]
        })

        # Simulate optimized design
        optimized_results = self.simulate_battery({
            "Positive electrode thickness [m]": optimized_params[0],
            "Negative electrode thickness [m]": optimized_params[1]
        })

        if initial_results["success"] and optimized_results["success"]:
            ax1.plot(initial_results["time"] / 60, initial_results["voltage_profile"],
                     'b-', label='Initial Design', linewidth=2)
            ax1.plot(optimized_results["time"] / 60, optimized_results["voltage_profile"],
                     'r--', label='Optimized Design', linewidth=2)
            ax1.set_xlabel('Time [min]')
            ax1.set_ylabel('Voltage [V]')
            ax1.set_title('Discharge Voltage Profiles')
            ax1.legend()
            ax1.grid(True)

        # --- Multi-parameter comparison ---
        labels = [
            'Pos. thickness [μm]',
            'Neg. thickness [μm]',
            'Separator [μm]',
            'Pos. porosity',
            'Neg. porosity',
            'Pos. particle radius [μm]',
            'Neg. particle radius [μm]'
        ]

        # Convert to appropriate scales
        init_scaled = [
            initial_params[0] * 1e6,  # μm
            initial_params[1] * 1e6,
            20,                       # Assume mid separator for baseline (dummy)
            0.3,                      # Assume mid porosity
            0.3,
            5.0,                      # μm
            6.0                       # μm
        ]

        opt_scaled = [
            optimized_params[0] * 1e6,
            optimized_params[1] * 1e6,
            optimized_params[2] * 1e6,
            optimized_params[3],
            optimized_params[4],
            optimized_params[5] * 1e6,
            optimized_params[6] * 1e6
        ]

        x = np.arange(len(labels))
        width = 0.35

        ax2.bar(x - width/2, init_scaled, width, label='Initial', alpha=0.7)
        ax2.bar(x + width/2, opt_scaled, width, label='Optimized', alpha=0.7)

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Scaled parameter value')
        ax2.set_title('Comparison of All Optimized Parameters')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Possible extension to higher number of parameters
def enhanced_optimization():
    """More realistic optimization with additional parameters"""

    # Additional design variables
    design_variables = {
        'Positive electrode thickness [m]': (5e-5, 15e-5),
        'Negative electrode thickness [m]': (5e-5, 15e-5),
        'Positive electrode porosity': (0.2, 0.4),
        'Negative electrode porosity': (0.2, 0.4),
        'Positive particle radius [m]': (1e-6, 10e-6),
    }

    # Multiple objectives
    objectives = {
        'energy_density': 'maximize',
        'capacity_fade_100_cycles': 'minimize',
        'heat_generation': 'minimize'
    }

    # Multiple constraints
    constraints = {
        'min_voltage_1C': '>= 3.0',
        'max_temperature': '<= 60.0',
        'power_density_5C': '>= 1000'
    }


def main():
    """Main function to run the battery optimization"""
    # Initialize optimizer
    optimizer = BatteryOptimizer()

    # Set initial guess
    initial_guess = [7.0e-5, 8.5e-5]  # 70μm positive, 85μm negative

    # Run optimization
    result = optimizer.optimize_design(initial_guess)

    # Print results
    if result.success:
        print("\n" + "=" * 50)
        print("OPTIMIZATION RESULTS")
        print("=" * 50)
        print(f"Optimization successful: {result.message}")
        print(f"Number of iterations: {result.nit}")
        print(f"Final objective value: {-result.fun:.2f} Wh/L")
        print(f"Optimal positive electrode thickness: {result.x[0] * 1e6:.2f} μm")
        print(f"Optimal negative electrode thickness: {result.x[1] * 1e6:.2f} μm")
        print(f"Total thickness: {(result.x[0] + result.x[1]) * 1e6:.2f} μm")

        # Compare with initial design
        initial_results = optimizer.simulate_battery({
            "Positive electrode thickness [m]": initial_guess[0],
            "Negative electrode thickness [m]": initial_guess[1]
        })

        if initial_results["success"]:
            print(f"Initial energy density: {initial_results['energy_density']:.2f} Wh/L")
            print(f"Optimized energy density: {-result.fun:.2f} Wh/L")
            improvement = ((-result.fun - initial_results["energy_density"]) /
                           initial_results["energy_density"] * 100)
            print(f"Energy density improvement: {improvement:.1f}%")

        # Plot results
        optimizer.plot_results(initial_guess, result.x)

    else:
        print(f"Optimization failed: {result.message}")


if __name__ == "__main__":
    main()