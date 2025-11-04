"""
Battery Optimization with Bayesian Optimization
===============================================
"""
import pybamm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import GPy
import GPyOpt
import datetime


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
        positive_thickness = x[0]
        negative_thickness = x[1]

        params_dict = {
            "Positive electrode thickness [m]": positive_thickness,
            "Negative electrode thickness [m]": negative_thickness
        }

        results = self.simulate_battery(params_dict)

        if not results["success"]:
            return 1e6  # Large penalty for failed simulations

        # Return negative energy density (we want to maximize energy density)
        return -results["energy_density"]


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
            {'name': 'positive_thickness', 'type': 'continuous', 'domain': (bounds[0][0], bounds[0][1])},
            {'name': 'negative_thickness', 'type': 'continuous', 'domain': (bounds[1][0], bounds[1][1])}
        ]

        # Wrap the objective function so it matches GPyOpt’s expected input shape
        def f(X):
            # X is a 2D array of shape (n_samples, n_dimensions)
            results = [self.objective_function(x) for x in X]
            return np.array(results).reshape(-1, 1)

        # Create and run the Bayesian optimizer
        bo = GPyOpt.methods.BayesianOptimization(
            f=f,
            domain=domain,
            acquisition_type='MPI',
            maximize=False
        )

        bo.run_optimization(max_iter=20, verbosity=True)

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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Simulate initial design
        initial_results = self.simulate_battery({
            "Positive electrode thickness [m]": initial_params[0],
            "Negative electrode thickness [m]": initial_params[1]
        })

        # Simulate optimized design
        optimized_results = self.simulate_battery({
            "Positive electrode thickness [m]": optimized_params[0],
            "Negative electrode thickness [m]": optimized_params[1]
        })

        # Plot voltage profiles
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

        # Plot design parameters
        labels = ['Positive\nElectrode', 'Negative\nElectrode']
        initial_thickness = [initial_params[0] * 1e6, initial_params[1] * 1e6]
        optimized_thickness = [optimized_params[0] * 1e6, optimized_params[1] * 1e6]

        x = np.arange(len(labels))
        width = 0.35

        ax2.bar(x - width / 2, initial_thickness, width, label='Initial', alpha=0.7)
        ax2.bar(x + width / 2, optimized_thickness, width, label='Optimized', alpha=0.7)

        ax2.set_xlabel('Electrode Type')
        ax2.set_ylabel('Thickness [μm]')
        ax2.set_title('Electrode Thickness Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add method to save image in a folder with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_results/bo_optimization_results_{timestamp}.png"

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()


def main():
    """Main function to run the battery optimization"""
    # Initialize optimizer
    optimizer = BatteryOptimizer()

    # Set initial guess
    initial_guess = [7.0e-5, 8.5e-5]

    # Run optimization
    result = optimizer.optimize_design(initial_guess)

    # Print results
    if result.success:
        print("\n" + "=" * 50)
        print("BO OPTIMIZATION RESULTS")
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