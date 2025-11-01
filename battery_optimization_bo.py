"""
Battery Optimization with Bayesian Optimization (RP17)
Extended class approach - adds BO methods to existing BatteryOptimizer
"""

from battery_optimization import BatteryOptimizer
import numpy as np
import matplotlib.pyplot as plt
import GPyOpt
import time


class BatteryOptimizerBO(BatteryOptimizer):
    """
    Extended BatteryOptimizer class with Bayesian Optimization capabilities
    
    Inherits all original methods from BatteryOptimizer and adds:
    - Bayesian Optimization with GPyOpt
    - Constraint handling for BO
    - BO-specific visualization methods
    """
    
    def __init__(self):
        """Initialize with parent class and add BO-specific attributes"""
        super().__init__()
        self.evaluation_count = 0
        self.evaluation_history = []
    
    # ============================================
    # BAYESIAN OPTIMIZATION METHODS
    # ============================================
    
    def bo_objective(self, x):
        """
        Wrapper for Bayesian Optimization objective
        Compatible with GPyOpt format
        Includes constraint handling via penalty method
        
        Args:
            x: 2D array from GPyOpt [[pos_thick, neg_thick]]
            
        Returns:
            2D array: [[objective_value]]
        """
        self.evaluation_count += 1
        
        # Extract parameters
        positive_thickness = x[0, 0]
        negative_thickness = x[0, 1]
        
        params_dict = {
            "Positive electrode thickness [m]": positive_thickness,
            "Negative electrode thickness [m]": negative_thickness
        }
        
        print(f"\nBO Evaluation {self.evaluation_count}:")
        print(f"  Positive: {positive_thickness*1e6:.2f} μm")
        print(f"  Negative: {negative_thickness*1e6:.2f} μm")
        
        results = self.simulate_battery(params_dict)
        
        if not results["success"]:
            obj_value = 1e6
            print(f"  Simulation FAILED")
        else:
            # Negative energy density (minimize negative = maximize positive)
            obj_value = -results["energy_density"]
            
            # Add penalty for constraint violations
            penalty = 0
            
            # Constraint 1: Voltage must be >= 3.0V
            if results["min_voltage"] < 3.0:
                voltage_violation = 3.0 - results["min_voltage"]
                penalty += 1000 * voltage_violation  # Large penalty
                #print(f"  ⚠ Voltage constraint violated: {results['min_voltage']:.3f} V < 3.0 V")
            
            # Constraint 2: Total thickness must be <= 200 μm
            total_thickness = positive_thickness + negative_thickness
            if total_thickness > 20e-5:  # 200 μm in meters
                thickness_violation = (total_thickness - 20e-5) * 1e6  # Convert to μm
                penalty += 1000 * thickness_violation
                #print(f"  ⚠ Thickness constraint violated: {total_thickness*1e6:.1f} μm > 200 μm")
            
            obj_value += penalty
            
            # if penalty == 0:
            #     print(f"  ✓ Energy Density: {results['energy_density']:.2f} Wh/L")
            #     print(f"  ✓ Min Voltage: {results['min_voltage']:.3f} V")
            #     print(f"  ✓ Total Thickness: {total_thickness*1e6:.1f} μm")
            # else:
            #     print(f"  Energy Density: {results['energy_density']:.2f} Wh/L (with penalty)")
        
        # Store evaluation history
        self.evaluation_history.append({
            'iteration': self.evaluation_count,
            'positive_thickness': positive_thickness,
            'negative_thickness': negative_thickness,
            'objective': obj_value,
            'results': results
        })
        
        return np.array([[obj_value]])
    
    def bo_constraint_voltage(self, x):
        """
        Voltage constraint for BO: min_voltage >= 3.0V
        Returns positive value when constraint is satisfied
        
        Args:
            x: 2D array [[pos_thick, neg_thick]]
            
        Returns:
            2D array: [[constraint_value]]
        """
        positive_thickness = x[0, 0]
        negative_thickness = x[0, 1]
        
        params_dict = {
            "Positive electrode thickness [m]": positive_thickness,
            "Negative electrode thickness [m]": negative_thickness
        }
        
        results = self.simulate_battery(params_dict)
        
        if not results["success"]:
            return np.array([[-1.0]])  # Constraint violated
        
        # Constraint: min_voltage - 3.0 >= 0
        constraint_value = results["min_voltage"] - 3.0
        
        return np.array([[constraint_value]])
    
    def bo_constraint_thickness(self, x):
        """
        Total thickness constraint: total <= 20e-5 m (200 μm)
        Returns positive value when constraint is satisfied
        
        Args:
            x: 2D array [[pos_thick, neg_thick]]
            
        Returns:
            2D array: [[constraint_value]]
        """
        total_thickness = x[0, 0] + x[0, 1]
        
        # Constraint: 20e-5 - total_thickness >= 0
        constraint_value = 20e-5 - total_thickness
        
        return np.array([[constraint_value]])
    
    def optimize_with_bayesian(self, max_iter=10, initial_design_numdata=5, 
                               acquisition_type='EI', use_constraints=True):
        """
        Optimize battery design using Bayesian Optimization
        
        Args:
            max_iter: Maximum number of BO iterations
            initial_design_numdata: Number of initial random samples
            acquisition_type: Acquisition function ('EI', 'LCB', 'MPI')
            use_constraints: Whether to use constraints (set False if having issues)
            
        Returns:
            dict: Optimization results including best parameters and history
        """
        print("="*60)
        print("BAYESIAN OPTIMIZATION")
        print("="*60)
        print(f"Acquisition function: {acquisition_type}")
        print(f"Max iterations: {max_iter}")
        print(f"Initial design points: {initial_design_numdata}")
        print(f"Using constraints: {use_constraints}")
        
        # Reset evaluation counter and history
        self.evaluation_count = 0
        self.evaluation_history = []
        
        # Define domain (search space bounds)
        domain = [
            {'name': 'positive_thickness', 'type': 'continuous', 
             'domain': (5.0e-5, 12.0e-5)},  # 50-120 μm
            {'name': 'negative_thickness', 'type': 'continuous', 
             'domain': (6.0e-5, 12.0e-5)}   # 60-120 μm
        ]
        
        # Define constraints (GPyOpt format - optional)
        constraints = None
        if use_constraints:
            # Note: GPyOpt constraints can be tricky. If they cause issues,
            # constraints are handled via penalties in the objective function
            print("Note: Constraints will be handled via penalty method in objective")
            constraints = None
        
        # Create Bayesian Optimization object
        bo_optimizer = GPyOpt.methods.BayesianOptimization(
            f=self.bo_objective,
            domain=domain,
            constraints=constraints,  # Set to None to avoid GPyOpt constraint issues
            acquisition_type=acquisition_type,
            exact_feval=True,  # Deterministic evaluations
            normalize_Y=True,  # Normalize objective values
            initial_design_numdata=initial_design_numdata,
            maximize=False  # We're minimizing negative energy density
        )
        
        # Run optimization
        start_time = time.time()
        bo_optimizer.run_optimization(max_iter=max_iter)
        elapsed_time = time.time() - start_time
        
        # Extract best solution
        x_best = bo_optimizer.x_opt
        f_best = bo_optimizer.fx_opt
        
        print("\n" + "="*60)
        print("BAYESIAN OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Total evaluations: {self.evaluation_count}")
        print(f"Time elapsed: {elapsed_time:.1f} seconds")
        print(f"Best energy density: {-f_best:.2f} Wh/L")
        print(f"Optimal positive thickness: {x_best[0]*1e6:.2f} μm")
        print(f"Optimal negative thickness: {x_best[1]*1e6:.2f} μm")
        print(f"Total thickness: {(x_best[0] + x_best[1])*1e6:.2f} μm")
        
        return {
            'optimizer': bo_optimizer,
            'x_best': x_best,
            'f_best': f_best,
            'elapsed_time': elapsed_time,
            'evaluation_history': self.evaluation_history,
            'method': 'Bayesian Optimization',
            'acquisition_type': acquisition_type
        }
    
    # ============================================
    # BO-SPECIFIC VISUALIZATION
    # ============================================
    
    def plot_bo_convergence(self, bo_results):
        """
        Plot Bayesian Optimization convergence
        
        Args:
            bo_results: Results dictionary from optimize_with_bayesian()
        """
        history = bo_results['evaluation_history']
        
        iterations = [h['iteration'] for h in history]
        objectives = [-h['objective'] for h in history if h['objective'] < 1e5]
        
        # Calculate best so far
        best_so_far = []
        current_best = -np.inf
        for h in history:
            obj = -h['objective']
            if h['objective'] < 1e5 and obj > current_best:
                current_best = obj
            best_so_far.append(current_best if current_best > -np.inf else np.nan)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Convergence curve
        valid_iterations = [iterations[i] for i in range(len(history)) 
                           if history[i]['objective'] < 1e5]
        
        ax1.plot(valid_iterations, objectives, 'bo', alpha=0.5, 
                label='Evaluated points', markersize=6)
        ax1.plot(iterations, best_so_far, 'r-', linewidth=2.5, 
                label='Best so far')
        
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Energy Density [Wh/L]', fontsize=12)
        ax1.set_title(f'BO Convergence ({bo_results["acquisition_type"]})', 
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Parameter evolution
        pos_thick = [h['positive_thickness']*1e6 for h in history]
        neg_thick = [h['negative_thickness']*1e6 for h in history]
        
        ax2.plot(iterations, pos_thick, 'b.-', label='Positive electrode', 
                linewidth=1.5, markersize=5)
        ax2.plot(iterations, neg_thick, 'r.-', label='Negative electrode', 
                linewidth=1.5, markersize=5)
        ax2.axhline(y=(bo_results['x_best'][0]*1e6), color='b', 
                   linestyle='--', alpha=0.5)
        ax2.axhline(y=(bo_results['x_best'][1]*1e6), color='r', 
                   linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Thickness [μm]', fontsize=12)
        ax2.set_title('Parameter Evolution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)


def main():
    """
    Simple test of Bayesian Optimization on battery design
    """
    print("\n" + "="*60)
    print("TESTING BAYESIAN OPTIMIZATION")
    print("="*60 + "\n")
    
    # Initialize optimizer
    optimizer = BatteryOptimizerBO()
    
    # Run Bayesian Optimization
    bo_results = optimizer.optimize_with_bayesian(
        max_iter=5,
        initial_design_numdata=5,
        acquisition_type='EI'
    )

    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION TEST COMPLETE")
    print("="*60)
    
    # Plot convergence
    optimizer.plot_bo_convergence(bo_results)



if __name__ == "__main__":
    main()