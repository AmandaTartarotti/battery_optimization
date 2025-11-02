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
                penalty += 500 * voltage_violation  # Reduced from 1000
                print(f"  ⚠ Voltage constraint violated: {results['min_voltage']:.3f} V < 3.0 V")
            
            # Constraint 2: Total thickness must be <= 200 μm
            total_thickness = positive_thickness + negative_thickness
            if total_thickness > 20e-5:  # 200 μm in meters
                thickness_violation = (total_thickness - 20e-5) * 1e6  # Convert to μm
                penalty += 500 * thickness_violation  # Reduced from 1000
                print(f"  ⚠ Thickness constraint violated: {total_thickness*1e6:.1f} μm > 200 μm")
            
            obj_value += penalty
            
            if penalty == 0:
                print(f"  ✓ Energy Density: {results['energy_density']:.2f} Wh/L")
                print(f"  ✓ Min Voltage: {results['min_voltage']:.3f} V")
                print(f"  ✓ Total Thickness: {total_thickness*1e6:.1f} μm")
            else:
                print(f"  Energy Density: {results['energy_density']:.2f} Wh/L (with penalty)")
        
        # Store evaluation history
        self.evaluation_history.append({
            'iteration': self.evaluation_count,
            'positive_thickness': positive_thickness,
            'negative_thickness': negative_thickness,
            'objective': obj_value,
            'results': results
        })
        
        return np.array([[obj_value]])
    
    # Note: Constraints are handled via penalty method in bo_objective()
    # No separate constraint functions needed for GPyOpt
    
    def optimize_with_bayesian(self, max_iter=25, initial_design_numdata=5, 
                               acquisition_type='EI', use_constraints=True,
                               initial_guess=None):
        """
        Optimize battery design using Bayesian Optimization
        
        Args:
            max_iter: Maximum number of BO iterations
            initial_design_numdata: Number of initial random samples
            acquisition_type: Acquisition function ('EI', 'LCB', 'MPI')
            use_constraints: Whether to use constraints (set False if having issues)
            initial_guess: Initial design for comparison [pos_thick, neg_thick] in meters
            
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
        
        # Set default initial guess if not provided
        if initial_guess is None:
            initial_guess = [7.0e-5, 8.5e-5]  # 70μm positive, 85μm negative
        
        # Evaluate initial design
        print("\n" + "-"*60)
        print("INITIAL DESIGN EVALUATION")
        print("-"*60)
        initial_results = self.simulate_battery({
            "Positive electrode thickness [m]": initial_guess[0],
            "Negative electrode thickness [m]": initial_guess[1]
        })
        
        if initial_results["success"]:
            print(f"Initial positive thickness: {initial_guess[0]*1e6:.2f} μm")
            print(f"Initial negative thickness: {initial_guess[1]*1e6:.2f} μm")
            print(f"Initial energy density: {initial_results['energy_density']:.2f} Wh/L")
            print(f"Initial min voltage: {initial_results['min_voltage']:.3f} V")
            print(f"Initial total thickness: {(initial_guess[0] + initial_guess[1])*1e6:.2f} μm")
        else:
            print("⚠ Initial design simulation failed!")
        
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
        #Note: Constraints will be handled via penalty method in objective")
        
        # Create Bayesian Optimization object
        print("\n" + "-"*60)
        print("STARTING OPTIMIZATION")
        print("-"*60)
        
        # Create initial design that includes the initial guess
        from GPyOpt.experiment_design import initial_design
        
        # Generate random initial points
        random_design = initial_design('random', 
                                      GPyOpt.core.task.space.Design_space(domain), 
                                      initial_design_numdata - 1)
        
        # Add initial guess as first point
        X_init = np.vstack([np.array([initial_guess]), random_design])
        
        bo_optimizer = GPyOpt.methods.BayesianOptimization(
            f=self.bo_objective,
            domain=domain,
            constraints=constraints,
            acquisition_type=acquisition_type,
            exact_feval=True,  # Deterministic evaluations
            normalize_Y=True,  # Normalize objective values
            X=X_init,  # Use our custom initial design
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
        print(f"\nOptimal design:")
        print(f"  Positive thickness: {x_best[0]*1e6:.2f} μm")
        print(f"  Negative thickness: {x_best[1]*1e6:.2f} μm")
        print(f"  Total thickness: {(x_best[0] + x_best[1])*1e6:.2f} μm")
        print(f"  Energy density: {-f_best:.2f} Wh/L")
        
        # Comparison with initial design
        if initial_results["success"]:
            print("\n" + "="*60)
            print("IMPROVEMENT OVER INITIAL DESIGN")
            print("="*60)
            
            # Calculate improvements
            energy_improvement = -f_best - initial_results['energy_density']
            energy_improvement_pct = (energy_improvement / initial_results['energy_density']) * 100
            
            pos_change = (x_best[0] - initial_guess[0]) * 1e6
            neg_change = (x_best[1] - initial_guess[1]) * 1e6
            
            print(f"\nEnergy Density:")
            print(f"  Initial:  {initial_results['energy_density']:.2f} Wh/L")
            print(f"  Optimal:  {-f_best:.2f} Wh/L")
            print(f"  Change:   {energy_improvement:+.2f} Wh/L ({energy_improvement_pct:+.2f}%)")
            
            print(f"\nPositive Electrode:")
            print(f"  Initial:  {initial_guess[0]*1e6:.2f} μm")
            print(f"  Optimal:  {x_best[0]*1e6:.2f} μm")
            print(f"  Change:   {pos_change:+.2f} μm")
            
            print(f"\nNegative Electrode:")
            print(f"  Initial:  {initial_guess[1]*1e6:.2f} μm")
            print(f"  Optimal:  {x_best[1]*1e6:.2f} μm")
            print(f"  Change:   {neg_change:+.2f} μm")
            
            if energy_improvement_pct > 0:
                print(f"\n✓ Bayesian Optimization found a better design!")
            elif energy_improvement_pct > -1:
                print(f"\n→ Similar performance to initial design")
            else:
                print(f"\n⚠ Note: Optimal design has lower energy density")
        
        return {
            'optimizer': bo_optimizer,
            'x_best': x_best,
            'f_best': f_best,
            'elapsed_time': elapsed_time,
            'evaluation_history': self.evaluation_history,
            'method': 'Bayesian Optimization',
            'acquisition_type': acquisition_type,
            'initial_guess': initial_guess,
            'initial_results': initial_results
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
        plt.show()


def main():
    """
    Simple test of Bayesian Optimization on battery design
    """
    print("\n" + "="*60)
    print("TESTING BAYESIAN OPTIMIZATION")
    print("="*60 + "\n")
    
    # Initialize optimizer
    optimizer = BatteryOptimizerBO()
    
    # Define initial design
    initial_guess = [7.0e-5, 8.5e-5]  # 70μm positive, 85μm negative
    
    # Run Bayesian Optimization
    bo_results = optimizer.optimize_with_bayesian(
        max_iter=30,  # Increased from 20
        initial_design_numdata=8,  # Increased from 5
        acquisition_type='EI',
        initial_guess=initial_guess
    )
    
    # Plot convergence
    optimizer.plot_bo_convergence(bo_results)
    
    # Plot initial vs optimal design
    optimizer.plot_results(initial_guess, bo_results['x_best'])
    
    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()