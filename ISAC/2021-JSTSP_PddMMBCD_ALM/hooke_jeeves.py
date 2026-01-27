#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 21:15:05 2026

@author: jack
"""

import numpy as np

def hooke_jeeves_solver(f, x0, bounds=None, d_init=1.0, d_th=1e-6,
                        N_max=1000, use_pattern_move=True):
    """
    Hooke-Jeeves Pattern Search Method for constrained optimization

    Parameters:
    -----------
    f : callable
        Objective function f(x)
    x0 : array_like
        Initial guess (must satisfy constraints)
    bounds : list of tuples or None
        Bounds for variables [(min1, max1), (min2, max2), ...]
        Use None for no bound in that direction, e.g., (0, None) for non-negative
    d_init : float
        Initial step size
    d_th : float
        Minimum step size threshold
    N_max : int
        Maximum number of iterations
    use_pattern_move : bool
        Whether to use pattern moves

    Returns:
    --------
    x_opt : ndarray
        Optimal solution
    f_opt : float
        Optimal objective value
    history : dict
        Optimization history
    """
    n = len(x0)
    x_base = np.array(x0, dtype=float)
    x_current = x_base.copy()
    d = d_init
    iter_count = 0

    # Create unit directions
    directions = [np.eye(n)[i] for i in range(n)]

    # Store history
    history = {
        'x': [x_base.copy()],
        'f': [f(x_base)],
        'd': [d],
        'iter': [0]
    }

    def apply_bounds(x):
        """Apply bounds to variable x, handling None values properly"""
        if bounds is None:
            return x

        x_bounded = x.copy()
        for i in range(n):
            if bounds[i][0] is not None and x_bounded[i] < bounds[i][0]:
                x_bounded[i] = bounds[i][0]
            if bounds[i][1] is not None and x_bounded[i] > bounds[i][1]:
                x_bounded[i] = bounds[i][1]
        return x_bounded

    # Make sure initial point satisfies bounds
    x_base = apply_bounds(x_base)
    x_current = x_base.copy()

    while iter_count < N_max and d >= d_th:
        improved = False

        # Exploration moves
        for i in range(n):
            # Positive direction
            x_test = x_current + d * directions[i]
            x_test = apply_bounds(x_test)

            if f(x_test) < f(x_current):
                x_current = x_test.copy()
                improved = True
                break  # Restart exploration

            # Negative direction
            x_test = x_current - d * directions[i]
            x_test = apply_bounds(x_test)

            if f(x_test) < f(x_current):
                x_current = x_test.copy()
                improved = True
                break  # Restart exploration

        # Check if we found improvement
        f_current = f(x_current)
        f_base = f(x_base)

        if f_current >= f_base and not improved:
            # No improvement found - reduce step size
            if d > d_th:
                d = d / 2
                x_current = x_base.copy()
            else:
                break
        else:
            if f_current < f_base:
                # Pattern move (if enabled)
                if use_pattern_move and iter_count > 0:
                    pattern_dir = x_current - x_base
                    x_pattern = x_current + d * pattern_dir
                    x_pattern = apply_bounds(x_pattern)

                    if f(x_pattern) < f(x_current):
                        x_current = x_pattern.copy()

                # Update base point
                x_base = x_current.copy()

        iter_count += 1

        # Store history
        history['x'].append(x_base.copy())
        history['f'].append(f(x_base))
        history['d'].append(d)
        history['iter'].append(iter_count)

    return x_base, f(x_base), history

# Example usage with a sample problem
def example_objective(x):
    """Example objective function (replace with your actual function)"""
    # This should match your specific optimization problem
    # For example: f(x) = x^T A x + b^T x + c
    A = np.array([[2, 1], [1, 3]])
    b = np.array([1, -2])
    return 0.5 * x.T @ A @ x + b.T @ x

def constrained_example(x):
    """Example with constraints handled via penalty method"""
    # Main objective
    f_val = example_objective(x)

    # Add penalty for constraints (example)
    penalty = 0
    if x[0] < 0:
        penalty += 100 * (-x[0])**2
    if x[1] < 0:
        penalty += 100 * (-x[1])**2

    return f_val + penalty

# Run optimization
if __name__ == "__main__":
    # Initial guess
    x0 = np.array([1.0, 1.0])

    # Define bounds (e.g., non-negativity constraints)
    # Using None for upper bound means no upper limit
    bounds = [(0, None), (0, None)]

    # Run Hooke-Jeeves
    x_opt, f_opt, history = hooke_jeeves_solver(
        f=constrained_example,
        x0=x0,
        bounds=bounds,
        d_init=0.5,
        d_th=1e-4,
        N_max=500
    )

    print(f"Optimal solution: {x_opt}")
    print(f"Optimal value: {f_opt}")
    print(f"Iterations: {history['iter'][-1]}")
    print(f"Function evaluations: {len(history['f'])}")

    # Visualization (optional)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history['iter'], history['f'])
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Convergence History')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        x_hist = np.array(history['x'])
        plt.plot(x_hist[:, 0], x_hist[:, 1], 'bo-', markersize=3)
        plt.scatter(x_opt[0], x_opt[1], c='red', s=100, marker='*')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Solution Path')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")
