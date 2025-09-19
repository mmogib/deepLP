#!/usr/bin/env python3
"""
Simple test example for deepLP
"""

from deeplp import train, createProblem

def main():
    print("Running deepLP simple example...")
    
    # Define a simple LP problem
    # minimize: x1 + 2*x2
    # subject to: 3*x1 - 5*x2 <= 15
    #            3*x1 - x2 <= 21
    #            3*x1 + x2 <= 27
    c = [1.0, 2.0]  # Objective coefficients
    A = [[3, -5], [3, -1], [3, 1]]  # Constraint matrix
    b = [15, 21, 27]  # Right-hand side
    tspan = (0.0, 10.0)  # Time span
    
    # Create the problem
    problem = createProblem(
        c, A, b, tspan,
        name="Simple Test Problem"
    )
    
    # Train the model (using small numbers for quick test)
    print("Training model...")
    solutions = train(
        batches=1,
        batch_size=16,
        epochs=100,  # Small number for quick test
        problem=problem,
        cases=[1],  # Time-only case
        do_plot=False,  # Set to True if you want plots
        model_type="pinn"
    )
    
    print(f"Solution found: {solutions[0].solution}")
    print(f"Final loss: {solutions[0].loss_list[-1]:.6f}")
    
    return solutions

if __name__ == "__main__":
    main()
