# How to Run deepLP Examples

This guide shows different ways to run deepLP examples, especially when using Poetry 2.0+ package management.

## Prerequisites

Make sure you're in the project directory and have Poetry installed:

```bash
cd C:\Users\mmogi\Dropbox\MyPythonPackages\deepLP
poetry install
```

## Method 1: Using `poetry run` (Recommended)

This is the most reliable method that works across all terminals including CMDER:

### Basic Examples

```bash
# Run built-in example 1 with PINN model (inequality constraints)
poetry run deeplp --example 1 --case 1 --iterations 100 --batch_size 16 --do_plot

# Run built-in example 2 with RNN model (equality constraints)
poetry run deeplp --example 2 --case 3 --iterations 100 --batch_size 16 --model rnn --do_plot

# Run multiple cases for example 1
poetry run deeplp --example 1 --case 1 2 3 --iterations 200 --batch_size 16 --do_plot
```

### Custom Python Scripts

```bash
# Run the test example script
poetry run python test_example.py

# Run interactive Python session
poetry run python
```

### Inline Python Code

```bash
poetry run python -c "
from deeplp import train, createProblem

# Simple example
c = [1.0, 2.0]
A = [[3, -5], [3, -1], [3, 1]]
b = [15, 21, 27]
tspan = (0.0, 10.0)

problem = createProblem(c, A, b, tspan, name='Quick Test')
solutions = train(
    batches=1,
    batch_size=16,
    epochs=50,
    problem=problem,
    cases=[1],
    do_plot=True,
    model_type='pinn'
)
print('Solution found:', solutions[0].solution)
"
```

## Method 2: Environment Activation (VSCode and Compatible Terminals)

### For Poetry 2.0+

```bash
poetry env activate
deeplp --batches 1 --batch_size 32 --iterations 100 --case 1 --example 1 --do_plot
```

### Manual Activation (CMDER and other terminals)

```bash
# Get the virtual environment path
poetry env info --path

# Activate manually (replace the path with your actual path)
C:\Users\mmogi\AppData\Local\pypoetry\Cache\virtualenvs\deeplp-xyz-py3.x\Scripts\activate.bat

# Now run commands directly
deeplp --batches 1 --batch_size 32 --iterations 100 --case 1 --example 1 --do_plot
```

## Method 3: Using Shell Plugin (Optional)

If you want the old `poetry shell` behavior:

```bash
poetry self add poetry-plugin-shell
poetry shell
deeplp --example 1 --case 1 --iterations 100 --do_plot
```

## Method 4: Batch Scripts (Windows)

Use the provided `run_deeplp.bat` file:

```bash
run_deeplp.bat
```

Or create your own batch files for different examples:

```batch
@echo off
echo Running deepLP example 2 with RNN...
poetry run deeplp --example 2 --case 3 --iterations 200 --model rnn --do_plot
pause
```

## Quick Examples by Use Case

### 1. Quick Test (Fast execution)
```bash
poetry run deeplp --example 1 --case 1 --iterations 50 --batch_size 16
```

### 2. Full Training with Plots
```bash
poetry run deeplp --example 1 --case 1 --iterations 500 --batch_size 32 --do_plot
```

### 3. Equality Constraints (RNN Model)
```bash
poetry run deeplp --example 2 --case 3 --iterations 300 --model rnn --do_plot
```

### 4. Multiple Cases Comparison
```bash
poetry run deeplp --example 1 --case 1 2 3 --iterations 200 --do_plot
```

### 5. Save Models to Directory
```bash
poetry run deeplp --example 1 --case 1 --iterations 500 --folder saved_models --do_plot
```

## Command Line Arguments Reference

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--example` | Which built-in example to run | None | 1, 2, 3, 4 |
| `--case` | Training scenario | 1 | 1 (time only), 2 (time+b), 3 (time+D) |
| `--iterations` | Number of training epochs | 1000 | Any positive integer |
| `--batch_size` | Training batch size | 128 | Any positive integer |
| `--batches` | Number of batches | 1 | Any positive integer |
| `--model` | Model type | pinn | pinn (Ax≤b), rnn (Ax=b, x≥0) |
| `--do_plot` | Show training plots | False | Flag (no value needed) |
| `--folder` | Save models to folder | None | Any valid path |

## Using Custom Problems

Create a Python script with your own problem:

```python
from deeplp import train, createProblem

# Define your LP problem
c = [1.0, 2.0]  # Objective coefficients
A = [[3, -5], [3, -1], [3, 1], [3, 4], [1, 3]]  # Constraint matrix
b = [15, 21, 27, 45, 30]  # Right-hand side values
tspan = (0.0, 10.0)  # Time span

# Create the problem
problem = createProblem(
    c, A, b, tspan,
    name="My Custom Problem",
    b_testing_points=[[15, 21, 27, 45, 30]],  # Optional
    c_testing_points=[[2, 5]]  # Optional
)

# Train the model
solutions = train(
    batches=1,
    batch_size=32,
    epochs=100,
    problem=problem,
    cases=[1, 2, 3],
    do_plot=True,
    model_type="pinn"
)

print(f"Solution: {solutions[0].solution}")
```

Then run:
```bash
poetry run python my_custom_problem.py
```

## Troubleshooting

### Poetry 2.0+ Shell Issues
If you get "shell command is not available" error, use `poetry run` instead of trying to activate the environment.

### CMDER Compatibility
CMDER works best with `poetry run` commands. Avoid environment activation if possible.

### GPU Usage
The package automatically detects and uses CUDA if available. No special configuration needed.

### Memory Issues
If you run out of memory, reduce `batch_size` or `batches`:
```bash
poetry run deeplp --example 1 --case 1 --batch_size 16 --batches 1
```

### Quick Testing
For quick tests, use fewer iterations:
```bash
poetry run deeplp --example 1 --case 1 --iterations 50 --batch_size 8
```

## Example Workflows

### Research/Development Workflow
```bash
# 1. Quick test to ensure everything works
poetry run deeplp --example 1 --case 1 --iterations 50

# 2. Full training with visualization
poetry run deeplp --example 1 --case 1 --iterations 500 --do_plot

# 3. Compare different cases
poetry run deeplp --example 1 --case 1 2 3 --iterations 300 --do_plot

# 4. Test equality constraints
poetry run deeplp --example 2 --case 3 --model rnn --iterations 300 --do_plot
```

### Production/Benchmark Workflow
```bash
# 1. Train and save models
poetry run deeplp --example 1 --case 1 --iterations 1000 --folder models --batch_size 64

# 2. Load and test saved model
poetry run deeplp --load models/example_1_time_only_pinn_1000_2025_XX_XX_out_dim_X.pt --in_dim 1 --out_dim X --T 10.0
```
