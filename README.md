# deeplp

[![PyPI version](https://img.shields.io/pypi/v/deeplp.svg)](https://pypi.org/project/deeplp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency--management-poetry-blue.svg)](https://python-poetry.org/)

**deeplp** is a Python package for solving linear programming problems using deep learning techniques. It leverages Physics-Informed Neural Networks (PINNs) and Recurrent Neural Networks (RNNs) with PyTorch for its backend computations and provides a simple API for defining problems and training models.

## 🚀 Quick Start

```bash
pip install deeplp
```

**New users**: See **[📖 HowToRun.md](HowToRun.md)** for comprehensive setup instructions, especially for Poetry 2.0+ and CMDER users!

## ✨ Features

- **Physics-Informed Neural Networks (PINNs)** for inequality constraints (Ax ≤ b)
- **Recurrent Neural Networks (RNNs)** for equality constraints (Ax = b, x ≥ 0)
- **Simple API** for defining linear programming problems
- **Command-line interface (CLI)** for running experiments
- **Built-in visualization** with training plots and results
- **Model saving/loading** capabilities
- **GPU acceleration** with CUDA support
- **Multiple training scenarios** (time-only, parameterized problems)

## 📋 Requirements

**deeplp** requires:
- Python 3.10+
- PyTorch (CPU or GPU)
- Poetry (recommended for development)

### Installing PyTorch

Visit the [PyTorch website](https://pytorch.org/get-started/locally/) for installation instructions:

```bash
# For CUDA support (if you have a compatible GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only support
pip install torch torchvision torchaudio
```

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install deeplp
```

### From Source with Poetry

```bash
git clone https://github.com/mmogib/deeplp.git
cd deeplp
poetry install
```

## 📖 Documentation

- **[HowToRun.md](HowToRun.md)** - Complete guide with Poetry 2.0+, CMDER support, and examples
- **[Contributing](#-contributing)** - Guidelines for contributors
- **[Examples](#-basic-usage)** - Code examples below

## 🎯 Basic Usage

### Example 1: Inequality Constraints (PINN Model)

```python
from deeplp import train, createProblem

# Define your LP problem: minimize c^T x subject to Ax ≤ b
c = [1.0, 2.0]  # Objective coefficients
A = [[3, -5], [3, -1], [3, 1], [3, 4], [1, 3]]  # Constraint matrix
b = [15, 21, 27, 45, 30]  # Right-hand side values
tspan = (0.0, 10.0)  # Time span for solution evolution

# Create the problem
problem = createProblem(
    c, A, b, tspan,
    name="PINN Example",
    b_testing_points=[[15, 21, 27, 45, 30]],
    c_testing_points=[[2, 5]]
)

# Train the model
solutions = train(
    batches=1,
    batch_size=32,
    epochs=100,
    problem=problem,
    cases=[1],  # Time-only case
    do_plot=True,
    model_type="pinn"
)

print(f"Solution: {solutions[0].solution}")
```

### Example 2: Equality Constraints (RNN Model)

```python
from deeplp import train, createProblem

# Define your LP problem: minimize c^T x subject to Ax = b, x ≥ 0
c = [1.0, 2.0, -1.0, -2.0, 0, 0, 0, 0, 0]  # Objective coefficients
A = [
    [3, -5, -3, 5, 1, 0, 0, 0, 0],
    [3, -1, -3, 1, 0, 1, 0, 0, 0],
    [3, 1, -3, -1, 0, 0, 1, 0, 0],
    [3, 4, -3, -4, 0, 0, 0, 1, 0],
    [1, 3, -1, -3, 0, 0, 0, 0, 1],
]  # Constraint matrix
b = [15, 21, 27, 45, 30]  # Right-hand side values
tspan = (0.0, 10.0)  # Time span

# Create the problem
problem = createProblem(
    c, A, b, tspan,
    name="RNN Example (Equality)",
    c_testing_points=[c]
)

# Train the model
solutions = train(
    batches=1,
    batch_size=32,
    epochs=100,
    problem=problem,
    cases=[3],  # Parameterized case
    do_plot=True,
    model_type="rnn"
)
```

## 🖥️ Command Line Interface

### Quick Examples

```bash
# Basic example with PINN model
deeplp --example 1 --case 1 --iterations 100 --batch_size 32 --do_plot

# Equality constraints with RNN model
deeplp --example 2 --case 3 --iterations 200 --model rnn --do_plot

# Multiple cases with model saving
deeplp --example 1 --case 1 2 3 --iterations 500 --folder saved_models --do_plot
```

### Using Poetry (Poetry 2.0+ Compatible)

```bash
# Recommended approach - no environment activation needed
poetry run deeplp --example 1 --case 1 --iterations 100 --do_plot

# Or activate environment first (if supported)
poetry env activate
deeplp --example 1 --case 1 --iterations 100 --do_plot
```

### All Available Options

```bash
deeplp --help
```

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--example` | Built-in example to run | None | 1, 2, 3, 4 |
| `--case` | Training scenario | 1 | 1 (time only), 2 (time+b), 3 (time+D) |
| `--iterations` | Number of training epochs | 1000 | Any positive integer |
| `--batch_size` | Training batch size | 128 | Any positive integer |
| `--batches` | Number of batches | 1 | Any positive integer |
| `--model` | Model type | pinn | pinn (Ax≤b), rnn (Ax=b, x≥0) |
| `--do_plot` | Show training plots | False | Flag (no value needed) |
| `--folder` | Save models to folder | None | Any valid path |

## 🔬 How It Works

**deeplp** transforms linear programming into a differential equation problem:

1. **Physics-Informed Approach**: LP constraints become "physics laws" that the neural network must satisfy
2. **Time Evolution**: Solutions evolve over time, converging to the optimal point
3. **Two Model Types**:
   - **PINN**: For inequality constraints (Ax ≤ b) using complementary slackness
   - **RNN**: For equality constraints (Ax = b, x ≥ 0) using barrier methods
4. **Automatic Differentiation**: PyTorch computes gradients for the physics constraints

## 🏃‍♂️ Quick Test

Run a quick test to verify installation:

```bash
# Using pip installation
python -c "
from deeplp import train, createProblem
problem = createProblem([1, 2], [[3, -5], [3, -1]], [15, 21], (0, 5), name='Test')
solutions = train(batches=1, batch_size=8, epochs=50, problem=problem, cases=[1])
print('✅ deeplp is working!')
"

# Using Poetry
poetry run python test_example.py
```

## 🎨 Examples and Tutorials

After installation, try these examples:

```bash
# Quick test (fast execution)
deeplp --example 1 --case 1 --iterations 50 --batch_size 16

# Full training with visualization
deeplp --example 1 --case 1 --iterations 500 --batch_size 32 --do_plot

# Compare PINN vs RNN
deeplp --example 2 --case 3 --model rnn --iterations 300 --do_plot
```

## 💾 Saving and Loading Models

```bash
# Save models to a directory
deeplp --example 1 --case 1 --iterations 500 --folder my_models

# Load a saved model
deeplp --load my_models/example_1_time_only_pinn_500_2025_XX_XX_out_dim_X.pt --in_dim 1 --out_dim X --T 10.0
```

## 🎓 Mathematical Background

This package implements a novel approach to linear programming by:

- Converting LP problems into physics-informed neural differential equations
- Using neural networks to approximate solution trajectories
- Leveraging automatic differentiation for constraint satisfaction
- Applying time-based evolution to reach optimal solutions

**Key insight**: Linear programming can be viewed as finding equilibrium points of dynamical systems, which neural networks can learn to approximate.

## 🛠️ Development Setup

### For Contributors

```bash
# Clone the repository
git clone https://github.com/mmogib/deeplp.git
cd deeplp

# Install Poetry if not already installed
pip install poetry

# Install dependencies
poetry install

# Run tests
poetry run python test_example.py

# See comprehensive development guide
open HowToRun.md
```

### Development Tools

- **Package Management**: Poetry
- **Testing**: Custom test scripts
- **Documentation**: Markdown
- **CI/CD**: GitHub Actions (planned)

## 🤝 Contributing

We welcome contributions! Whether you're:

- 🐛 Reporting bugs
- 💡 Suggesting features  
- 📖 Improving documentation
- 🔧 Contributing code

Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test your changes: `poetry run python test_example.py`
5. Submit a pull request

## 📝 Citation

If you use **deeplp** in your research, please cite:

```bibtex
@software{alshahrani2025deeplp,
  title={deepLP: Deep Learning for Linear Programming},
  author={Alshahrani, Mohammed},
  year={2025},
  url={https://github.com/mmogib/deeplp},
  version={0.7.1}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch** team for the excellent deep learning framework
- **Poetry** for modern Python packaging
- The **optimization** and **machine learning** communities for inspiration

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/mmogib/deeplp/issues)
- 💬 **Questions**: [GitHub Discussions](https://github.com/mmogib/deeplp/discussions)
- 📧 **Email**: mmogib@gmail.com
- 📖 **Documentation**: [HowToRun.md](HowToRun.md)

---

**Happy Optimizing! 🚀**
