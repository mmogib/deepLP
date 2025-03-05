from collections import namedtuple
import importlib
import inspect
from typing import List
import torch
import numpy as np

import os

from deeplp.problems import pretty_print_lp
from deeplp.models import (
    train_model,
    save_model,
    test_model,
)


# Import the module using importlib (the file must be in your Python path)
problem_module = importlib.import_module("deeplp.problems")

# Use inspect.getmembers to retrieve all functions defined in the module.
# This returns a list of tuples (name, function).


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt


Solution = namedtuple("Solution", ["solution", "model", "loss_list", "mov_lis"])


def plot_data(filename, title, ylabel):
    # Assume loss_list is a list of float loss values collected during training
    # Load loss_list from the text file.
    data_array = np.loadtxt(filename)
    # Convert to a Python list if needed:
    data_list = data_array.tolist()
    plt.figure(figsize=(8, 5))
    plt.plot(data_list, label=title)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(f"{title} Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


def save_lists_to_file(loss_list, mov_list, filename):
    np_loss = np.array(loss_list)
    loss_filename = f"{filename}_loss.txt"
    print(f"saving... loss list in {loss_filename}")
    np.savetxt(loss_filename, np_loss, fmt="%.6f")

    movs_filename = f"{filename}_mov.txt"
    np_movs = np.array(mov_list)
    print(f"saving... movs list in {movs_filename}")
    np.savetxt(movs_filename, np_movs, fmt="%.6f")


# Example 1: Solving one LP with 4 variables and 1 constraint


def train(
    batches: int = 1,
    batch_size: int = 128,
    epochs: int = 1000,
    problems_ids: List[int] = [1],
    cases: List[int] = [1],
    do_plot: bool = True,
    saving_dir: str | None = "saved_models",
):

    torch.manual_seed(2025)
    problems = [
        func
        for name, func in inspect.getmembers(problem_module, inspect.isfunction)
        if name.startswith("problem")
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for problem_indx in problems_ids:
        problem = problems[problem_indx - 1]
        prb = problem()
        pretty_print_lp(prb)
        D, A, b, tspan, name, test_points, D_testing_points = prb

        D = torch.tensor(D, dtype=torch.float32, device=device)
        A = torch.tensor(A, dtype=torch.float32, device=device)
        b = torch.tensor(b, dtype=torch.float32, device=device)
        for case in cases:
            case_saving_dir = (
                f"{saving_dir}/case_{case}"
                if saving_dir is not None
                else f"case_{case}"
            )
            os.makedirs(case_saving_dir, exist_ok=True)
            model, loss_list, mov_list = train_model(
                A,
                b,
                D,
                f"{name} (CASE {case})",
                tspan,
                case,
                epochs,
                batch_size,
                batches,
                device,
            )
            test_points = (
                ([D.tolist()] if D_testing_points is None else D_testing_points)
                if case == 3
                else test_points
            )
            filename = test_model(
                model,
                device,
                test_points,
                case,
                tspan[1],
                name,
                epochs,
                dir_name=case_saving_dir,
            )
            if saving_dir is not None:
                save_model(model, filename)
                save_lists_to_file(loss_list, mov_list, filename)

            movs_filename = f"{filename}_mov.txt"
            loss_filename = f"{filename}_loss.txt"
            if do_plot:
                plot_data(loss_filename, "Trainig Loss", "Loss")
                plot_data(movs_filename, "Trainig MOV", "MOV")
            t_tensor = torch.tensor(
                tspan[1], dtype=torch.float32, device=device, requires_grad=True
            )
            y_pred = model(t_tensor)
            y_pred_np = y_pred.cpu().detach().numpy()
            return Solution(y_pred_np, model, loss_list, mov_list)


if __name__ == "__main__":
    pass
