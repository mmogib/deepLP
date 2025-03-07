import argparse
import os
from time import sleep

from .problems import get_all_problems, pretty_print_lp
from .train import train
from .utils import in_notebook
from .models import load_model
from .solvers import solve_lp, solve_ode


def main():
    parser = argparse.ArgumentParser(
        description="Train the PINN model using named arguments"
    )
    parser.add_argument("--no-action", help="No action needed", action="store_true")

    parser.add_argument(
        "--iterations", type=int, default=1000, help="Number of training iterations"
    )

    parser.add_argument(
        "--batches", type=int, default=1, help="Number of training batches"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")

    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        help="Path to the saving folder",
        # action="store_const",
        # const=None,
    )
    parser.add_argument(
        "--load",
        "-l",
        type=str,
        help="Path to model file",
        # action="store_const",
        # const=None,
    )
    parser.add_argument(
        "--in_dim",
        "-in",
        type=int,
        # action="store_const",
        # const=None,
        help="Number of Input Variables.",
    )
    parser.add_argument(
        "--out_dim",
        "-out",
        type=int,
        # action="store_const",
        # const=None,
        help="Number of Output Variables.",
    )
    parser.add_argument(
        "--T",
        "-T",
        type=float,
        # action="store_const",
        # const=None,
        help="Upper limit of the time interval.",
    )
    parser.add_argument("--do_plot", action="store_true", help="Plot them")
    parser.add_argument(
        "--case",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        default="1",
        help="Which case to run (1: time only, 2: time and b, 3: time and D)",
    )
    parser.add_argument(
        "--example",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Which example to run (1, 2, 3, ...)",
    )
    args = parser.parse_args()

    if args.no_action:
        print("No action flag is set.")
        # read_mps("mps_files/problem2.mps")
        # plot_loss()
        from tqdm import tqdm

        if in_notebook():
            from tqdm import tqdm_notebook

            rnag1 = tqdm_notebook(range(10), desc="Outer loop")
            rnag2 = tqdm_notebook(range(20), desc="Inner loop", leave=False)
        else:
            rnag1 = tqdm(range(10), desc="Outer loop")
            rnag2 = tqdm(range(20), desc="Inner loop", leave=False)

        for i in rnag1:
            # Inner loop; using leave=False so it doesn't keep each inner bar on a new line
            for j in rnag2:
                # Simulate some work
                sleep(0.01)
        exit(0)
    # examples = [example_1, example_2, example_3]
    if args.load:
        filename = args.load
        assert (
            args.in_dim is not None and args.out_dim is not None
        ), "You must provide the number of variables in and out."
        assert os.path.exists(filename), "The file does not exist; check the name."
        print(f"Loading file {filename}")
        T = 10.0 if args.T is None else args.T
        val = load_model(args.load, args.in_dim, args.out_dim)(T)
        print(val)
        if (problem_indx := args.example) is not None:
            print(problem_indx)
            problems = get_all_problems()
            problem = problems[problem_indx[0] - 1]
            prb = problem()
            pretty_print_lp(prb)
            D, A, b, tspan, name, test_points, D_testing_points = prb
            sol, valu = solve_lp(D, A, b)
            sol_ode = solve_ode(prb)
            print(f"{sol}, {valu}")
            print(f"{sol_ode.y[:,-1]}, {valu}")

        exit(0)
    print(f"Running example {args.example} for {args.iterations} epochs.")
    train(
        batches=args.batches,
        batch_size=args.batch_size,
        epochs=args.iterations,
        cases=args.case,
        problems_ids=args.example,
        do_plot=args.do_plot,
        saving_dir=args.folder,
    )


if __name__ == "__main__":
    main()
