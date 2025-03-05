import argparse
from time import sleep
from typing import List
import numpy as np

from deeplp.train import train
from deeplp.utils import in_notebook

def add(x:int, y:List[int]):
    s = np.array(y) + x
    return s 

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
        default="saved_models",
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
        default="1",
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
            rnag2= tqdm_notebook(range(20), desc="Inner loop", leave=False)
        else:
            rnag1 = tqdm(range(10), desc="Outer loop")
            rnag2= tqdm(range(20), desc="Inner loop", leave=False)

        for i in rnag1:
            # Inner loop; using leave=False so it doesn't keep each inner bar on a new line
            for j in rnag2:
                # Simulate some work
                sleep(0.01)
        exit(0)
    # examples = [example_1, example_2, example_3]

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