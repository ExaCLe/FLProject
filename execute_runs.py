import pandas as pd
from types import SimpleNamespace
from main import main, generate_run_name
import wandb
from run_agent import run_sweep, DEFAULT_CONFIG


def execute_direct_runs(tsv_path: str):
    """Execute runs directly from TSV configuration file."""
    # Read TSV file
    df = pd.read_csv(tsv_path, sep="\t")

    for _, row in df.iterrows():
        print(f"\nStarting run with configuration:")
        print(row.to_dict())

        # Create config from row
        config = DEFAULT_CONFIG.copy()
        config.update(row.to_dict())

        # Generate experiment name
        config["experiment_name"] = generate_run_name(config)

        # Convert to SimpleNamespace for compatibility with main()
        config = SimpleNamespace(**config)

        try:
            # Initialize wandb for this run
            wandb.init(
                project="federated-xnli",
                name=config.experiment_name,
                config=vars(config),
                reinit=True,
            )

            # Run training
            main(config)

            # Finish wandb run
            wandb.finish()

        except Exception as e:
            print(f"Error in run {config.experiment_name}: {str(e)}")
            wandb.finish()
            continue


def execute_sweeps(tsv_path: str):
    """Execute sweeps from TSV configuration file."""
    # Read TSV file
    df = pd.read_csv(tsv_path, sep="\t")

    for _, row in df.iterrows():
        sweep_id = row["sweep_id"]
        count = int(row["count"])
        print(f"\nStarting sweep {sweep_id} with {count} runs")

        try:
            run_sweep(sweep_id, count)
        except Exception as e:
            print(f"Error in sweep {sweep_id}: {str(e)}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--direct_tsv",
        type=str,
        help="Path to TSV file containing direct run configurations",
    )
    parser.add_argument(
        "--sweep_tsv", type=str, help="Path to TSV file containing sweep configurations"
    )

    args = parser.parse_args()

    if args.direct_tsv:
        print(f"Executing direct runs from {args.direct_tsv}")
        execute_direct_runs(args.direct_tsv)

    if args.sweep_tsv:
        print(f"Executing sweeps from {args.sweep_tsv}")
        execute_sweeps(args.sweep_tsv)
