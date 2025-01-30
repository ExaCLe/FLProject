import argparse
import wandb
import yaml
from datetime import datetime


def load_sweep_config(yaml_path):
    """Load sweep configuration from YAML file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Initialize a W&B sweep")
    parser.add_argument(
        "--sweep_config",
        type=str,
        required=True,
        help="Path to the sweep configuration YAML file",
    )
    args = parser.parse_args()

    # Load sweep configuration
    sweep_config = load_sweep_config(args.sweep_config)

    # Initialize the sweep and get sweep ID
    sweep_id = wandb.sweep(
        sweep_config,
        project="federated-xnli",  # Change to your project name
    )

    print(f"Sweep initialized with ID: {sweep_id}")
    print(f"To run an agent, use: python run_agent.py --sweep_id {sweep_id}")


if __name__ == "__main__":
    main()
