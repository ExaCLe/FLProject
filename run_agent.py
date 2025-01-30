import argparse
import wandb
import subprocess
import sys
from datetime import datetime


def generate_run_name(config):
    """Generate descriptive name for individual runs based on their parameters."""
    model = config.get("model_name", "unknown")
    mode = config.get("mode", "unknown")

    # Get key hyperparameters
    lora_params = f"r{config.get('lora_r', '?')}_a{config.get('lora_alpha', '?')}"
    batch = f"b{config.get('batch_size', '?')}"

    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%H%M%S")

    return f"{mode}_{model}_{lora_params}_{batch}_{timestamp}"


def main():
    parser = argparse.ArgumentParser(description="Run a W&B sweep agent")
    parser.add_argument(
        "--sweep_id", type=str, required=True, help="Sweep ID to run agent for"
    )
    parser.add_argument(
        "--count", type=int, default=10, help="Number of runs to execute in the sweep"
    )
    args = parser.parse_args()

    def run_training():
        # Initialize wandb for this run
        wandb.init(
            project="federated-xnli",  # Change to your project name
        )
        # Override experiment_name with descriptive name
        config = dict(wandb.config)
        exp_name = generate_run_name(config)
        config["experiment_name"] = exp_name

        # Run the training with all parameters
        subprocess.run(
            [
                sys.executable,
                "main.py",
                *[f"--{k}={v}" for k, v in config.items()],
            ],
            check=True,
        )

    # Start the sweep agent
    wandb.agent(
        args.sweep_id, function=run_training, count=args.count, project="federated-xnli"
    )


if __name__ == "__main__":
    main()
