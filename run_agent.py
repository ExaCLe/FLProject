import argparse
import wandb
from types import SimpleNamespace
from main import main, generate_run_name

# Define default configuration that matches argparse defaults
DEFAULT_CONFIG = {
    "num_supernodes": 5,
    "model_name": "gpt2",
    "num_rounds": 5,
    "mode": "federated",
    "batch_size": 8,
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "learning_rate": 5e-5,
    "seed": 42,
    "language_set": "full",
    "sa_samples": 0,
    "sa_interval": 0.5,
    "sa_epochs": 1,
}


def run_sweep(sweep_id: str, count: int):
    """Run a W&B sweep with the given ID and count."""

    def run_training():
        # Initialize wandb for this run
        wandb.init()

        # Merge default config with wandb config
        config = DEFAULT_CONFIG.copy()
        config.update(dict(wandb.config))

        # Generate experiment name
        config["experiment_name"] = generate_run_name(config)

        # Convert to SimpleNamespace
        config = SimpleNamespace(**config)

        # Run the training
        main(config)

    # Start the sweep agent
    wandb.agent(sweep_id, function=run_training, count=count, project="federated-xnli")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a W&B sweep agent")
    parser.add_argument(
        "--sweep_id", type=str, required=True, help="Sweep ID to run agent for"
    )
    parser.add_argument(
        "--count", type=int, default=20, help="Number of runs to execute in the sweep"
    )
    args = parser.parse_args()

    run_sweep(args.sweep_id, args.count)
