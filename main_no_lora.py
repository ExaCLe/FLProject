import argparse
import time
import flwr as fl
from flwr.client import ClientApp
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import wandb
from dataset import load_validation_data
import shutil
import os
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

from client import GPT2FLClient
from dataset import load_data
from model import test, train

MODEL_CONFIGS = {
    "gpt2": {
        "path": "openai-community/gpt2",
        "pad_token_strategy": "eos_token",
    },
    "t5-small": {
        "path": "google-t5/t5-small",
        "pad_token_strategy": "pad_token",
    },
    "distilbert": {
        "path": "distilbert/distilbert-base-uncased",
        "pad_token_strategy": "pad_token",
    },
    "distilroberta": {
        "path": "distilbert/distilroberta-base",
        "pad_token_strategy": "pad_token",
    },
    "all-minilm": {
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "pad_token_strategy": "pad_token",
    },
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_model(model, experiment_name):
    """Save model checkpoint."""
    save_dir = f"models/{experiment_name}"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"model_round_{time.time()}")
    model.save_pretrained(model_path)
    return model_path


def centralized_training(model, languages, tokenizer, device, args):
    """Run centralized training on all data."""
    # Combine data from all languages
    all_trainloaders = [
        load_data(lang, tokenizer, batch_size=args.batch_size) for lang in languages
    ]
    all_datasets = [loader.dataset for loader in all_trainloaders]
    combined_dataset = ConcatDataset(all_datasets)

    # Create a single dataloader with all data
    trainloader = DataLoader(
        combined_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    validation_loader = load_validation_data(tokenizer, batch_size=args.batch_size)

    print(f"\nStarting centralized training for {args.num_rounds} epochs")
    print(f"Model: {args.model_name}, Batch size: {args.batch_size}")
    print("-" * 50)

    # Training loop
    val_accuracy = 0
    for epoch in range(args.num_rounds):
        print(f"\nEpoch {epoch + 1}/{args.num_rounds}")

        # Train for one epoch
        metrics = train(model, trainloader, epochs=1, device=device)

        # Evaluate
        val_loss, val_accuracy = test(model, validation_loader, device)

        # Save model
        model_path = save_model(model, args.experiment_name)

        # Print progress
        print(
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Train Acc: {metrics['train_accuracy']:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # Log metrics
        wandb.log(
            {
                "train/loss": metrics["train_loss"],
                "train/accuracy": metrics["train_accuracy"],
                "validation/loss": val_loss,
                "validation/accuracy": val_accuracy,
                "epoch": epoch,
                "model_checkpoint": model_path,
            }
        )

    print("\nTraining completed!")
    print(f"Final validation accuracy: {val_accuracy:.4f}")


def federated_training(model, languages, tokenizer, device, args, experiment_id):
    # Load validation dataset once
    validation_loader = load_validation_data(tokenizer)

    def validate_global_model(model):
        model.eval()
        loss, accuracy = test(model, validation_loader, device)
        wandb.log({"validation/loss": loss, "validation/accuracy": accuracy})
        return loss, accuracy

    experiment_name = args.experiment_name

    class DeviceAgnosticFedAvg(FedAvg):
        def __init__(self, device, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.device = device

        def aggregate_fit(self, *args, **kwargs):
            # Move current model parameters to CPU before aggregation
            with torch.device("cpu"):
                results = super().aggregate_fit(*args, **kwargs)
            if results is not None:
                loss, accuracy = validate_global_model(model)
                model_path = save_model(model, experiment_name)
                wandb.log(
                    {
                        "validation/loss": loss,
                        "validation/accuracy": accuracy,
                        "model_checkpoint": model_path,
                    }
                )
            return results

    """Run federated training."""
    strategy = DeviceAgnosticFedAvg(
        device=device,
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
    )

    def client_fn(context: Context):
        partition_id: int = int(context.node_config["partition-id"])
        language = languages[partition_id % len(languages)]
        trainloader = load_data(language, tokenizer, batch_size=args.batch_size)
        testloader = load_data(language, tokenizer, batch_size=args.batch_size)
        return GPT2FLClient(
            model,
            trainloader,
            testloader,
            device,
            client_id=f"{language}_{partition_id}",
            wandb_group=experiment_id,  # Pass the group ID to client
            experiment_name=args.experiment_name,
        ).to_client()

    def server_fn(context: Context) -> ServerAppComponents:
        config = ServerConfig(num_rounds=args.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    server = ServerApp(server_fn=server_fn)
    client = ClientApp(client_fn=client_fn)

    backend_config = {"client_resources": {"num_gpus": 1, "num_cpus": 1}}
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=args.num_supernodes,
        backend_config=backend_config,  # type: ignore
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_supernodes", type=int, default=5, help="Number of clients to simulate"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        choices=MODEL_CONFIGS.keys(),
        help="Model architecture to use",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="federated_xnli",
        help="Name for the wandb experiment",
    )
    parser.add_argument(
        "--num_rounds", type=int, default=5, help="Number of federated rounds"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["federated", "centralized"],
        default="federated",
        help="Training mode: federated or centralized",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and validation",
    )

    args = parser.parse_args()

    # Create a unique experiment ID for grouping
    experiment_id = wandb.util.generate_id()  # type: ignore

    # Initialize wandb for the server
    wandb.init(
        project="federated-xnli",
        name=f"{args.mode}_{args.experiment_name}",
        group=experiment_id,
        config={
            "mode": args.mode,
            "model_name": args.model_name,
            "num_supernodes": args.num_supernodes,
            "num_rounds": args.num_rounds,
        },
        reinit=True,
    )

    # delete the .wandb_runs folder to delete exising run IDs
    if os.path.exists("./.wandb_runs"):
        shutil.rmtree("./.wandb_runs")

    # Initialize model and tokenizer based on selection
    model_config = MODEL_CONFIGS[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_config["path"])

    # Handle padding token based on model type
    if model_config["pad_token_strategy"] == "eos_token":
        tokenizer.pad_token = tokenizer.eos_token

    device = get_device()
    print(f"Using device: {device}")

    # Initialize the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["path"],
        num_labels=3,
    ).to(device)

    # Set pad token ID if needed
    if (
        hasattr(model.config, "pad_token_id")
        and model_config["pad_token_strategy"] == "eos_token"
    ):
        model.config.pad_token_id = tokenizer.pad_token_id

    # # Freeze all parameters except the classification head
    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    # Verify and print trainable parameters
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"Trainable layer: {name}")

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(
        f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%"
    )

    device = get_device()
    print(f"Using device: {device}")

    languages = ["en", "de", "es", "fr", "ru"]

    if args.mode == "federated":
        federated_training(model, languages, tokenizer, device, args, experiment_id)
    else:
        centralized_training(model, languages, tokenizer, device, args)

    wandb.finish()

    # Clean up wandb run ID files after experiment is complete
    if os.path.exists("./.wandb_runs"):
        shutil.rmtree("./.wandb_runs")


if __name__ == "__main__":
    main()
