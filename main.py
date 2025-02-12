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
from peft.tuners.lora import LoraConfig
from peft.utils.peft_types import TaskType
from peft.mapping import get_peft_model
import wandb
from dataset import load_test_data, load_validation_data
import shutil
import os
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import random
import numpy as np
from datetime import datetime

from client import GPT2FLClient
from dataset import load_data
from model import test, train
from semantic_alignment import semantic_alignment_training  # import the function

MODEL_CONFIGS = {
    "gpt2": {
        "path": "openai-community/gpt2",
        "pad_token_strategy": "eos_token",
    },
    "t5-small": {
        "path": "google-t5/t5-small",
        "pad_token_strategy": "pad_token",
    },
    "multi-distilbert": {
        "path": "distilbert/distilbert-base-multilingual-cased",
        "pad_token_strategy": "pad_token",
    },
    "distilbert": {
        "path": "distilbert/distilbert-base-cased",
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

    validation_loader = load_validation_data(
        tokenizer, languages=languages, batch_size=args.batch_size
    )

    print(f"\nStarting centralized training for {args.num_rounds} epochs")
    print(f"Model: {args.model_name}, Batch size: {args.batch_size}")
    print(
        f"LoRA params - r: {args.lora_r}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}"
    )
    print("-" * 50)

    # Training loop
    best_val_accuracy = 0
    best_model_state = None

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

        # Save best model state
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    # Load best model for final test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final test evaluation
    test_loader = load_test_data(
        tokenizer, languages=languages, batch_size=args.batch_size
    )
    test_loss, test_accuracy = test(model, test_loader, device)
    print("\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    wandb.log({"test/loss": test_loss, "test/accuracy": test_accuracy})

    print("\nTraining completed!")
    print(f"Final validation accuracy: {best_val_accuracy:.4f}")


class MetricsAggregationStrategy(FedAvg):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.current_round = 0
        self.global_weights = None  # New attribute

    def aggregate_fit(self, server_round, results, failures):
        self.current_round = server_round
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        if aggregated_result is not None:
            self.global_weights = aggregated_result  # Store latest aggregated weights

            client_metrics = [res.metrics for _, res in results]
            if client_metrics:
                avg_loss = sum(m["train_loss"] for m in client_metrics) / len(client_metrics)  # type: ignore
                avg_accuracy = sum(m["train_accuracy"] for m in client_metrics) / len(client_metrics)  # type: ignore
                wandb.log(
                    {
                        "round": server_round,
                        "aggregated/train_loss": avg_loss,
                        "aggregated/train_accuracy": avg_accuracy,
                        **{
                            f"client_{m['client_id']}/train_loss": m["train_loss"]
                            for m in client_metrics
                        },
                        **{
                            f"client_{m['client_id']}/train_accuracy": m[
                                "train_accuracy"
                            ]
                            for m in client_metrics
                        },
                    }
                )
        return aggregated_result

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)
        if results:
            client_metrics = [res.metrics for _, res in results]
            avg_accuracy = sum(m["accuracy"] for m in client_metrics) / len(client_metrics)  # type: ignore
            wandb.log(
                {
                    "round": server_round,
                    "aggregated/eval_accuracy": avg_accuracy,
                    **{
                        f"client_{m['client_id']}/eval_accuracy": m["accuracy"]
                        for m in client_metrics
                    },
                }
            )
        return aggregated_result


def convert_weight(weight, target_dtype, device):
    print(type(weight))
    # If weight is an instance of Parameters, convert its first ByteBuffer to a numpy array.
    if hasattr(weight, "tensors"):
        # Example conversion: adjust dtype mapping as needed based on weight.tensorType.
        buf = weight.tensors[0]
        weight_np = np.frombuffer(buf, dtype=np.float32)  # adjust dtype if necessary
    else:
        weight_np = weight.cpu().numpy() if isinstance(weight, torch.Tensor) else weight
    return torch.as_tensor(weight_np, dtype=target_dtype, device=device)


from flwr.common import parameters_to_ndarrays, Parameters


def federated_training(model, languages, tokenizer, device, args, experiment_id):
    # Calculate total clients (languages * clients per language)
    total_clients = len(languages) * args.num_clients_per_language

    strategy = MetricsAggregationStrategy(
        device=device,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=total_clients,
        min_evaluate_clients=total_clients,
        min_available_clients=total_clients,
    )

    def client_fn(context: Context):
        partition_id: int = int(context.node_config["partition-id"])
        language_idx = partition_id // args.num_clients_per_language
        client_idx = partition_id % args.num_clients_per_language
        language = languages[language_idx]

        # Load data with partition information
        trainloader = load_data(
            language,
            tokenizer,
            batch_size=args.batch_size,
            partition_id=client_idx,
            total_partitions=args.num_clients_per_language,
        )
        testloader = load_validation_data(
            tokenizer, languages, batch_size=args.batch_size
        )

        client = GPT2FLClient(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            device=device,
            client_id=f"{language}_{partition_id}",
            sa_interval=args.sa_interval,
            sa_epochs=args.sa_epochs,
            sa_samples=args.sa_samples,
            language=language,
            tokenizer=tokenizer,
        )
        return client.to_client()

    def server_fn(context: Context) -> ServerAppComponents:
        config = ServerConfig(num_rounds=args.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    server = ServerApp(server_fn=server_fn)
    client = ClientApp(client_fn=client_fn)

    backend_config = {
        "client_resources": {
            "num_gpus": 1 if device == torch.device("cuda") else 0,
            "num_cpus": 1,
        }
    }
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=args.num_supernodes,
        backend_config=backend_config,  # type: ignore
    )

    # New code: If final global weights exist, load them into the model before final evaluation.
    if strategy.global_weights is not None and strategy.global_weights[0] is not None:
        # Now convert Parameters to numpy arrays
        ndarrays = parameters_to_ndarrays(strategy.global_weights[0])

        # Load state into model
        new_state = {}
        model_state = model.state_dict()
        for key, array in zip(model_state.keys(), ndarrays):
            new_state[key] = torch.as_tensor(
                array, dtype=model_state[key].dtype, device=device
            )
        model.load_state_dict(new_state)

    # After simulation, evaluate on test set
    test_loader = load_test_data(
        tokenizer, languages=languages, batch_size=args.batch_size
    )
    test_loss, test_accuracy = test(model, test_loader, device)
    print("\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    wandb.log({"test/loss": test_loss, "test/accuracy": test_accuracy})


def generate_run_name(config):
    """Generate descriptive name for individual runs based on their parameters."""
    model = config.get("model_name", "unknown")
    if model == "distilbert":
        model = "d"
    elif model == "multi-distilbert":
        model = "md"
    mode = config.get("mode", "unknown")

    # Get key hyperparameters
    lora_params = f"r{config.get('lora_r', '?')}_a{config.get('lora_alpha', '?')}"
    batch = f"b{config.get('batch_size', '?')}"
    language = config.get("language_set", "unknown")

    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%H%M%S")

    return f"{mode}_{language}_{model}_{lora_params}_{batch}_{timestamp}"


def main(config):
    # Set seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Update num_supernodes based on number of clients per language
    if config.mode == "federated":
        if config.language_set == "full":
            config.num_supernodes = 5 * config.num_clients_per_language  # 5 languages
        elif config.language_set == "limited":
            config.num_supernodes = 4 * config.num_clients_per_language  # 4 languages

    # Determine languages based on language_set
    if config.language_set == "full":
        languages = ["en", "de", "es", "fr", "zh"]
    elif config.language_set == "limited":
        languages = ["en", "de", "es", "fr"]
    else:
        if config.mode == "federated":
            raise ValueError(
                "Single language selection is only supported in centralized mode."
            )
        languages = [config.language_set]
        available_languages = ["en", "de", "es", "fr", "zh"]
        if languages[0] not in available_languages:
            raise ValueError(
                f"Unsupported language '{languages[0]}'. Available languages: {available_languages}"
            )

    # Create a unique experiment ID for grouping
    experiment_id = wandb.util.generate_id()  # type: ignore

    # Initialize wandb for the server
    wandb.init(
        project="federated-xnli",
        name=config.experiment_name,
        group=experiment_id,
        config=vars(config),
        reinit=True,
    )

    # delete the .wandb_runs folder to delete existing run IDs
    if os.path.exists("./.wandb_runs"):
        shutil.rmtree("./.wandb_runs")

    # Initialize model and tokenizer based on selection
    model_config = MODEL_CONFIGS[config.model_name]
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

    # Apply LoRA adapters with CLI arguments
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=(
            ["q_lin", "v_lin"] if "distilbert" in config.model_name else None
        ),
    )
    model = get_peft_model(model, peft_config)

    if config.mode == "federated":
        federated_training(model, languages, tokenizer, device, config, experiment_id)
    else:
        centralized_training(model, languages, tokenizer, device, config)

    wandb.finish()

    # Clean up wandb run ID files after experiment is complete
    if os.path.exists("./.wandb_runs"):
        shutil.rmtree("./.wandb_runs")


if __name__ == "__main__":
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
    # Add LoRA hyperparameters
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="Rank of the LoRA update matrices",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Scaling factor for the LoRA update",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--language_set",
        type=str,
        default="full",
        help="Set of languages to use: 'full', 'limited', or a single language code (e.g., 'en')",
    )
    # In the argument parser section, add new semantic alignment hyperparameters:
    parser.add_argument(
        "--sa_interval",
        type=float,
        default=1.0,
        help="Fraction of an epoch (if <1) or full batches (if >=1) after which to run semantic alignment training.",
    )
    parser.add_argument(
        "--sa_epochs",
        type=int,
        default=1,
        help="Number of epochs to run semantic alignment training.",
    )
    parser.add_argument(
        "--sa_samples",
        type=int,
        default=0,
        help="Number of samples to choose for semantic alignment training. If 0, semantic alignment is skipped.",
    )
    parser.add_argument(
        "--num_clients_per_language",
        type=int,
        default=1,
        help="Number of clients per language (default: 1)",
    )

    args = parser.parse_args()

    # Generate experiment name
    args.experiment_name = generate_run_name(vars(args))

    # Run main with processed arguments
    main(args)
