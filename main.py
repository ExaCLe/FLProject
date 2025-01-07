import argparse
import flwr as fl
from flwr.client import ClientApp
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
import torch
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from peft import LoraConfig, TaskType, get_peft_model

from client import GPT2FLClient
from dataset import load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_supernodes", type=int, default=5, help="Number of clients to simulate"
    )
    args = parser.parse_args()

    # Initialize model and tokenizer
    model_name = "gpt2"  # Use a base GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Use GPT-2 for sequence classification
    model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA adapters
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    languages = ["en", "de", "es", "fr", "ru"]

    def client_fn(context: Context):
        partition_id: int = int(context.node_config["partition-id"])
        language = languages[partition_id % len(languages)]
        trainloader = load_data(language, tokenizer)
        testloader = load_data(language, tokenizer)
        return GPT2FLClient(model, trainloader, testloader, device).to_client()

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
    )

    def server_fn(context: Context) -> ServerAppComponents:
        config = ServerConfig(num_rounds=5)
        return ServerAppComponents(strategy=strategy, config=config)

    server = ServerApp(server_fn=server_fn)
    client = ClientApp(client_fn=client_fn)

    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=args.num_supernodes,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )


if __name__ == "__main__":
    main()
