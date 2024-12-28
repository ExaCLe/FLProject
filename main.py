import argparse
import flwr as fl
from flwr.client import NumPyClient, ClientApp
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from torch.optim import AdamW
from peft import LoraConfig, TaskType, get_peft_model


# Define a classification dataset for XNLI
class CustomTextDataset(Dataset):
    def __init__(self, sentence_pairs, labels, tokenizer, max_length):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        sentence1, sentence2 = self.sentence_pairs[idx]
        # Combine the two sentences. GPT-2 can handle raw text input.
        combined_text = f"{sentence1} [SEP] {sentence2}"
        tokenized = self.tokenizer(
            combined_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return (
            tokenized["input_ids"].squeeze(),
            tokenized["attention_mask"].squeeze(),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def load_data(language, tokenizer, max_length=128):
    # This CSV should contain XNLI data filtered by language
    # Columns: sentence1, sentence2, gold_label
    # For example: "entailment", "neutral", "contradiction"
    csv_path = "./data/xnli/xnli_filtered_dev.csv"
    df = pd.read_csv(csv_path)
    df = df[df["language"] == language]

    # Map gold_label to integers
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    labels = df["gold_label"].map(label_map).tolist()

    sentence_pairs = list(zip(df["sentence1"], df["sentence2"]))
    dataset = CustomTextDataset(sentence_pairs, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    return dataloader


# Training function
def train(net, trainloader, epochs, device):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    net.to(device)
    for _ in range(epochs):
        for input_ids, attention_mask, labels in trainloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = net(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


# Evaluation function with accuracy calculation
def test(net, testloader, device):
    net.eval()
    net.to(device)
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in testloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = net(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss.item()
            total_loss += loss

            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(testloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


# Flower client for federated learning
class GPT2FLClient(NumPyClient):
    def __init__(self, model, trainloader, testloader, device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters, config):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        train(self.model, self.trainloader, epochs=1, device=self.device)
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        loss, accuracy = test(self.model, self.testloader, device=self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


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
