from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from flwr.client import NumPyClient


# Define the training function
def train(net, trainloader, epochs, device):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for input_ids, attention_mask in trainloader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            outputs = net(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


# Define the evaluation function
def test(net, testloader, device):
    net.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask in testloader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            outputs = net(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            total_loss += outputs.loss.item()
    return total_loss / len(testloader)


# Define Flower client
class GPT2FLClient(NumPyClient):
    def __init__(self, model, trainloader, testloader, device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1, device=self.device)
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = test(self.model, self.testloader, device=self.device)
        return loss, len(self.testloader.dataset), {}


# Load dataset and tokenize
def load_data(language, tokenizer, max_length=128):
    csv_path = "./data/xnli/xnli_filtered_dev.csv"
    df = pd.read_csv(csv_path)
    df = df[df["language"] == language]

    class CustomTextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            tokenized = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return (
                tokenized["input_ids"].squeeze(),
                tokenized["attention_mask"].squeeze(),
            )

    dataset = CustomTextDataset(df["text"].tolist(), tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    return dataloader


# Start the client
def start_client(language, model_name="openai-community/gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainloader = load_data(language, tokenizer)
    testloader = load_data(language, tokenizer)

    client = GPT2FLClient(model, trainloader, testloader, device)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
