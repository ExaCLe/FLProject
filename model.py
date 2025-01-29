import torch
from torch.optim import AdamW
from tqdm import trange, tqdm
from contextlib import nullcontext


def get_autocast(device):
    """Helper function to get the appropriate autocast context."""
    if device.type == "cuda":
        return torch.autocast("cuda")
    elif device.type == "mps":
        return nullcontext()  # MPS doesn't support autocast yet
    return nullcontext()  # CPU doesn't need autocast


# Training function
def train(net, trainloader, epochs, device):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    net.to(device)
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0

    autocast_context = get_autocast(device)
    for _ in range(epochs):
        with autocast_context:
            for input_ids, attention_mask, labels in tqdm(trainloader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = net(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return {"train_loss": avg_loss, "train_accuracy": correct / total}


# Evaluation function with accuracy calculation
def test(net, testloader, device):
    net.eval()
    net.to(device)
    total_loss = 0
    correct = 0
    total = 0

    autocast_context = get_autocast(device)
    with torch.no_grad():
        with autocast_context:
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
