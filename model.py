import torch
from torch.optim import AdamW
from tqdm import trange, tqdm


# Training function
def train(net, trainloader, epochs, device):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    net.to(device)
    for _ in range(epochs):
        for input_ids, attention_mask, labels in tqdm(trainloader):
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
