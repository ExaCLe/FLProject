import torch
from torch.optim import AdamW
from tqdm import tqdm
from semantic_alignment import semantic_alignment_training


def train(
    net,
    trainloader,
    epochs,
    device,
    learning_rate=5e-5,
    sa_interval: float = 0.0,
    sa_samples: int = 0,
    sa_epochs: int = 0,
    language: str = None,  # type: ignore
    tokenizer=None,
):
    """Train a neural network with optional semantic alignment.

    This function handles both standard training and semantic alignment training
    at specified intervals for non-English languages.

    Args:
        net: Neural network model to train
        trainloader: DataLoader containing training data
        epochs (int): Number of training epochs
        device: Device to run training on (cuda/cpu)
        learning_rate (float, optional): Learning rate for optimization. Defaults to 5e-5
        sa_interval (float, optional): Interval for semantic alignment (fraction of epoch or batches). Defaults to 0.0
        sa_samples (int, optional): Number of samples for semantic alignment. Defaults to 0
        sa_epochs (int, optional): Number of epochs for semantic alignment training. Defaults to 0
        language (str, optional): Language code for current training data. Defaults to None
        tokenizer (optional): Tokenizer for text processing. Defaults to None

    Returns:
        dict: Training metrics including:
            - train_loss (float): Average training loss
            - train_accuracy (float): Training accuracy
    """
    optimizer = AdamW(net.parameters(), lr=learning_rate)
    net.train()
    net.to(device)
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0

    total_batches = len(trainloader)
    # Calculate after how many batches to do semantic alignment
    sa_threshold = int(total_batches * sa_interval)

    for _ in range(epochs):
        batch_counter = 0
        for batch in tqdm(trainloader):
            batch_counter += 1

            # Process batch
            if isinstance(batch, dict):
                input_ids = batch.get("input_ids").to(device)  # type: ignore
                attention_mask = batch.get("attention_mask").to(device)  # type: ignore
                labels = batch.get("labels").to(device)  # type: ignore
            else:
                input_ids, attention_mask, labels = (x.to(device) for x in batch)

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

            # Check if we should do semantic alignment
            if (
                sa_samples > 0
                and batch_counter % sa_threshold == 0
                and language != "en"
            ):
                print(
                    f"\nPerforming semantic alignment at batch {batch_counter}/{total_batches}"
                )
                net = semantic_alignment_training(
                    model=net,
                    language=language,
                    select_samples=sa_samples,
                    tokenizer=tokenizer,
                    num_epochs=sa_epochs,
                    batch_size=trainloader.batch_size,
                )
                batch_counter = 0
                net.train()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return {"train_loss": avg_loss, "train_accuracy": correct / total}


def test(net, testloader, device):
    """Evaluate a neural network model on test data.

    Args:
        net: Neural network model to evaluate
        testloader: DataLoader containing test data
        device: Device to run evaluation on (cuda/cpu)

    Returns:
        tuple: (avg_loss, accuracy)
            - avg_loss (float): Average loss on test set
            - accuracy (float): Classification accuracy on test set
    """
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
