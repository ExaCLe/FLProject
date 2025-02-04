import torch
from torch.optim import AdamW
from tqdm import trange, tqdm
from contextlib import nullcontext
from semantic_alignment import semantic_alignment_training


# Updated Training function
def train(
    net,
    trainloader,
    epochs,
    device,
    learning_rate=5e-5,
    # Add semantic alignment parameters
    sa_interval: float = 0.0,
    sa_samples: int = 0,
    sa_epochs: int = 0,
    language: str = None,  # type: ignore
    tokenizer=None,
):
    """Training function that includes semantic alignment at specified intervals."""
    optimizer = AdamW(net.parameters(), lr=learning_rate)
    net.train()
    net.to(device)
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0

    total_batches = len(trainloader)
    # Calculate after how many batches to do semantic alignment
    sa_threshold = (
        int(total_batches * sa_interval)
        if sa_interval < 1
        else int(sa_interval * total_batches)
    )

    for epoch in range(epochs):
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
            if sa_samples > 0 and batch_counter == sa_threshold:
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
                # Reset batch counter to handle multiple alignments per epoch if sa_interval < 1
                batch_counter = 0
                net.train()  # Ensure we're back in training mode

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return {"train_loss": avg_loss, "train_accuracy": correct / total}


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
