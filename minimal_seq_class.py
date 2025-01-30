import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, num_samples=20):
        self.tokenizer = tokenizer
        # Minimal texts and labels
        self.texts = ["Hello world"] * (num_samples // 2) + ["Foo bar"] * (
            num_samples // 2
        )
        self.labels = [0] * (num_samples // 2) + [1] * (num_samples // 2)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=8,
            return_tensors="pt",
        )
        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for input_ids, attention_mask, labels in dataloader:
        input_ids, attention_mask, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = TinyDataset(tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(20):
        # evaluate accuracy here
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for input_ids, attention_mask, labels in dataloader:
                input_ids, attention_mask, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    labels.to(device),
                )
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        if (epoch + 0) % 1 == 0:
            accuracy = correct / total
            print(f"Epoch {epoch+1} done.")
            print(f"Accuracy: {accuracy:.2f}")
        train_one_epoch(model, dataloader, optimizer, device)


if __name__ == "__main__":
    main()
