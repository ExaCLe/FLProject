import torch
import torch.nn as nn
import torch.optim as optim


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=20, seq_len=5, vocab_size=10):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TinyModel(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        pooled = embedded.mean(dim=1)  # (batch_size, embed_dim)
        return self.fc(pooled)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TinyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    model = TinyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
        train_one_epoch(model, dataloader, optimizer, device)
        # evaluate accuracy here
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        if (epoch + 1) % 10 == 0:
            accuracy = correct / total
            print(f"Epoch {epoch+1} done.")
            print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
