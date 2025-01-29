import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import GPT2Tokenizer


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
        tokenized = self.tokenizer(
            sentence1,
            sentence2,
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


def load_data(language, tokenizer, max_length=128, batch_size=8):
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
    # Add drop_last=True to ensure consistent batch sizes
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return dataloader


def load_validation_data(tokenizer, max_length=128, batch_size=32):
    """Load the validation dataset."""
    csv_path = "./data/xnli/xnli_validation.csv"
    df = pd.read_csv(csv_path)

    # Map gold_label to integers
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    labels = df["gold_label"].map(label_map).tolist()

    sentence_pairs = list(zip(df["sentence1"], df["sentence2"]))
    dataset = CustomTextDataset(sentence_pairs, labels, tokenizer, max_length)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return dataloader
