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


def load_data(
    language: str,
    tokenizer,
    batch_size: int = 8,
    partition_id: int = 0,
    total_partitions: int = 1,
    max_length: int = 128,
):
    """
    Load and partition training data for a specific language from the main CSV.

    Args:
        language: Language code to filter by
        tokenizer: Tokenizer to use
        batch_size: Batch size for DataLoader
        partition_id: Which partition of the data to load (0 to total_partitions-1)
        total_partitions: Total number of partitions to split data into
        max_length: Max sequence length for tokenization
    """
    # Load the main XNLI training data
    csv_path = "./data/xnli/xnli_filtered_dev.csv"
    df = pd.read_csv(csv_path)

    # Filter by language first
    df = df[df["language"] == language]

    # Then partition the language-specific data
    total_samples = len(df)
    samples_per_partition = total_samples // total_partitions
    start_idx = partition_id * samples_per_partition
    end_idx = (
        start_idx + samples_per_partition
        if partition_id < total_partitions - 1
        else total_samples
    )

    # Select partition
    df = df.iloc[start_idx:end_idx]

    # Map labels to integers
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    labels = df["gold_label"].map(label_map).tolist()

    # Create sentence pairs
    sentence_pairs = list(zip(df["sentence1"], df["sentence2"]))

    # Create dataset and dataloader
    dataset = CustomTextDataset(sentence_pairs, labels, tokenizer, max_length)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return dataloader


def load_validation_data(tokenizer, languages=None, max_length=128, batch_size=32):
    """Load the validation dataset for specified languages or all languages."""
    csv_path = "./data/xnli/xnli_validation.csv"
    df = pd.read_csv(csv_path)

    # Filter by languages if specified
    if languages is not None:
        df = df[df["language"].isin(languages)]

    # Map gold_label to integers
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    labels = df["gold_label"].map(label_map).tolist()

    sentence_pairs = list(zip(df["sentence1"], df["sentence2"]))
    dataset = CustomTextDataset(sentence_pairs, labels, tokenizer, max_length)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return dataloader


def load_test_data(tokenizer, languages=None, max_length=128, batch_size=32):
    """Load the test dataset for specified languages or all languages."""
    csv_path = "./data/xnli/xnli_test.csv"
    df = pd.read_csv(csv_path)

    # Filter by languages if specified
    if languages is not None:
        df = df[df["language"].isin(languages)]

    # Map gold_label to integers
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    labels = df["gold_label"].map(label_map).tolist()

    sentence_pairs = list(zip(df["sentence1"], df["sentence2"]))
    dataset = CustomTextDataset(sentence_pairs, labels, tokenizer, max_length)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return dataloader
