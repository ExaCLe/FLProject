import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
)
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
from peft.utils.peft_types import TaskType
from torch.nn.functional import cosine_similarity


class MLMDataset(Dataset):
    """Dataset for masked language modeling with parallel text pairs.

    This dataset handles pairs of texts in different languages, preparing them
    for masked language modeling tasks during semantic alignment training.

    Args:
        first_texts (list): First set of texts in the parallel corpus
        second_texts (list): Second set of texts (translations) in the parallel corpus
        tokenizer: Tokenizer for text processing
    """

    def __init__(self, first_texts, second_texts, tokenizer):
        # Convert all texts to strings and handle potential NaN values
        self.first_texts = [str(text) for text in first_texts]
        self.second_texts = [str(text) for text in second_texts]
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        """Get a tokenized text pair at the specified index.

        Args:
            idx (int): Index of the text pair to retrieve

        Returns:
            dict: Tokenized inputs including input_ids and attention_mask
        """
        # Ensure we're passing strings to the tokenizer
        first_text = str(self.first_texts[idx])
        second_text = str(self.second_texts[idx])

        encodings = self.tokenizer(
            text=first_text,
            text_pair=second_text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encodings.items()}

    def __len__(self):
        """Get the total number of text pairs in the dataset.

        Returns:
            int: Number of text pairs
        """
        return len(self.first_texts)


def compute_pair_similarities(model, texts_1, texts_2, tokenizer, device, num_pairs=5):
    """Compute semantic similarities between pairs of texts using cosine similarity.

    Args:
        model: Model to use for text encoding
        texts_1 (list): First set of texts
        texts_2 (list): Second set of texts (parallel to first set)
        tokenizer: Tokenizer for processing texts
        device: Device to run computations on
        num_pairs (int, optional): Number of pairs to compute similarities for. Defaults to 5

    Returns:
        float: Average cosine similarity across the text pairs
    """
    model.eval()
    similarities = []

    with torch.no_grad():
        for t1, t2 in zip(texts_1[:num_pairs], texts_2[:num_pairs]):
            inputs1 = tokenizer(
                t1, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            inputs2 = tokenizer(
                t2, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            outputs1 = model.distilbert(**inputs1).last_hidden_state.mean(dim=1)
            outputs2 = model.distilbert(**inputs2).last_hidden_state.mean(dim=1)

            sim = cosine_similarity(outputs1, outputs2).item()
            similarities.append(sim)

    return sum(similarities) / len(similarities)


def semantic_alignment_training(
    model,
    language: str,
    select_samples: int,
    tokenizer,
    num_epochs: int = 2,
    batch_size: int = 8,
):
    """Train model for cross-lingual semantic alignment using masked language modeling.

    This function fine-tunes a model using parallel text pairs to improve the alignment
    between semantic representations across languages.

    Args:
        model: Model to fine-tune
        language (str): Target language code (e.g., "de", "fr")
        select_samples (int): Number of parallel text pairs to use for training
        tokenizer: Tokenizer for text processing
        num_epochs (int, optional): Number of training epochs. Defaults to 2
        batch_size (int, optional): Training batch size. Defaults to 8

    Returns:
        model: Fine-tuned model with improved cross-lingual alignment

    Raises:
        ValueError: If there aren't enough samples in the dataset
    """
    data_folder = "./data/semantic_alignment/"
    csv_path = os.path.join(data_folder, f"{language}.csv")
    df = pd.read_csv(csv_path)

    # Clean the DataFrame: remove any rows with NaN values
    df = df.dropna()

    if len(df) < select_samples:
        raise ValueError(f"Not enough samples in {csv_path}")

    df_selected = df.sample(n=select_samples, random_state=42).reset_index(drop=True)

    # Convert DataFrame columns to strings explicitly
    df_selected = df_selected.astype(str)

    half = select_samples // 2
    # For first half: (non-English, English)
    texts_lang_first = df_selected.iloc[:half][language].tolist()
    texts_en_first = df_selected.iloc[:half]["en"].tolist()
    # For second half: (English, non-English)
    texts_en_second = df_selected.iloc[half:]["en"].tolist()
    texts_lang_second = df_selected.iloc[half:][language].tolist()

    # Concatenate pairs accordingly
    first_texts = texts_lang_first + texts_en_second
    second_texts = texts_en_first + texts_lang_second

    # Identify device
    device = next(model.parameters()).device

    # Create dataset with raw texts
    dataset = MLMDataset(first_texts, second_texts, tokenizer)

    # Data collator that applies the masking.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator
    )

    # Get the base model through proper PEFT API
    base_model = model.distilbert

    # Freeze all parameters except those belonging to LoRA.
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Initialize MLM head and tie weights (but keep it frozen)
    mlm_head = nn.Linear(base_model.config.dim, base_model.config.vocab_size).to(device)
    mlm_head.weight = base_model.embeddings.word_embeddings.weight  # tie weights
    mlm_head.weight.requires_grad = False
    mlm_head.bias.requires_grad = False

    # Set up optimizer using only the trainable parameters.
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5
    )

    # Ensure model is in training mode
    model.train()
    base_model.train()
    mlm_head.eval()

    pbar = trange(num_epochs)
    for epoch in pbar:
        epoch_loss = 0
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.pop("labels")

            # Forward pass with gradient computation
            with torch.set_grad_enabled(True):
                outputs = base_model(**inputs)
                hidden_states = outputs.last_hidden_state
                logits = mlm_head(hidden_states)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()

        pbar.set_description(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.4f}")

    return model


def get_sample_texts(language: str, num_samples: int = 5):
    """Retrieve sample parallel texts from the dataset for evaluation.

    Args:
        language (str): Language code for the target language
        num_samples (int, optional): Number of text pairs to retrieve. Defaults to 5

    Returns:
        dict: Dictionary with language codes as keys and lists of texts as values
    """
    data_folder = "./data/semantic_alignment/"
    csv_path = os.path.join(data_folder, f"{language}.csv")
    df = pd.read_csv(csv_path)
    df = df.dropna()  # Remove any rows with NaN values
    df = df.head(num_samples)  # Get first num_samples rows
    return {language: df[language].tolist(), "en": df["en"].tolist()}


if __name__ == "__main__":
    """Demonstration script for semantic alignment training.

    This script shows how to:
    1. Initialize a model with LoRA adapters
    2. Load parallel text pairs
    3. Compute initial cross-lingual similarities
    4. Perform semantic alignment training
    5. Evaluate the improvement in cross-lingual alignment
    """
    model_path = "distilbert/distilbert-base-cased"
    model_path = "distilbert/distilbert-base-multilingual-cased"

    # Initialize model and tokenizer based on selection
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    language = "zh"

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Initialize the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=3,
    ).to(device)

    # Apply LoRA adapters with CLI arguments
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=(["q_lin", "v_lin"]),
    )
    model = get_peft_model(model, peft_config)
    model.to(device)
    model.print_trainable_parameters()

    # Get actual samples from dataset instead of hardcoded texts
    texts_to_compare = get_sample_texts(language=language, num_samples=5)

    # Compute initial similarities
    avg_sim_before = compute_pair_similarities(
        model,
        texts_to_compare[language],  # Use the language we're training on
        texts_to_compare["en"],
        tokenizer,
        device,
    )
    print(f"\nAverage similarity before training: {avg_sim_before:.4f}")

    # Run the semantic alignment training with same language
    model = semantic_alignment_training(
        model,
        language=language,
        select_samples=200,
        tokenizer=tokenizer,
        num_epochs=10,
        batch_size=8,
    )

    # Compute similarities after training with same text pairs
    print("\nComputing similarities after training...")
    avg_sim_after = compute_pair_similarities(
        model,
        texts_to_compare[language],  # Use the same texts as before
        texts_to_compare["en"],
        tokenizer,
        device,
    )
    print(f"\nAverage similarity before: {avg_sim_before:.4f}")
    print(f"Average similarity after:  {avg_sim_after:.4f}")
    print(f"Similarity improvement:    {avg_sim_after - avg_sim_before:.4f}")
