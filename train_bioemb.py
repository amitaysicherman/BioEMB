"""
Main script for training the BioEmb model.

This script orchestrates the entire training pipeline:
1.  Loads configuration from a YAML file.
2.  Sets up the dataset and data loaders.
3.  Initializes the pretrained encoder and tokenizer.
4.  Performs Residual Vector Quantization (RVQ) to create discrete targets.
5.  Builds a prefix trie for constrained decoding.
6.  Initializes the BioEmb model.
7.  Sets up the Hugging Face Trainer with arguments and callbacks.
8.  Launches the training and evaluation loop.
"""

import logging
import torch
import os
from transformers import Trainer, TrainingArguments
from tdc.single_pred import ADME
from tdc.single_pred import Tox

from bioemb.trie import build_trie_from_text
from bioemb.pretrained import get_model_and_tokenizer, QuantizeTokenizer, MODEL_TO_DIM
from bioemb.data_manager import BioEmbDataset
from bioemb.rvq import ResidualVectorQuantizer
from bioemb.downstream_eval import compute_downstream_metrics
from bioemb.models import BioEmbModel
from utils import get_config, EvalLoggingCallback, set_seed

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_device() -> torch.device:
    """Detects and returns the available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(dataset_name: str, seq_col: str, label_col: str, split_method: str) -> tuple:
    """Loads and splits the dataset using TDC."""
    logger.info(f"Loading dataset: {dataset_name}")
    try:
        data = ADME(name=dataset_name)
    except:  # try Tox
        data = Tox(name=dataset_name)
    logger.info(f"Dataset {dataset_name} loaded successfully.")
    split = data.get_split(method=split_method, seed=42, frac=[0.7, 0.1, 0.2])

    train_data = {
        'sequences': split["train"][seq_col].tolist(),
        'labels': split["train"][label_col].tolist()
    }
    validation_data = {
        'sequences': split["valid"][seq_col].tolist(),
        'labels': split["valid"][label_col].tolist()
    }
    test_data = {
        'sequences': split["test"][seq_col].tolist(),
        'labels': split["test"][label_col].tolist()
    }

    return train_data, validation_data, test_data


def main():
    """Main function to orchestrate the BioEmb training pipeline."""
    config = get_config()
    device = setup_device()
    logger.info(f"Using device: {device}")

    set_seed(config.get("random_seed", 42))

    # --- 1. Load Data ---
    train_data, validation_data, test_data = load_dataset(
        config["dataset"], config["seq_col_name"], config["label_col_name"], config["split_method"]
    )
    all_sequences = train_data['sequences'] + validation_data['sequences'] + test_data['sequences']
    all_labels = train_data['labels'] + validation_data['labels'] + test_data['labels']

    # --- 2. Initialize Models and Tokenizers ---
    encoder, src_tokenizer = get_model_and_tokenizer(
        config["model_type"], config["model_name"]
    )
    encoder.to(device)

    # --- 3. Run RVQ to get discrete targets ---
    logger.info("Starting Residual Vector Quantization (RVQ)...")
    rvq = ResidualVectorQuantizer(
        n_clusters=config["n_clusters"],
        model=encoder,
        tokenizer=src_tokenizer,
        random_fit=config.get("random_tgt", False)
    )
    rvq.fit(all_sequences)

    tgt_sequences_str = rvq.transform(all_sequences)
    train_tgt_sequences = rvq.transform(train_data['sequences'])
    test_tgt_sequences = rvq.transform(test_data['sequences'])

    tgt_tokenizer = QuantizeTokenizer(max_token=config["n_clusters"])

    # --- 4. Build Trie for Constrained Decoding ---
    logger.info("Building prefix trie for constrained decoding...")
    trie = build_trie_from_text(list(set(tgt_sequences_str)), tgt_tokenizer)

    # --- 5. Create Datasets ---
    encoder_dim = MODEL_TO_DIM.get(config["model_name"], 768)

    full_dataset = BioEmbDataset(
        src_texts=all_sequences,
        tgt_texts=tgt_sequences_str,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        labels=all_labels,
        src_encoder=encoder,
        pooling=config.get("pooling", True)
    )

    test_dataset = BioEmbDataset(
        src_texts=test_data['sequences'],
        tgt_texts=test_tgt_sequences,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        labels=test_data['labels'],
        src_encoder=encoder,
        pooling=config.get("pooling", True)
    )

    # --- 6. Initialize BioEmb Model ---
    logger.info("Initializing BioEmb model...")
    bioemb_model = BioEmbModel(
        trie=trie,
        encoder_dim=encoder_dim,
        bottleneck_dim=config["bottleneck_dim"],
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        dropout=config["dropout"],
        tokenizer=tgt_tokenizer,
        entropy_normalize=config.get("entropy_normalize", True),
        constraint_mode=config.get("constraint_mode", "always")
    ).to(device)

    logger.info(
        f"BioEmb model initialized. Trainable parameters: {sum(p.numel() for p in bioemb_model.parameters() if p.requires_grad):,}")

    # --- 7. Setup Trainer ---
    metrics_calculator = lambda eval_preds: compute_downstream_metrics(
        eval_preds,
        model=bioemb_model,
        train_dataset=train_data,
        validation_data=validation_data,
        test_dataset=test_dataset,
        bottleneck_dim=config["bottleneck_dim"],
        device=device
    )

    output_dir = os.path.join(config["output_dir"], config["dataset"])
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(config.get("logs_base_dir", "logs"), config["dataset"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logs_dir,
        eval_strategy="steps",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config.get("epochs", 10),  # More intuitive than steps
        logging_steps=config["log_steps"],
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=config.get("metric_for_best_model", "eval_downstream_auc"),
        report_to=[config.get("report_to", "tensorboard")],
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        save_safetensors=False
    )

    trainer = Trainer(
        model=bioemb_model,
        args=training_args,
        train_dataset=full_dataset,
        eval_dataset=test_dataset,
        compute_metrics=metrics_calculator,
        callbacks=[EvalLoggingCallback(output_dir=output_dir)]
    )

    # --- 8. Start Training ---
    logger.info("Starting BioEmb model training...")
    trainer.evaluate()
    trainer.train()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
