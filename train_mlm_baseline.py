"""
Script for fine-tuning a pretrained encoder using the Masked Language
Modeling (MLM) objective.

This script serves as a baseline to compare against the BioEmb fine-tuning
approach. It performs the following steps:
1.  Loads a configuration file.
2.  Loads the specified pretrained model and tokenizer.
3.  Loads the dataset and tokenizes it for MLM.
4.  Sets up a Hugging Face Trainer with a data collator for MLM.
5.  Runs the fine-tuning process.
6.  Evaluates the fine-tuned encoder on a downstream task.
"""
import logging
import torch
from transformers import (
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

from utils import get_config, set_seed
from bioemb.pretrained import get_model_and_tokenizer
from bioemb.downstream_eval import train_prediction_head
# (Assuming you have a function to load data similar to train_bioemb.py)
from train_bioemb import load_dataset, setup_device

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """Main function to orchestrate the MLM fine-tuning and evaluation."""
    config = get_config()
    device = setup_device()
    logger.info(f"Using device: {device}")

    set_seed(config.get("random_seed", 42))

    # --- 1. Load Model and Tokenizer ---
    logger.info(f"Loading base model: {config['model_name']}")
    model, tokenizer = get_model_and_tokenizer(
        config["model_type"], config["model_name"]
    )
    # We need the MLM head for this training
    mlm_model = AutoModelForMaskedLM.from_pretrained(config['model_name']).to(device)

    # --- 2. Load and Prepare Data ---
    train_data, test_data = load_dataset(
        config["dataset"], config["seq_col_name"], config["label_col_name"]
    )
    all_sequences = train_data['sequences'] + test_data['sequences']
    
    hf_dataset = Dataset.from_dict({"text": all_sequences})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=256, padding="max_length"
        )

    logger.info("Tokenizing dataset for MLM...")
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Data collator will create MLM labels on-the-fly
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # --- 3. Define Training Arguments ---
    output_dir = config["output_dir"] + "_mlm_baseline"
    logs_dir = os.path.join(config.get("logs_base_dir", "logs"), f"{config['dataset']}_mlm")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.get("mlm_epochs", 5),
        per_device_train_batch_size=config.get("mlm_batch_size", 16),
        save_strategy="epoch",
        logging_dir=logs_dir,
        logging_steps=100,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # --- 4. Initialize Trainer and Start Training ---
    trainer = Trainer(
        model=mlm_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    logger.info("Starting MLM fine-tuning...")
    trainer.train()
    logger.info("MLM fine-tuning complete.")

    # --- 5. Evaluate on Downstream Task ---
    logger.info("Evaluating fine-tuned model on downstream task...")
    
    # We only need the base encoder part for downstream tasks
    finetuned_encoder = mlm_model.base_model 
    
    # Re-create PyTorch datasets for the evaluation function
    # (This part would need to be adapted based on your data_manager)
    from bioemb.data_manager import BioEmbDataset
    train_eval_dataset = BioEmbDataset(
        src_texts=train_data['sequences'],
        tgt_texts=[""]*len(train_data['sequences']), # Dummy targets
        src_tokenizer=tokenizer,
        tgt_tokenizer=None, # Not needed
        labels=train_data['labels'],
        src_encoder=finetuned_encoder,
    )
    test_eval_dataset = BioEmbDataset(
        src_texts=test_data['sequences'],
        tgt_texts=[""]*len(test_data['sequences']),
        src_tokenizer=tokenizer,
        tgt_tokenizer=None,
        labels=test_data['labels'],
        src_encoder=finetuned_encoder,
    )
    
    bottleneck_dim = finetuned_encoder.config.hidden_size
    final_score = train_prediction_head(
        finetuned_encoder, # Pass the model to generate embeddings
        train_eval_dataset,
        test_eval_dataset,
        bottleneck_dim,
        device,
        task_type='classification' # Assuming classification
    )

    logger.info(f"Downstream task score (AUC) for MLM-finetuned model: {final_score:.4f}")

if __name__ == "__main__":
    main()
