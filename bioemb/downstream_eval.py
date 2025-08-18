"""
This module handles the evaluation of fine-tuned embeddings on downstream tasks.

It provides a standardized pipeline for training a simple classification or
regression head on top of the frozen embeddings produced by a model (like BioEmb
or an MLM baseline) and computes the relevant performance metrics (e.g., AUC, RMSE).
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm import tqdm

logger = logging.getLogger(__name__)

class PredictionHead(nn.Module):
    """A simple linear prediction head for classification or regression."""
    def __init__(self, bottleneck_dim: int, output_dim: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(bottleneck_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.head(x)

def train_prediction_head(
    model: nn.Module,
    train_dataset,
    test_dataset,
    bottleneck_dim: int,
    device: torch.device,
    task_type: str = 'classification',
    mlm: bool = False

):
    """
    Trains and evaluates a prediction head on top of frozen model embeddings.

    Args:
        model: The fine-tuned model (e.g., BioEmb) used to generate embeddings.
        train_dataset: The dataset for training the prediction head.
        test_dataset: The dataset for evaluating the prediction head.
        bottleneck_dim: The dimension of the embeddings from the model.
        device: The device to run training on.
        task_type: 'classification' or 'regression'.

    Returns:
        The best evaluation score (AUC or RMSE).
    """
    model.eval()
    
    # 1. Generate all embeddings first
    def get_embeddings(dataset):
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        all_embeds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Generating embeddings"):
                batch = {k: v.to(device) for k, v in batch.items()}
                if mlm:
                    output=model(batch['input_ids'],batch['attention_mask'])
                    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
                        bottleneck = output.pooler_output.cpu()
                    else:
                        last_hidden_state = output.last_hidden_state
                        attention_mask = batch['attention_mask']
                        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                        pooled_output = (last_hidden_state * mask_expanded).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                        bottleneck = pooled_output.cpu()
                else:
                    bottleneck = model(
                        encoder_outputs=batch['encoder_outputs'],
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=None,
                        return_bottleneck=True
                    )
                all_embeds.append(bottleneck.cpu())
                all_labels.append(batch['downstream_label'].cpu())
        return torch.cat(all_embeds), torch.cat(all_labels)

    train_embeds, train_labels = get_embeddings(train_dataset)
    test_embeds, test_labels = get_embeddings(test_dataset)

    # 2. Train a simple head on the embeddings
    head = PredictionHead(bottleneck_dim).to(device)
    optimizer = optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    
    if task_type == 'classification':
        criterion = nn.BCEWithLogitsLoss()
        best_score = 0.0
    else: # regression
        criterion = nn.MSELoss()
        best_score = float('inf')

    train_loader = DataLoader(TensorDataset(train_embeds, train_labels), batch_size=128, shuffle=True)
    
    for epoch in range(50): # Train for a fixed number of epochs
        head.train()
        for embeds, labels in train_loader:
            embeds, labels = embeds.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = head(embeds).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

        # 3. Evaluate the head
        head.eval()
        with torch.no_grad():
            test_preds = head(test_embeds.to(device)).squeeze().cpu().numpy()
            if task_type == 'classification':
                score = roc_auc_score(test_labels.numpy(), test_preds)
                best_score = max(best_score, score)
            else:
                score = np.sqrt(mean_squared_error(test_labels.numpy(), test_preds))
                best_score = min(best_score, score)

    return best_score


def compute_downstream_metrics(
    eval_preds, model, train_dataset, test_dataset, bottleneck_dim, device
) -> dict:
    """
    Computes both sequence generation and downstream task metrics.
    
    This function is designed as a `compute_metrics` callback for the
    Hugging Face Trainer.
    """
    results = {}
    
    # --- Sequence Generation Metrics ---
    if eval_preds:
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        
        # Shift preds and labels for accuracy calculation
        preds = preds[:, :-1]
        labels = labels[:, 1:]
        
        # Mask out padding tokens (-100)
        mask = labels != -100
        
        correct_tokens = np.sum((preds == labels) & mask)
        total_tokens = np.sum(mask)
        results["token_accuracy"] = correct_tokens / total_tokens if total_tokens > 0 else 0
        
        # Sample accuracy (exact match)
        correct_samples = sum(
            np.array_equal(p[m], l[m]) for p, l, m in zip(preds, labels, mask)
        )
        results["sample_accuracy"] = correct_samples / len(labels)

    # --- Downstream Task Metrics ---
    # Assuming classification for now, can be extended
    downstream_auc = train_prediction_head(
        model, train_dataset, test_dataset, bottleneck_dim, device, task_type='classification'
    )
    print("Downstream AUC:", downstream_auc)
    results["downstream_auc"] = downstream_auc
    
    return results
