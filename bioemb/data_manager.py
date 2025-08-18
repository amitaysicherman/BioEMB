"""
This module provides the PyTorch Dataset class for the BioEmb project.

It handles tokenization of source and target sequences and, importantly,
manages the pre-computation and caching of source encoder outputs to
accelerate training when the source encoder is frozen.
"""
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import List, Dict, Any

class BioEmbDataset(TorchDataset):
    """
    PyTorch Dataset for BioEmb training.

    This class tokenizes source (e.g., SMILES/FASTA) and target (I-RVQ codes)
    sequences. When the source encoder is not being trained, it pre-computes
    and caches the encoder's output for each unique source sequence to avoid
    redundant computations during training, significantly speeding up the process.
    """

    def __init__(
        self,
        src_texts: List[str],
        tgt_texts: List[str],
        src_tokenizer: Any,
        tgt_tokenizer: Any,
        labels: List[int],
        src_encoder: torch.nn.Module,
        max_length: int = 256,
        pooling: bool = True
    ):
        """
        Initializes the dataset.

        Args:
            src_texts (List[str]): List of source sequences (e.g., SMILES).
            tgt_texts (List[str]): List of target I-RVQ code sequences.
            src_tokenizer: Tokenizer for the source sequences.
            tgt_tokenizer: Tokenizer for the target I-RVQ codes.
            labels (List[int]): List of downstream task labels.
            src_encoder (torch.nn.Module): The pretrained source encoder model.
            max_length (int): Maximum sequence length for tokenization.
            pooling (bool): Whether to pool the source encoder's last hidden state.
        """
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.src_encoder = src_encoder
        self.max_length = max_length
        self.pooling = pooling
        self._cache: Dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.src_texts)

    def _get_encoder_output(self, src_text: str) -> torch.Tensor:
        """
        Gets the source encoder output for a text, using a cache.
        This is the key optimization for training with a frozen encoder.
        """
        if src_text in self._cache:
            return self._cache[src_text]

        device = next(self.src_encoder.parameters()).device
        src_tokens = self.src_tokenizer(
            src_text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        src_tokens = {k: v.to(device) for k, v in src_tokens.items()}

        with torch.no_grad():
            encoder_output = self.src_encoder(**src_tokens)
        
        # Pool the output to get a fixed-size representation
        if self.pooling:
            last_hidden_state = encoder_output.last_hidden_state
            attention_mask = src_tokens['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = encoder_output.last_hidden_state

        # Cache the result on CPU to save GPU memory
        self._cache[src_text] = pooled_output.squeeze(0).cpu()
        return self._cache[src_text]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # Get pre-computed encoder output (from cache or by computing it)
        encoder_output = self._get_encoder_output(src_text)

        # Tokenize target sequence
        tgt_tokens = self.tgt_tokenizer(
            tgt_text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        
        # Prepare labels for loss calculation (shift and mask padding)
        decoder_labels = tgt_tokens["input_ids"].clone().squeeze(0)
        decoder_labels[decoder_labels == self.tgt_tokenizer.pad_token_id] = -100

        return {
            "encoder_outputs": encoder_output,
            "encoder_attention_mask": torch.ones(encoder_output.shape[0]), # Simplified mask after pooling
            "input_ids": tgt_tokens["input_ids"].squeeze(0),
            "attention_mask": tgt_tokens["attention_mask"].squeeze(0),
            "labels": decoder_labels,
            "downstream_label": self.labels[idx],
        }
