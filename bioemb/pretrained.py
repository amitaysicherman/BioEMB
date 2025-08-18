"""
This module manages the loading of pretrained models and tokenizers,
and defines a custom tokenizer for the discrete I-RVQ codes.
"""

from transformers import AutoTokenizer, AutoModel
import torch
from typing import Dict, List, Tuple

# --- Constants ---
MODEL_TO_DIM = {
    "ibm/MoLFormer-XL-both-10pct": 768,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "seyonec/ChemBERTa-zinc-base-v1": 768,
    "Rostlab/prot_bert": 1024
}

def get_model_and_tokenizer(model_type: str, model_name: str) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Loads a pretrained model and its corresponding tokenizer from Hugging Face.

    Args:
        model_type (str): The type of data ("molecules", "proteins").
                          Currently unused but kept for future compatibility.
        model_name (str): The Hugging Face model identifier.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    model_kwargs = {}
    if "MoLFormer" in model_name:
        model_kwargs["trust_remote_code"] = True
        model_kwargs["deterministic_eval"] = True

    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
    return model, tokenizer


class QuantizeTokenizer:
    """
    A custom tokenizer for handling the discrete integer codes from RVQ.

    This class mimics the behavior of a Hugging Face tokenizer for seamless
    integration with the rest of the pipeline. It defines special tokens
    (EOS, PAD, BOS) and handles the conversion between string-formatted
    sequences of codes and tensor representations.
    """
    def __init__(self, max_token: int = 15):
        self.max_token = max_token
        self.eos_token_id = max_token
        self.pad_token_id = max_token + 1
        self.bos_token_id = max_token + 2
        self.vocab_size = max_token + 3

    def get_vocab(self) -> Dict[int, int]:
        """Returns the vocabulary as a dictionary."""
        return {i: i for i in range(self.vocab_size)}

    def __call__(self, sequences: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a batch of code sequences.

        Args:
            sequences (List[str]): A list of space-separated code strings.

        Returns:
            A dictionary with 'input_ids' and 'attention_mask' tensors.
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        tokenized_batch = []
        for seq_str in sequences:
            tokens = [int(x) for x in seq_str.split()]
            tokenized_seq = [self.bos_token_id] + tokens + [self.eos_token_id]
            tokenized_batch.append(torch.tensor(tokenized_seq, dtype=torch.long))
        
        # Pad the batch to the longest sequence
        padded_ids = torch.nn.utils.rnn.pad_sequence(
            tokenized_batch, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask = (padded_ids != self.pad_token_id).long()
        
        return {"input_ids": padded_ids, "attention_mask": attention_mask}

    def encode(self, sequence: str, **kwargs) -> List[int]:
        """Encodes a single sequence string into a list of token IDs."""
        return self(sequence, **kwargs)["input_ids"][0].tolist()

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs back to a space-separated string."""
        # Filter out special tokens for decoding
        cleaned_ids = [
            str(t) for t in token_ids 
            if t not in [self.bos_token_id, self.eos_token_id, self.pad_token_id]
        ]
        return " ".join(cleaned_ids)
