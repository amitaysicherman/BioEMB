"""
This module provides a Trie data structure implementation tailored for
use in constrained sequence generation.

The Trie stores valid sequences of discrete tokens (from I-RVQ). During
generation, it can be queried with a prefix to find all valid next tokens,
which are then used to create a mask for the decoder's output logits.
"""
import torch
from tqdm import tqdm
from typing import List, Dict, Any

TrieNode = Dict[int, Any]

class Trie:
    """
    A Trie data structure for storing and querying sequences of token IDs.
    """
    def __init__(self, sequences: List[List[int]]):
        """
        Initializes and builds the Trie from a list of token sequences.

        Args:
            sequences (List[List[int]]): A list of token ID sequences.
        """
        self.root: TrieNode = {}
        self.node_count = 0
        
        for seq in tqdm(sequences, desc="Building Trie"):
            self._insert(seq)

    def _insert(self, sequence: List[int]):
        """Inserts a single sequence into the Trie."""
        node = self.root
        for token in sequence:
            if token not in node:
                node[token] = {}
                self.node_count += 1
            node = node[token]

    def get_valid_next_tokens(self, prefix: List[int]) -> List[int]:
        """
        Finds all valid tokens that can follow a given prefix.

        Args:
            prefix (List[int]): The prefix sequence of token IDs.

        Returns:
            A list of valid next token IDs. Returns an empty list if the
            prefix is not in the trie or if it's a complete sequence.
        """
        node = self.root
        for token in prefix:
            if token not in node:
                return []
            node = node[token]
        return list(node.keys())


def build_trie_from_text(corpus: List[str], tokenizer: Any) -> Trie:
    """
    Utility function to build a Trie directly from a corpus of text.

    Args:
        corpus (List[str]): A list of strings (e.g., I-RVQ code sequences).
        tokenizer: The tokenizer to convert strings to token ID sequences.

    Returns:
        An initialized Trie object.
    """
    token_sequences = [
        tokenizer.encode(text) for text in corpus
    ]
    # We only need the token IDs, not special tokens for the trie structure itself.
    # The tokenizer handles BOS/EOS, but the trie just needs the body.
    cleaned_sequences = []
    for seq in token_sequences:
        cleaned_seq = [
            t for t in seq if t not in 
            [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]
        ]
        cleaned_sequences.append(cleaned_seq)
        
    return Trie(cleaned_sequences)


def build_mask_from_trie(
    trie: Trie,
    sequences: torch.Tensor,
    vocab_size: int
) -> torch.Tensor:
    """
    Generates a batch of masks indicating valid next tokens based on a trie.

    For each sequence in the batch and each position in that sequence, this
    function queries the trie to find valid next tokens and creates a mask.

    Args:
        trie (Trie): The trie containing valid sequences.
        sequences (torch.Tensor): A batch of input sequences (batch_size, seq_len).
        vocab_size (int): The total size of the target vocabulary.

    Returns:
        A float tensor mask of shape (batch_size, seq_len, vocab_size), where
        a value of 1.0 indicates a valid next token.
    """
    batch_size, seq_length = sequences.shape
    mask = torch.zeros(
        (batch_size, seq_length, vocab_size),
        dtype=torch.float32,
        device=sequences.device
    )

    for i in range(batch_size):
        for j in range(seq_length):
            prefix = sequences[i, :j+1].tolist()
            # The trie should not contain special tokens
            cleaned_prefix = [
                t for t in prefix if t < trie.root.get('bos_token_id', float('inf'))
            ]

            valid_next = trie.get_valid_next_tokens(cleaned_prefix)
            if valid_next:
                mask[i, j, valid_next] = 1.0
    
    return mask
