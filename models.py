"""
This module defines the core PyTorch models for the BioEmb framework.

- BioEmbModel: The main encoder-decoder architecture for self-constrained
  generative fine-tuning. It integrates the source encoder, a generative
  decoder, a bottleneck layer for embedding extraction, and the trie-based
  constrained decoding logic.
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertGenerationConfig, BertGenerationDecoder
from typing import Optional

from .trie import Trie, build_mask_from_trie

class BioEmbModel(nn.Module):
    """
    The main BioEmb model for self-constrained generative fine-tuning.

    This model integrates a source encoder with a generative decoder.
    It learns to reconstruct a unique discrete sequence of an entity from its
    own embedding. The final fine-tuned embedding is extracted from a
    bottleneck layer that processes the decoder's hidden states.
    """
    def __init__(
        self,
        encoder: nn.Module,
        trie: Trie,
        encoder_dim: int,
        bottleneck_dim: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        dropout: float,
        tgt_vocab_size: int,
        entropy_normalize: bool = True,
        constraint_mode: str = "always", # "always", "never", "eval_only"
    ):
        super().__init__()
        self.encoder = encoder
        self.trie = trie
        self.entropy_normalize = entropy_normalize
        self.constraint_mode = constraint_mode

        # Bottleneck layers to project encoder output to a compressed space
        self.to_bottleneck = nn.Linear(encoder_dim, bottleneck_dim)
        self.from_bottleneck = nn.Linear(bottleneck_dim, hidden_size)
        self.relu = nn.ReLU()

        # Generative decoder
        decoder_config = BertGenerationConfig(
            is_decoder=True,
            add_cross_attention=True,
            vocab_size=tgt_vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.decoder = BertGenerationDecoder(decoder_config)

    def _apply_trie_constraints(self, decoder_outputs, input_ids, labels):
        """Applies the trie mask to logits and calculates loss."""
        # The mask shape is (batch, seq_len, vocab_size)
        trie_mask = build_mask_from_trie(
            self.trie, input_ids, self.decoder.config.vocab_size
        )
        
        # We predict the next token, so shift mask and logits
        logits_for_loss = decoder_outputs.logits[:, :-1, :]
        mask_for_loss = trie_mask[:, :-1, :]

        # Identify steps with only one valid token (no real choice)
        is_deterministic_step = mask_for_loss.sum(dim=-1) <= 1
        
        # Create the penalty mask for invalid tokens
        penalty = torch.full_like(mask_for_loss, -1e9)
        final_mask = torch.where(mask_for_loss.bool(), 0.0, penalty)
        
        # Apply the mask to the logits
        constrained_logits = logits_for_loss + final_mask

        # Optionally apply information weighting to the loss
        if self.entropy_normalize and self.training:
            valid_token_count = mask_for_loss.sum(dim=-1)
            # Add epsilon for stability, log scale to dampen effect
            information_weights = torch.log(valid_token_count + 1e-6) + 1.0 
            constrained_logits = constrained_logits / information_weights.unsqueeze(-1)
        
        decoder_outputs.logits[:, :-1, :] = constrained_logits

        # Adjust labels to ignore deterministic steps
        if labels is not None:
            shifted_labels = labels[:, 1:].clone()
            shifted_labels[is_deterministic_step] = -100 # Ignore loss for these steps
            
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                constrained_logits.reshape(-1, constrained_logits.size(-1)),
                shifted_labels.reshape(-1)
            )
            decoder_outputs.loss = loss
        
        return decoder_outputs

    def forward(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_bottleneck: bool = False,
    ):
        """
        Forward pass for the BioEmb model.

        Args:
            src_input_ids: Token IDs for the source encoder.
            src_attention_mask: Attention mask for the source encoder.
            input_ids: Token IDs for the decoder (target sequence).
            attention_mask: Attention mask for the decoder.
            labels: Ground truth labels for loss calculation.
            return_bottleneck: If True, returns the bottleneck embedding instead
                               of decoder outputs. Used for downstream evaluation.

        Returns:
            If return_bottleneck is True, returns the bottleneck tensor.
            Otherwise, returns the standard Hugging Face model output object.
        """
        # 1. Get encoder hidden states
        encoder_outputs = self.encoder(
            input_ids=src_input_ids, attention_mask=src_attention_mask
        )
        # Use mean pooling of last hidden state for a fixed-size representation
        encoder_hidden_states = encoder_outputs.last_hidden_state.mean(dim=1)

        # 2. Pass through bottleneck
        bottleneck = self.to_bottleneck(encoder_hidden_states)
        if return_bottleneck:
            return bottleneck

        projected_encoder_states = self.relu(self.from_bottleneck(bottleneck))
        # Add a sequence dimension for cross-attention
        projected_encoder_states = projected_encoder_states.unsqueeze(1)

        # 3. Pass through decoder
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=projected_encoder_states,
        )

        # 4. Apply trie constraints if necessary
        should_constrain = (
            self.constraint_mode == "always" or
            (self.constraint_mode == "eval_only" and not self.training)
        )
        if should_constrain:
            return self._apply_trie_constraints(decoder_outputs, input_ids, labels)

        # 5. Calculate standard loss if not constraining
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            shifted_logits = decoder_outputs.logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            decoder_outputs.loss = loss_fct(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1)
            )
        
        return decoder_outputs

