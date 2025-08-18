"""
This module implements Residual Vector Quantization (RVQ).

RVQ is a multi-stage quantization method used here to convert
high-dimensional biochemical embeddings into unique, discrete integer sequences.
This process is a core component of the BioEmb framework, creating the
target sequences for the self-constrained generative task.
"""
import logging
import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import List, Dict, Any

# --- Setup Logging ---
logger = logging.getLogger(__name__)

class ResidualVectorQuantizer:
    """
    Implements the Residual Vector Quantization (RVQ) algorithm.

    In each stage (layer), a K-Means model is fitted to the residual vectors
    from the previous stage. This process continues iteratively until each
    input embedding is mapped to a unique sequence of cluster indices, or
    a stopping criterion (e.g., max layers, stagnation) is met.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        n_clusters: int = 15,
        max_layers: int = 50,
        random_fit: bool = False,
        kmeans_kwargs: Dict[str, Any] = None,
    ):
        """
        Initializes the RVQ instance.

        Args:
            model: The pretrained encoder model to generate embeddings.
            tokenizer: The tokenizer for the encoder.
            n_clusters (int): Number of clusters (K) for each K-Means layer.
            max_layers (int): Maximum number of quantization layers.
            random_fit (bool): If True, assign random codes instead of training
                               K-Means. Used for baseline/ablation studies.
            kmeans_kwargs (dict): Additional arguments for scikit-learn's KMeans.
        """
        if n_clusters <= 1:
            raise ValueError("n_clusters must be greater than 1.")
        
        self.model = model
        self.tokenizer = tokenizer
        self.n_clusters = n_clusters
        self.max_layers = max_layers
        self.random_fit = random_fit
        self.kmeans_kwargs = kmeans_kwargs or {}
        self.quantizers_: List[KMeans] = []
        self._is_fitted = False
        self.line_to_code_map: Dict[str, str] = {}

    def _generate_embeddings(self, lines: List[str], batch_size: int = 32) -> np.ndarray:
        """Generates embeddings for a list of input sequences."""
        logger.info(f"Generating embeddings for {len(lines)} sequences...")
        self.model.eval()
        device = next(self.model.parameters()).device
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(lines), batch_size), desc="Embedding Generation"):
                batch = lines[i: i + batch_size]
                tokens = self.tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=512
                ).to(device)
                
                outputs = self.model(**tokens)
                
                # Mean pooling of the last hidden state
                last_hidden = outputs.last_hidden_state
                attention_mask = tokens['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())
                sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                
                all_embeddings.append((sum_embeddings / sum_mask).cpu().numpy())

        return np.vstack(all_embeddings)

    def fit(self, lines: List[str]):
        """
        Fits the RVQ model to the data.

        Args:
            lines (List[str]): The input sequences (e.g., SMILES or FASTA).
        """
        unique_lines = sorted(list(set(lines)))
        embeddings = self._generate_embeddings(unique_lines)
        
        residuals = embeddings.copy()
        layered_labels = []
        num_samples = embeddings.shape[0]

        logger.info("Starting RVQ fitting process...")
        for layer_idx in range(self.max_layers):
            logger.info(f"Fitting layer {layer_idx + 1}/{self.max_layers}...")
            
            if self.random_fit:
                # For ablation: assign random cluster indices
                labels = np.random.randint(0, self.n_clusters, size=num_samples)
                quantized_vectors = np.zeros_like(residuals) # No real centroids
            else:
                kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    n_init='auto',
                    random_state=42 + layer_idx, # Ensure different init per layer
                    **self.kmeans_kwargs,
                )
                labels = kmeans.fit_predict(residuals)
                self.quantizers_.append(kmeans)
                quantized_vectors = kmeans.cluster_centers_[labels]

            layered_labels.append(labels)
            residuals -= quantized_vectors
            
            # Check for uniqueness
            codes_as_tuples = [tuple(c) for c in np.array(layered_labels).T]
            num_unique_codes = len(set(codes_as_tuples))
            logger.info(f"Layer {layer_idx + 1}: Found {num_unique_codes}/{num_samples} unique codes.")

            if num_unique_codes == num_samples:
                logger.info("All samples have unique codes. Stopping RVQ fitting.")
                break
        
        # Store the final mapping
        final_codes = np.array(layered_labels).T
        self.line_to_code_map = {
            line: " ".join(map(str, code))
            for line, code in zip(unique_lines, final_codes)
        }
        self._is_fitted = True
        logger.info(f"RVQ fitting complete with {len(self.quantizers_)} layers.")

    def transform(self, lines: List[str]) -> List[str]:
        """
        Transforms input sequences into their discrete RVQ code strings.

        Args:
            lines (List[str]): A list of input sequences.

        Returns:
            A list of space-separated code strings corresponding to the inputs.
        """
        if not self._is_fitted:
            raise RuntimeError("RVQ must be fitted before calling transform.")
        
        return [self.line_to_code_map[line] for line in lines]
