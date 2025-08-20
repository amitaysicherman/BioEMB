"""
This module provides the PyTorch Dataset class for the BioEmb project.

It handles tokenization of source and target sequences and, importantly,
manages the pre-computation and caching of source encoder outputs to
accelerate training when the source encoder is frozen.
"""
import torch
from torch.utils.data import Dataset as TorchDataset
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

dataset_to_task_type = {
    # Molecule datasets
    "BBB_Martins": "classification",
    "CYP2C19_Veith": "classification",
    "CYP2D6_Veith": "classification",
    "CYP3A4_Veith": "classification",
    "CYP1A2_Veith": "classification",
    "CYP2C9_Veith": "classification",
    "CYP2C9_Substrate_CarbonMangels": "classification",
    "CYP2D6_Substrate_CarbonMangels": "classification",
    "CYP3A4_Substrate_CarbonMangels": "classification",
    "AMES": "classification",
    "ClinTox": "classification",
    "Carcinogens_Lagunin": "classification",
    "PAMPA_NCATS": "classification",
    "HIA_Hou": "classification",
    "Pgp_Broccatelli": "classification",
    "Bioavailability_Ma": "classification",
    "hERG": "classification",
    "hERG_Karim": "classification",
    "DILI": "classification",
    "Skin Reaction": "classification",
    "Caco2_Wang": "regression",
    "Lipophilicity_AstraZeneca": "regression",
    "Solubility_AqSolDB": "regression",
    "HydrationFreeEnergy_FreeSolv": "regression",
    "PPBR_AZ": "regression",
    "VDss_Lombardo": "regression",
    "Half_Life_Obach": "regression",
    "Clearance_Hepatocyte_AZ": "regression",
    "LD50_Zhu": "regression",
    # Protein datasets
    "Solubility": "classification",
    "BinaryLocalization": "classification",
    "BetaLactamase":"regression",
    "Fluorescence":"regression",
    "Stability":"regression",
}



def get_protein_dataset_splits(dataset_name: str, max_length: int = 1024) -> tuple:
    from torchdrug import datasets, transforms

    dataset_class = getattr(datasets, dataset_name)
    transform = transforms.Compose([
        transforms.TruncateProtein(max_length=max_length, random=False),
        transforms.ProteinView(view="residue")
    ])
    dataset = dataset_class(path="~/torchdrug_data/",
                            atom_feature=None,
                            bond_feature=None,
                            residue_feature="default",
                            transform=transform)
    train_set, valid_set, test_set = dataset.split()
    splits_data = {"train": train_set, "valid": valid_set, "test": test_set}
    output = {}
    target_field = dataset_class.target_fields[0]
    for split_name, split_dataset in splits_data.items():
        sequences = [data['graph'].to_sequence().replace(".", "") for data in split_dataset]
        labels = [data[target_field] for data in split_dataset]
        output[split_name] = {
            "sequences": sequences,
            "labels": labels
        }
        print(f"Processed {split_name} split: Found {len(sequences)} sequences and {len(labels)} labels.")
    return output["train"], output["valid"], output["test"]


def load_molecule_dataset_splits(dataset_name: str, seq_col: str, label_col: str, split_method: str) -> tuple:
    from tdc.single_pred import ADME, Tox

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


def load_dataset(dataset_name: str, seq_col: str, label_col: str, split_method: str, data_type: str) -> tuple:
    """Loads and splits the dataset using TDC."""
    logger.info(f"Loading dataset: {dataset_name}")
    if data_type == "molecules":
        train_data, validation_data, test_data = load_molecule_dataset_splits(
            dataset_name, seq_col, label_col, split_method
        )
    elif data_type == "proteins":
        train_data, validation_data, test_data = get_protein_dataset_splits(dataset_name)
    else:
        raise ValueError(f"Unsupported data type: {data_type}. Supported types are 'molecules' and 'proteins'.")
    logger.info(f"Dataset split into train ({len(train_data['sequences'])}), "
                f"validation ({len(validation_data['sequences'])}), "
                f"and test ({len(test_data['sequences'])}) sets.")

    return train_data, validation_data, test_data


def get_dataset(sequences, tgt_sequences, labels, src_tokenizer, tgt_tokenizer, encoder, pooling=True):
    return BioEmbDataset(
        src_texts=sequences,
        tgt_texts=tgt_sequences,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        labels=labels,
        src_encoder=encoder,
        pooling=pooling
    )


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
            "input_ids": tgt_tokens["input_ids"].squeeze(0),
            "attention_mask": tgt_tokens["attention_mask"].squeeze(0),
            "labels": decoder_labels,
            "downstream_label": self.labels[idx],
        }
