from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from utils.tokenizer import TextTokenizer


@dataclass
class LMDatasetConfig:
    seq_len: int = 512


class Wikitext2LMDataset(Dataset):
    """
    Causal language modeling dataset built from Wikitext-2.
    Creates contiguous token streams and chunks them into fixed-length sequences.
    """

    def __init__(
        self,
        split: str,
        tokenizer: TextTokenizer,
        config: LMDatasetConfig,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split for Wikitext-2: {split}")
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = config.seq_len

        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Flatten into a single stream of token ids.
        token_ids: List[int] = []
        for ex in raw:
            text = ex.get("text", "")
            if not text or text.isspace():
                continue
            ids = tokenizer.encode(text)
            token_ids.extend(ids)

        # Chunk into (seq_len + 1) blocks for input/target pairs.
        inputs: List[List[int]] = []
        targets: List[List[int]] = []
        block_size = self.seq_len + 1
        for i in range(0, len(token_ids) - block_size + 1, block_size):
            block = token_ids[i : i + block_size]
            x = block[:-1]
            y = block[1:]
            inputs.append(x)
            targets.append(y)

        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


SNLI_LABEL_MAP: Dict[str, int] = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}


@dataclass
class SNLIDatasetConfig:
    max_seq_len: int = 256


class SNLIDataset(Dataset):
    """
    SNLI dataset for NLI classification.
    Uses the same tokenizer as the LM and produces padded input_ids and attention masks.
    """

    def __init__(
        self,
        split: str,
        tokenizer: TextTokenizer,
        config: SNLIDatasetConfig,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split for SNLI: {split}")

        self.tokenizer = tokenizer
        self.max_seq_len = config.max_seq_len

        raw = load_dataset("snli", split=split)

        self.examples: List[Dict[str, torch.Tensor]] = []

        for ex in raw:
            label = ex.get("label")
            # Some SNLI examples have label -1 / "-" meaning no gold label.
            if label is None or label == -1:
                continue

            premise = ex.get("premise")
            hypothesis = ex.get("hypothesis")
            if not premise or not hypothesis:
                continue

            label_str = raw.features["label"].int2str(label)
            if label_str not in SNLI_LABEL_MAP:
                continue

            ids = tokenizer.encode_pair(premise, hypothesis, add_special_tokens=True)
            if len(ids) > self.max_seq_len:
                ids = ids[: self.max_seq_len]

            attn_mask = [1] * len(ids)

            # Pad to max_seq_len.
            pad_length = self.max_seq_len - len(ids)
            if pad_length > 0:
                ids = ids + [tokenizer.pad_id] * pad_length
                attn_mask = attn_mask + [0] * pad_length

            self.examples.append(
                {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
                    "label": torch.tensor(SNLI_LABEL_MAP[label_str], dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def create_lm_dataloader(
    split: str,
    tokenizer: TextTokenizer,
    config: LMDatasetConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = Wikitext2LMDataset(split=split, tokenizer=tokenizer, config=config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_snli_dataloader(
    split: str,
    tokenizer: TextTokenizer,
    config: SNLIDatasetConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = SNLIDataset(split=split, tokenizer=tokenizer, config=config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

