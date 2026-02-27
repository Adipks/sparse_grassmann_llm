from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "sep_token": "<sep>",
}


class TextTokenizer:
    """
    Lightweight wrapper around a Hugging Face `tokenizers.Tokenizer`
    with helpers for causal LM and pairwise inputs (SNLI).
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.token_to_id(SPECIAL_TOKENS["pad_token"])
        self.unk_id = self.tokenizer.token_to_id(SPECIAL_TOKENS["unk_token"])
        self.bos_id = self.tokenizer.token_to_id(SPECIAL_TOKENS["bos_token"])
        self.eos_id = self.tokenizer.token_to_id(SPECIAL_TOKENS["eos_token"])
        self.sep_id = self.tokenizer.token_to_id(SPECIAL_TOKENS["sep_token"])

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        if add_special_tokens:
            output = self.tokenizer.encode(text)
            return [self.bos_id] + output.ids + [self.eos_id]
        return self.tokenizer.encode(text).ids

    def encode_pair(
        self,
        text_a: str,
        text_b: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Encode a pair of sentences with a separator token for SNLI-style tasks.
        """
        if add_special_tokens:
            a = self.tokenizer.encode(text_a)
            b = self.tokenizer.encode(text_b)
            return [self.bos_id] + a.ids + [self.sep_id] + b.ids + [self.eos_id]
        # Fallback: simple concatenation without BOS/EOS/SEP.
        a = self.tokenizer.encode(text_a)
        b = self.tokenizer.encode(text_b)
        return a.ids + b.ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)


def _iter_wikitext2_text() -> Iterable[str]:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    for split in ("train", "validation", "test"):
        for ex in ds[split]:
            text = ex.get("text", "")
            if text and not text.isspace():
                yield text


def _iter_snli_text() -> Iterable[Tuple[str, str]]:
    ds = load_dataset("snli")
    for split in ("train", "validation", "test"):
        for ex in ds[split]:
            prem = ex.get("premise")
            hyp = ex.get("hypothesis")
            if prem and hyp:
                yield prem, hyp


def train_or_load_tokenizer(
    tokenizer_path: os.PathLike | str,
    vocab_size: int = 8000,
) -> TextTokenizer:
    """
    Train a BPE tokenizer on Wikitext-2 + SNLI (if not already present),
    then wrap it in `TextTokenizer`.
    """
    tokenizer_path = Path(tokenizer_path)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    if tokenizer_path.exists():
        tok = Tokenizer.from_file(str(tokenizer_path))
        return TextTokenizer(tok)

    # Build a new tokenizer.
    tok = Tokenizer(BPE(unk_token=SPECIAL_TOKENS["unk_token"]))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = Whitespace()

    special_tokens = list(SPECIAL_TOKENS.values())

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens,
    )

    def batch_iterator() -> Iterable[str]:
        for text in _iter_wikitext2_text():
            yield text
        for prem, hyp in _iter_snli_text():
            yield prem
            yield hyp

    tok.train_from_iterator(batch_iterator(), trainer=trainer)

    # Set post-processor for BOS/EOS handling (we still manually add them in encode).
    tok.post_processor = TemplateProcessing(
        single=f"{SPECIAL_TOKENS['bos_token']} $0 {SPECIAL_TOKENS['eos_token']}",
        pair=(
            f"{SPECIAL_TOKENS['bos_token']} $A {SPECIAL_TOKENS['sep_token']} "
            f"$B {SPECIAL_TOKENS['eos_token']}"
        ),
        special_tokens=[(t, tok.token_to_id(t)) for t in special_tokens],
    )

    tok.save(str(tokenizer_path))
    return TextTokenizer(tok)

