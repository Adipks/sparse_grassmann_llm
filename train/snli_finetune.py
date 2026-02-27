from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data.datasets import SNLIDatasetConfig, create_snli_dataloader
from models import GrassmannConfig, GrassmannLM, TransformerConfig, TransformerLM
from utils.tokenizer import train_or_load_tokenizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_lm_backbone(
    model_type: Literal["transformer", "grassmann"],
    vocab_size: int,
    max_seq_len: int,
) -> nn.Module:
    if model_type == "transformer":
        cfg = TransformerConfig(vocab_size=vocab_size, max_seq_len=max_seq_len)
        return TransformerLM(cfg)
    cfg_g = GrassmannConfig(vocab_size=vocab_size, max_seq_len=max_seq_len)
    return GrassmannLM(cfg_g)


def encode_backbone(
    lm: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Run the LM up to the pre-softmax hidden states.

    Both TransformerLM and GrassmannLM share the attributes used here.
    """
    x = lm.token_embedding(input_ids)
    x = lm.pos_encoding(x)
    x = lm.dropout(x)

    for block in lm.blocks:
        # GrassmannBlock ignores attention_mask; TransformerBlock accepts it.
        try:
            x = block(x, attn_mask=None)
        except TypeError:
            x = block(x)

    x = lm.ln_f(x)
    return x


class SNLIClassifier(nn.Module):
    def __init__(self, lm_backbone: nn.Module, d_model: int, num_labels: int = 3) -> None:
        super().__init__()
        self.lm = lm_backbone
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = encode_backbone(self.lm, input_ids, attention_mask)
        # Use the last non-padded token representation as sentence pair embedding.
        lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = hidden[batch_indices, lengths]  # (batch, d_model)
        logits = self.classifier(pooled)
        return logits


def train_epoch(
    model: SNLIClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(input_ids, attention_mask)
            loss = nn.functional.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_snli(
    model: SNLIClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = nn.functional.cross_entropy(logits, labels)
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="SNLI fine-tuning for Transformer and Grassmann models.")
    parser.add_argument("--model", choices=["transformer", "grassmann"], default="transformer")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parent.parent
    tokenizer_path = project_root / "data" / "tokenizer.json"
    tokenizer = train_or_load_tokenizer(tokenizer_path, vocab_size=8000)

    snli_cfg = SNLIDatasetConfig(max_seq_len=args.max_seq_len)
    train_loader = create_snli_dataloader(
        split="train",
        tokenizer=tokenizer,
        config=snli_cfg,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = create_snli_dataloader(
        split="validation",
        tokenizer=tokenizer,
        config=snli_cfg,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Build LM backbone and load pre-trained weights if available.
    lm = build_lm_backbone(args.model, vocab_size=tokenizer.vocab_size, max_seq_len=args.max_seq_len)
    ckpt_path = project_root / "checkpoints" / f"{args.model}_lm.pt"
    if ckpt_path.exists():
        print(f"Loading LM weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        lm.load_state_dict(ckpt["state_dict"])
    else:
        print(f"Warning: LM checkpoint {ckpt_path} not found; fine-tuning from scratch.")

    lm.to(device)

    d_model = lm.config.d_model  # type: ignore[attr-defined]
    model = SNLIClassifier(lm, d_model=d_model, num_labels=3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=device.type == "cuda")

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_acc = evaluate_snli(model, val_loader, device)
        print(f"  train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / f"{args.model}_snli.pt"
            torch.save(
                {
                    "model_type": args.model,
                    "state_dict": model.state_dict(),
                    "d_model": d_model,
                },
                ckpt_path,
            )
            print(f"  Saved SNLI checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()

