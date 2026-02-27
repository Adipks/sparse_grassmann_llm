from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Literal, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data.datasets import LMDatasetConfig, create_lm_dataloader
from models import GrassmannConfig, GrassmannLM, TransformerConfig, TransformerLM
from utils.tokenizer import train_or_load_tokenizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(
    model_type: Literal["transformer", "grassmann"],
    vocab_size: int,
    max_seq_len: int,
) -> nn.Module:
    if model_type == "transformer":
        cfg = TransformerConfig(vocab_size=vocab_size, max_seq_len=max_seq_len)
        return TransformerLM(cfg)
    cfg_g = GrassmannConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=512,
        n_layers=6,
        d_ff=2048,
        reduced_dim=32,
        window_sizes=[1, 2, 4, 8, 12, 16],
        dropout=0.1,
    )
    return GrassmannLM(cfg_g)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    max_steps: int | None = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    step = 0
    for batch in dataloader:
        step += 1
        if max_steps is not None and step > max_steps:
            break

        input_ids, targets = batch
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_tokens = targets.numel()
        total_loss += loss.item()
        total_tokens += batch_tokens

        if step % 10 == 0:
            elapsed = time.time() - start_time
            toks_per_sec = total_tokens / max(1e-8, elapsed)
            print(
                f"    step={step} loss={loss.item():.4f} "
                f"tokens={total_tokens} toks/sec={toks_per_sec:.1f}",
                flush=True,
            )

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, step)
    toks_per_sec = total_tokens / max(1e-8, elapsed)
    return avg_loss, toks_per_sec


@torch.no_grad()
def evaluate_lm(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / max(1, n_batches)
    perplexity = float(torch.exp(torch.tensor(avg_loss)))
    return perplexity


def main() -> None:
    parser = argparse.ArgumentParser(description="Wikitext-2 LM training for baseline and Grassmann models.")
    parser.add_argument("--model", choices=["transformer", "grassmann"], default="transformer")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--max-train-steps", type=int, default=0, help="Optional cap on train steps per epoch.")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}", flush=True)

    project_root = Path(__file__).resolve().parent.parent
    tokenizer_path = project_root / "data" / "tokenizer.json"
    print(f"Loading or training tokenizer at {tokenizer_path}", flush=True)
    tokenizer = train_or_load_tokenizer(tokenizer_path, vocab_size=8000)

    lm_cfg = LMDatasetConfig(seq_len=args.seq_len)
    train_loader = create_lm_dataloader(
        split="train",
        tokenizer=tokenizer,
        config=lm_cfg,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = create_lm_dataloader(
        split="validation",
        tokenizer=tokenizer,
        config=lm_cfg,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(args.model, vocab_size=tokenizer.vocab_size, max_seq_len=args.seq_len)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built {args.model} model with {n_params/1e6:.2f}M parameters", flush=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=device.type == "cuda")

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_ppl = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        max_steps = args.max_train_steps or None
        train_loss, toks_per_sec = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            max_steps=max_steps,
        )
        print(f"  train_loss={train_loss:.4f} tokens/sec={toks_per_sec:.1f}")

        val_ppl = evaluate_lm(model, val_loader, device)
        print(f"  val_perplexity={val_ppl:.2f}")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            ckpt_path = output_dir / f"{args.model}_lm.pt"
            torch.save(
                {
                    "model_type": args.model,
                    "config": {
                        "vocab_size": tokenizer.vocab_size,
                        "d_model": getattr(model.config, "d_model", None),
                        "n_layers": getattr(model.config, "n_layers", None),
                    },
                    "state_dict": model.state_dict(),
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()

