from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Literal, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data.datasets import LMDatasetConfig, create_lm_dataloader
from models import GrassmannConfig, GrassmannLM, TransformerConfig, TransformerLM
from models.blocks import MaskedLinear
from utils.tokenizer import train_or_load_tokenizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Pruning-Aware Training (PAT) helper
# Re-zeros masked weights after every optimizer step so the 2:4 sparsity
# pattern is enforced throughout training, not just at initialisation.
# ---------------------------------------------------------------------------
def reapply_masks(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, MaskedLinear):
            with torch.no_grad():
                module.weight.mul_(module.weight_mask)


def build_model(
    model_type: Literal["transformer", "grassmann"],
    vocab_size: int,
    max_seq_len: int,
    apply_sparse: bool = True,
) -> Tuple[nn.Module, object]:
    """Return (model, config) so callers can save the full config in checkpoints."""
    if model_type == "transformer":
        cfg = TransformerConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=432,        # matched to Grassmann ~21.75M → 21.64M params
            n_layers=8,
            n_heads=8,
            d_ff=432 * 4,
            dropout=0.1,
        )
        return TransformerLM(cfg), cfg
    cfg_g = GrassmannConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=512,
        n_layers=6,
        d_ff=2048,
        reduced_dim=32,
        window_sizes=[1, 2, 4, 8, 16, 32],   # covers 2048-token contexts
        dropout=0.1,
        apply_sparse=apply_sparse,
    )
    return GrassmannLM(cfg_g), cfg_g


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_steps: int | None = None,
    is_sparse: bool = False,
    grad_clip: float = 1.0,
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

        # Unscale before clipping so the clip threshold is in the original scale
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # --- PAT: re-enforce 2:4 sparsity pattern after every optimizer step ---
        if is_sparse:
            reapply_masks(model)

        if scheduler is not None:
            scheduler.step()

        batch_tokens = targets.numel()
        total_loss += loss.item()
        total_tokens += batch_tokens

        if step % 10 == 0:
            elapsed = time.time() - start_time
            toks_per_sec = total_tokens / max(1e-8, elapsed)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"    step={step} loss={loss.item():.4f} lr={current_lr:.2e} "
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
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear LR warmup steps.")
    parser.add_argument("--total-epochs", type=int, default=0,
                        help="Total planned epochs across ALL runs (used to build the cosine schedule). "
                             "If 0, defaults to --epochs. Set this consistently across resume runs.")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--max-train-steps", type=int, default=0, help="Optional cap on train steps per epoch (smoke-test).")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint if available.")
    parser.add_argument("--no-sparse", action="store_true",
                        help="Disable 2:4 sparsity masks (dense Grassmann ablation). "
                             "Checkpoint saved as <model>_dense_lm.pt.")
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

    model, cfg = build_model(args.model, vocab_size=tokenizer.vocab_size, max_seq_len=args.seq_len,
                             apply_sparse=(not args.no_sparse))
    is_sparse = args.model == "grassmann" and not args.no_sparse

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built {args.model} model with {n_params/1e6:.2f}M parameters", flush=True)

    # Count and report sparse layers
    sparse_layers = [m for m in model.modules() if isinstance(m, MaskedLinear)]
    if sparse_layers:
        total_w = sum(m.weight.numel() for m in sparse_layers)
        zero_w  = sum((m.weight_mask == 0).sum().item() for m in sparse_layers)
        print(f"  MaskedLinear layers: {len(sparse_layers)} | "
              f"sparsity: {zero_w}/{total_w} = {100*zero_w/total_w:.1f}%", flush=True)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scaler = GradScaler(enabled=device.type == "cuda")

    # ------------------------------------------------------------------
    # Resume from existing checkpoint (loads weights + optimizer state)
    # Must happen BEFORE building the scheduler so we know start_epoch
    # and the original total_epochs saved in the checkpoint.
    # ------------------------------------------------------------------
    start_epoch = 1
    best_val_ppl = float("inf")
    ckpt_total_epochs = 0  # will be overwritten if checkpoint has it

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_suffix = "_dense_lm.pt" if args.no_sparse else "_lm.pt"
    ckpt_path = output_dir / f"{args.model}{ckpt_suffix}"

    if args.resume and ckpt_path.exists():
        print(f"Resuming from checkpoint: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("  Loaded optimizer state.", flush=True)
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
            print(f"  Resuming from epoch {start_epoch}", flush=True)
        if "best_val_ppl" in ckpt:
            best_val_ppl = ckpt["best_val_ppl"]
            print(f"  Previous best val_ppl: {best_val_ppl:.2f}", flush=True)
        if "total_epochs" in ckpt:
            ckpt_total_epochs = ckpt["total_epochs"]

        # PAT: re-derive masks from loaded weights (weight_mask is non-persistent
        # so it is not in the checkpoint). Zeros in weights == masked positions.
        if is_sparse:
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, MaskedLinear):
                        module.weight_mask.data = (module.weight != 0).float()
            print("  Re-derived 2:4 sparsity masks from loaded weights.", flush=True)
    else:
        if args.resume:
            print(f"No checkpoint found at {ckpt_path}, starting from scratch.", flush=True)

    # ------------------------------------------------------------------
    # Cosine LR schedule — built AFTER loading checkpoint so we can
    # account for the full planned training horizon correctly.
    #
    # Priority for total_epochs:
    #   1. --total-epochs CLI flag (explicit override)
    #   2. total_epochs saved in checkpoint
    #   3. start_epoch - 1 + args.epochs  (best guess from current run)
    # ------------------------------------------------------------------
    end_epoch = start_epoch + args.epochs - 1
    if args.total_epochs > 0:
        total_epochs = args.total_epochs
    elif ckpt_total_epochs > 0:
        total_epochs = max(ckpt_total_epochs, end_epoch)
    else:
        total_epochs = end_epoch

    steps_per_epoch = len(train_loader)
    total_steps  = total_epochs  * steps_per_epoch
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    # global step offset so the schedule continues from where it left off
    global_step_offset = (start_epoch - 1) * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        g = current_step + global_step_offset  # global step
        if g < warmup_steps:
            return float(g) / float(max(1, warmup_steps))
        progress = float(g - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(
        f"  LR schedule: warmup={warmup_steps} steps | total={total_steps} steps "
        f"| epochs {start_epoch}→{end_epoch} of {total_epochs}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, end_epoch + 1):
        print(f"Epoch {epoch}/{total_epochs}")
        max_steps = args.max_train_steps or None
        train_loss, toks_per_sec = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            scheduler=scheduler,
            max_steps=max_steps,
            is_sparse=is_sparse,
            grad_clip=args.grad_clip,
        )
        print(f"  train_loss={train_loss:.4f} tokens/sec={toks_per_sec:.1f}")

        val_ppl = evaluate_lm(model, val_loader, device)
        print(f"  val_perplexity={val_ppl:.2f}")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(
                {
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "model_type": args.model if not args.no_sparse else f"{args.model}_dense",
                    "best_val_ppl": best_val_ppl,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": {
                        "vocab_size": tokenizer.vocab_size,
                        "d_model": getattr(cfg, "d_model", None),
                        "n_layers": getattr(cfg, "n_layers", None),
                        "d_ff": getattr(cfg, "d_ff", None),
                        "reduced_dim": getattr(cfg, "reduced_dim", None),
                        "max_seq_len": getattr(cfg, "max_seq_len", None),
                        "apply_sparse": not args.no_sparse,
                    },
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best checkpoint to {ckpt_path} (val_ppl={best_val_ppl:.2f})")

    print(f"\nBest val perplexity: {best_val_ppl:.2f}")


if __name__ == "__main__":
    main()

