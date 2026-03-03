"""
bench_inference.py — Section 8 of the project spec.

Benchmarks FP16/FP32 and INT8 variants of Grassmann vs Transformer models:
  - Tokens/sec at context lengths 128, 256, 512
  - Peak VRAM (MB) at each context length
  - Latency per token (ms)
  - Perplexity on Wikitext-2 validation set
  - Model weight size on disk (MB)

Usage:
    python eval/bench_inference.py
    python eval/bench_inference.py --seq-lens 128 256 512 --runs 30 --quantize
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_vram_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    return 0.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_model(
    model: nn.Module,
    seq_lens: List[int],
    device: torch.device,
    batch_size: int = 1,
    n_warmup: int = 5,
    n_runs: int = 30,
    vocab_size: int = 8000,
) -> Dict[int, Dict[str, float]]:
    model.eval()
    results: Dict[int, Dict[str, float]] = {}

    for seq_len in seq_lens:
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # --- warm-up ---
        for _ in range(n_warmup):
            _ = model(ids)
        _sync(device)

        # --- timed runs ---
        _reset_peak(device)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(ids)
        _sync(device)
        elapsed = time.perf_counter() - t0

        total_tokens = n_runs * batch_size * seq_len
        tps = total_tokens / elapsed
        latency_ms = (elapsed / n_runs) * 1000
        vram_mb = _peak_vram_mb(device)

        results[seq_len] = {
            "tokens_per_sec": round(tps, 1),
            "latency_ms_per_fwd": round(latency_ms, 3),
            "peak_vram_mb": round(vram_mb, 1),
        }
        print(
            f"  seq={seq_len:>5}  tps={tps:>9,.0f}  "
            f"lat={latency_ms:.1f}ms  vram={vram_mb:.0f}MB"
        )

    return results


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, str]:
    import sys
    project_root = ckpt_path.parent.parent
    sys.path.insert(0, str(project_root))

    from models import GrassmannConfig, GrassmannLM, TransformerConfig, TransformerLM
    from models.blocks import MaskedLinear

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_type: str = ckpt.get("model_type", ckpt_path.stem.split("_")[0])
    cfg_dict: dict = ckpt.get("config", {})

    if model_type in ("grassmann", "grassmann_dense"):
        apply_sparse = cfg_dict.get("apply_sparse", model_type == "grassmann")
        cfg = GrassmannConfig(
            vocab_size=cfg_dict.get("vocab_size", 8000),
            d_model=cfg_dict.get("d_model", 512),
            n_layers=cfg_dict.get("n_layers", 6),
            d_ff=cfg_dict.get("d_ff", 2048),
            reduced_dim=cfg_dict.get("reduced_dim", 32),
            max_seq_len=cfg_dict.get("max_seq_len", 512),
            apply_sparse=apply_sparse,
        )
        model = GrassmannLM(cfg)
    else:
        cfg = TransformerConfig(
            vocab_size=cfg_dict.get("vocab_size", 8000),
            d_model=cfg_dict.get("d_model", 432),
            n_layers=cfg_dict.get("n_layers", 8),
            n_heads=8,
            d_ff=cfg_dict.get("d_ff", 432 * 4),
            max_seq_len=cfg_dict.get("max_seq_len", 512),
        )
        model = TransformerLM(cfg)

    model.load_state_dict(ckpt["state_dict"])

    # weight_mask is non-persistent so it wasn't saved.  Re-derive it from
    # the loaded weights: wherever a weight is exactly 0 it was masked during
    # PAT, so the mask should be 0 there as well.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, MaskedLinear):
                module.weight_mask.data = (module.weight != 0).float()

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    weight_mb = ckpt_path.stat().st_size / 1e6
    print(f"  Loaded {model_type} ({n_params/1e6:.2f}M params, {weight_mb:.1f}MB on disk) from {ckpt_path.name}")
    return model, model_type


# ---------------------------------------------------------------------------
# INT8 quantization helper (PTQ)
# ---------------------------------------------------------------------------

def quantize_int8(model: nn.Module, calibration_input: torch.Tensor) -> nn.Module:
    """
    Apply dynamic INT8 quantization to all Linear layers.
    Dynamic quant doesn't need a calibration pass — weights are quantized
    statically, activations are quantized on-the-fly.
    """
    model_cpu = model.cpu()
    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized


# ---------------------------------------------------------------------------
# Perplexity on Wikitext-2 validation set
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_perplexity(
    model: nn.Module,
    device: torch.device,
    project_root: Path,
    seq_len: int = 512,
    batch_size: int = 8,
) -> float:
    """Compute perplexity on the Wikitext-2 validation split."""
    import sys
    sys.path.insert(0, str(project_root))
    from data.datasets import LMDatasetConfig, create_lm_dataloader
    from utils.tokenizer import train_or_load_tokenizer

    tokenizer_path = project_root / "data" / "tokenizer.json"
    tokenizer = train_or_load_tokenizer(tokenizer_path, vocab_size=8000)
    val_loader = create_lm_dataloader(
        split="validation",
        tokenizer=tokenizer,
        config=LMDatasetConfig(seq_len=seq_len),
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()
    total_loss = 0.0
    n_batches = 0
    for input_ids, targets in val_loader:
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
    return float(math.exp(avg_loss))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inference benchmark: Grassmann vs Transformer")
    parser.add_argument("--ckpt-dir",  type=str, default="checkpoints")
    parser.add_argument("--seq-lens",  type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--runs",      type=int, default=30)
    parser.add_argument("--warmup",    type=int, default=5)
    parser.add_argument("--quantize",  action="store_true", help="Also benchmark INT8 dynamic quantization")
    parser.add_argument("--output",    type=str, default="eval/bench_results.json")
    parser.add_argument("--ppl-seq-len", type=int, default=512, help="Seq len used for perplexity evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    project_root = Path(__file__).resolve().parent.parent
    ckpt_dir = project_root / args.ckpt_dir
    output_path = project_root / args.output

    # All checkpoints to benchmark (in display order)
    ckpt_names = [
        "grassmann_lm.pt",
        "grassmann_dense_lm.pt",
        "transformer_lm.pt",
    ]

    all_results: Dict[str, dict] = {}
    meta: Dict[str, dict] = {}   # per-model metadata (ppl, weight_mb)

    for ckpt_name in ckpt_names:
        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"[SKIP] {ckpt_path.name} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Benchmarking: {ckpt_name}")
        model, model_type = load_model(ckpt_path, device)
        weight_mb = round(ckpt_path.stat().st_size / 1e6, 1)

        # --- Perplexity on Wikitext-2 val ---
        # Use the model's own max_seq_len as a cap so old checkpoints don't crash.
        max_model_seq = getattr(getattr(model, "config", None), "max_seq_len", args.ppl_seq_len)
        ppl_seq = min(args.ppl_seq_len, max_model_seq)
        print(f"\n[Perplexity] evaluating on Wikitext-2 val (seq_len={ppl_seq})…")
        val_ppl = measure_perplexity(model, device, project_root, seq_len=ppl_seq)
        print(f"  val_ppl = {val_ppl:.2f}")
        meta[model_type] = {"val_ppl": round(val_ppl, 2), "weight_mb": weight_mb, "trained_seq_len": max_model_seq}

        # --- FP benchmark: only use seq_lens the model actually supports ---
        bench_seq_lens = [s for s in args.seq_lens if s <= max_model_seq]
        skipped = [s for s in args.seq_lens if s > max_model_seq]
        if skipped:
            print(f"  [note] skipping seq_lens {skipped} — model max_seq_len={max_model_seq}")
        print(f"\n[FP] seq_lens={bench_seq_lens}")
        fp_results = benchmark_model(
            model, bench_seq_lens, device,
            batch_size=args.batch_size,
            n_warmup=args.warmup,
            n_runs=args.runs,
        )
        label = model_type.replace("_", "-")
        all_results[f"{label}_fp"] = fp_results

        # --- INT8 dynamic quantization benchmark ---
        if args.quantize:
            print(f"\n[INT8 dynamic] seq_lens={args.seq_lens}")
            q_model = quantize_int8(model, calibration_input=None)
            q_device = torch.device("cpu")  # dynamic quant runs on CPU
            q_results = benchmark_model(
                q_model, args.seq_lens, q_device,
                batch_size=args.batch_size,
                n_warmup=args.warmup,
                n_runs=args.runs,
            )
            all_results[f"{label}_int8"] = q_results

    # --- Save JSON (results + metadata) ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_output = {"metadata": meta, "throughput": all_results}
    with open(output_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # --- Print metadata table ---
    print("\n" + "="*50)
    print(f"{'Model':<25} {'val_ppl':>9} {'weight_MB':>10}")
    print("-"*50)
    for mtype, m in meta.items():
        print(f"{mtype:<25} {m['val_ppl']:>9.2f} {m['weight_mb']:>10.1f}")

    # --- Print throughput table ---
    print("\n" + "="*70)
    print(f"{'Model':<25} {'seq_len':>8} {'tok/s':>10} {'lat(ms)':>10} {'VRAM(MB)':>10}")
    print("-"*70)
    for label, res in all_results.items():
        for seq_len, metrics in res.items():
            print(
                f"{label:<25} {seq_len:>8} "
                f"{metrics['tokens_per_sec']:>10,.0f} "
                f"{metrics['latency_ms_per_fwd']:>10.1f} "
                f"{metrics['peak_vram_mb']:>10.0f}"
            )

    # --- Save CSV for plotting ---
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seq_len", "tokens_per_sec", "latency_ms_per_fwd", "peak_vram_mb"])
        writer.writeheader()
        for label, res in all_results.items():
            for seq_len, metrics in res.items():
                writer.writerow({"model": label, "seq_len": seq_len, **metrics})
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
