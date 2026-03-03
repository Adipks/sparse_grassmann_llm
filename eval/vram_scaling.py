"""
eval/vram_scaling.py  —  VRAM & throughput vs sequence-length benchmark.

For each model measures, at every seq_len in [64, 128, 256, 512, 1024, 2048]:
  • Peak VRAM (MB)
  • Tokens / sec
  • Latency per forward pass (ms)

Key architectural difference this exposes
──────────────────────────────────────────
  Transformer:  KV-cache grows O(n) and attention is O(n²) → VRAM grows
  Grassmann:    No KV-cache, sliding-window mixing O(n)    → VRAM stays flat

Positional-embedding extension
───────────────────────────────
Old 512-trained checkpoints only have a 512-row embedding table.
When testing at seq_len > 512 we linearly interpolate the table to the
target length so the model runs without crashing.
Quality will be imperfect at those lengths — the banner says so clearly.
After retraining with --seq-len 2048 this becomes moot.

Usage
─────
    cd sparse_grassmann_llm
    $env:PYTHONPATH = "."
    .venv\\Scripts\\python.exe eval/vram_scaling.py
    .venv\\Scripts\\python.exe eval/vram_scaling.py --seq-lens 64 128 256 512 1024 2048
    .venv\\Scripts\\python.exe eval/vram_scaling.py --no-plots      # tables only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.blocks import LearnedPositionalEmbedding, MaskedLinear, PositionalEncodingConfig
from models.grassmann_sparse import GrassmannConfig, GrassmannLM
from models.transformer_baseline import TransformerConfig, TransformerLM
from utils.tokenizer import train_or_load_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Positional embedding extension
# ─────────────────────────────────────────────────────────────────────────────

def _extend_pos_embedding(model: nn.Module, target_seq_len: int) -> None:
    """
    Extend every LearnedPositionalEmbedding whose table is smaller than
    target_seq_len.  Extra rows are linearly interpolated from the existing
    table so the model doesn't crash on longer sequences.
    """
    for module in model.modules():
        if not isinstance(module, LearnedPositionalEmbedding):
            continue
        old_len = module.max_seq_len
        if old_len >= target_seq_len:
            continue

        # Interpolate: (1, old_len, d_model) → (1, target_seq_len, d_model)
        old_w = module.embedding.weight.data          # (old_len, d_model)
        d_model = old_w.shape[1]
        # F.interpolate needs (N, C, L) format
        interp = F.interpolate(
            old_w.T.unsqueeze(0),                    # (1, d_model, old_len)
            size=target_seq_len,
            mode="linear",
            align_corners=True,
        ).squeeze(0).T                                # (target_seq_len, d_model)

        new_emb = nn.Embedding(target_seq_len, d_model)
        new_emb.weight = nn.Parameter(interp.to(old_w.device))
        module.embedding = new_emb
        module.max_seq_len = target_seq_len


def _rederive_masks(model: nn.Module) -> None:
    """Re-derive 2:4 masks from loaded weights (zeros in weight → mask=0)."""
    for module in model.modules():
        if isinstance(module, MaskedLinear):
            module.weight_mask.data = (module.weight.data != 0).float()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    ckpt_path: Path,
    vocab_size: int,
    target_seq_len: int,
    device: torch.device,
) -> tuple[nn.Module, str, bool]:
    """
    Load a checkpoint, extending its positional embedding if needed.
    Returns (model, model_type, was_extended).
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_type: str = payload.get("model_type", ckpt_path.stem.rsplit("_lm", 1)[0])
    cfg_dict: dict = payload.get("config", {})

    trained_seq_len = cfg_dict.get("max_seq_len", 512)

    if model_type in ("grassmann", "grassmann_dense"):
        apply_sparse = cfg_dict.get("apply_sparse", model_type == "grassmann")
        cfg = GrassmannConfig(
            vocab_size=cfg_dict.get("vocab_size", vocab_size),
            d_model=cfg_dict.get("d_model", 512),
            n_layers=cfg_dict.get("n_layers", 6),
            d_ff=cfg_dict.get("d_ff", 2048),
            reduced_dim=cfg_dict.get("reduced_dim", 32),
            max_seq_len=trained_seq_len,
            apply_sparse=apply_sparse,
        )
        model = GrassmannLM(cfg)
        model.load_state_dict(payload["state_dict"], strict=True)
        if apply_sparse:
            _rederive_masks(model)

    else:  # transformer
        cfg = TransformerConfig(
            vocab_size=cfg_dict.get("vocab_size", vocab_size),
            d_model=cfg_dict.get("d_model", 432),
            n_layers=cfg_dict.get("n_layers", 8),
            n_heads=cfg_dict.get("n_heads", 8),
            d_ff=cfg_dict.get("d_ff", 432 * 4),
            max_seq_len=trained_seq_len,
        )
        model = TransformerLM(cfg)
        model.load_state_dict(payload["state_dict"], strict=True)

    # Extend pos embedding if we need to run beyond trained length
    was_extended = target_seq_len > trained_seq_len
    if was_extended:
        _extend_pos_embedding(model, target_seq_len)

    model.to(device).eval()
    return model, model_type, was_extended


# ─────────────────────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_one(
    model: nn.Module,
    seq_len: int,
    device: torch.device,
    batch_size: int = 1,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> dict:
    ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

    # warm-up (outside timing window)
    for _ in range(n_warmup):
        _ = model(ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model(ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    vram_mb = (torch.cuda.max_memory_allocated(device) / 1e6
               if device.type == "cuda" else 0.0)
    tps = (n_runs * batch_size * seq_len) / elapsed
    lat = elapsed / n_runs * 1000

    return {
        "peak_vram_mb":   round(vram_mb, 1),
        "tokens_per_sec": round(tps, 1),
        "latency_ms":     round(lat, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY_NAMES = {
    "grassmann":       "Grassmann+PAT (O(n), no KV-cache)",
    "grassmann_dense": "Grassmann Dense (O(n), no KV-cache)",
    "transformer":     "Transformer     (O(n²), KV-cache)",
}

CKPT_NAMES = {
    "grassmann":       "grassmann_lm.pt",
    "grassmann_dense": "grassmann_dense_lm.pt",
    "transformer":     "transformer_lm.pt",
}


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = ROOT / args.ckpt_dir
    out_dir  = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = train_or_load_tokenizer(
        ROOT / "data" / "tokenizer.json", vocab_size=8000
    )
    vocab_size = tokenizer.vocab_size

    seq_lens = sorted(args.seq_lens)
    model_keys = ["grassmann", "grassmann_dense", "transformer"]

    all_rows: list[dict] = []

    for key in model_keys:
        ckpt_path = ckpt_dir / CKPT_NAMES[key]
        if not ckpt_path.exists():
            print(f"\n[SKIP] {ckpt_path.name} not found")
            continue

        print(f"\n{'═'*70}")
        print(f"  {DISPLAY_NAMES[key]}")
        print(f"  checkpoint: {ckpt_path.name}")
        print(f"{'─'*70}")
        print(f"  {'seq_len':>8}  {'VRAM (MB)':>10}  {'tok/s':>12}  {'lat (ms)':>10}  note")
        print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*20}")

        for seq_len in seq_lens:
            try:
                model, _, extended = load_model(
                    ckpt_path, vocab_size, seq_len, device
                )
                m = measure_one(
                    model, seq_len, device,
                    batch_size=args.batch_size,
                    n_warmup=args.n_warmup,
                    n_runs=args.n_runs,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                note = "pos-interp (retrain for best quality)" if extended else ""
                row = {"model": key, "seq_len": seq_len, "extended": extended, **m}
                all_rows.append(row)

                print(
                    f"  {seq_len:>8}  "
                    f"{m['peak_vram_mb']:>9.1f}  "
                    f"{m['tokens_per_sec']:>12,.0f}  "
                    f"{m['latency_ms']:>9.2f}  "
                    f"{note}"
                )

            except torch.cuda.OutOfMemoryError:
                print(f"  {seq_len:>8}  {'OOM':>10}  {'---':>12}  {'---':>10}")
                all_rows.append({
                    "model": key, "seq_len": seq_len, "extended": False,
                    "peak_vram_mb": None, "tokens_per_sec": None, "latency_ms": None,
                })
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # ── Save CSV / JSON ───────────────────────────────────────────────────────
    csv_path = out_dir / "vram_scaling.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = out_dir / "vram_scaling.json"
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)

    print(f"\n✓  Results → {out_dir}")
    print(f"   {csv_path.name}  |  {json_path.name}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        _plot(all_rows, seq_lens, out_dir)


def _plot(rows: list[dict], seq_lens: list[int], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("\n[warn] matplotlib not installed — skipping plots")
        print("       pip install matplotlib")
        return

    COLORS = {
        "grassmann":       "#2ecc71",   # green  — Grassmann+PAT
        "grassmann_dense": "#3498db",   # blue   — Grassmann dense
        "transformer":     "#e74c3c",   # red    — Transformer
    }
    LINESTYLES = {
        "grassmann":       "-",
        "grassmann_dense": "--",
        "transformer":     "-.",
    }

    def _get_series(metric: str):
        series = {}
        for key in DISPLAY_NAMES:
            xs, ys = [], []
            for r in rows:
                if r["model"] == key and r.get(metric) is not None:
                    xs.append(r["seq_len"])
                    ys.append(r[metric])
            if xs:
                series[key] = (xs, ys)
        return series

    # ── Figure 1: VRAM vs seq_len ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, (xs, ys) in _get_series("peak_vram_mb").items():
        ax.plot(xs, ys,
                marker="o", linewidth=2.5, markersize=8,
                color=COLORS[key],
                linestyle=LINESTYLES[key],
                label=DISPLAY_NAMES[key])

    # Annotate the architectural difference
    ax.annotate(
        "Transformer: KV-cache\n→ VRAM grows with n",
        xy=(seq_lens[-1], next(
            (r["peak_vram_mb"] for r in reversed(rows)
             if r["model"] == "transformer" and r.get("peak_vram_mb")), 200
        )),
        xytext=(-160, -30), textcoords="offset points",
        fontsize=9, color="#e74c3c",
        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2),
    )
    ax.annotate(
        "Grassmann: no KV-cache\n→ VRAM stays flat",
        xy=(seq_lens[-1], next(
            (r["peak_vram_mb"] for r in reversed(rows)
             if r["model"] == "grassmann" and r.get("peak_vram_mb")), 180
        )),
        xytext=(-170, 20), textcoords="offset points",
        fontsize=9, color="#2ecc71",
        arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.2),
    )

    ax.set_xlabel("Sequence Length (tokens)", fontsize=13)
    ax.set_ylabel("Peak VRAM (MB)", fontsize=13)
    ax.set_title("VRAM Usage vs Sequence Length  (batch=1, FP32)", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    p = out_dir / "fig_vram_vs_seqlen.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"✓  Plot → {p}")

    # ── Figure 2: Throughput (tok/s) vs seq_len ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, (xs, ys) in _get_series("tokens_per_sec").items():
        ax.plot(xs, ys,
                marker="s", linewidth=2.5, markersize=8,
                color=COLORS[key],
                linestyle=LINESTYLES[key],
                label=DISPLAY_NAMES[key])

    ax.set_xlabel("Sequence Length (tokens)", fontsize=13)
    ax.set_ylabel("Throughput (tokens / sec)", fontsize=13)
    ax.set_title("Throughput vs Sequence Length  (batch=1, FP32)", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    p = out_dir / "fig_tps_vs_seqlen.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"✓  Plot → {p}")

    # ── Figure 3: Latency (ms) vs seq_len ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, (xs, ys) in _get_series("latency_ms").items():
        ax.plot(xs, ys,
                marker="^", linewidth=2.5, markersize=8,
                color=COLORS[key],
                linestyle=LINESTYLES[key],
                label=DISPLAY_NAMES[key])

    ax.set_xlabel("Sequence Length (tokens)", fontsize=13)
    ax.set_ylabel("Latency per forward pass (ms)", fontsize=13)
    ax.set_title("Latency vs Sequence Length  (batch=1, FP32)", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    p = out_dir / "fig_latency_vs_seqlen.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"✓  Plot → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VRAM & throughput vs sequence length — Grassmann vs Transformer"
    )
    parser.add_argument(
        "--seq-lens", nargs="+", type=int,
        default=[64, 128, 256, 512, 1024, 2048],
        help="Sequence lengths to sweep",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-warmup",   type=int, default=3)
    parser.add_argument("--n-runs",     type=int, default=10)
    parser.add_argument("--ckpt-dir",   type=str, default="checkpoints")
    parser.add_argument("--no-plots",   action="store_true",
                        help="Skip matplotlib output (tables only)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
