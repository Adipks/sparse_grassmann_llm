"""
plot_results.py — Generate charts and a formatted summary from bench_results.json.

Usage:
    python scripts/plot_results.py                        # reads eval/bench_results.json
    python scripts/plot_results.py --input path/to/results.json
    python scripts/plot_results.py --no-plots             # print tables only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_LABELS: Dict[str, str] = {
    "grassmann-fp":       "Grassmann+PAT (FP32)",
    "grassmann_dense-fp": "Grassmann Dense (FP32)",
    "transformer-fp":     "Transformer (FP32)",
    "grassmann-int8":     "Grassmann+PAT (INT8)",
    "grassmann_dense-int8": "Grassmann Dense (INT8)",
    "transformer-int8":   "Transformer (INT8)",
}

MODEL_COLORS: Dict[str, str] = {
    "grassmann-fp":        "#2196F3",   # blue
    "grassmann_dense-fp":  "#64B5F6",   # light blue
    "transformer-fp":      "#F44336",   # red
    "grassmann-int8":      "#4CAF50",   # green
    "grassmann_dense-int8":"#A5D6A7",   # light green
    "transformer-int8":    "#FF9800",   # orange
}


def print_metadata_table(meta: Dict[str, dict]) -> None:
    print("\n" + "=" * 60)
    print("  MODEL QUALITY & SIZE SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<30} {'Val PPL':>9} {'Size (MB)':>10}")
    print("  " + "-" * 52)
    for mtype, m in meta.items():
        label = MODEL_LABELS.get(f"{mtype}-fp", mtype)
        ppl = m.get("val_ppl", "N/A")
        mb  = m.get("weight_mb", "N/A")
        ppl_str = f"{ppl:.2f}" if isinstance(ppl, float) else str(ppl)
        mb_str  = f"{mb:.1f}"  if isinstance(mb,  float) else str(mb)
        print(f"  {label:<30} {ppl_str:>9} {mb_str:>10}")
    print()


def print_throughput_table(throughput: Dict[str, dict]) -> None:
    print("=" * 75)
    print("  THROUGHPUT BENCHMARK (tokens/sec, latency, VRAM)")
    print("=" * 75)
    print(f"  {'Model':<28} {'seq':>5} {'tok/s':>10} {'lat(ms)':>9} {'VRAM MB':>9}")
    print("  " + "-" * 66)

    fp_keys   = [k for k in throughput if k.endswith("_fp")  or k.endswith("-fp")]
    int8_keys = [k for k in throughput if k.endswith("_int8") or k.endswith("-int8")]

    for group_label, keys in [("FP32", fp_keys), ("INT8 (CPU)", int8_keys)]:
        if not keys:
            continue
        print(f"\n  -- {group_label} --")
        for key in keys:
            label = MODEL_LABELS.get(key, key)
            for seq_len, metrics in throughput[key].items():
                tps   = metrics["tokens_per_sec"]
                lat   = metrics["latency_ms_per_fwd"]
                vram  = metrics["peak_vram_mb"]
                print(
                    f"  {label:<28} {seq_len:>5} "
                    f"{tps:>10,.0f} {lat:>9.1f} {vram:>9.0f}"
                )
    print()


def make_plots(meta: Dict[str, dict], throughput: Dict[str, dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[WARNING] matplotlib not installed — skipping plots. Run: pip install matplotlib")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Figure 1: Throughput scaling (FP, tok/s vs seq_len, per model)
    # ------------------------------------------------------------------ #
    fp_keys = [k for k in throughput if k.endswith("-fp") or k.endswith("_fp")]
    if fp_keys:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for key in fp_keys:
            data = throughput[key]
            xs = sorted(int(s) for s in data)
            ys = [data[str(s)]["tokens_per_sec"] for s in xs]
            label = MODEL_LABELS.get(key, key)
            color = MODEL_COLORS.get(key, None)
            ax.plot(xs, ys, marker="o", label=label, color=color, linewidth=2)

        ax.set_xlabel("Sequence length (tokens)", fontsize=11)
        ax.set_ylabel("Throughput (tokens / sec)", fontsize=11)
        ax.set_title("Inference Throughput — FP32 (batch=1)", fontsize=12)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        path = out_dir / "fig1_throughput_fp.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved {path}")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Figure 2: Latency scaling (FP, ms per fwd pass vs seq_len)
    # ------------------------------------------------------------------ #
    if fp_keys:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for key in fp_keys:
            data = throughput[key]
            xs = sorted(int(s) for s in data)
            ys = [data[str(s)]["latency_ms_per_fwd"] for s in xs]
            label = MODEL_LABELS.get(key, key)
            color = MODEL_COLORS.get(key, None)
            ax.plot(xs, ys, marker="s", label=label, color=color, linewidth=2)

        ax.set_xlabel("Sequence length (tokens)", fontsize=11)
        ax.set_ylabel("Latency per forward pass (ms)", fontsize=11)
        ax.set_title("Forward-Pass Latency — FP32 (batch=1)", fontsize=12)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        path = out_dir / "fig2_latency_fp.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved {path}")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Figure 3: Val perplexity bar chart (lower = better)
    # ------------------------------------------------------------------ #
    if meta:
        fig, ax = plt.subplots(figsize=(6, 4))
        models = list(meta.keys())
        ppls   = [meta[m].get("val_ppl", 0) for m in models]
        labels = [MODEL_LABELS.get(f"{m}-fp", m) for m in models]
        colors = [MODEL_COLORS.get(f"{m}-fp", "#999") for m in models]
        bars = ax.bar(labels, ppls, color=colors, edgecolor="black", linewidth=0.8)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.set_ylabel("Validation Perplexity (↓ better)", fontsize=11)
        ax.set_title("Wikitext-2 Validation Perplexity", fontsize=12)
        ax.set_ylim(0, max(ppls) * 1.2)
        plt.xticks(rotation=15, ha="right", fontsize=8)
        fig.tight_layout()
        path = out_dir / "fig3_val_ppl.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved {path}")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Figure 4: FP vs INT8 throughput comparison (grouped bar, seq=512)
    # ------------------------------------------------------------------ #
    seq_key = "512"
    all_keys_512 = [k for k in throughput if seq_key in throughput[k]]
    if all_keys_512:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x_pos = range(len(all_keys_512))
        bar_vals = [throughput[k][seq_key]["tokens_per_sec"] for k in all_keys_512]
        bar_labels = [MODEL_LABELS.get(k, k) for k in all_keys_512]
        bar_colors = [MODEL_COLORS.get(k, "#aaa") for k in all_keys_512]
        bars = ax.bar(bar_labels, bar_vals, color=bar_colors, edgecolor="black", linewidth=0.8)
        ax.bar_label(bars, fmt=lambda v: f"{int(v):,}", padding=3, fontsize=8)
        ax.set_ylabel("Tokens / sec", fontsize=11)
        ax.set_title(f"Throughput at seq_len={seq_key} (batch=1)", fontsize=12)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
        plt.xticks(rotation=20, ha="right", fontsize=8)
        fig.tight_layout()
        path = out_dir / "fig4_throughput_all.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved {path}")
        plt.close(fig)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot and summarise bench_results.json")
    parser.add_argument("--input",     type=str, default="eval/bench_results.json")
    parser.add_argument("--out-dir",   type=str, default="eval/figures")
    parser.add_argument("--no-plots",  action="store_true", help="Only print tables, skip matplotlib")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    results_path = project_root / args.input
    out_dir      = project_root / args.out_dir

    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        return

    with open(results_path) as f:
        data = json.load(f)

    # Support both old format (flat dict) and new format (metadata + throughput)
    if "throughput" in data:
        meta       = data.get("metadata", {})
        throughput = data["throughput"]
    else:
        meta       = {}
        throughput = data

    if meta:
        print_metadata_table(meta)
    print_throughput_table(throughput)

    if not args.no_plots:
        print(f"Generating plots → {out_dir}")
        make_plots(meta, throughput, out_dir)
    else:
        print("(--no-plots: skipping figure generation)")


if __name__ == "__main__":
    main()
