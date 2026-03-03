"""
prompt.py  —  Interactive text-generation REPL for the trained LMs.

Supports all three checkpoints: Grassmann+PAT, Grassmann Dense, Transformer.
Decoding strategies: greedy | top-k | top-p (nucleus).
Tokens are streamed character-by-character as they are generated.

Usage
─────
    cd sparse_grassmann_llm
    $env:PYTHONPATH = "."

    # Default: Grassmann+PAT, top-p sampling
    .venv\\Scripts\\python.exe prompt.py

    # Transformer baseline
    .venv\\Scripts\\python.exe prompt.py --model transformer

    # Dense Grassmann ablation
    .venv\\Scripts\\python.exe prompt.py --model grassmann_dense

    # Greedy decoding, 300 tokens max
    .venv\\Scripts\\python.exe prompt.py --strategy greedy --max-new-tokens 300

    # Top-k with custom temperature
    .venv\\Scripts\\python.exe prompt.py --strategy top_k --top-k 40 --temperature 0.85

    # CPU-only
    .venv\\Scripts\\python.exe prompt.py --device cpu
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.blocks import LearnedPositionalEmbedding, MaskedLinear
from models.grassmann_sparse import GrassmannConfig, GrassmannLM
from models.transformer_baseline import TransformerConfig, TransformerLM
from utils.tokenizer import train_or_load_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rederive_masks(model: nn.Module) -> None:
    """Re-derive 2:4 PAT masks from loaded weights (weight==0 → mask=0)."""
    for m in model.modules():
        if isinstance(m, MaskedLinear):
            m.weight_mask.data = (m.weight.data != 0).float()


def _extend_pos_embedding(model: nn.Module, target_seq_len: int) -> None:
    """
    Linearly interpolate the learned positional embedding table to
    target_seq_len if the stored table is shorter.  This lets 512-trained
    checkpoints accept longer prompts at generation time.
    """
    for module in model.modules():
        if not isinstance(module, LearnedPositionalEmbedding):
            continue
        old_len = module.max_seq_len
        if old_len >= target_seq_len:
            continue
        old_w  = module.embedding.weight.data          # (old_len, d_model)
        d_model = old_w.shape[1]
        interp = F.interpolate(
            old_w.T.unsqueeze(0),                      # (1, d_model, old_len)
            size=target_seq_len,
            mode="linear",
            align_corners=True,
        ).squeeze(0).T                                  # (target_seq_len, d_model)
        new_emb = nn.Embedding(target_seq_len, d_model)
        new_emb.weight = nn.Parameter(interp.to(old_w.device))
        module.embedding  = new_emb
        module.max_seq_len = target_seq_len


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    model_key: str,
    ckpt_path: Path,
    vocab_size: int,
    device: torch.device,
    max_gen_len: int,
) -> nn.Module:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict: dict = payload.get("config", {})
    trained_seq_len: int = cfg_dict.get("max_seq_len", 512)

    if model_key in ("grassmann", "grassmann_dense"):
        apply_sparse = cfg_dict.get("apply_sparse", model_key == "grassmann")
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
    else:
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

    # Extend pos embedding so prompt + generation fits
    needed = max_gen_len
    if needed > trained_seq_len:
        _extend_pos_embedding(model, needed)

    model.to(device).eval()
    return model, trained_seq_len


# ─────────────────────────────────────────────────────────────────────────────
# Decoding
# ─────────────────────────────────────────────────────────────────────────────

def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    k = min(k, logits.size(-1))
    threshold = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # shift: include the token that pushed cumulative probability over p
    remove = (cum_probs - F.softmax(sorted_logits, dim=-1)) > p
    sorted_logits[remove] = float("-inf")
    return logits.scatter(-1, sorted_idx, sorted_logits)


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: List[int],
    max_new_tokens: int,
    strategy: str,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_id: Optional[int],
    device: torch.device,
    tokenizer,
    stream: bool = True,
) -> List[int]:
    """
    Auto-regressive generation.  Streams decoded text to stdout as it goes.
    Returns list of generated token IDs (prompt not included).
    """
    max_ctx = getattr(model.config, "max_seq_len", 512)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: List[int] = []
    # For streaming: track the decoded text so far so we can print only new chars
    prev_decoded_len = 0

    for step in range(max_new_tokens):
        ctx = ids[:, -max_ctx:]                   # sliding window over context

        logits = model(ctx)                        # (1, ctx_len, vocab)
        next_logits = logits[0, -1, :]             # (vocab,)

        if temperature != 1.0 and strategy != "greedy":
            next_logits = next_logits / max(temperature, 1e-8)

        # Suppress special tokens (BOS, EOS, PAD) during generation so the
        # model is forced to produce real text.  The training data has EOS
        # after every article, so greedy always picks it first otherwise.
        suppress_ids = [tokenizer.eos_id, tokenizer.bos_id, tokenizer.pad_id]
        for suppress_id in suppress_ids:
            if suppress_id is not None:
                next_logits = next_logits.clone()  # ensure it's not a read-only view
                next_logits[suppress_id] = float("-inf")

        if strategy == "greedy":
            next_id = int(next_logits.argmax())

        elif strategy == "top_k":
            next_logits = _top_k_filter(next_logits, top_k)
            probs = F.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1))

        elif strategy == "top_p":
            next_logits = _top_p_filter(next_logits, top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1))

        else:
            raise ValueError(f"Unknown strategy '{strategy}'")

        generated.append(next_id)
        ids = torch.cat(
            [ids, torch.tensor([[next_id]], device=device)], dim=1
        )

        if stream:
            # Decode all generated tokens at once so the tokenizer has full
            # context to reconstruct spaces/subword merges correctly, then
            # print only the newly-added characters.
            full_text = tokenizer.decode(generated)
            new_chars = full_text[prev_decoded_len:]
            if new_chars:
                print(f"\033[32m{new_chars}\033[0m", end="", flush=True)
                prev_decoded_len = len(full_text)

    return generated


# ─────────────────────────────────────────────────────────────────────────────
# REPL
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
 ╔══════════════════════════════════════════════════════════════════════╗
 ║        Sparse Grassmann LLM  —  Interactive Prompt Interface        ║
 ╠══════════════════════════════════════════════════════════════════════╣
 ║  Type any text and press Enter to generate a continuation.          ║
 ║                                                                     ║
 ║  Commands  (prefix with :)                                          ║
 ║    :help              show this help                                ║
 ║    :model             show current model name + param count         ║
 ║    :info              show all current settings                     ║
 ║    :strategy greedy|top_k|top_p   change decoding strategy          ║
 ║    :temp   <float>    temperature  (default 1.0)                    ║
 ║    :topk   <int>      top-k cutoff  (default 50)                    ║
 ║    :topp   <float>    top-p nucleus  (default 0.95)                 ║
 ║    :maxtok <int>      max new tokens  (default 150)                 ║
 ║    :stream on|off     toggle token streaming                        ║
 ║    :q  /  :quit       exit                                          ║
 ╚══════════════════════════════════════════════════════════════════════╝
"""

MODEL_DISPLAY = {
    "grassmann":       "Grassmann + 2:4 PAT",
    "grassmann_dense": "Grassmann Dense (ablation)",
    "transformer":     "Transformer Baseline",
}


def _print_settings(state: dict, model_key: str, n_params: int, device: torch.device) -> None:
    print(f"\n  model      : {MODEL_DISPLAY.get(model_key, model_key)}  ({n_params:,} params)")
    print(f"  device     : {device}")
    print(f"  strategy   : {state['strategy']}")
    print(f"  temperature: {state['temperature']}")
    print(f"  top_k      : {state['top_k']}")
    print(f"  top_p      : {state['top_p']}")
    print(f"  max_tokens : {state['max_new_tokens']}")
    print(f"  streaming  : {state['stream']}\n")


def repl(
    args: argparse.Namespace,
    model: nn.Module,
    tokenizer,
    trained_seq_len: int,
) -> None:
    device    = next(model.parameters()).device
    n_params  = sum(p.numel() for p in model.parameters())
    eos_id    = getattr(tokenizer, "eos_id", None)

    state = {
        "strategy":       args.strategy,
        "temperature":    args.temperature,
        "top_k":          args.top_k,
        "top_p":          args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "stream":         True,
    }

    print(BANNER)

    if trained_seq_len < args.max_new_tokens + 512:
        print(
            f"  \033[33m[note]\033[0m  Checkpoint was trained at seq_len={trained_seq_len}. "
            f"Positional embeddings have been interpolated to "
            f"support longer sequences — generation quality is best "
            f"within {trained_seq_len} tokens.\n"
            f"          Retrain with --seq-len 2048 for full 2048-token quality.\n"
        )

    _print_settings(state, args.model, n_params, device)

    while True:
        # ── prompt input ─────────────────────────────────────────────────────
        try:
            raw = input("\033[1;36m>>> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        prompt = raw.strip()
        if not prompt:
            continue

        # ── commands ─────────────────────────────────────────────────────────
        if prompt.startswith(":"):
            parts = prompt.split()
            cmd   = parts[0].lower()

            if cmd in (":q", ":quit", ":exit"):
                print("  Bye!")
                break

            elif cmd in (":help", ":h"):
                print(BANNER)

            elif cmd == ":model":
                print(f"  {MODEL_DISPLAY.get(args.model, args.model)}  |  {n_params:,} params  |  {device}")

            elif cmd == ":info":
                _print_settings(state, args.model, n_params, device)

            elif cmd == ":strategy" and len(parts) > 1:
                s = parts[1].lower()
                if s in ("greedy", "top_k", "top_p"):
                    state["strategy"] = s
                    print(f"  strategy → {s}")
                else:
                    print("  valid strategies: greedy | top_k | top_p")

            elif cmd == ":temp" and len(parts) > 1:
                state["temperature"] = float(parts[1])
                print(f"  temperature → {state['temperature']}")

            elif cmd == ":topk" and len(parts) > 1:
                state["top_k"] = int(parts[1])
                print(f"  top_k → {state['top_k']}")

            elif cmd == ":topp" and len(parts) > 1:
                state["top_p"] = float(parts[1])
                print(f"  top_p → {state['top_p']}")

            elif cmd == ":maxtok" and len(parts) > 1:
                state["max_new_tokens"] = int(parts[1])
                print(f"  max_new_tokens → {state['max_new_tokens']}")

            elif cmd == ":stream" and len(parts) > 1:
                state["stream"] = parts[1].lower() in ("on", "true", "1", "yes")
                print(f"  streaming → {state['stream']}")

            else:
                print(f"  unknown command '{prompt}'.  Type :help for usage.")

            continue

        # ── generation ───────────────────────────────────────────────────────
        # Encode without EOS — appending EOS to the prompt tells the model
        # the sequence is finished and causes immediate degenerate output.
        prompt_ids = [tokenizer.bos_id] + tokenizer.tokenizer.encode(prompt).ids
        n_prompt   = len(prompt_ids)

        print(f"\n\033[90m[{n_prompt} prompt tokens | strategy={state['strategy']} "
              f"| temp={state['temperature']} | "
              f"max={state['max_new_tokens']} new tokens]\033[0m")
        print("\033[90m" + "─" * 68 + "\033[0m")
        # echo prompt in white, then stream generation in green
        print(f"\033[97m{prompt}\033[0m", end="", flush=True)

        try:
            gen_ids = generate(
                model          = model,
                prompt_ids     = prompt_ids,
                max_new_tokens = state["max_new_tokens"],
                strategy       = state["strategy"],
                temperature    = state["temperature"],
                top_k          = state["top_k"],
                top_p          = state["top_p"],
                eos_id         = eos_id,
                device         = device,
                tokenizer      = tokenizer,
                stream         = state["stream"],
            )
            if not state["stream"]:
                # print everything at once
                full_text = tokenizer.decode(gen_ids)
                print(f"\033[32m{full_text}\033[0m", end="")

        except torch.cuda.OutOfMemoryError:
            print("\n\033[31m[OOM — reduce :maxtok or use a shorter prompt]\033[0m")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n\033[31m[error: {e}]\033[0m")

        print(f"\n\033[90m[{len(gen_ids)} tokens generated]\033[0m")
        print("\033[90m" + "─" * 68 + "\033[0m\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

CKPT_MAP = {
    "grassmann":       "grassmann_lm.pt",
    "grassmann_dense": "grassmann_dense_lm.pt",
    "transformer":     "transformer_lm.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive LM prompt interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python prompt.py
          python prompt.py --model transformer --strategy greedy
          python prompt.py --model grassmann_dense --temperature 0.9 --top-p 0.92
          python prompt.py --device cpu --max-new-tokens 100
        """),
    )
    parser.add_argument(
        "--model", default="grassmann",
        choices=["grassmann", "grassmann_dense", "transformer"],
        help="Which checkpoint to load (default: grassmann)",
    )
    parser.add_argument("--ckpt-dir",       default="checkpoints")
    parser.add_argument("--vocab-size",     type=int,   default=8000)
    parser.add_argument("--device",         default="auto",
                        help="cuda | cpu | auto  (default: auto)")
    parser.add_argument("--strategy",       default="top_p",
                        choices=["greedy", "top_k", "top_p"])
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--top-k",          type=int,   default=50)
    parser.add_argument("--top-p",          type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int,   default=150)
    args = parser.parse_args()

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt_path = ROOT / args.ckpt_dir / CKPT_MAP[args.model]
    if not ckpt_path.exists():
        print(f"[error] Checkpoint not found: {ckpt_path}")
        print(f"        Train it with:  python train/train_lm.py --model "
              f"{'grassmann' if 'grassmann' in args.model else 'transformer'}"
              f"{' --no-sparse' if args.model == 'grassmann_dense' else ''}")
        sys.exit(1)

    tokenizer = train_or_load_tokenizer(
        ROOT / "data" / "tokenizer.json", vocab_size=args.vocab_size
    )

    # max context needed = trained length OR prompt+gen, whichever larger
    max_needed = 2048  # always extend to at least 2048
    print(f"Loading {MODEL_DISPLAY.get(args.model, args.model)} … ", end="", flush=True)
    model, trained_seq_len = load_model(
        args.model, ckpt_path, tokenizer.vocab_size, device, max_needed
    )
    print(f"done  ({sum(p.numel() for p in model.parameters()):,} params, {device})")

    repl(args, model, tokenizer, trained_seq_len)


if __name__ == "__main__":
    main()
