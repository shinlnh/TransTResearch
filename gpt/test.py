from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

try:
    from torch.amp import autocast as amp_autocast

    _USE_NEW_AMP = True
except Exception:  # pragma: no cover - fallback for older torch
    from torch.cuda.amp import autocast as amp_autocast

    _USE_NEW_AMP = False

try:
    from tokenizers import Tokenizer

    _HAS_TOKENIZERS = True
except Exception:
    Tokenizer = None
    _HAS_TOKENIZERS = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.model import GPT, GPTConfig


@dataclass
class TestConfig:
    checkpoint: str = "gpt/checkpoints/ckpt_last.pt"
    tokenizer_json: str = "data/Token/tokenizers_bpe/tokenizer.json"
    prompt: str = ""
    prompt_ids: str = ""
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    eos_token_id: int = -1
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

    # Fallback model config if checkpoint has no train_config.
    seq_len: int = 512
    n_layers: int = 6
    n_heads: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True
    vocab_size: int = 0
    use_ckpt_config: bool = True


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def vocab_size_from_tokenizer_json(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    model = data.get("model", {})
    vocab = model.get("vocab")
    max_id = -1
    if isinstance(vocab, dict):
        max_id = max(vocab.values()) if vocab else -1
    elif isinstance(vocab, list):
        max_id = len(vocab) - 1
    added = data.get("added_tokens", [])
    for tok in added:
        tok_id = tok.get("id")
        if isinstance(tok_id, int):
            max_id = max(max_id, tok_id)
    return max_id + 1 if max_id >= 0 else None


def parse_args() -> TestConfig:
    p = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint.")
    p.add_argument("--checkpoint", type=str, default=TestConfig.checkpoint)
    p.add_argument("--tokenizer_json", type=str, default=TestConfig.tokenizer_json)
    p.add_argument("--prompt", type=str, default=TestConfig.prompt)
    p.add_argument(
        "--prompt_ids",
        type=str,
        default=TestConfig.prompt_ids,
        help="Comma-separated token ids. Use this if tokenizers package is not installed.",
    )
    p.add_argument("--max_new_tokens", type=int, default=TestConfig.max_new_tokens)
    p.add_argument("--temperature", type=float, default=TestConfig.temperature)
    p.add_argument("--top_k", type=int, default=TestConfig.top_k)
    p.add_argument("--top_p", type=float, default=TestConfig.top_p)
    p.add_argument("--repetition_penalty", type=float, default=TestConfig.repetition_penalty)
    p.add_argument("--no_repeat_ngram_size", type=int, default=TestConfig.no_repeat_ngram_size)
    p.add_argument(
        "--eos_token_id",
        type=int,
        default=TestConfig.eos_token_id,
        help="Set >= 0 to stop generation when eos token is sampled.",
    )
    p.add_argument("--seed", type=int, default=TestConfig.seed)
    p.add_argument("--device", type=str, default=TestConfig.device)
    p.add_argument("--amp", dest="amp", action="store_true")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--seq_len", type=int, default=TestConfig.seq_len)
    p.add_argument("--n_layers", type=int, default=TestConfig.n_layers)
    p.add_argument("--n_heads", type=int, default=TestConfig.n_heads)
    p.add_argument("--n_embd", type=int, default=TestConfig.n_embd)
    p.add_argument("--dropout", type=float, default=TestConfig.dropout)
    p.add_argument("--bias", dest="bias", action="store_true")
    p.add_argument("--no_bias", dest="bias", action="store_false")
    p.add_argument("--vocab_size", type=int, default=TestConfig.vocab_size)
    p.add_argument("--use_ckpt_config", dest="use_ckpt_config", action="store_true")
    p.add_argument("--no_use_ckpt_config", dest="use_ckpt_config", action="store_false")
    p.set_defaults(
        amp=TestConfig.amp,
        bias=TestConfig.bias,
        use_ckpt_config=TestConfig.use_ckpt_config,
    )
    args = p.parse_args()
    return TestConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    ckpt_path = resolve_path(cfg.checkpoint)
    tok_path = resolve_path(cfg.tokenizer_json)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not tok_path.exists() and not cfg.prompt_ids.strip():
        raise FileNotFoundError(f"Tokenizer JSON not found: {tok_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Invalid checkpoint format: expected dict.")

    ckpt_cfg = ckpt.get("train_config", {})
    if cfg.use_ckpt_config and isinstance(ckpt_cfg, dict) and ckpt_cfg:
        cfg.seq_len = int(ckpt_cfg.get("seq_len", cfg.seq_len))
        cfg.n_layers = int(ckpt_cfg.get("n_layers", cfg.n_layers))
        cfg.n_heads = int(ckpt_cfg.get("n_heads", cfg.n_heads))
        cfg.n_embd = int(ckpt_cfg.get("n_embd", cfg.n_embd))
        cfg.dropout = float(ckpt_cfg.get("dropout", cfg.dropout))
        cfg.bias = bool(ckpt_cfg.get("bias", cfg.bias))
        cfg.vocab_size = int(ckpt_cfg.get("vocab_size", cfg.vocab_size))

    model_state = ckpt.get("model_state")
    if not isinstance(model_state, dict):
        raise RuntimeError("Checkpoint missing model_state.")

    emb_weight = model_state.get("tok_emb.weight")
    if isinstance(emb_weight, torch.Tensor) and emb_weight.ndim == 2:
        cfg.vocab_size = int(emb_weight.shape[0])
        cfg.n_embd = int(emb_weight.shape[1])

    if cfg.vocab_size <= 0:
        inferred_vocab = vocab_size_from_tokenizer_json(tok_path)
        if inferred_vocab is None:
            raise RuntimeError(
                "Could not infer vocab_size. Pass --vocab_size or ensure tokenizer JSON is valid."
            )
        cfg.vocab_size = inferred_vocab

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    tokenizer = None
    if _HAS_TOKENIZERS and tok_path.exists():
        tokenizer = Tokenizer.from_file(str(tok_path))

    prompt = cfg.prompt.strip()
    input_ids = []
    if cfg.prompt_ids.strip():
        input_ids = [int(x.strip()) for x in cfg.prompt_ids.split(",") if x.strip()]
    else:
        if tokenizer is None:
            raise RuntimeError(
                "Missing dependency 'tokenizers'. Install it with: pip install tokenizers "
                "or pass --prompt_ids."
            )
        if not prompt:
            prompt = input("Prompt: ").strip()
        if not prompt:
            raise ValueError("Prompt is empty.")
        enc = tokenizer.encode(prompt)
        input_ids = enc.ids

    if not input_ids:
        raise ValueError("Prompt produced no tokens.")

    model_cfg = GPTConfig(
        vocab_size=cfg.vocab_size,
        block_size=cfg.seq_len,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    model = GPT(model_cfg).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    top_k = cfg.top_k if cfg.top_k > 0 else None
    top_p = cfg.top_p if 0.0 < cfg.top_p < 1.0 else None
    eos_token_id = cfg.eos_token_id if cfg.eos_token_id >= 0 else None
    repetition_penalty = max(float(cfg.repetition_penalty), 1.0)
    no_repeat_ngram_size = max(int(cfg.no_repeat_ngram_size), 0)
    use_amp = cfg.amp and device.type == "cuda"

    t0 = time.time()
    with torch.no_grad():
        if _USE_NEW_AMP:
            with amp_autocast(device.type, enabled=use_amp):
                out = model.generate(
                    idx=idx,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    eos_token_id=eos_token_id,
                )
        else:
            with amp_autocast(enabled=use_amp):
                out = model.generate(
                    idx=idx,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    eos_token_id=eos_token_id,
                )
    elapsed = time.time() - t0

    full_ids = out[0].tolist()
    new_ids = full_ids[len(input_ids) :]
    full_text = None
    new_text = None
    if tokenizer is not None:
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        new_text = tokenizer.decode(new_ids, skip_special_tokens=True) if new_ids else ""
    tps = len(new_ids) / max(elapsed, 1e-8)

    print(
        "Test result:",
        {
            "checkpoint": str(ckpt_path),
            "device": str(device),
            "amp": use_amp,
            "temperature": cfg.temperature,
            "top_k": top_k if top_k is not None else 0,
            "top_p": top_p if top_p is not None else 0.0,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "prompt_tokens": len(input_ids),
            "new_tokens": len(new_ids),
            "tokens_per_s": int(tps),
        },
    )
    if prompt:
        print("\n=== Prompt ===")
        print(prompt)
    if new_text is not None:
        print("\n=== Completion ===")
        print(new_text)
        print("\n=== Full Text ===")
        print(full_text)
    else:
        print("\n=== Generated IDs ===")
        print(",".join(str(x) for x in full_ids))


if __name__ == "__main__":
    main()
