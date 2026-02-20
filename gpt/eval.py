from __future__ import annotations

import argparse
import json
import math
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
    from tqdm import tqdm

    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.dataloader.dataloader import LoaderConfig, build_dataloader
from gpt.model import GPT, GPTConfig


@dataclass
class EvalConfig:
    # Data/meta
    meta_path: str = "gpt/CutSequences/gpt_meta.json"
    x_bin: str = ""
    y_bin: str = ""
    seq_len: int = 512
    dtype: str = "uint32"

    # Tokenizer / vocab
    vocab_json: str = "data/Token/tokenizers_bpe/vocab.json"
    tokenizer_json: str = "data/Token/tokenizers_bpe/tokenizer.json"
    vocab_size: int = 0  # if 0, infer from vocab/tokenizer/ckpt

    # Model (fallback if ckpt has no config)
    n_layers: int = 6
    n_heads: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True

    # Eval
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    max_batches: int = 0  # 0 = full
    checkpoint: str = "gpt/checkpoints/ckpt_last.pt"
    use_ckpt_config: bool = True


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_meta(cfg: EvalConfig) -> None:
    meta_path = resolve_path(cfg.meta_path)
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg.seq_len = int(meta.get("seq_len", cfg.seq_len))
    cfg.dtype = str(meta.get("dtype", cfg.dtype)).replace("numpy.", "")
    if not cfg.x_bin:
        cfg.x_bin = str(meta.get("x_bin", cfg.x_bin))
    if not cfg.y_bin:
        cfg.y_bin = str(meta.get("y_bin", cfg.y_bin))


def vocab_size_from_vocab_json(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        max_id = max(data.values()) if data else -1
        return max_id + 1
    if isinstance(data, list):
        return len(data)
    return None


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


def resolve_vocab_size(cfg: EvalConfig, ckpt_vocab: Optional[int]) -> int:
    if ckpt_vocab is not None and ckpt_vocab > 0:
        return ckpt_vocab
    if cfg.vocab_size > 0:
        return cfg.vocab_size

    vocab_json = resolve_path(cfg.vocab_json)
    tokenizer_json = resolve_path(cfg.tokenizer_json)

    vocab_size = vocab_size_from_vocab_json(vocab_json)
    if vocab_size is None:
        vocab_size = vocab_size_from_tokenizer_json(tokenizer_json)

    if vocab_size is None:
        raise FileNotFoundError(
            "Could not infer vocab_size. Provide --vocab_size or a valid "
            "vocab_json/tokenizer_json path."
        )
    return vocab_size


def build_model(cfg: EvalConfig, vocab_size: int) -> GPT:
    model_cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=cfg.seq_len,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    return GPT(model_cfg)


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate GPT model on bin data.")
    p.add_argument("--meta_path", type=str, default=EvalConfig.meta_path)
    p.add_argument("--x_bin", type=str, default=EvalConfig.x_bin)
    p.add_argument("--y_bin", type=str, default=EvalConfig.y_bin)
    p.add_argument("--seq_len", type=int, default=EvalConfig.seq_len)
    p.add_argument("--dtype", type=str, default=EvalConfig.dtype)
    p.add_argument("--vocab_json", type=str, default=EvalConfig.vocab_json)
    p.add_argument("--tokenizer_json", type=str, default=EvalConfig.tokenizer_json)
    p.add_argument("--vocab_size", type=int, default=EvalConfig.vocab_size)
    p.add_argument("--n_layers", type=int, default=EvalConfig.n_layers)
    p.add_argument("--n_heads", type=int, default=EvalConfig.n_heads)
    p.add_argument("--n_embd", type=int, default=EvalConfig.n_embd)
    p.add_argument("--dropout", type=float, default=EvalConfig.dropout)
    p.add_argument("--bias", dest="bias", action="store_true")
    p.add_argument("--no_bias", dest="bias", action="store_false")
    p.add_argument("--batch_size", type=int, default=EvalConfig.batch_size)
    p.add_argument("--num_workers", type=int, default=EvalConfig.num_workers)
    p.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    p.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    p.add_argument("--device", type=str, default=EvalConfig.device)
    p.add_argument("--amp", dest="amp", action="store_true")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--max_batches", type=int, default=EvalConfig.max_batches)
    p.add_argument("--checkpoint", type=str, default=EvalConfig.checkpoint)
    p.add_argument("--use_ckpt_config", dest="use_ckpt_config", action="store_true")
    p.add_argument("--no_use_ckpt_config", dest="use_ckpt_config", action="store_false")
    p.set_defaults(
        bias=EvalConfig.bias,
        pin_memory=EvalConfig.pin_memory,
        amp=EvalConfig.amp,
        use_ckpt_config=EvalConfig.use_ckpt_config,
    )
    args = p.parse_args()
    return EvalConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    load_meta(cfg)
    if not cfg.x_bin or not cfg.y_bin:
        raise FileNotFoundError("x_bin/y_bin not set. Provide --meta_path or --x_bin/--y_bin")

    ckpt_path = resolve_path(cfg.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_cfg = ckpt.get("train_config", {}) if isinstance(ckpt, dict) else {}

    if cfg.use_ckpt_config and ckpt_cfg:
        cfg.seq_len = int(ckpt_cfg.get("seq_len", cfg.seq_len))
        cfg.n_layers = int(ckpt_cfg.get("n_layers", cfg.n_layers))
        cfg.n_heads = int(ckpt_cfg.get("n_heads", cfg.n_heads))
        cfg.n_embd = int(ckpt_cfg.get("n_embd", cfg.n_embd))
        cfg.dropout = float(ckpt_cfg.get("dropout", cfg.dropout))
        cfg.bias = bool(ckpt_cfg.get("bias", cfg.bias))

    model_state = ckpt.get("model_state") if isinstance(ckpt, dict) else None
    ckpt_vocab = None
    if isinstance(model_state, dict):
        emb_weight = model_state.get("tok_emb.weight")
        if isinstance(emb_weight, torch.Tensor) and emb_weight.ndim == 2:
            ckpt_vocab = int(emb_weight.shape[0])
            cfg.n_embd = int(emb_weight.shape[1])

    vocab_size = resolve_vocab_size(cfg, ckpt_vocab)

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")

    loader_cfg = LoaderConfig(
        x_bin=str(resolve_path(cfg.x_bin)),
        y_bin=str(resolve_path(cfg.y_bin)),
        seq_len=cfg.seq_len,
        dtype=cfg.dtype,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2,
    )
    _, dl = build_dataloader(loader_cfg)

    model = build_model(cfg, vocab_size).to(device)
    if isinstance(model_state, dict):
        model.load_state_dict(model_state, strict=True)
    model.eval()

    use_amp = cfg.amp and device.type == "cuda"

    total_loss = 0.0
    total_tokens = 0
    start = time.time()

    if _HAS_TQDM:
        pbar = tqdm(dl, total=len(dl), desc="eval", ncols=100)
        iterator = pbar
    else:
        pbar = None
        iterator = dl

    with torch.no_grad():
        for step, (xb, yb) in enumerate(iterator, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if _USE_NEW_AMP:
                with amp_autocast(device.type, enabled=use_amp):
                    _, loss = model(xb, yb)
            else:
                with amp_autocast(enabled=use_amp):
                    _, loss = model(xb, yb)

            if loss is None:
                raise RuntimeError("Loss is None. Check targets.")

            tokens = xb.numel()
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens

            if pbar is not None:
                avg_loss = total_loss / max(total_tokens, 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

            if cfg.max_batches and step >= cfg.max_batches:
                break

    if pbar is not None:
        pbar.close()

    avg_loss = total_loss / max(total_tokens, 1)
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")

    elapsed = time.time() - start
    tps = total_tokens / max(elapsed, 1e-8)

    print(
        "Eval result:",
        {
            "avg_loss": round(avg_loss, 6),
            "ppl": round(ppl, 6) if math.isfinite(ppl) else "inf",
            "tokens": int(total_tokens),
            "tokens_per_s": int(tps),
        },
    )


if __name__ == "__main__":
    main()
