from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False

try:
    from torch.amp import autocast as amp_autocast
    from torch.amp import GradScaler as AmpGradScaler

    _USE_NEW_AMP = True
except Exception:  # pragma: no cover - fallback for older torch
    from torch.cuda.amp import autocast as amp_autocast
    from torch.cuda.amp import GradScaler as AmpGradScaler

    _USE_NEW_AMP = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpt.dataloader.dataloader import LoaderConfig, build_dataloader
from gpt.model import GPT, GPTConfig


@dataclass
class TrainConfig:
    # Data/meta
    meta_path: str = "gpt/CutSequences/gpt_meta.json"
    x_bin: str = ""
    y_bin: str = ""
    seq_len: int = 512
    dtype: str = "uint32"

    # Tokenizer / vocab
    vocab_json: str = "data/Token/tokenizers_bpe/vocab.json"
    tokenizer_json: str = "data/Token/tokenizers_bpe/tokenizer.json"
    vocab_size: int = 0  # if 0, infer from vocab/tokenizer

    # Model
    n_layers: int = 6
    n_heads: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True

    # Train
    batch_size: int = 8
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 1337
    num_workers: int = 0
    pin_memory: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    log_every: int = 100
    save_every: int = 1000
    out_dir: str = "gpt/checkpoints"
    max_steps: int = 0  # 0 = run full epochs
    resume: str = ""
    early_stop_patience: int = 0  # 0 = disable
    early_stop_min_delta: float = 0.0


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_meta(cfg: TrainConfig) -> None:
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


def resolve_vocab_size(cfg: TrainConfig) -> int:
    if cfg.vocab_size > 0:
        return cfg.vocab_size

    vocab_size = None
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


def build_model(cfg: TrainConfig, vocab_size: int) -> GPT:
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


def save_checkpoint(
    out_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[AmpGradScaler],
    cfg: TrainConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "train_config": asdict(cfg),
    }
    ckpt_path = out_dir / f"ckpt_step_{step}.pt"
    torch.save(ckpt, ckpt_path)
    torch.save(ckpt, out_dir / "ckpt_last.pt")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[AmpGradScaler],
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return int(ckpt.get("step", 0))


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train GPT model on bin data.")
    p.add_argument("--meta_path", type=str, default=TrainConfig.meta_path)
    p.add_argument("--x_bin", type=str, default=TrainConfig.x_bin)
    p.add_argument("--y_bin", type=str, default=TrainConfig.y_bin)
    p.add_argument("--seq_len", type=int, default=TrainConfig.seq_len)
    p.add_argument("--dtype", type=str, default=TrainConfig.dtype)
    p.add_argument("--vocab_json", type=str, default=TrainConfig.vocab_json)
    p.add_argument("--tokenizer_json", type=str, default=TrainConfig.tokenizer_json)
    p.add_argument("--vocab_size", type=int, default=TrainConfig.vocab_size)
    p.add_argument("--n_layers", type=int, default=TrainConfig.n_layers)
    p.add_argument("--n_heads", type=int, default=TrainConfig.n_heads)
    p.add_argument("--n_embd", type=int, default=TrainConfig.n_embd)
    p.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    p.add_argument("--bias", dest="bias", action="store_true")
    p.add_argument("--no_bias", dest="bias", action="store_false")
    p.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--beta1", type=float, default=TrainConfig.beta1)
    p.add_argument("--beta2", type=float, default=TrainConfig.beta2)
    p.add_argument("--grad_clip", type=float, default=TrainConfig.grad_clip)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    p.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    p.add_argument("--device", type=str, default=TrainConfig.device)
    p.add_argument("--amp", dest="amp", action="store_true")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--log_every", type=int, default=TrainConfig.log_every)
    p.add_argument("--save_every", type=int, default=TrainConfig.save_every)
    p.add_argument("--out_dir", type=str, default=TrainConfig.out_dir)
    p.add_argument("--max_steps", type=int, default=TrainConfig.max_steps)
    p.add_argument("--resume", type=str, default=TrainConfig.resume)
    p.add_argument("--early_stop_patience", type=int, default=TrainConfig.early_stop_patience)
    p.add_argument("--early_stop_min_delta", type=float, default=TrainConfig.early_stop_min_delta)
    p.set_defaults(
        bias=TrainConfig.bias,
        pin_memory=TrainConfig.pin_memory,
        amp=TrainConfig.amp,
    )
    args = p.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    load_meta(cfg)
    if not cfg.x_bin or not cfg.y_bin:
        raise FileNotFoundError("x_bin/y_bin not set. Provide --meta_path or --x_bin/--y_bin")

    vocab_size = resolve_vocab_size(cfg)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    loader_cfg = LoaderConfig(
        x_bin=str(resolve_path(cfg.x_bin)),
        y_bin=str(resolve_path(cfg.y_bin)),
        seq_len=cfg.seq_len,
        dtype=cfg.dtype,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2,
    )
    _, dl = build_dataloader(loader_cfg)

    model = build_model(cfg, vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    use_amp = cfg.amp and device.type == "cuda"
    scaler = AmpGradScaler(enabled=use_amp)

    start_step = 0
    print(
        "Train config:",
        {
            "vocab_size": vocab_size,
            "seq_len": cfg.seq_len,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "device": str(device),
            "amp": use_amp,
        },
    )
    if cfg.resume:
        start_step = load_checkpoint(resolve_path(cfg.resume), model, optimizer, scaler)
        move_optimizer_state(optimizer, device)
        print(f"Resumed from {cfg.resume} at step {start_step}")

    model.train()
    global_step = start_step
    t0 = time.time()
    tokens_per_step = cfg.batch_size * cfg.seq_len
    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        if _HAS_TQDM:
            pbar = tqdm(dl, total=len(dl), desc=f"epoch {epoch+1}/{cfg.epochs}", ncols=100)
            iterator = pbar
        else:
            pbar = None
            iterator = dl

        epoch_loss_sum = 0.0
        epoch_steps = 0
        for xb, yb in iterator:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if _USE_NEW_AMP:
                with amp_autocast(device.type, enabled=use_amp):
                    _, loss = model(xb, yb)
            else:
                with amp_autocast(enabled=use_amp):
                    _, loss = model(xb, yb)

            if loss is None:
                raise RuntimeError("Loss is None. Check targets.")

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            epoch_loss_sum += float(loss.item())
            epoch_steps += 1
            if cfg.log_every and (global_step % cfg.log_every == 0):
                dt = time.time() - t0
                toks = tokens_per_step * cfg.log_every
                tps = toks / max(dt, 1e-8)
                if pbar is not None:
                    pbar.set_postfix(loss=f"{loss.item():.4f}", tps=f"{tps:,.0f}", step=global_step)
                else:
                    print(
                        f"epoch={epoch+1} step={global_step} "
                        f"loss={loss.item():.4f} tokens/s={tps:,.0f}"
                    )
                t0 = time.time()

            if cfg.save_every and (global_step % cfg.save_every == 0):
                save_checkpoint(resolve_path(cfg.out_dir), global_step, model, optimizer, scaler, cfg)

            if cfg.max_steps and global_step >= cfg.max_steps:
                break

        if pbar is not None:
            pbar.close()

        if epoch_steps > 0:
            epoch_loss = epoch_loss_sum / epoch_steps
            print(f"epoch {epoch+1} done | avg_loss={epoch_loss:.4f} | steps={epoch_steps}")
        else:
            epoch_loss = None

        if cfg.early_stop_patience > 0 and epoch_loss is not None:
            if best_loss - epoch_loss > cfg.early_stop_min_delta:
                best_loss = epoch_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                print(
                    f"early stop: no improvement (best={best_loss:.4f}) "
                    f"bad_epochs={bad_epochs}/{cfg.early_stop_patience}"
                )
                if bad_epochs >= cfg.early_stop_patience:
                    print("early stop triggered.")
                    break

        if cfg.max_steps and global_step >= cfg.max_steps:
            break

    save_checkpoint(resolve_path(cfg.out_dir), global_step, model, optimizer, scaler, cfg)
    print("Done.")


if __name__ == "__main__":
    main()
