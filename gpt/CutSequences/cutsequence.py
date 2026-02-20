from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ===== CONFIG =====
BASE_DIR = Path(__file__).resolve().parents[2]
IN_IDS_BIN = (
    BASE_DIR / "data" / "TokenizeCorpus" / "data_tokGPT" / "train_ids.bin"
)
# Output directly into the CutSequences folder (same as this script)
OUT_DIR = Path(__file__).resolve().parent
SEQ_LEN = 512                      # context length
STRIDE = 512                       # = SEQ_LEN (no overlap). set 256 for 50% overlap
DTYPE = np.uint32
LOG_EVERY = 2000                   # progress print; set 0 to disable
# ==================


def count_tokens(bin_path: Path, dtype: np.dtype) -> int:
    size = bin_path.stat().st_size
    itemsize = np.dtype(dtype).itemsize
    if size % itemsize != 0:
        raise ValueError(
            f"File size {size} not divisible by itemsize {itemsize}: {bin_path}"
        )
    return size // itemsize


def main() -> None:
    if STRIDE <= 0:
        raise ValueError("STRIDE must be > 0")
    if SEQ_LEN <= 0:
        raise ValueError("SEQ_LEN must be > 0")
    if not IN_IDS_BIN.exists():
        raise FileNotFoundError(f"Missing ids bin: {IN_IDS_BIN}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n = count_tokens(IN_IDS_BIN, DTYPE)
    win = SEQ_LEN + 1
    if n < win:
        raise ValueError(f"Not enough tokens: n={n} < seq_len+1={win}")

    num_samples = (n - win) // STRIDE + 1

    ids = np.memmap(IN_IDS_BIN, dtype=DTYPE, mode="r")
    x_path = OUT_DIR / "gpt_x.bin"
    y_path = OUT_DIR / "gpt_y.bin"
    x = np.memmap(x_path, dtype=DTYPE, mode="w+", shape=(num_samples, SEQ_LEN))
    y = np.memmap(y_path, dtype=DTYPE, mode="w+", shape=(num_samples, SEQ_LEN))

    for i, start in enumerate(range(0, num_samples * STRIDE, STRIDE)):
        s = start
        e = s + SEQ_LEN
        x[i] = ids[s:e]
        y[i] = ids[s + 1 : e + 1]

        if LOG_EVERY and (i + 1) % LOG_EVERY == 0:
            print(f"Wrote {i + 1:,}/{num_samples:,} samples...")

    x.flush()
    y.flush()

    meta = {
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
        "num_samples": int(num_samples),
        "dtype": str(np.dtype(DTYPE)),
        "ids_bin": str(IN_IDS_BIN.resolve()),
        "x_bin": str(x_path.resolve()),
        "y_bin": str(y_path.resolve()),
        "format": "gpt_shifted",
    }

    meta_path = OUT_DIR / "gpt_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Total tokens: {n:,}")
    print(f"Window (seq_len+1): {win}")
    print(f"Stride: {STRIDE}")
    print(f"Num samples: {num_samples:,}")
    print(f"Saved: {x_path} and {y_path}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
