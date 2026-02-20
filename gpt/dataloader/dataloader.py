from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


DTYPE_MAP = {
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
    "int64": np.int64,
}


def _infer_num_sequences(bin_path: Union[str, Path], seq_len: int, np_dtype: np.dtype) -> int:
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"Không thấy file: {bin_path}")

    n_bytes = bin_path.stat().st_size
    itemsize = np.dtype(np_dtype).itemsize
    bytes_per_seq = seq_len * itemsize

    if n_bytes % bytes_per_seq != 0:
        raise ValueError(
            f"File size không khớp seq_len/dtype.\n"
            f"- file: {bin_path}\n"
            f"- size bytes: {n_bytes}\n"
            f"- seq_len: {seq_len}\n"
            f"- dtype: {np_dtype} (itemsize={itemsize})\n"
            f"- bytes_per_seq: {bytes_per_seq}\n"
            f"size % bytes_per_seq = {n_bytes % bytes_per_seq} (phải = 0)"
        )

    return n_bytes // bytes_per_seq


class GPTBinDataset(Dataset):
    """
    Dataset đọc trực tiếp (memmap) từ 2 file:
      - x_bin: chứa N*seq_len token ids
      - y_bin: chứa N*seq_len token ids (shift-right labels)

    Trả về:
      x: torch.LongTensor [seq_len]
      y: torch.LongTensor [seq_len]
    """

    def __init__(
        self,
        x_bin: Union[str, Path],
        y_bin: Union[str, Path],
        seq_len: int,
        dtype: str = "uint32",
        device: Optional[torch.device] = None,
    ):
        self.x_bin = Path(x_bin)
        self.y_bin = Path(y_bin)
        self.seq_len = int(seq_len)

        if dtype not in DTYPE_MAP:
            raise ValueError(f"dtype '{dtype}' không hỗ trợ. Chọn một trong: {list(DTYPE_MAP.keys())}")
        self.np_dtype = DTYPE_MAP[dtype]

        # Infer N từ size file (đảm bảo khớp seq_len & dtype)
        n_x = _infer_num_sequences(self.x_bin, self.seq_len, self.np_dtype)
        n_y = _infer_num_sequences(self.y_bin, self.seq_len, self.np_dtype)
        if n_x != n_y:
            raise ValueError(f"Số sequence không khớp: x={n_x}, y={n_y}")

        self.n_sequences = int(n_x)
        self.device = device  # thường để None, chuyển lên GPU ở training loop

        # Memmap dạng 2D để slice nhanh: [N, seq_len]
        self._shape = (self.n_sequences, self.seq_len)
        self._x = None
        self._y = None

    def _ensure_memmaps(self) -> None:
        if self._x is None:
            self._x = np.memmap(self.x_bin, mode="r", dtype=self.np_dtype, shape=self._shape)
        if self._y is None:
            self._y = np.memmap(self.y_bin, mode="r", dtype=self.np_dtype, shape=self._shape)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_x"] = None
        state["_y"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._x = None
        self._y = None

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_memmaps()
        # np.memmap slice is read-only; copy to avoid non-writable tensor warning
        x = torch.from_numpy(np.array(self._x[idx], copy=True)).long()
        y = torch.from_numpy(np.array(self._y[idx], copy=True)).long()

        if self.device is not None:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

        return x, y


@dataclass
class LoaderConfig:
    x_bin: Union[str, Path]
    y_bin: Union[str, Path]
    seq_len: int = 512
    dtype: str = "uint32"
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2  # chỉ có tác dụng khi num_workers > 0


def build_dataloader(cfg: LoaderConfig) -> Tuple[GPTBinDataset, DataLoader]:
    ds = GPTBinDataset(
        x_bin=cfg.x_bin,
        y_bin=cfg.y_bin,
        seq_len=cfg.seq_len,
        dtype=cfg.dtype,
        device=None,  # chuyển lên GPU ở train loop sẽ tốt hơn
    )

    # Lưu ý: persistent_workers chỉ hợp lệ khi num_workers > 0
    persistent = cfg.persistent_workers and cfg.num_workers > 0

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        persistent_workers=persistent,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    return ds, dl


if __name__ == "__main__":
    # Ví dụ chạy thử nhanh: python dataloader.py
    cfg = LoaderConfig(
        x_bin="gpt_x.bin",
        y_bin="gpt_y.bin",
        seq_len=512,
        dtype="uint32",
        batch_size=8,
        shuffle=True,
        num_workers=2,  # Windows: dùng num_workers > 0 với memmap
    )

    ds, dl = build_dataloader(cfg)
    print(f"Dataset size: {len(ds):,} sequences | seq_len={cfg.seq_len}")

    xb, yb = next(iter(dl))
    print("Batch shapes:", xb.shape, yb.shape)
    print("Dtypes:", xb.dtype, yb.dtype)
    print("First few tokens x[0]:", xb[0, :10].tolist())
    print("First few tokens y[0]:", yb[0, :10].tolist())
