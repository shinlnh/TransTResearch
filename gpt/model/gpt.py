from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layers: int = 6
    n_heads: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        if cfg.n_embd % cfg.n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.n_embd // cfg.n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, emb = x.size()

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=2)

        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, emb)
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.tok_emb.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len = idx.size()
        if seq_len > self.cfg.block_size:
            raise ValueError(f"seq_len {seq_len} exceeds block_size {self.cfg.block_size}")

        pos = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if repetition_penalty > 1.0:
                for row in range(idx.size(0)):
                    seen_ids = torch.unique(idx[row])
                    row_logits = logits[row, seen_ids]
                    row_logits = torch.where(
                        row_logits < 0,
                        row_logits * repetition_penalty,
                        row_logits / repetition_penalty,
                    )
                    logits[row, seen_ids] = row_logits

            if no_repeat_ngram_size > 1:
                n = int(no_repeat_ngram_size)
                idx_list = idx.tolist()
                for row in range(idx.size(0)):
                    tokens = idx_list[row]
                    if len(tokens) < (n - 1):
                        continue
                    prefix = tuple(tokens[-(n - 1) :])
                    banned = []
                    for i in range(len(tokens) - n + 1):
                        ngram = tokens[i : i + n]
                        if tuple(ngram[:-1]) == prefix:
                            banned.append(ngram[-1])
                    if banned:
                        logits[row, torch.tensor(banned, device=idx.device)] = float("-inf")

            if top_k is not None:
                top_k_value = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k_value)
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_remove = cumulative_probs > top_p
                sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
                sorted_remove[:, 0] = False
                remove_mask = torch.zeros_like(logits, dtype=torch.bool)
                remove_mask.scatter_(1, sorted_indices, sorted_remove)
                logits = logits.masked_fill(remove_mask, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None:
                finished = finished | (next_id.squeeze(1) == eos_token_id)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break

        return idx
