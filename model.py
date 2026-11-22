"""Small modern decoder-only Transformer.

Components: RMSNorm (pre-norm), SwiGLU MLP, grouped-query attention with
QK-norm, optional RoPE (pos='rope') or no positional encoding (pos='nope'),
tied input/output embeddings.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int
    d_model: int = 192
    n_layer: int = 4
    n_head: int = 6
    n_kv_head: int = 2
    mlp_mult: float = 8 / 3
    pos: str = "nope"  # 'rope' | 'nope'
    max_len: int = 256
    dropout: float = 0.0

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(seq, dim, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq, device=device).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):  # x: B,H,T,D
    x1, x2 = x[..., ::2], x[..., 1::2]
    T = x.shape[2]
    c, s = cos[:T], sin[:T]
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        assert c.d_model % c.n_head == 0 and c.n_head % c.n_kv_head == 0
        self.c = c
        self.hd = c.d_model // c.n_head
        self.q = nn.Linear(c.d_model, c.n_head * self.hd, bias=False)
        self.kv = nn.Linear(c.d_model, 2 * c.n_kv_head * self.hd, bias=False)
        self.o = nn.Linear(c.d_model, c.d_model, bias=False)
        self.qn = RMSNorm(self.hd)
        self.kn = RMSNorm(self.hd)

    def forward(self, x, rope=None, attn_mask=None, cache=None):
        B, T, _ = x.shape
        c = self.c
        q = self.q(x).view(B, T, c.n_head, self.hd).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, c.n_kv_head, self.hd).permute(2, 0, 3, 1, 4)
        q, k = self.qn(q), self.kn(k)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if cache is not None:  # cache: dict with 'k','v' (pre-repeat), incremental decoding
            if "k" in cache:
                k = torch.cat([cache["k"], k], 2)
                v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
            if T == 1:
                attn_mask = torch.ones(1, 1, 1, k.shape[2], dtype=torch.bool)
        rep = c.n_head // c.n_kv_head
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        h = int(c.d_model * c.mlp_mult)
        h = (h + 7) // 8 * 8
        self.up = nn.Linear(c.d_model, 2 * h, bias=False)
        self.down = nn.Linear(h, c.d_model, bias=False)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n1, self.attn = RMSNorm(c.d_model), Attention(c)
        self.n2, self.mlp = RMSNorm(c.d_model), SwiGLU(c)

    def forward(self, x, rope=None, attn_mask=None, cache=None):
        x = x + self.attn(self.n1(x), rope, attn_mask, cache)
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.d_model)
        self.apply(self._init)
        for b in self.blocks:  # scaled residual init
            nn.init.normal_(b.attn.o.weight, std=0.02 / math.sqrt(2 * c.n_layer))
            nn.init.normal_(b.mlp.down.weight, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, attn_mask=None, caches=None, offset=0):
        x = self.emb(idx)
        rope = None
        if self.c.pos == "rope":
            hd = self.c.d_model // self.c.n_head
            cos, sin = rope_cache(offset + idx.shape[1], hd, idx.device)
            rope = (cos[offset:], sin[offset:])
        n_ckpt = 0
        if self.training and caches is None:  # memory efficiency: recompute some blocks in backward
            n_ckpt = round(getattr(self, "ckpt_frac", 0.0) * len(self.blocks))
        for i, b in enumerate(self.blocks):
            if i < n_ckpt:
                x = torch.utils.checkpoint.checkpoint(b, x, rope, attn_mask, None, use_reentrant=False)
            else:
                x = b(x, rope, attn_mask, None if caches is None else caches[i])
        return self.norm(x) @ self.emb.weight.T

    @torch.no_grad()
    def generate(self, prompts, max_new, eos_id, pad_id, temperature=0.0):
        """prompts: LongTensor B,T (all same length). Returns B,max_new tokens."""
        caches = [{} for _ in self.blocks]
        done = torch.zeros(prompts.shape[0], dtype=torch.bool)
        out = []
        logits = self(prompts, caches=caches)[:, -1]
        pos = prompts.shape[1]
        for step in range(max_new):
            if step > 0:
                logits = self(nxt[:, None], caches=caches, offset=pos)[:, -1]
                pos += 1
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1)[:, 0]
            nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
            out.append(nxt)
            done |= nxt == eos_id
            if done.all():
                break
        return torch.stack(out, 1)

    def grow_depth(self):
        """Function-preserving growth: append a block whose output projections
        are zeroed so the network computes the same function initially."""
        c = self.c
        c.n_layer += 1
        nb = Block(c)
        nn.init.zeros_(nb.attn.o.weight)
        nn.init.zeros_(nb.mlp.down.weight)
        self.blocks.append(nb)
