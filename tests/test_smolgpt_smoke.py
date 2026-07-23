# Standalone smoke test for the SmolGPT architecture.
#
# train_smolgpt.py is meant to be pasted straight into a Colab cell and
# has Colab-only side effects at import time (!pip install, drive.mount),
# so it can't be imported directly in CI. This file re-declares the same
# model classes in isolation and runs a tiny forward/backward pass on CPU
# to catch basic breakage (shape errors, NaNs, etc.) on every push.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

TEST_CONFIG = {
    "vocab_size": 65,
    "block_size": 16,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 32,
    "dropout": 0.0,
}


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg["n_head"]
        self.head_dim = cfg["n_embd"] // cfg["n_head"]
        self.qkv = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"])
        self.proj = nn.Linear(cfg["n_embd"], cfg["n_embd"])
        self.attn_drop = nn.Dropout(cfg["dropout"])
        self.resid_drop = nn.Dropout(cfg["dropout"])
        mask = torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
        self.register_buffer("mask", mask.view(1, 1, cfg["block_size"], cfg["block_size"]))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"]),
            nn.GELU(),
            nn.Linear(4 * cfg["n_embd"], cfg["n_embd"]),
            nn.Dropout(cfg["dropout"]),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SmolGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg["block_size"], cfg["n_embd"]))
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb[:, :T, :]
        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def test_forward_backward_pass():
    model = SmolGPT(TEST_CONFIG)
    x = torch.randint(0, TEST_CONFIG["vocab_size"], (4, TEST_CONFIG["block_size"]))
    y = torch.randint(0, TEST_CONFIG["vocab_size"], (4, TEST_CONFIG["block_size"]))

    logits, loss = model(x, y)

    assert logits.shape == (4, TEST_CONFIG["block_size"], TEST_CONFIG["vocab_size"])
    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"


def test_causal_mask_blocks_future_tokens():
    model = SmolGPT(TEST_CONFIG)
    model.eval()
    x = torch.randint(0, TEST_CONFIG["vocab_size"], (1, TEST_CONFIG["block_size"]))

    with torch.no_grad():
        logits_full, _ = model(x)

    x_truncated = x[:, :8]
    with torch.no_grad():
        logits_truncated, _ = model(x_truncated)

    assert torch.allclose(logits_full[:, :8, :], logits_truncated, atol=1e-5)
