# ============================================================
# MINI GPT — Train a small language model from scratch on Colab
# ============================================================
# HOW TO USE:
# 1. Open a new notebook at https://colab.research.google.com
# 2. Runtime -> Change runtime type -> GPU (T4 is fine)
# 3. Paste each "# %%" block below into its own cell (or just
#    paste the whole file into one cell and run it)
# ============================================================

# %% [Cell 1] Install dependencies
!pip install -q torch datasets tiktoken tqdm

# %% [Cell 2] Imports & config
import math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---- Model / training hyperparameters (tune to your GPU) ----
config = {
    "vocab_size": 50257,      # GPT-2 tokenizer vocab size
    "block_size": 256,        # context length (tokens per training sample)
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "dropout": 0.1,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "max_iters": 3000,
    "eval_interval": 300,
    "eval_iters": 50,
}

# %% [Cell 3] Load & tokenize a dataset
# Swap "roneneldan/TinyStories" for any dataset in the suggestions
# section below (e.g. "wikitext", "wikitext-103-raw-v1").
raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:2%]")  # small slice for a quick demo
text = "\n".join(raw_dataset["text"])
print(f"Loaded {len(text):,} characters of raw text")

enc = tiktoken.get_encoding("gpt2")
data = torch.tensor(enc.encode(text), dtype=torch.long)
print(f"Encoded into {len(data):,} tokens")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# %% [Cell 4] Data loading helper
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - config["block_size"] - 1, (config["batch_size"],))
    x = torch.stack([d[i:i + config["block_size"]] for i in ix])
    y = torch.stack([d[i + 1:i + config["block_size"] + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# %% [Cell 5] Model definition — a small GPT (decoder-only transformer)
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["n_embd"] % cfg["n_head"] == 0
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

class MiniGPT(nn.Module):
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

# %% [Cell 6] Train
model = MiniGPT(config).to(device)
print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

start = time.time()
for it in range(config["max_iters"]):
    if it % config["eval_interval"] == 0 or it == config["max_iters"] - 1:
        losses = estimate_loss(model)
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
              f"elapsed {time.time()-start:.0f}s")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# %% [Cell 7] Generate sample text
prompt = "Once upon a time"
ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
out = model.generate(ids, max_new_tokens=200, temperature=0.8, top_k=50)
print(enc.decode(out[0].tolist()))

# %% [Cell 8] Save checkpoint (download it from Colab's file browser)
torch.save(model.state_dict(), "mini_gpt.pt")
print("Saved to mini_gpt.pt")
