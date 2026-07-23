!pip install -q torch datasets tiktoken tqdm matplotlib wandb




from google.colab import drive
drive.mount('/content/drive')

CHECKPOINT_DIR = "/content/drive/MyDrive/smolgpt_checkpoints"
import os
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "smolgpt_ckpt.pt")


import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tiktoken
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

USE_WANDB = False
if USE_WANDB:
    import wandb
    wandb.login()




config = {
    "vocab_size": 50257,
    "block_size": 256,
    "n_layer": 8,
    "n_head": 8,
    "n_embd": 512,
    "dropout": 0.1,
    "batch_size": 32,
    "grad_accum_steps": 2,
    "learning_rate": 3e-4,
    "min_lr": 3e-5,
    "warmup_iters": 200,
    "max_iters": 6000,
    "eval_interval": 300,
    "eval_iters": 50,
    "grad_clip": 1.0,
    "checkpoint_interval": 500,
}








import numpy as np

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

TRAIN_BIN = os.path.join(CHECKPOINT_DIR, "train.bin")
VAL_BIN = os.path.join(CHECKPOINT_DIR, "val.bin")

if os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN):
    print("Found cached tokenized data on Drive, skipping re-tokenization.")
else:
    raw_dataset = load_dataset("roneneldan/TinyStories", split="train")
    split_dataset = raw_dataset.train_test_split(test_size=0.02, seed=42)
    split_dataset["val"] = split_dataset.pop("test")

    def tokenize(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(EOT)
        return {"ids": ids, "len": len(ids)}

    for split_name, dset in split_dataset.items():


        tokenized = dset.map(
            tokenize,
            remove_columns=["text"],
            desc=f"tokenizing {split_name}",
            num_proc=2,
            writer_batch_size=1000,
        )
        arr_len = np.sum(tokenized["len"], dtype=np.uint64)
        out_path = TRAIN_BIN if split_name == "train" else VAL_BIN
        arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(arr_len,))

        idx = 0
        total_batches = 1024
        for batch_idx in range(total_batches):
            batch = tokenized.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"Wrote {idx:,} tokens to {out_path}")


def get_batch(split):


    path = TRAIN_BIN if split == "train" else VAL_BIN
    d = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(d) - config["block_size"] - 1, (config["batch_size"],))
    x = torch.stack([torch.from_numpy(d[i:i + config["block_size"]].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + config["block_size"] + 1].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            X, Y = get_batch(split)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it):

    if it < config["warmup_iters"]:
        return config["learning_rate"] * (it + 1) / config["warmup_iters"]
    if it > config["max_iters"]:
        return config["min_lr"]
    decay_ratio = (it - config["warmup_iters"]) / (config["max_iters"] - config["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])


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


model = SmolGPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95), weight_decay=0.1)

start_iter = 0
train_losses, val_losses, loss_steps = [], [], []

if os.path.exists(CHECKPOINT_PATH):
    print("Found existing checkpoint — resuming training from it.")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_iter = ckpt["iter"] + 1
    train_losses = ckpt.get("train_losses", [])
    val_losses = ckpt.get("val_losses", [])
    loss_steps = ckpt.get("loss_steps", [])
    print(f"Resumed at iteration {start_iter}")
else:
    print(f"No checkpoint found at {CHECKPOINT_PATH} — starting fresh.")

print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

if USE_WANDB:
    wandb.init(project="smolgpt", config=config, resume="allow")

def save_checkpoint(it):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": it,
        "config": config,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "loss_steps": loss_steps,
    }, CHECKPOINT_PATH)
    print(f"  -> checkpoint saved at iter {it}")


start = time.time()
for it in range(start_iter, config["max_iters"]):
    lr = get_lr(it)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    if it % config["eval_interval"] == 0 or it == config["max_iters"] - 1:
        losses = estimate_loss(model)
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
              f"lr {lr:.2e}, elapsed {time.time()-start:.0f}s")
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])
        loss_steps.append(it)
        if USE_WANDB:
            wandb.log({"train_loss": losses["train"], "val_loss": losses["val"], "lr": lr, "iter": it})

    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(config["grad_accum_steps"]):
        xb, yb = get_batch("train")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            _, loss = model(xb, yb)
            loss = loss / config["grad_accum_steps"]
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
    optimizer.step()

    if it > 0 and it % config["checkpoint_interval"] == 0:
        save_checkpoint(it)

save_checkpoint(config["max_iters"] - 1)
print(f"Training complete in {time.time()-start:.0f}s")


plt.figure(figsize=(8, 5))
plt.plot(loss_steps, train_losses, label="train loss")
plt.plot(loss_steps, val_losses, label="val loss")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("SmolGPT training curves")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(CHECKPOINT_DIR, "loss_curve.png"))
plt.show()


model.eval()
prompt = "Once upon a time"
ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
out = model.generate(ids, max_new_tokens=200, temperature=0.8, top_k=50)
print(enc.decode(out[0].tolist()))


torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "smolgpt_final_weights.pt"))
print("Saved final weights to Drive.")
