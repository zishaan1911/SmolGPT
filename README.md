# Mini GPT — Colab

A small GPT-style decoder-only transformer, trained from scratch on Colab,
plus dataset suggestions for language model pretraining.

## Features

- Custom decoder-only transformer (causal self-attention, MLP blocks) —
  ~40M parameters (8 layers, 512 dim) by default
- Trains on the full [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset
- Memory-safe tokenization: streams tokens straight to a memory-mapped
  `.bin` file instead of holding the whole dataset in RAM (fixes free-tier
  Colab OOM crashes)
- Checkpointing + auto-resume via Google Drive — safe against Colab
  disconnects/timeouts
- Mixed precision (bf16 autocast), cosine LR schedule with warmup,
  gradient clipping, gradient accumulation
- Loss curve plotting, optional Weights & Biases logging

## Usage

1. Open `train_mini_gpt.py` in Google Colab (Runtime -> Change runtime
   type -> GPU, T4 is fine on the free tier).
2. Paste each `# %%` block into its own cell (or paste the whole file
   into one cell and run it top to bottom).
3. First run will mount your Google Drive, tokenize TinyStories (~5-15
   min), then start training. Re-running later resumes automatically
   from the last checkpoint.

## Requirements

See `requirements.txt`. Installed automatically by Cell 1 in the script.

## Dataset suggestions

- **TinyStories** (`roneneldan/TinyStories`) — small, simple stories; trains
  fast and produces coherent output even from a tiny from-scratch model.
- **WikiText-2 / WikiText-103** — clean Wikipedia text, standard benchmark.
- **OpenWebText** — Reddit-sourced web crawl, GPT-2-style training data.
- **C4 / The Pile** — large-scale pretraining corpora (need serious compute).
- **BookCorpus** — long-form narrative text.
- Domain-specific corpora (code, legal, medical, multilingual) for
  specialized models.
