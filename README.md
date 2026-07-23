# Mini GPT — Colab

A small GPT-style decoder-only transformer, trained from scratch on Colab,
plus dataset suggestions for language model pretraining.

## Usage

Open `train_mini_gpt.py` in Google Colab (Runtime -> GPU), and paste each
`# %%` block into its own cell, or paste the whole file into one cell and
run it top to bottom.

## Dataset suggestions

- **TinyStories** (`roneneldan/TinyStories`) — small, simple stories; trains
  fast and produces coherent output even from a tiny from-scratch model.
- **WikiText-2 / WikiText-103** — clean Wikipedia text, standard benchmark.
- **OpenWebText** — Reddit-sourced web crawl, GPT-2-style training data.
- **C4 / The Pile** — large-scale pretraining corpora (need serious compute).
- **BookCorpus** — long-form narrative text.
- Domain-specific corpora (code, legal, medical, multilingual) for
  specialized models.
