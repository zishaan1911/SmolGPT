# SmolGPT

A small GPT-style decoder-only transformer, trained from scratch on Colab on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), with a
story editor that runs the model entirely in your browser.

**[Try it →](https://zishaan1911.github.io/SmolGPT/)** · [Model on Hugging Face](https://huggingface.co/zishaan1911/smolGPT)

## The model

| | |
| --- | --- |
| Parameters | 51.1M (embedding and output layers share weights) |
| Architecture | decoder-only transformer, 8 layers, 8 heads, width 512 |
| Context | 256 tokens |
| Tokenizer | GPT-2 BPE, 50,257 tokens |
| Training data | TinyStories, 465M tokens after tokenization |

The exported ONNX file stores the output layer as a separate copy of the
embedding matrix, so it is ~300 MB (about 77M fp32 values) even though the
model has 51.1M distinct parameters.

## Browser demo

`index.html`, `app.js` and `worker.js` form a static site (served by GitHub
Pages) that runs the ONNX export with
[transformers.js](https://github.com/huggingface/transformers.js):

- Inference runs in a Web Worker on WebGPU when available, otherwise on the
  CPU via WebAssembly. Nothing leaves the device.
- Generation runs in 4-token steps that reuse the key/value cache, so **stop**
  takes effect within a step and chunking costs nothing.
- The prompt is trimmed by tokens so the story plus the requested length
  always fits the 256-token context. A meter shows how much of the story
  the model can currently see.
- Stories are kept in `localStorage`, with undo, rewrite and copy, plus
  controls for length, temperature, top-k and repetition penalty.
- About one TinyStories row in ten stores curly quotes as mojibake (`â€œ`
  for `“`), and the model learned to write them that way. The worker maps
  these back to the intended characters while streaming.

To run it locally, serve the folder over HTTP:

```bash
python -m http.server 8000
```

## Training

- Custom decoder-only transformer (causal self-attention, MLP blocks)
- Memory-safe tokenization: streams tokens straight to a memory-mapped
  `.bin` file instead of holding the whole dataset in RAM (fixes free-tier
  Colab OOM crashes)
- Checkpoints and tokenized data persisted to the Hugging Face Hub, with
  auto-resume after Colab disconnects
- Mixed precision (bf16 autocast), cosine LR schedule with warmup,
  gradient clipping, gradient accumulation
- Loss curve plotting, optional Weights & Biases logging

### Usage

1. Open a new notebook in Google Colab (Runtime -> Change runtime type
   -> GPU, T4 is fine on the free tier).
2. Set `HF_REPO_ID` (and optionally `HF_TOKEN`) at the top of
   `train_smolgpt.py`, paste the whole script into a single cell and run it.
3. The first run tokenizes TinyStories (~5-15 min) and uploads the token
   files to your Hub repo, then trains. Re-running resumes from the last
   checkpoint on the Hub.

See `requirements.txt` for dependencies; the script installs them itself.

## Dataset suggestions

- **TinyStories** (`roneneldan/TinyStories`): small, simple stories; trains
  fast and produces coherent output even from a tiny from-scratch model.
- **WikiText-2 / WikiText-103**: clean Wikipedia text, a standard benchmark.
- **OpenWebText**: Reddit-sourced web crawl, GPT-2-style training data.
- **C4 / The Pile**: large-scale pretraining corpora (need serious compute).
- **BookCorpus**: long-form narrative text.
- Domain-specific corpora (code, legal, medical, multilingual) for
  specialized models.
