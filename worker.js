// SmolGPT inference worker.
// Runs transformers.js off the main thread so the page stays responsive
// while the model downloads and generates (wasm inference is CPU-heavy).

import {
  AutoTokenizer,
  Tensor,
  TextStreamer,
  env,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODEL_ID = "zishaan1911/smolGPT";
const CONTEXT = 256; // n_positions of the model
const EOS = 50256; // <|endoftext|>
// Tokens per generate() call. WASM inference never yields to the event loop,
// so a stop message can only be read between calls; the KV cache is carried
// over, so chunking costs almost nothing.
const CHUNK = 4;

env.allowLocalModels = false;
env.allowRemoteModels = true;

let tokenizer = null;
let generator = null;
let stopRequested = false;

// Aggregate download progress over every file the pipeline fetches.
const files = new Map();
function onProgress(p) {
  if (p.status === "progress" && p.total) {
    files.set(p.file, { loaded: p.loaded, total: p.total });
  } else if (p.status === "done" && files.has(p.file)) {
    const f = files.get(p.file);
    files.set(p.file, { loaded: f.total, total: f.total });
  } else {
    return;
  }
  let loaded = 0;
  let total = 0;
  for (const f of files.values()) {
    loaded += f.loaded;
    total += f.total;
  }
  self.postMessage({ type: "progress", loaded, total, file: p.file });
}

async function hasWebGPU() {
  if (!("gpu" in navigator)) return false;
  try {
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

async function load() {
  try {
    // The tokenizer is small; load it first so the context meter works
    // while the 300 MB of weights are still downloading.
    tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback: onProgress });
    self.postMessage({ type: "tokenizer" });

    let device = (await hasWebGPU()) ? "webgpu" : "wasm";
    self.postMessage({ type: "device", device });
    try {
      generator = await pipeline("text-generation", MODEL_ID, {
        dtype: "fp32",
        device,
        progress_callback: onProgress,
      });
    } catch (err) {
      if (device !== "webgpu") throw err;
      // An adapter can exist and still fail at session creation; fall back to CPU.
      device = "wasm";
      self.postMessage({ type: "device", device, fallback: true });
      generator = await pipeline("text-generation", MODEL_ID, {
        dtype: "fp32",
        device,
        progress_callback: onProgress,
      });
    }
    self.postMessage({ type: "ready", device });
  } catch (err) {
    self.postMessage({ type: "error", message: String(err?.message ?? err) });
  }
}

function countTokens(text) {
  return text ? tokenizer.encode(text).length : 0;
}

// Keep only the most recent tokens that fit next to the requested output.
function windowed(text, maxNew) {
  const ids = tokenizer.encode(text);
  const budget = CONTEXT - maxNew;
  return ids.length <= budget
    ? { ids, dropped: 0 }
    : { ids: ids.slice(-budget), dropped: ids.length - budget };
}

// About one TinyStories row in ten stores curly quotes and dashes as mojibake
// (UTF-8 bytes misread as Windows-1252, e.g. "â€œ" for "“"), and the model
// learned to write them that way. Map such sequences back to the characters
// they encode.
const CP1252 = {
  0x20ac: 0x80, 0x201a: 0x82, 0x0192: 0x83, 0x201e: 0x84, 0x2026: 0x85, 0x2020: 0x86,
  0x2021: 0x87, 0x02c6: 0x88, 0x2030: 0x89, 0x0160: 0x8a, 0x2039: 0x8b, 0x0152: 0x8c,
  0x017d: 0x8e, 0x2018: 0x91, 0x2019: 0x92, 0x201c: 0x93, 0x201d: 0x94, 0x2022: 0x95,
  0x2013: 0x96, 0x2014: 0x97, 0x02dc: 0x98, 0x2122: 0x99, 0x0161: 0x9a, 0x203a: 0x9b,
  0x0153: 0x9c, 0x017e: 0x9e, 0x0178: 0x9f,
};
const CONT = "[\\u0080-\\u00bf\\u0152\\u0153\\u0160\\u0161\\u0178\\u017d\\u017e\\u0192\\u02c6\\u02dc" +
  "\\u2013\\u2014\\u2018-\\u201a\\u201c-\\u201e\\u2020-\\u2022\\u2026\\u2030\\u2039\\u203a\\u20ac\\u2122]";
// The lead byte fixes the length: C2-DF take one continuation, E0-EF two, F0-F4 three.
const MOJIBAKE = new RegExp(
  `[\\u00c2-\\u00df]${CONT}|[\\u00e0-\\u00ef]${CONT}{2}|[\\u00f0-\\u00f4]${CONT}{3}`,
  "g",
);
const PENDING_TAIL = new RegExp(`(?:[\\u00c2-\\u00f4]${CONT}{0,2}|\\uFFFD)$`);
const utf8 = new TextDecoder("utf-8", { fatal: true });

function repairMojibake(text) {
  return text.replace(MOJIBAKE, (run) => {
    const bytes = Uint8Array.from(run, (ch) => {
      const code = ch.codePointAt(0);
      return code <= 0xff ? code : CP1252[code];
    });
    try {
      return utf8.decode(bytes);
    } catch {
      return run;
    }
  });
}

const toTensor = (ids) => new Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
const nextTask = () => new Promise((resolve) => setTimeout(resolve, 0));

async function generate({ id, text, maxNew, temperature, topK, repetitionPenalty }) {
  const { ids, dropped } = windowed(text, maxNew);
  const model = generator.model;
  const t0 = performance.now();
  let first = null;
  let sequence = ids;
  let past = null;
  let sent = "";
  const produced = [];
  stopRequested = false;

  // Decode everything generated so far and send only the new characters, so
  // multi-token words and chunk boundaries never garble the text. A trailing
  // U+FFFD (half a UTF-8 character) or the start of a mojibake sequence is
  // held back until the next token completes it.
  const flush = (final) => {
    const full = repairMojibake(tokenizer.decode(produced, { skip_special_tokens: true }));
    const stable = final ? full : full.replace(PENDING_TAIL, "");
    if (stable.length > sent.length) {
      self.postMessage({ type: "chunk", id, text: stable.slice(sent.length) });
      sent = stable;
    }
  };
  const onToken = (tokens) => {
    if (first === null) first = performance.now() - t0;
    produced.push(...tokens.map(Number).filter((t) => t !== EOS));
    flush(false);
  };

  let ended = false;
  try {
    while (produced.length < maxNew && !ended && !stopRequested) {
      const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true,
        callback_function: () => {},
        token_callback_function: onToken,
      });
      const n = Math.min(CHUNK, maxNew - produced.length);
      const out = await model.generate({
        input_ids: toTensor(sequence),
        attention_mask: toTensor(sequence.map(() => 1)),
        past_key_values: past,
        max_new_tokens: n,
        do_sample: true,
        temperature,
        top_k: topK > 0 ? topK : 0,
        repetition_penalty: repetitionPenalty,
        streamer,
        return_dict_in_generate: true,
      });
      past = out.past_key_values;
      sequence = Array.from(out.sequences.data, Number);
      ended = sequence.at(-1) === EOS;
      await nextTask(); // let a pending "stop" message in
    }
  } catch (err) {
    self.postMessage({ type: "error", id, message: String(err?.message ?? err) });
    return;
  }
  flush(true);

  self.postMessage({
    type: "done",
    id,
    tokens: produced.length,
    promptTokens: ids.length,
    dropped,
    firstTokenMs: first,
    elapsedMs: performance.now() - t0,
    ended,
    stopped: stopRequested && !ended,
  });
}

self.addEventListener("message", (e) => {
  const msg = e.data;
  if (msg.type === "load") load();
  else if (msg.type === "count" && tokenizer) {
    self.postMessage({ type: "count", id: msg.id, tokens: countTokens(msg.text) });
  } else if (msg.type === "generate" && generator) generate(msg);
  else if (msg.type === "stop") stopRequested = true;
});
