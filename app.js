// SmolGPT story editor. The model runs in worker.js; this file is the UI.

const CONTEXT = 256;
const STORIES_KEY = "smolgpt_stories_v2";
const LEGACY_KEY = "smolgpt_conversations_v1";
const SETTINGS_KEY = "smolgpt_settings_v1";
const DEFAULTS = { len: 120, temp: 0.8, topk: 40, rep: 1.1 };

const $ = (id) => document.getElementById(id);
const el = {
  loader: $("loader"),
  loaderTitle: $("loaderTitle"),
  loaderMeta: $("loaderMeta"),
  loaderFill: $("loaderFill"),
  loaderNote: $("loaderNote"),
  title: $("storyTitle"),
  undo: $("undoBtn"),
  retry: $("retryBtn"),
  copy: $("copyBtn"),
  starter: $("starter"),
  chips: $("starterChips"),
  story: $("story"),
  composer: $("composer"),
  input: $("input"),
  go: $("goBtn"),
  goLabel: $("goLabel"),
  goKey: $("goKey"),
  library: $("library"),
  newStory: $("newStoryBtn"),
  ctxPrompt: $("ctxPrompt"),
  ctxNew: $("ctxNew"),
  ctxPromptN: $("ctxPromptN"),
  ctxNewN: $("ctxNewN"),
  ctxUsed: $("ctxUsed"),
  ctxNote: $("ctxNote"),
  rStatus: $("rStatus"),
  rBackend: $("rBackend"),
  rFirst: $("rFirst"),
  rSpeed: $("rSpeed"),
  rLast: $("rLast"),
  panel: $("panel"),
  panelToggle: $("panelToggle"),
  scrim: $("scrim"),
  toast: $("toast"),
  knobs: {
    len: [$("len"), $("lenOut"), (v) => String(v)],
    temp: [$("temp"), $("tempOut"), (v) => v.toFixed(2)],
    topk: [$("topk"), $("topkOut"), (v) => (v === 0 ? "off" : String(v))],
    rep: [$("rep"), $("repOut"), (v) => v.toFixed(2)],
  },
};

// ---------------------------------------------------------------- storage

function read(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function write(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* private mode or quota: the editor still works for this visit */
  }
}

function newStory() {
  return { id: crypto.randomUUID(), title: "", titled: false, segments: [], updated: Date.now() };
}

function loadStories() {
  const stories = read(STORIES_KEY, null);
  if (stories) return stories;
  // Carry over conversations saved by the previous chat-style UI.
  const legacy = read(LEGACY_KEY, []);
  return legacy
    .filter((c) => c.messages?.length)
    .map((c) => ({
      id: c.id,
      title: c.title === "New story" ? "" : c.title,
      titled: false,
      segments: c.messages.map((m) => ({ by: m.role === "user" ? "you" : "model", text: m.text })),
      updated: Date.now(),
    }));
}

let stories = loadStories();
if (!stories.length) stories = [newStory()];
let active = stories[0];
const settings = { ...DEFAULTS, ...read(SETTINGS_KEY, {}) };

function save() {
  active.updated = Date.now();
  write(STORIES_KEY, stories);
}

// ---------------------------------------------------------------- story text

const storyText = (segments = active.segments) => segments.map((s) => s.text).join("");

// Join a new passage to the story the way a writer would: with a space,
// unless the story already ends in whitespace or the passage starts with punctuation.
function joinText(prev, next) {
  if (!prev || /\s$/.test(prev) || /^[\s,.;:!?'")\]]/.test(next)) return next;
  return " " + next;
}

// Title from the first sentence, minus the fairy-tale opener, cut at a word boundary.
function titleFrom(text) {
  let first = text.trim().replace(/\s+/g, " ").split(/(?<=[.!?])\s/)[0] ?? "";
  first = first.replace(/^once upon a time,?\s*/i, "").replace(/[.!?,;:]+$/, "");
  if (first.length > 44) first = first.slice(0, 44).replace(/\s+\S*$/, "") + "…";
  return first.charAt(0).toUpperCase() + first.slice(1);
}

// ---------------------------------------------------------------- rendering

let live = null; // { segment, node } while generating

function renderStory() {
  el.story.textContent = "";
  el.starter.hidden = active.segments.length > 0;
  active.segments.forEach((seg, i) => {
    const span = document.createElement("span");
    span.className = `seg ${seg.by}`;
    span.textContent = seg.text;
    if (seg.by === "model") span.title = "Written by SmolGPT";
    el.story.appendChild(span);
    if (live && live.segment === seg) {
      span.classList.add("live");
      live.node = span;
      const caret = document.createElement("span");
      caret.className = "caret";
      el.story.appendChild(caret);
    }
    if (seg.ended && i === active.segments.length - 1) {
      const end = document.createElement("span");
      end.className = "the-end";
      end.textContent = "the end";
      el.story.appendChild(end);
    }
  });
  el.title.value = active.title;
  updateTools();
  scheduleCount();
}

function renderLibrary() {
  el.library.textContent = "";
  const sorted = [...stories].sort((a, b) => b.updated - a.updated);
  for (const s of sorted) {
    const li = document.createElement("li");
    if (s === active) li.classList.add("active");
    const open = document.createElement("button");
    open.className = "open";
    open.textContent = s.title || titleFrom(storyText(s.segments)) || "Untitled story";
    open.addEventListener("click", () => switchTo(s));
    const words = storyText(s.segments).trim().split(/\s+/).filter(Boolean).length;
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = words ? `${words}w` : "empty";
    const del = document.createElement("button");
    del.className = "del";
    del.textContent = "✕";
    del.title = "Delete story";
    del.setAttribute("aria-label", `Delete ${open.textContent}`);
    del.addEventListener("click", () => removeStory(s));
    li.append(open, meta, del);
    el.library.appendChild(li);
  }
}

function updateTools() {
  const last = active.segments.at(-1);
  const busy = !!live;
  el.undo.disabled = busy || !last;
  el.retry.disabled = busy || !modelReady || last?.by !== "model";
  el.copy.disabled = !last;
  if (busy) {
    el.go.disabled = false;
    el.go.classList.add("stop");
    el.goLabel.textContent = "stop";
    el.goKey.textContent = "esc";
  } else {
    el.go.classList.remove("stop");
    el.goLabel.textContent = modelReady ? (el.input.value.trim() ? "add & continue" : "continue") : "loading…";
    el.goKey.textContent = "↵";
    el.go.disabled = !modelReady;
  }
}

function toast(message) {
  el.toast.textContent = message;
  el.toast.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.toast.classList.remove("show"), 1800);
}

// ---------------------------------------------------------------- worker

const worker = new Worker(new URL("./worker.js?v=3", import.meta.url), { type: "module" });
let tokenizerReady = false;
let modelReady = false;
let runId = 0;
let countId = 0;

worker.addEventListener("message", ({ data: msg }) => {
  switch (msg.type) {
    case "progress": {
      const pct = msg.total ? (msg.loaded / msg.total) * 100 : 0;
      el.loaderFill.style.width = `${pct.toFixed(1)}%`;
      el.loaderMeta.textContent = `${(msg.loaded / 1e6).toFixed(0)} / ${(msg.total / 1e6).toFixed(0)} MB`;
      break;
    }
    case "tokenizer":
      tokenizerReady = true;
      scheduleCount();
      break;
    case "device":
      el.rBackend.textContent = msg.device === "webgpu" ? "WebGPU" : "WASM (CPU)";
      el.loaderTitle.textContent = msg.fallback
        ? "WebGPU unavailable, loading on the CPU"
        : `Fetching the model · ${msg.device === "webgpu" ? "WebGPU" : "CPU"}`;
      break;
    case "ready":
      modelReady = true;
      el.rStatus.textContent = "ready";
      el.rStatus.className = "ok";
      el.loaderTitle.textContent = "Ready";
      el.loaderFill.style.width = "100%";
      el.loader.classList.add("done");
      setTimeout(() => el.loader.classList.add("gone"), 450);
      updateTools();
      break;
    case "count":
      if (msg.id === countId) paintContext(msg.tokens, true);
      break;
    case "chunk":
      if (live && msg.id === runId) {
        live.segment.text += msg.text;
        live.node.textContent = live.segment.text;
        followCaret();
      }
      break;
    case "done":
      if (msg.id === runId) finishRun(msg);
      break;
    case "error":
      if (live && msg.id === runId) {
        finishRun({ tokens: 0, stopped: true });
        toast("Generation failed. See the console for details.");
        console.error(msg.message);
      } else {
        el.loader.classList.add("error");
        el.loaderTitle.textContent = "Couldn't load SmolGPT";
        el.loaderNote.textContent = `${msg.message}. Reload to try again.`;
        el.rStatus.textContent = "error";
        el.rStatus.className = "bad";
      }
      break;
  }
});
worker.postMessage({ type: "load" });

function followCaret() {
  const caret = el.story.querySelector(".caret");
  if (!caret) return;
  const r = caret.getBoundingClientRect();
  const limit = window.innerHeight - el.composer.offsetHeight - 48;
  if (r.bottom > limit) window.scrollBy({ top: r.bottom - limit, behavior: "auto" });
}

// ---------------------------------------------------------------- generation

function generate() {
  const text = storyText();
  if (!text.trim()) {
    toast("Write a first line to start the story");
    el.input.focus();
    return;
  }
  const segment = { by: "model", text: "" };
  active.segments.push(segment);
  live = { segment, node: null };
  runId += 1;
  el.rStatus.textContent = "writing";
  el.rStatus.className = "";
  renderStory();
  worker.postMessage({
    type: "generate",
    id: runId,
    text,
    maxNew: settings.len,
    temperature: settings.temp,
    topK: settings.topk,
    repetitionPenalty: settings.rep,
  });
}

function finishRun(msg) {
  const seg = live.segment;
  live = null;
  if (!seg.text.trim()) {
    active.segments.splice(active.segments.indexOf(seg), 1);
  } else if (msg.ended) {
    seg.ended = true;
  }
  el.rStatus.textContent = msg.stopped ? "stopped" : "ready";
  el.rStatus.className = "ok";
  if (msg.tokens) {
    const secs = msg.elapsedMs / 1000;
    el.rFirst.textContent = `${Math.round(msg.firstTokenMs)} ms`;
    el.rSpeed.textContent = `${(msg.tokens / secs).toFixed(1)} tok/s`;
    el.rLast.textContent = `${msg.tokens} tok · ${secs.toFixed(1)} s`;
  }
  if (msg.dropped) toast(`The oldest ${msg.dropped} tokens were outside the model's view`);
  save();
  renderStory();
  renderLibrary();
  el.input.focus();
}

function stop() {
  if (live) worker.postMessage({ type: "stop" });
}

// ---------------------------------------------------------------- context meter

let countTimer = 0;
function scheduleCount() {
  clearTimeout(countTimer);
  countTimer = setTimeout(() => {
    const text = storyText() + joinText(storyText(), el.input.value.trim());
    // Paint an estimate straight away; the exact count can queue behind a
    // generation or model start-up in the worker.
    paintContext(Math.ceil(text.length / 4), false);
    if (tokenizerReady) {
      countId += 1;
      worker.postMessage({ type: "count", id: countId, text });
    }
  }, 120);
}

function paintContext(tokens, exact) {
  const reserved = settings.len;
  const budget = CONTEXT - reserved;
  const seen = Math.min(tokens, budget);
  el.ctxPrompt.style.width = `${(seen / CONTEXT) * 100}%`;
  el.ctxNew.style.width = `${(reserved / CONTEXT) * 100}%`;
  el.ctxPromptN.textContent = `${exact ? "" : "≈"}${seen}`;
  el.ctxNewN.textContent = reserved;
  el.ctxUsed.textContent = seen + reserved;
  const over = tokens > budget;
  el.ctxNote.classList.toggle("over", over);
  el.ctxNote.textContent = over
    ? `The story is ${tokens} tokens long; SmolGPT sees only the last ${budget}. Earlier lines still stay in your story.`
    : "The model sees 256 tokens at once: the end of your story plus the words it is about to write.";
}

// ---------------------------------------------------------------- actions

function switchTo(story) {
  if (live) return toast("Stop the current passage first");
  active = story;
  renderStory();
  renderLibrary();
  closePanel();
}

function removeStory(story) {
  if (live && story === active) return toast("Stop the current passage first");
  stories = stories.filter((s) => s !== story);
  if (!stories.length) stories = [newStory()];
  if (story === active) active = stories[0];
  write(STORIES_KEY, stories);
  renderStory();
  renderLibrary();
}

el.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  if (live) return stop();
  if (!modelReady) return;
  const draft = el.input.value.trim();
  if (!draft && active.segments.at(-1)?.ended) {
    toast("This story has ended. Add a line to keep it going.");
    el.input.focus();
    return;
  }
  if (draft) {
    const passage = joinText(storyText(), draft);
    active.segments.push({ by: "you", text: passage });
    // A new line after "the end" starts a fresh chapter rather than ending again.
    active.segments.forEach((s) => delete s.ended);
    if (!active.titled && !active.title) active.title = titleFrom(storyText());
    el.input.value = "";
    autosize();
    save();
    renderLibrary();
  }
  generate();
});

el.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    el.composer.requestSubmit();
  }
});

function autosize() {
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(el.input.scrollHeight, 180)}px`;
}
el.input.addEventListener("input", () => {
  autosize();
  updateTools();
  scheduleCount();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (live) stop();
    else closePanel();
  }
});

el.chips.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  el.input.value = chip.textContent.trim();
  autosize();
  updateTools();
  scheduleCount();
  if (modelReady) el.composer.requestSubmit();
  else el.input.focus();
});

el.undo.addEventListener("click", () => {
  const last = active.segments.pop();
  if (!last) return;
  if (last.by === "you" && !el.input.value.trim()) {
    el.input.value = last.text.trim();
    autosize();
  }
  const prev = active.segments.at(-1);
  if (prev) delete prev.ended;
  save();
  renderStory();
  renderLibrary();
});

el.retry.addEventListener("click", () => {
  if (active.segments.at(-1)?.by !== "model") return;
  active.segments.pop();
  generate();
});

el.copy.addEventListener("click", async () => {
  const body = storyText().trim();
  const text = active.title ? `${active.title}\n\n${body}` : body;
  try {
    await navigator.clipboard.writeText(text);
    toast("Story copied");
  } catch {
    toast("Couldn't reach the clipboard");
  }
});

el.title.addEventListener("input", () => {
  active.title = el.title.value;
  active.titled = true;
  save();
  renderLibrary();
});

el.newStory.addEventListener("click", () => {
  if (live) return toast("Stop the current passage first");
  const empty = stories.find((s) => !s.segments.length);
  active = empty ?? newStory();
  if (!empty) stories.unshift(active);
  save();
  renderStory();
  renderLibrary();
  closePanel();
  el.input.focus();
});

// ---------------------------------------------------------------- knobs

function applyKnobs() {
  for (const [key, [input, output, fmt]] of Object.entries(el.knobs)) {
    input.value = settings[key];
    output.textContent = fmt(settings[key]);
  }
}
for (const [key, [input, output, fmt]] of Object.entries(el.knobs)) {
  input.addEventListener("input", () => {
    settings[key] = Number(input.value);
    output.textContent = fmt(settings[key]);
    write(SETTINGS_KEY, settings);
    if (key === "len") scheduleCount();
  });
}
$("resetKnobs").addEventListener("click", () => {
  Object.assign(settings, DEFAULTS);
  write(SETTINGS_KEY, settings);
  applyKnobs();
  scheduleCount();
});

// ---------------------------------------------------------------- mobile panel

function openPanel() {
  el.panel.classList.add("open");
  el.scrim.hidden = false;
  el.panelToggle.setAttribute("aria-expanded", "true");
}
function closePanel() {
  el.panel.classList.remove("open");
  el.scrim.hidden = true;
  el.panelToggle.setAttribute("aria-expanded", "false");
}
el.panelToggle.addEventListener("click", () =>
  el.panel.classList.contains("open") ? closePanel() : openPanel(),
);
el.scrim.addEventListener("click", closePanel);

// ---------------------------------------------------------------- start

applyKnobs();
renderStory();
renderLibrary();
