// ---------------------------------------------------------------
// SmolGPT — fully client-side inference via transformers.js
// Model: https://huggingface.co/zishaan1911/smolGPT (ONNX build)
// ---------------------------------------------------------------

import { pipeline, TextStreamer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0";

// ---- CONFIG ----------------------------------------------------
const MODEL_ID = "zishaan1911/smolGPT"; // change if you push ONNX weights to a different repo
const STORAGE_KEY = "smolgpt_conversations_v1";
const MAX_CONTEXT_CHARS = 700; // keep the running "story" short enough for the model's context window

// Let the browser cache weights across visits (IndexedDB), don't force local-only lookups
env.allowRemoteModels = true;
env.allowLocalModels = false;

// ---- DOM ---------------------------------------------------------
const $ = (id) => document.getElementById(id);
const sidebar = $("sidebar");
const sidebarToggle = $("sidebarToggle");
const historyList = $("historyList");
const newChatBtn = $("newChatBtn");
const statusDot = $("statusDot");
const statusText = $("statusText");
const readoutBackend = $("readoutBackend");
const readoutSpeed = $("readoutSpeed");
const loadBanner = $("loadBanner");
const loadTitle = $("loadTitle");
const progressFill = $("progressFill");
const loadDetail = $("loadDetail");
const chatScroll = $("chatScroll");
const emptyState = $("emptyState");
const messagesEl = $("messages");
const composerForm = $("composerForm");
const promptInput = $("promptInput");
const sendBtn = $("sendBtn");
const lengthSelect = $("lengthSelect");

// ---- State ---------------------------------------------------------
let generator = null;
let modelReady = false;
let conversations = loadConversations();
let activeId = conversations.length ? conversations[0].id : createConversation();
let isGenerating = false;

renderSidebar();
renderActiveConversation();

// ---- Persistence ---------------------------------------------------
function loadConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}
function saveConversations() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  } catch {
    /* storage full or unavailable — fail silently, chat still works this session */
  }
}
function createConversation() {
  const conv = { id: crypto.randomUUID(), title: "New story", messages: [] };
  conversations.unshift(conv);
  saveConversations();
  return conv.id;
}
function getActive() {
  return conversations.find((c) => c.id === activeId);
}

// ---- Sidebar rendering ----------------------------------------------
function renderSidebar() {
  historyList.innerHTML = "";
  for (const conv of conversations) {
    const item = document.createElement("div");
    item.className = "history-item" + (conv.id === activeId ? " active" : "");
    item.innerHTML = `<span class="title">${escapeHtml(conv.title)}</span>
      <button class="del-btn" title="Delete">✕</button>`;
    item.querySelector(".title").addEventListener("click", () => {
      activeId = conv.id;
      renderSidebar();
      renderActiveConversation();
    });
    item.querySelector(".del-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      conversations = conversations.filter((c) => c.id !== conv.id);
      if (activeId === conv.id) {
        activeId = conversations.length ? conversations[0].id : createConversation();
      }
      saveConversations();
      renderSidebar();
      renderActiveConversation();
    });
    historyList.appendChild(item);
  }
}

function renderActiveConversation() {
  const conv = getActive();
  messagesEl.innerHTML = "";
  if (!conv || conv.messages.length === 0) {
    emptyState.style.display = "block";
    return;
  }
  emptyState.style.display = "none";
  for (const msg of conv.messages) {
    messagesEl.appendChild(renderMessage(msg.role, msg.text));
  }
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function renderMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML = `<span class="role">${role === "user" ? "you" : "smolgpt"}</span>
    <div class="bubble"></div>`;
  wrap.querySelector(".bubble").textContent = text;
  return wrap;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---- Sidebar toggle --------------------------------------------------
sidebarToggle.addEventListener("click", () => sidebar.classList.toggle("collapsed"));
newChatBtn.addEventListener("click", () => {
  activeId = createConversation();
  renderSidebar();
  renderActiveConversation();
  promptInput.focus();
});

// ---- Suggestion chips -------------------------------------------------
document.querySelectorAll(".suggestion-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    promptInput.value = chip.dataset.prompt;
    autoResize();
    if (modelReady) composerForm.requestSubmit();
  });
});

// ---- Textarea auto-resize ---------------------------------------------
function autoResize() {
  promptInput.style.height = "auto";
  promptInput.style.height = Math.min(promptInput.scrollHeight, 160) + "px";
}
promptInput.addEventListener("input", autoResize);
promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composerForm.requestSubmit();
  }
});

// ---- Model loading ------------------------------------------------------
function setStatus(kind, text) {
  statusDot.className = `status-dot ${kind}`;
  statusText.textContent = text;
}

async function initModel() {
  const useWebGPU = "gpu" in navigator;
  readoutBackend.textContent = useWebGPU ? "webgpu" : "wasm";
  setStatus("loading", useWebGPU ? "loading on webgpu…" : "loading on cpu (wasm)…");

  try {
    generator = await pipeline("text-generation", MODEL_ID, {
      dtype: "fp32",
      device: useWebGPU ? "webgpu" : "wasm",
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          const pct = Math.round((p.loaded / p.total) * 100);
          progressFill.style.width = pct + "%";
          loadDetail.textContent = `${p.file || "weights"} — ${pct}% (${(p.loaded / 1e6).toFixed(1)}MB / ${(p.total / 1e6).toFixed(1)}MB)`;
        } else if (p.status === "done") {
          loadDetail.textContent = `${p.file || "file"} ready`;
        }
      },
    });
    modelReady = true;
    setStatus("ready", "ready — running locally");
    loadTitle.textContent = "SmolGPT is ready 🌱";
    loadDetail.textContent = "Weights are cached in your browser — future visits load instantly.";
    progressFill.style.width = "100%";
    setTimeout(() => loadBanner.classList.add("hidden"), 1600);
    promptInput.disabled = false;
    sendBtn.disabled = false;
    promptInput.placeholder = "Start a story, or continue one…";
  } catch (err) {
    console.error(err);
    setStatus("error", "failed to load model");
    loadTitle.textContent = "Couldn't load SmolGPT";
    loadDetail.textContent = String(err.message || err);
    loadBanner.classList.remove("hidden");
  }
}
initModel();

// ---- Generation -----------------------------------------------------------
composerForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!modelReady || isGenerating) return;
  const text = promptInput.value.trim();
  if (!text) return;

  const conv = getActive();
  emptyState.style.display = "none";

  // Save + render user turn
  conv.messages.push({ role: "user", text });
  if (conv.title === "New story") conv.title = text.slice(0, 40) || "New story";
  saveConversations();
  renderSidebar();
  messagesEl.appendChild(renderMessage("user", text));

  promptInput.value = "";
  autoResize();
  isGenerating = true;
  sendBtn.disabled = true;
  promptInput.disabled = true;

  // Assistant bubble with live cursor
  const assistantWrap = document.createElement("div");
  assistantWrap.className = "msg assistant";
  assistantWrap.innerHTML = `<span class="role">smolgpt</span><div class="bubble"><span class="cursor"></span></div>`;
  messagesEl.appendChild(assistantWrap);
  const bubble = assistantWrap.querySelector(".bubble");
  chatScroll.scrollTop = chatScroll.scrollHeight;

  // Build a running-story prompt: prior context (trimmed) + the new line,
  // since SmolGPT continues text rather than answering turn by turn.
  const priorText = conv.messages
    .slice(0, -1)
    .map((m) => m.text)
    .join(" ");
  let prompt = (priorText + " " + text).trim();
  if (prompt.length > MAX_CONTEXT_CHARS) {
    prompt = prompt.slice(-MAX_CONTEXT_CHARS);
  }

  const maxNewTokens = Number(lengthSelect.value);
  let generatedText = "";
  const startTime = performance.now();
  let tokenCount = 0;

  try {
    const streamer = new TextStreamer(generator.tokenizer, {
      skip_prompt: true,
      callback_function: (chunk) => {
        generatedText += chunk;
        tokenCount += 1;
        bubble.innerHTML = escapeHtml(generatedText) + '<span class="cursor"></span>';
        chatScroll.scrollTop = chatScroll.scrollHeight;
        const elapsed = (performance.now() - startTime) / 1000;
        if (elapsed > 0) readoutSpeed.textContent = (tokenCount / elapsed).toFixed(1);
      },
    });

    await generator(prompt, {
      max_new_tokens: maxNewTokens,
      temperature: 0.8,
      top_k: 40,
      do_sample: true,
      streamer,
    });
  } catch (err) {
    console.error(err);
    generatedText = generatedText || "( generation failed — see console for details )";
  }

  bubble.textContent = generatedText.trim() || "…";
  conv.messages.push({ role: "assistant", text: generatedText.trim() });
  saveConversations();
  renderSidebar();

  isGenerating = false;
  sendBtn.disabled = false;
  promptInput.disabled = false;
  promptInput.focus();
});
