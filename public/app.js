"use strict";

const $ = (id) => document.getElementById(id);
const STORE_KEY = "subjectmate.settings";
const CHAT_KEY = "subjectmate.chat";
const HISTORY_TURNS = 3; // previous Q&A pairs sent along for follow-up questions
const SUGGESTIONS = [
  "How does Dijkstra's algorithm work?",
  "What are eigenvalues and eigenvectors?",
  "How do I build a confidence interval for a mean?",
  "What is the difference between a stack and a queue?",
];

// Citations like [Lecture_03.pdf, p. 12], [Regression.pptx, slide 4], [Quiz.docx, part 2]
const CITATION = /\[([^\[\]<>]+?\.(?:pdf|pptx|docx|ipynb|md|txt)),\s*(pp?\.|slides?|parts?)\s*([\d\s,–-]+)\]/gi;
// Math: $$...$$, \[...\], \(...\), and $...$ (no space just inside the dollars, so "$5 and $10" stays text)
const MATH = /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$(?=\S)[^$\n]+?(?<=\S)\$/g;

let providers = [];
let settings = loadSettings();
let chat = loadChat();
let controller = null; // AbortController of the request in flight

/* ---------- storage ---------- */

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
    return { provider: saved.provider || null, keys: saved.keys || {}, models: saved.models || {} };
  } catch {
    return { provider: null, keys: {}, models: {} };
  }
}

function saveSettings() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(settings)); } catch { /* storage unavailable */ }
}

function loadChat() {
  try {
    const saved = JSON.parse(localStorage.getItem(CHAT_KEY) || "[]");
    return Array.isArray(saved) ? saved : [];
  } catch {
    return [];
  }
}

function saveChat() {
  try { localStorage.setItem(CHAT_KEY, JSON.stringify(chat.slice(-50))); } catch { /* storage unavailable */ }
}

/* ---------- rendering helpers ---------- */

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderMarkdown(text) {
  // Keep math out of the Markdown parser (it would eat underscores and backslashes).
  const math = [];
  const protectedText = text.replace(MATH, (m) => `@@MATH${math.push(m) - 1}@@`);
  let html;
  if (window.marked && window.DOMPurify) {
    html = DOMPurify.sanitize(marked.parse(protectedText));
  } else {
    html = `<p>${escapeHtml(protectedText).replace(/\n/g, "<br>")}</p>`;
  }
  html = html.replace(/@@MATH(\d+)@@/g, (_, i) => escapeHtml(math[Number(i)]));
  return html.replace(CITATION, (_, file, unit, page) => `<span class="cite">${file}, ${unit} ${page.trim()}</span>`);
}

function renderMath(el) {
  if (!window.renderMathInElement) return;
  try {
    renderMathInElement(el, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  } catch { /* leave the raw TeX visible */ }
}

function renderSources(sources) {
  if (!sources || !sources.length) return "";
  const items = sources.map((s) => `
    <div class="source">
      <div class="source-head"><span>${escapeHtml(s.source)}, ${escapeHtml(s.unit || "p.")} ${s.page}${
          s.subject ? ` <span class="score">· ${escapeHtml(s.subject)}</span>` : ""}</span>
        <span class="score">match ${(s.score * 100).toFixed(0)}%</span></div>
      <p class="snippet">${escapeHtml(s.snippet)}${s.snippet.length >= 400 ? "…" : ""}</p>
    </div>`).join("");
  return `<details class="sources"><summary>Retrieved passages (${sources.length})</summary>${items}</details>`;
}

function renderMeta(turn) {
  const label = providers.find((p) => p.id === turn.provider)?.label || turn.provider;
  const parts = [];
  if (turn.provider) parts.push(`Answered by ${escapeHtml(label)}${turn.model ? ` · ${escapeHtml(turn.model)}` : ""}`);
  if (turn.searchQuery && turn.searchQuery !== turn.question) {
    parts.push(`Searched for: “${escapeHtml(turn.searchQuery)}”`);
  }
  const note = turn.note ? `<div class="note">${escapeHtml(turn.note)}</div>` : "";
  return parts.length || note ? `<div class="meta">${parts.join(" · ")}${note}</div>` : "";
}

/* ---------- chat ---------- */

function addRow(role) {
  $("empty")?.remove();
  const row = document.createElement("div");
  row.className = `msg ${role}`;
  row.innerHTML = `<div class="bubble"></div>`;
  $("chat").appendChild(row);
  return row.firstElementChild;
}

function scrollToBottom() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function renderAnswer(bubble, turn, { streaming = false } = {}) {
  bubble.className = "bubble";
  if (turn.error) {
    bubble.classList.add("error");
    bubble.textContent = turn.error;
    return;
  }
  if (turn.abstained) bubble.classList.add("abstain");
  const body = turn.answer
    ? `<div class="answer">${renderMarkdown(turn.answer)}${streaming ? '<span class="cursor"></span>' : ""}</div>`
    : '<span class="typing"><span></span><span></span><span></span></span>';
  const actions = !streaming && turn.answer
    ? `<div class="actions"><button type="button" class="link-btn" data-copy>Copy</button></div>` : "";
  bubble.innerHTML = body + (streaming ? "" : renderMeta(turn) + renderSources(turn.sources) + actions);
  if (!streaming) renderMath(bubble);
  bubble.querySelector("[data-copy]")?.addEventListener("click", async (e) => {
    try {
      await navigator.clipboard.writeText(turn.answer);
      e.target.textContent = "Copied";
      setTimeout(() => { e.target.textContent = "Copy"; }, 1500);
    } catch { e.target.textContent = "Copy failed"; }
  });
}

function renderTurn(turn) {
  addRow("user").textContent = turn.question;
  const bubble = addRow("bot");
  renderAnswer(bubble, turn);
  return bubble;
}

function historyMessages() {
  return chat
    .filter((t) => t.answer && !t.error)
    .slice(-HISTORY_TURNS)
    .flatMap((t) => [{ role: "user", content: t.question }, { role: "assistant", content: t.answer.slice(0, 4000) }]);
}

async function postWithRetry(url, body, signal) {
  const options = { method: "POST", headers: { "Content-Type": "application/json" }, body, signal };
  try {
    return await fetch(url, options);
  } catch (err) {
    if (signal.aborted) throw err;
    await new Promise((r) => setTimeout(r, 2000)); // server restarting / cold start
    try {
      return await fetch(url, options);
    } catch (err2) {
      if (signal.aborted) throw err2;
      throw new Error("Can't reach the SubjectMate server. Check that it is running, then try again.");
    }
  }
}

function setBusy(isBusy) {
  const btn = $("sendBtn");
  btn.textContent = isBusy ? "Stop" : "Ask";
  btn.classList.toggle("stop", isBusy);
  btn.type = isBusy ? "button" : "submit";
}

async function ask(question) {
  question = question.trim();
  if (controller || !question) return;
  const provider = settings.provider;
  const turn = { question, answer: "", sources: [] };
  const history = historyMessages();
  addRow("user").textContent = question;
  const bubble = addRow("bot");
  renderAnswer(bubble, turn, { streaming: true });
  scrollToBottom();

  controller = new AbortController();
  setBusy(true);
  let pending = false;
  const repaint = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; renderAnswer(bubble, turn, { streaming: true }); });
  };

  try {
    const res = await postWithRetry("/api/ask/stream", JSON.stringify({
      question,
      provider,
      model: settings.models[provider] || null,
      api_key: settings.keys[provider] || null,
      history,
    }), controller.signal);
    if (!res.ok) {
      let detail = `Server error (${res.status}).`;
      try { const data = await res.json(); if (typeof data.detail === "string") detail = data.detail; } catch { /* keep default */ }
      throw new Error(detail);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.type === "meta") {
          Object.assign(turn, { sources: event.sources, provider: event.provider, model: event.model,
                                note: event.note, searchQuery: event.search_query });
        } else if (event.type === "token") {
          turn.answer += event.text;
          repaint();
        } else if (event.type === "done") {
          turn.answer = event.answer;
          turn.abstained = event.abstained;
        } else if (event.type === "error") {
          throw new Error(event.message);
        }
      }
    }
    if (!turn.answer) throw new Error("The server returned no answer. Please try again.");
  } catch (err) {
    if (err.name === "AbortError") {
      turn.answer = turn.answer ? `${turn.answer}\n\n*(Stopped.)*` : "";
      if (!turn.answer) turn.error = "Stopped.";
    } else {
      turn.error = err.message || "Something went wrong. Please try again.";
    }
  } finally {
    controller = null;
    setBusy(false);
    renderAnswer(bubble, turn);
    chat.push(turn);
    saveChat();
    scrollToBottom();
  }
}

function newChat() {
  if (controller) controller.abort();
  chat = [];
  saveChat();
  $("chat").innerHTML = "";
  $("chat").appendChild(emptyState);
  $("question").focus();
}

/* ---------- providers & settings ---------- */

function hasKey(p) {
  return p.ready || Boolean(settings.keys[p.id]);
}

function renderProviderSelect() {
  const select = $("providerSelect");
  select.innerHTML = providers
    .map((p) => `<option value="${p.id}">${escapeHtml(p.label)}${hasKey(p) ? "" : " (needs key)"}</option>`)
    .join("");
  select.value = settings.provider;
}

function renderProviderSettings() {
  $("providerSettings").innerHTML = providers.filter((p) => p.id !== "extractive").map((p) => `
    <div class="provider-row">
      <div class="row-head">
        <strong>${escapeHtml(p.label)}</strong>
        <span class="badge ${hasKey(p) ? "ok" : ""}">${!p.needs_key ? "No key needed"
          : p.ready ? "Server key available" : settings.keys[p.id] ? "Using your key" : "No key"}</span>
      </div>
      ${p.needs_key ? `<label for="key-${p.id}">Your API key${p.ready ? " (optional)" : ""} ·
        <a href="${p.key_url}" target="_blank" rel="noopener">get one</a></label>
      <input id="key-${p.id}" data-provider="${p.id}" data-field="keys" type="password"
        autocomplete="off" placeholder="Paste API key" value="${escapeHtml(settings.keys[p.id] || "")}">` : ""}
      <label for="model-${p.id}">Model</label>
      <input id="model-${p.id}" data-provider="${p.id}" data-field="models" type="text"
        placeholder="${escapeHtml(p.default_model)}" value="${escapeHtml(settings.models[p.id] || "")}">
    </div>`).join("");
}

function openSettings() {
  renderProviderSettings();
  $("settings").showModal();
}

async function loadProviders() {
  try {
    const res = await fetch("/api/providers");
    const data = await res.json();
    providers = data.providers;
    const current = providers.find((p) => p.id === settings.provider);
    if (!current || !hasKey(current)) settings.provider = data.default;
  } catch {
    providers = [];
  }
  renderProviderSelect();
}

async function loadDocuments() {
  const btn = $("docsBtn");
  try {
    const res = await fetch("/api/documents");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    const docs = data.documents;
    btn.textContent = `${docs.length} course file${docs.length === 1 ? "" : "s"} · view list`;
    $("docList").innerHTML = docs
      .map((d) => `<li><strong>${escapeHtml(d.name)}</strong>${
        d.subject ? ` <span class="muted">· ${escapeHtml(d.subject)}</span>` : ""}</li>`)
      .join("");
  } catch (err) {
    btn.textContent = "Course material not indexed yet";
    $("docList").innerHTML = `<li>${escapeHtml(err.message || "Could not load the document list.")}</li>`;
  }
}

/* ---------- events ---------- */

$("providerSettings").addEventListener("input", (e) => {
  const { provider, field } = e.target.dataset;
  if (!provider) return;
  const value = e.target.value.trim();
  if (value) settings[field][provider] = value; else delete settings[field][provider];
  saveSettings();
  renderProviderSelect();
});
$("settings").addEventListener("close", renderProviderSettings);
$("providerSelect").addEventListener("change", (e) => {
  settings.provider = e.target.value;
  saveSettings();
  const p = providers.find((x) => x.id === settings.provider);
  if (p && !hasKey(p)) openSettings();
});
$("settingsBtn").addEventListener("click", openSettings);
$("docsBtn").addEventListener("click", () => $("docs").showModal());
$("newChatBtn").addEventListener("click", newChat);

const textarea = $("question");
textarea.addEventListener("input", () => {
  textarea.style.height = "auto";
  textarea.style.height = `${textarea.scrollHeight}px`;
});
textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!controller) $("composer").requestSubmit();
  }
});
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = textarea.value;
  textarea.value = "";
  textarea.style.height = "auto";
  ask(q);
});
$("sendBtn").addEventListener("click", () => {
  if (controller) controller.abort();
});

$("suggestions").innerHTML = SUGGESTIONS.map((s) => `<button type="button">${escapeHtml(s)}</button>`).join("");
$("suggestions").addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") ask(e.target.textContent);
});
const emptyState = $("empty");

/* ---------- start ---------- */

loadProviders().then(() => {
  for (const turn of chat) renderTurn(turn);
  if (chat.length) scrollToBottom();
});
loadDocuments();
