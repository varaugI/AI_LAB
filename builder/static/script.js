const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const state = { mode: "chat", lastPrompt: "", lastResponse: "", lastSources: [] };

const modeHints = {
  chat: "Natural conversation with optional cited library context.",
  explain: "Step-by-step teaching with definitions and examples.",
  study: "Tutor mode with a recap and self-check questions.",
  code: "Programming-focused answers and runnable examples.",
  law: "Careful source-grounded explanation; not legal advice.",
  summary: "Summarize the selected category or document library.",
};

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({ error: "Unreadable server response." }));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}

function addMessage(text, sender = "assistant", payload = {}) {
  const article = element("article", `message ${sender}`);
  article.appendChild(element("div", "avatar", sender === "user" ? "YOU" : "AI"));
  const bubble = element("div", "bubble");
  bubble.appendChild(element("p", "", text));

  if (sender === "assistant" && payload.backend) {
    const meta = element("div", "answer-meta");
    const model = payload.model ? ` · ${payload.model}` : "";
    meta.appendChild(element("span", "", `${payload.backend}${model}`));
    if (typeof payload.confidence === "number") {
      meta.appendChild(element("span", "", `grounding ${Math.round(payload.confidence * 100)}%`));
    }
    const actions = element("div", "feedback-actions");
    const good = element("button", "", "👍 Good");
    const bad = element("button", "", "👎 Wrong");
    const correct = element("button", "", "✎ Correct");
    const feedbackPrompt = payload.prompt || state.lastPrompt;
    const feedbackSources = payload.sources || [];
    good.addEventListener("click", () => saveFeedback(feedbackPrompt, text, feedbackSources, 1, "", true));
    bad.addEventListener("click", () => saveFeedback(feedbackPrompt, text, feedbackSources, -1, "", false));
    correct.addEventListener("click", () => {
      const correction = window.prompt("Enter the answer the model should learn:", text);
      if (correction?.trim()) saveFeedback(feedbackPrompt, text, feedbackSources, 0, correction.trim(), true);
    });
    actions.append(good, bad, correct);
    meta.appendChild(actions);
    bubble.appendChild(meta);
  }

  if (payload.sources?.length) {
    const details = element("details");
    details.appendChild(element("summary", "", `${payload.sources.length} cited source${payload.sources.length === 1 ? "" : "s"}`));
    payload.sources.forEach(source => {
      const card = element("div", "source");
      card.appendChild(element("strong", "", `[${source.number}] ${source.title}`));
      card.appendChild(element("div", "", source.location));
      card.appendChild(element("p", "", source.excerpt));
      details.appendChild(card);
    });
    bubble.appendChild(details);
  }

  article.appendChild(bubble);
  $("#chat").appendChild(article);
  $("#chat").scrollTop = $("#chat").scrollHeight;
  return article;
}

async function saveFeedback(prompt, response, sources, rating, correctedResponse, approved) {
  try {
    const data = await api("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        response,
        corrected_response: correctedResponse,
        rating,
        approved,
        sources,
      }),
    });
    addMessage(data.message, "assistant");
  } catch (error) {
    addMessage(error.message, "assistant");
  }
}

async function refreshStatus() {
  try {
    const status = await api("/api/status");
    $("#library-status").textContent = `${status.documents} documents · ${status.chunks} searchable chunks`;
    $("#ready-dot").classList.toggle("ready", status.ready);
    $("#engine-status").textContent = status.model ? `${status.backend} · ${status.model}` : status.backend;
    $("#memory-status").textContent = `${status.memory_turns} remembered turns`;
    $("#domain-stats").replaceChildren();
    Object.entries(status.domains || {}).forEach(([name, count]) => {
      $("#domain-stats").appendChild(element("span", "chip", `${name}: ${count}`));
    });
    await refreshDocuments();
  } catch (error) {
    $("#library-status").textContent = "Server unavailable";
  }
}

async function refreshDocuments() {
  const payload = await api("/api/documents");
  const list = $("#document-list");
  list.replaceChildren();
  payload.documents.forEach(documentRecord => {
    const row = element("div", "document");
    const info = element("div");
    info.appendChild(element("strong", "", documentRecord.title));
    info.appendChild(element("span", "", `${documentRecord.domain} · ${documentRecord.chunk_count} chunks · #${documentRecord.id}`));
    const remove = element("button", "icon-button", "×");
    remove.title = "Delete document and indexed knowledge";
    remove.addEventListener("click", async () => {
      if (!window.confirm(`Delete “${documentRecord.title}” from files and retrieval? This does not erase it from checkpoints already trained on it.`)) return;
      try {
        const result = await api(`/api/documents/${documentRecord.id}`, { method: "DELETE" });
        addMessage(result.message, "assistant");
        await refreshStatus();
      } catch (error) {
        addMessage(error.message, "assistant");
      }
    });
    row.append(info, remove);
    list.appendChild(row);
  });
}

$$('.mode').forEach(button => button.addEventListener("click", () => {
  state.mode = button.dataset.mode;
  $$('.mode').forEach(item => item.classList.toggle("active", item === button));
  $("#mode-hint").textContent = modeHints[state.mode];
  $("#prompt").placeholder = state.mode === "summary" ? "Optional: describe the desired summary focus…" : "Ask a question…";
}));

$("#composer").addEventListener("submit", async event => {
  event.preventDefault();
  const message = $("#prompt").value.trim();
  if (!message && state.mode !== "summary") return;
  state.lastPrompt = message || "Summarize the library";
  addMessage(state.lastPrompt, "user");
  $("#prompt").value = "";
  const loading = addMessage("Thinking…", "assistant");
  $("#send").disabled = true;
  try {
    const data = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        mode: state.mode,
        domain: $("#domain").value,
        scope: $("#scope").value,
      }),
    });
    loading.remove();
    state.lastResponse = data.answer;
    state.lastSources = data.sources || [];
    addMessage(data.answer, "assistant", { ...data, prompt: state.lastPrompt });
    await refreshStatus();
  } catch (error) {
    loading.remove();
    addMessage(error.message, "assistant");
  } finally {
    $("#send").disabled = false;
    $("#prompt").focus();
  }
});

$("#prompt").addEventListener("input", event => {
  event.target.style.height = "auto";
  event.target.style.height = `${Math.min(event.target.scrollHeight, 180)}px`;
});
$("#prompt").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#composer").requestSubmit();
  }
});

$("#upload-button").addEventListener("click", async () => {
  const files = $("#files").files;
  if (!files.length) {
    $("#upload-message").textContent = "Select at least one file.";
    return;
  }
  const form = new FormData();
  [...files].forEach(file => form.append("files", file));
  form.append("domain", $("#upload-domain").value);
  form.append("ocr_scanned", $("#ocr-scanned").checked);
  form.append("replace_same_name", $("#replace-same-name").checked);
  $("#upload-message").textContent = "Extracting and indexing…";
  try {
    const result = await api("/api/upload", { method: "POST", body: form });
    $("#upload-message").textContent = `${result.added} added, ${result.duplicates} duplicates skipped.`;
    $("#files").value = "";
    addMessage(result.results.map(item => item.message).join("\n"), "assistant");
    await refreshStatus();
  } catch (error) {
    $("#upload-message").textContent = error.message;
  }
});

$("#backend-button").addEventListener("click", async () => {
  $("#backend-message").textContent = "Loading…";
  try {
    const result = await api("/api/backend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        backend: $("#backend").value,
        checkpoint: $("#checkpoint").value.trim(),
        model: $("#model").value.trim(),
        base_url: $("#base-url").value.trim(),
      }),
    });
    $("#backend-message").textContent = result.message;
    addMessage(result.message, "assistant");
    await refreshStatus();
  } catch (error) {
    $("#backend-message").textContent = error.message;
  }
});

$("#export-button").addEventListener("click", async () => {
  try {
    const result = await api("/api/training/export", { method: "POST" });
    $("#export-message").textContent = `${result.documents} documents and ${result.chat_examples} approved chats exported.`;
    addMessage(`Corpus: ${result.corpus}\nChat examples: ${result.chats}`, "assistant");
  } catch (error) {
    $("#export-message").textContent = error.message;
  }
});

$("#clear-memory").addEventListener("click", async () => {
  try {
    const result = await api("/api/memory/clear", { method: "POST" });
    addMessage(result.message, "assistant");
    await refreshStatus();
  } catch (error) {
    addMessage(error.message, "assistant");
  }
});

$("#refresh-documents").addEventListener("click", refreshDocuments);
refreshStatus();
