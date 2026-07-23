const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-btn");
const modeButtons = [...document.querySelectorAll(".mode")];
const modeHint = document.getElementById("mode-hint");
const uploadButton = document.getElementById("upload-btn");
const fileInput = document.getElementById("book-files");
const uploadDomain = document.getElementById("upload-domain");
const chatDomain = document.getElementById("chat-domain");
const chatScope = document.getElementById("chat-scope");
const uploadStatus = document.getElementById("upload-status");
const clearMemoryButton = document.getElementById("clear-memory-btn");
const backendSelect = document.getElementById("backend-select");
const modelName = document.getElementById("model-name");
const backendButton = document.getElementById("backend-btn");
const backendMessage = document.getElementById("backend-message");
let mode = "chat";

const modeCopy = {
    chat: ["Ask anything about your imported knowledge…", "Chat mode answers from relevant library passages and remembers recent turns."],
    explain: ["What should I explain simply?", "Explain mode teaches the retrieved concept step by step."],
    study: ["Ask a textbook question or request a lesson…", "Study mode explains key points and gives self-check questions."],
    code: ["Ask about an imported coding book or source file…", "Code mode focuses on programming explanations and examples."],
    legal: ["Ask what your imported legal material says…", "Law mode explains cited material but is not legal advice."],
    search: ["Search exact names, phrases, sections, APIs, or cases…", "Search returns matching passages without composing a reply."],
    summary: ["Enter an exact document title, or leave blank for the library…", "Summary selects representative sentences and cites their locations."],
    characters: ["Click send to list recurring fictional characters…", "Character analysis is most useful for novels and stories."],
};

function setMode(nextMode) {
    mode = nextMode;
    modeButtons.forEach(button => button.classList.toggle("active", button.dataset.mode === mode));
    chatInput.placeholder = modeCopy[mode][0];
    modeHint.textContent = modeCopy[mode][1];
    chatInput.focus();
}

modeButtons.forEach(button => button.addEventListener("click", () => setMode(button.dataset.mode)));

function makeElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
}

function addMessage(text, sender = "assistant", payload = {}) {
    const article = makeElement("article", `message ${sender}`);
    article.appendChild(makeElement("div", "avatar", sender === "user" ? "YOU" : "AI"));
    const bubble = makeElement("div", "bubble");
    if (payload.heading) bubble.appendChild(makeElement("h3", "message-heading", payload.heading));
    bubble.appendChild(makeElement("p", "message-text", text || ""));

    if (payload.backend) {
        const label = payload.model ? `${payload.backend} · ${payload.model}` : payload.backend;
        bubble.appendChild(makeElement("div", "engine-label", `Reply engine: ${label}`));
    }
    if (typeof payload.confidence === "number") {
        bubble.appendChild(makeElement("div", "confidence", `Grounding confidence: ${Math.round(payload.confidence * 100)}%`));
    }

    if (payload.sources?.length) {
        const details = makeElement("details", "source-list");
        details.appendChild(makeElement("summary", "", `${payload.sources.length} cited source${payload.sources.length === 1 ? "" : "s"}`));
        payload.sources.forEach(source => {
            const card = makeElement("div", "source-card");
            card.appendChild(makeElement("strong", "", `[${source.number}] ${source.title}`));
            card.appendChild(makeElement("div", "source-location", source.location));
            card.appendChild(makeElement("p", "", source.excerpt || ""));
            details.appendChild(card);
        });
        bubble.appendChild(details);
    }

    if (payload.results?.length) {
        const container = makeElement("div", "result-list");
        payload.results.forEach(result => {
            const card = makeElement("div", "result-card");
            card.appendChild(makeElement("strong", "", result.title));
            card.appendChild(makeElement("div", "source-location", `${result.location} · score ${result.score.toFixed(2)}`));
            card.appendChild(makeElement("p", "", result.excerpt));
            container.appendChild(card);
        });
        bubble.appendChild(container);
    }

    const profiles = payload.characters || (payload.character ? [payload.character] : []);
    if (profiles.length) {
        const grid = makeElement("div", "character-grid");
        profiles.forEach(profile => {
            const card = makeElement("div", "character-card");
            card.appendChild(makeElement("h3", "", profile.name));
            card.appendChild(makeElement("div", "badge", `${profile.mentions} mentions`));
            if (profile.first_appearance) card.appendChild(makeElement("p", "small", `First: ${profile.first_appearance}`));
            (profile.descriptions || []).slice(0, 3).forEach(description => card.appendChild(makeElement("p", "quote", description)));
            grid.appendChild(card);
        });
        bubble.appendChild(grid);
    }

    article.appendChild(bubble);
    chatWindow.appendChild(article);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return article;
}

async function requestJSON(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({ error: "The server returned an unreadable response." }));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
}

chatForm.addEventListener("submit", async event => {
    event.preventDefault();
    const message = chatInput.value.trim();
    if (!message && !["summary", "characters"].includes(mode)) return;
    addMessage(message || (mode === "summary" ? "Summarize the library" : "List recurring characters"), "user");
    chatInput.value = "";
    sendButton.disabled = true;
    const loading = addMessage("Searching and thinking…", "assistant");
    try {
        const data = await requestJSON("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message,
                mode,
                domain: chatDomain.value,
                scope: chatScope.value,
            }),
        });
        loading.remove();
        addMessage(data.answer, "assistant", data);
        await refreshStatus();
    } catch (error) {
        loading.remove();
        addMessage(error.message, "assistant");
    } finally {
        sendButton.disabled = false;
        chatInput.focus();
    }
});

chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = `${Math.min(chatInput.scrollHeight, 180)}px`;
});
chatInput.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});

uploadButton.addEventListener("click", async () => {
    if (!fileInput.files.length) {
        uploadStatus.textContent = "Select at least one document.";
        return;
    }
    const formData = new FormData();
    [...fileInput.files].forEach(file => formData.append("files", file));
    formData.append("domain", uploadDomain.value);
    uploadButton.disabled = true;
    uploadStatus.textContent = "Reading, classifying, and indexing…";
    try {
        const data = await requestJSON("/api/upload", { method: "POST", body: formData });
        uploadStatus.textContent = `${data.documents} file(s), ${data.words.toLocaleString()} words imported.`;
        fileInput.value = "";
        addMessage(`Imported ${data.documents} document(s) and created ${data.chunks} searchable passages.`, "assistant");
        await refreshStatus();
    } catch (error) {
        uploadStatus.textContent = error.message;
    } finally {
        uploadButton.disabled = false;
    }
});


backendButton.addEventListener("click", async () => {
    backendButton.disabled = true;
    backendMessage.textContent = "Connecting…";
    try {
        const data = await requestJSON("/api/backend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                backend: backendSelect.value,
                model: modelName.value.trim(),
                base_url: "http://127.0.0.1:11434",
            }),
        });
        backendMessage.textContent = data.message;
        addMessage(data.message, "assistant");
        await refreshStatus();
    } catch (error) {
        backendMessage.textContent = error.message;
    } finally {
        backendButton.disabled = false;
    }
});

clearMemoryButton.addEventListener("click", async () => {
    try {
        const data = await requestJSON("/api/memory/clear", { method: "POST" });
        addMessage(data.message, "assistant");
        await refreshStatus();
    } catch (error) {
        addMessage(error.message, "assistant");
    }
});

async function refreshStatus() {
    const statusElement = document.getElementById("library-status");
    const backendElement = document.getElementById("backend-status");
    const domainStats = document.getElementById("domain-stats");
    const list = document.getElementById("book-list");
    const memory = document.getElementById("memory-count");
    try {
        const data = await requestJSON("/api/status");
        statusElement.textContent = data.ready ? `${data.chunks} searchable passages` : "No library loaded";
        statusElement.classList.toggle("ready", data.ready);
        backendElement.textContent = `Reply engine: ${data.model ? `${data.backend} · ${data.model}` : data.backend}`;
        list.replaceChildren();
        data.titles.forEach(title => list.appendChild(makeElement("div", "book-chip", title)));
        domainStats.replaceChildren();
        Object.entries(data.domains || {}).forEach(([name, count]) => {
            domainStats.appendChild(makeElement("span", "domain-chip", `${name}: ${count}`));
        });
        memory.textContent = `${data.memory_turns} remembered turn${data.memory_turns === 1 ? "" : "s"}`;
    } catch (error) {
        statusElement.textContent = "Server unavailable";
    }
}

refreshStatus();
