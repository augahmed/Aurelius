const STORAGE_KEY = "aurelius.chat.history.v1";
const SETTINGS_KEY = "aurelius.chat.settings.v1";

const chatHistory = document.getElementById("chat-history");
const chatForm = document.getElementById("chat-form");
const promptInput = document.getElementById("prompt-input");
const maxTokensInput = document.getElementById("max-tokens");
const temperatureInput = document.getElementById("temperature");
const topKInput = document.getElementById("top-k");
const useCacheInput = document.getElementById("use-cache");
const submitButton = document.getElementById("submit-button");
const statusText = document.getElementById("status-text");
const clearHistoryButton = document.getElementById("clear-history");
const messageTemplate = document.getElementById("message-template");

const state = {
  messages: loadMessages(),
  pending: false,
};

applySettings();
renderMessages();

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.pending) {
    return;
  }

  const prompt = promptInput.value.trim();
  const maxTokens = normalizeMaxTokens();
  const temperature = normalizeTemperature();
  const topK = normalizeTopK();
  const useCache = useCacheInput.checked;

  if (!prompt) {
    setStatus("Enter a prompt to continue.", true);
    return;
  }
  if (!Number.isFinite(maxTokens) || maxTokens < 0) {
    setStatus("Max tokens must be zero or greater.", true);
    return;
  }

  const userMessage = {
    role: "user",
    content: prompt,
  };
  state.messages.push(userMessage);
  persistMessages();
  persistSettings();
  renderMessages();

  promptInput.value = "";
  state.pending = true;
  renderPendingMessage();
  setStatus("Generating…", false);
  submitButton.disabled = true;

  try {
    const response = await fetch("/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt,
        max_tokens: maxTokens,
        temperature,
        top_k: topK,
        use_cache: useCache,
        messages: state.messages,
      }),
    });

    const bodyText = await response.text();
    if (!response.ok) {
      throw new Error(bodyText || `request failed with status ${response.status}`);
    }

    const payload = JSON.parse(bodyText);
    const assistantMessage = {
      role: "assistant",
      content: payload.output || "",
    };
    state.messages.push(assistantMessage);
    persistMessages();
    renderMessages();
    setStatus("Reply generated.", false);
  } catch (error) {
    removePendingMessage();
    const message = error instanceof Error ? error.message : "generation failed";
    setStatus(message, true);
  } finally {
    state.pending = false;
    submitButton.disabled = false;
    promptInput.focus();
  }
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

clearHistoryButton.addEventListener("click", () => {
  state.messages = [];
  persistMessages();
  renderMessages();
  setStatus("Conversation cleared.", false);
});

function loadMessages() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter(isValidMessage);
  } catch {
    return [];
  }
}

function persistMessages() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.messages));
}

function persistSettings() {
  const settings = {
    maxTokens: String(normalizeMaxTokens()),
    temperature: String(normalizeTemperature()),
    topK: String(normalizeTopK()),
    useCache: useCacheInput.checked,
  };
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function applySettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) {
      return;
    }
    const settings = JSON.parse(raw);
    if (typeof settings.maxTokens === "string") {
      maxTokensInput.value = settings.maxTokens;
    }
    if (typeof settings.temperature === "string") {
      temperatureInput.value = settings.temperature;
    }
    if (typeof settings.topK === "string") {
      topKInput.value = settings.topK;
    }
    if (typeof settings.useCache === "boolean") {
      useCacheInput.checked = settings.useCache;
    }
  } catch {
    // Ignore invalid saved settings.
  }
  normalizeMaxTokens();
  normalizeTemperature();
  normalizeTopK();
}

function renderMessages() {
  chatHistory.innerHTML = "";
  if (state.messages.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = `
      <p class="empty-title">No conversation yet.</p>
      <p class="empty-copy">Start with a prompt to open a local Aurelius session. History stays in this browser via localStorage.</p>
    `;
    chatHistory.appendChild(empty);
    return;
  }

  for (const message of state.messages) {
    chatHistory.appendChild(createMessageNode(message));
  }
  scrollToBottom();
}

function renderPendingMessage() {
  const pending = createMessageNode({
    role: "assistant",
    content: "Thinking…",
  });
  pending.dataset.pending = "true";
  pending.classList.add("pending");
  chatHistory.appendChild(pending);
  scrollToBottom();
}

function removePendingMessage() {
  const pending = chatHistory.querySelector('[data-pending="true"]');
  if (pending) {
    pending.remove();
  }
}

function createMessageNode(message) {
  const node = messageTemplate.content.firstElementChild.cloneNode(true);
  node.classList.add(message.role === "assistant" ? "assistant" : "user");

  const meta = node.querySelector(".message-meta");
  const body = node.querySelector(".message-body");

  meta.textContent = message.role === "assistant" ? "Aurelius" : "You";
  if (message.role === "assistant") {
    body.innerHTML = renderMarkdown(message.content);
  } else {
    body.textContent = message.content;
  }
  return node
}

function renderMarkdown(source) {
  let html = escapeHtml(source);

  html = html.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  const paragraphs = html
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean)
    .map((block) => {
      if (block.startsWith("<pre><code>")) {
        return block;
      }
      return `<p>${block.replace(/\n/g, "<br>")}</p>`;
    });

  return paragraphs.join("");
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function isValidMessage(value) {
  return value && (value.role === "user" || value.role === "assistant") && typeof value.content === "string";
}

function setStatus(message, isError) {
  statusText.textContent = message;
  statusText.dataset.error = isError ? "true" : "false";
}

function scrollToBottom() {
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function normalizeMaxTokens() {
  const fallback = Number(maxTokensInput.defaultValue || "8");
  const min = Number(maxTokensInput.min || "0");
  const max = Number(maxTokensInput.max || "12");
  let value = Number(maxTokensInput.value);

  if (!Number.isFinite(value)) {
    value = fallback;
  }
  value = Math.max(min, Math.min(max, Math.trunc(value)));
  maxTokensInput.value = String(value);
  return value;
}

function normalizeTemperature() {
  const fallback = Number(temperatureInput.defaultValue || "0.8");
  const min = Number(temperatureInput.min || "0");
  const max = Number(temperatureInput.max || "1.5");
  let value = Number(temperatureInput.value);

  if (!Number.isFinite(value)) {
    value = fallback;
  }
  value = Math.max(min, Math.min(max, Math.round(value * 10) / 10));
  temperatureInput.value = String(value);
  return value;
}

function normalizeTopK() {
  const fallback = Number(topKInput.defaultValue || "40");
  const min = Number(topKInput.min || "0");
  const max = Number(topKInput.max || "80");
  let value = Number(topKInput.value);

  if (!Number.isFinite(value)) {
    value = fallback;
  }
  value = Math.max(min, Math.min(max, Math.trunc(value)));
  topKInput.value = String(value);
  return value;
}
