export async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

export const api = {
  health() {
    return fetchJson("/api/health");
  },
  knowledge() {
    return fetchJson("/api/knowledge");
  },
  rebuildKnowledge() {
    return fetchJson("/api/knowledge/rebuild", { method: "POST" });
  },
  sessions() {
    return fetchJson("/api/sessions");
  },
  session(id) {
    return fetchJson(`/api/sessions/${encodeURIComponent(id)}`);
  },
  createSession(title) {
    return fetchJson("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
  },
  chat(payload) {
    return fetchJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },
  retrievalTraining() {
    return fetchJson("/api/retrieval-training/latest");
  },
  repos() {
    return fetchJson("/api/repos");
  },
  cloneRepo(payload) {
    return fetchJson("/api/repos/clone", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },
};
