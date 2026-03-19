function asText(value) {
  return String(value == null ? "" : value).trim();
}

const relationPanel = document.querySelector("[data-testid='relation-panel']");
const statusPill = document.querySelector("[data-testid='status-pill']");
const primaryButton = document.querySelector("[data-testid='hero-action']");

function articleTitles() {
  return Array.from(document.querySelectorAll("[data-testid='article-card-title']"))
    .map((node) => asText(node.textContent))
    .filter(Boolean);
}

function setStatus(text, state) {
  if (!statusPill) return;
  statusPill.textContent = asText(text);
  statusPill.dataset.state = asText(state);
}

function resetState() {
  if (relationPanel) relationPanel.hidden = true;
  setStatus("READY", "ready");
  return getSnapshot();
}

function openRelationMap() {
  if (relationPanel) relationPanel.hidden = false;
  setStatus("RELATION MAP OPEN", "open");
  return getSnapshot();
}

function getSnapshot() {
  return {
    ready: true,
    status: asText(statusPill?.textContent || ""),
    statusState: asText(statusPill?.dataset?.state || ""),
    primaryButtonLabel: asText(primaryButton?.textContent || ""),
    relationOpen: relationPanel ? relationPanel.hidden !== true : false,
    relationText: relationPanel ? asText(relationPanel.textContent || "") : "",
    articleCount: articleTitles().length,
    articleTitles: articleTitles(),
  };
}

if (primaryButton) {
  primaryButton.addEventListener("click", () => {
    openRelationMap();
  });
}

globalThis.__toolSmoke = {
  ready: true,
  resetState,
  openRelationMap,
  getSnapshot,
};

resetState();
