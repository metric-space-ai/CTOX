const DEFAULT_HTML_FILE = "./index.html";

function remGetCollection(ctx, name) {
  if (!ctx || !ctx.db || typeof ctx.db.collection !== "function") return null;
  try { return ctx.db.collection(name); } catch (_) { return null; }
}

function remCreatePersistenceAdapter(ctx, moduleId) {
  return {
    mode: "ctox-business-os",
    async saveRecord(record) {
      const collection = remGetCollection(ctx, "business_commands");
      if (!collection || typeof collection.insert !== "function") return null;
      const now = new Date().toISOString();
      const recordId = record.id || moduleId + "-record-" + Date.now() + "-" + Math.random().toString(36).slice(2);
      const payload = {
        id: moduleId + "-persistence-" + Date.now() + "-" + Math.random().toString(36).slice(2),
        command_id: moduleId + "-persistence-" + Date.now(),
        module: moduleId,
        command_type: "rem.persistence.save",
        type: "rem.persistence.save",
        moduleId,
        record: { ...record, id: recordId, updatedAt: now },
        created_at: now,
        updated_at_ms: Date.now(),
        status: "pending",
      };
      await collection.insert(payload);
      return payload.record;
    },
    async listRecords() {
      const collection = remGetCollection(ctx, "business_commands");
      if (!collection || typeof collection.find !== "function") return [];
      const docs = await collection
        .find({ selector: { type: "rem.persistence.save", moduleId }, sort: [{ created_at: "desc" }], limit: 50 })
        .exec();
      const records = (docs || []).map((doc) => {
        const raw = typeof doc.toJSON === "function" ? doc.toJSON() : doc;
        return raw.record || raw;
      });
      const unique = new Map();
      records.forEach((record) => {
        if (record && record.id && !unique.has(record.id)) unique.set(record.id, record);
      });
      return Array.from(unique.values());
    },
  };
}

function remRenderPersistenceList(list, records) {
  if (!records.length) {
    list.innerHTML = '<div class="rem-ctox-empty">Noch keine gespeicherten Vorgänge.</div>';
    return;
  }
  list.innerHTML = records.map((record) => {
    const title = record.title || record.name || record.id || "Vorgang";
    const meta = record.updatedAt ? new Date(record.updatedAt).toLocaleString("de-DE") : "";
    return '<button class="rem-ctox-item" type="button" data-id="' + String(record.id || "").replace(/"/g, "&quot;") + '">' +
      '<span>' + String(title).replace(/</g, "&lt;") + '</span><small>' + String(meta).replace(/</g, "&lt;") + '</small></button>';
  }).join("");
}

export async function mount(ctx = {}) {
  const root = ctx.host || ctx.root || document.body;
  root.classList.add("rem-ctox-host");
  root.innerHTML = '<div class="ctox-workspace ctox-workspace--three-pane rem-ctox-shell" data-rem-shell-version="1">' +
    '<aside class="ctox-pane rem-ctox-persistence" data-rem-persistence-pane>' +
      '<div class="rem-ctox-pane-head"><strong>Vorgänge</strong><button type="button" data-rem-new aria-label="Neuer Vorgang">Neu</button></div>' +
      '<div class="rem-ctox-editor"><label for="rem-ctox-title">Bezeichnung</label><input id="rem-ctox-title" data-rem-title placeholder="Vorgang benennen"><button type="button" data-rem-save>Speichern</button></div>' +
      '<div class="rem-ctox-list" data-rem-record-list></div>' +
    '</aside>' +
    '<div class="ctox-column-resizer rem-ctox-resizer" data-rem-resizer aria-hidden="true"></div>' +
    '<main class="ctox-pane rem-ctox-main"><iframe class="rem-ctox-frame" title="Vertriebsmanagement" sandbox="allow-scripts allow-same-origin allow-downloads allow-forms allow-modals allow-popups"></iframe></main>' +
  '</div>';

  const adapter = remCreatePersistenceAdapter(ctx, "rem-vertriebsmanagement");
  window.__remCtoxPersistenceAdapter = adapter;
  const frame = root.querySelector(".rem-ctox-frame");
  const list = root.querySelector("[data-rem-record-list]");
  const titleInput = root.querySelector("[data-rem-title]");
  const saveButton = root.querySelector("[data-rem-save]");
  const newButton = root.querySelector("[data-rem-new]");
  let activeRecordId = "";
  let recordsById = new Map();

  async function refreshRecords() {
    try {
      const records = await adapter.listRecords();
      recordsById = new Map(records.map((record) => [record.id, record]));
      remRenderPersistenceList(list, records);
    }
    catch (_) { list.innerHTML = '<div class="rem-ctox-empty">Persistenz ist nicht erreichbar.</div>'; }
  }

  const htmlUrl = new URL(DEFAULT_HTML_FILE, import.meta.url);
  const html = await fetch(htmlUrl, { cache: "no-store" }).then((res) => {
    if (!res.ok) throw new Error("REM-App konnte nicht geladen werden (" + res.status + ").");
    return res.text();
  });
  const base = new URL("./", import.meta.url).href;
  const srcdoc = ensureStorageGuard(html).replace(/<head([^>]*)>/i, '<head$1><base href="' + base + '"><script>window.RemTargetMode="ctox";window.ctoxBusinessOsPersistence=parent.__remCtoxPersistenceAdapter;<\/script>');

  function appDocument() {
    try { return frame.contentDocument; } catch (_) { return null; }
  }

  function editableControls(doc) {
    return Array.from(doc?.querySelectorAll("input, textarea, select") || []).filter((control) => {
      if (!control.id && !control.name) return false;
      if (["password", "file", "button", "submit", "hidden", "radio", "checkbox"].includes(control.type)) return false;
      if (control.closest("#apiModal, dialog")) return false;
      return true;
    });
  }

  function captureState() {
    const doc = appDocument();
    const fields = {};
    editableControls(doc).forEach((control) => {
      fields[control.id || control.name] = control.value;
    });
    const fallback = Object.values(fields).find((value) => String(value || "").trim()) || "Vorgang";
    return {
      id: activeRecordId || undefined,
      title: titleInput.value.trim() || String(fallback).trim().slice(0, 72),
      fields,
    };
  }

  function restoreState(record) {
    const doc = appDocument();
    if (!doc || !record) return;
    activeRecordId = record.id || "";
    titleInput.value = record.title || "";
    editableControls(doc).forEach((control) => {
      const key = control.id || control.name;
      if (!Object.prototype.hasOwnProperty.call(record.fields || {}, key)) return;
      control.value = record.fields[key] ?? "";
      control.dispatchEvent(new Event("input", { bubbles: true }));
      control.dispatchEvent(new Event("change", { bubbles: true }));
    });
  }

  function loadFrame(record = null) {
    frame.onload = () => {
      if (record) restoreState(record);
    };
    frame.srcdoc = srcdoc;
  }

  saveButton.addEventListener("click", async () => {
    saveButton.disabled = true;
    try {
      const saved = await adapter.saveRecord(captureState());
      activeRecordId = saved?.id || activeRecordId;
      titleInput.value = saved?.title || titleInput.value;
      await refreshRecords();
    } finally {
      saveButton.disabled = false;
    }
  });
  newButton.addEventListener("click", () => {
    activeRecordId = "";
    titleInput.value = "";
    loadFrame();
  });
  list.addEventListener("click", (event) => {
    const button = event.target.closest("[data-id]");
    const record = button ? recordsById.get(button.dataset.id) : null;
    if (record) loadFrame(record);
  });
  loadFrame();
  refreshRecords();

  const shell = root.querySelector(".rem-ctox-shell");
  const resizer = root.querySelector("[data-rem-resizer]");
  let dragging = false;
  resizer.addEventListener("pointerdown", (event) => { dragging = true; resizer.setPointerCapture(event.pointerId); });
  resizer.addEventListener("pointerup", () => { dragging = false; });
  resizer.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const rect = shell.getBoundingClientRect();
    const width = Math.min(360, Math.max(180, event.clientX - rect.left));
    shell.style.setProperty("--rem-ctox-left", width + "px");
  });
}

export default { mount };
