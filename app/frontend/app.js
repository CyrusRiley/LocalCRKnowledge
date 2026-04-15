const state = {
  selectedNoteId: "",
  selectedNote: null,
  currentRawText: "",
  currentMarkdown: "",
  latestAnswer: "",
  organizeRunId: "",
  organizePollTimer: null,
};

const els = {
  tabs: document.querySelectorAll(".tab-button"),
  panels: document.querySelectorAll(".tab-panel"),
  healthBadge: document.getElementById("health-badge"),
  direction: document.getElementById("direction-select"),
  rawText: document.getElementById("raw-text"),
  importLog: document.getElementById("import-log"),
  fileInput: document.getElementById("file-input"),
  filePath: document.getElementById("file-path"),
  dirPath: document.getElementById("dir-path"),
  resultRaw: document.getElementById("result-raw"),
  markdownPreview: document.getElementById("markdown-preview"),
  filterTheme: document.getElementById("filter-theme"),
  filterType: document.getElementById("filter-type"),
  filterLimit: document.getElementById("filter-limit"),
  organizeStatus: document.getElementById("organize-status"),
  noteList: document.getElementById("note-list"),
  noteDetail: document.getElementById("note-detail"),
  editTitle: document.getElementById("edit-title"),
  editNoteType: document.getElementById("edit-note-type"),
  editThemes: document.getElementById("edit-themes"),
  editKeywords: document.getElementById("edit-keywords"),
  editSummary: document.getElementById("edit-summary"),
  editKeyPoints: document.getElementById("edit-key-points"),
  editUsageScenarios: document.getElementById("edit-usage-scenarios"),
  editUserInsights: document.getElementById("edit-user-insights"),
  editSourceExcerpt: document.getElementById("edit-source-excerpt"),
  keywordQuery: document.getElementById("keyword-query"),
  questionQuery: document.getElementById("question-query"),
  searchResultList: document.getElementById("search-result-list"),
  answerView: document.getElementById("answer-view"),
};

init();

function init() {
  bindEvents();
  refreshHealth();
  refreshNotes();
  loadLatestOrganizeRun();
}

function bindEvents() {
  els.tabs.forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });

  document.getElementById("btn-import-text").addEventListener("click", importText);
  document.getElementById("btn-import-files").addEventListener("click", importFiles);
  document.getElementById("btn-import-file-path").addEventListener("click", importFileByPath);
  document.getElementById("btn-import-dir").addEventListener("click", importDirectory);
  document.getElementById("btn-refresh-notes").addEventListener("click", refreshNotes);
  document.getElementById("btn-organize-library").addEventListener("click", startOrganizeLibrary);
  document.getElementById("btn-keyword-search").addEventListener("click", keywordSearch);
  document.getElementById("btn-ask").addEventListener("click", askQuestion);
  document.getElementById("btn-export-current").addEventListener("click", exportCurrentMarkdown);
  document.getElementById("btn-save-note").addEventListener("click", saveSelectedNote);
  document.getElementById("btn-reset-note").addEventListener("click", resetSelectedNoteEditor);

  [
    els.editTitle,
    els.editNoteType,
    els.editThemes,
    els.editKeywords,
    els.editSummary,
    els.editKeyPoints,
    els.editUsageScenarios,
    els.editUserInsights,
    els.editSourceExcerpt,
  ].forEach((field) => {
    field.addEventListener("input", () => {
      const markdown = buildMarkdownFromEditor();
      state.currentMarkdown = markdown;
      els.noteDetail.innerHTML = markdownToHtml(markdown, { hideSourceInfo: true, removeEmptySections: true });
      renderPreview(markdown);
    });
  });
}

async function loadLatestOrganizeRun() {
  try {
    const data = await apiGet("/api/library/organize/latest");
    const run = data.run;
    if (!run) {
      els.organizeStatus.textContent = "尚未运行知识库整理任务。";
      return;
    }
    state.organizeRunId = run.run_id;
    const stats = run.stats || {};
    els.organizeStatus.textContent = [
      `最近任务: ${run.status}`,
      `开始: ${run.started_at || ""}`,
      stats.total_notes_before !== undefined
        ? `前后条目: ${stats.total_notes_before} -> ${stats.total_notes_after}`
        : "",
      stats.removed_duplicates !== undefined ? `去重: ${stats.removed_duplicates}` : "",
      stats.relations_saved !== undefined ? `关系: ${stats.relations_saved}` : "",
    ]
      .filter(Boolean)
      .join(" | ");
    if (run.status === "running") {
      startOrganizePolling();
    }
  } catch (err) {
    els.organizeStatus.textContent = `整理任务状态加载失败：${err.message}`;
  }
}

async function refreshHealth() {
  try {
    const data = await apiGet("/api/health");
    els.healthBadge.textContent = `模型在线：${data.llm_model} @ ${data.llm_base_url}`;
    els.healthBadge.style.background = "#d9f5e9";
    els.healthBadge.style.color = "#185443";
    els.healthBadge.style.borderColor = "#9cd7bd";
  } catch (err) {
    els.healthBadge.textContent = `连接失败：${err.message}`;
  }
}

async function importText() {
  const text = els.rawText.value.trim();
  if (!text) {
    appendImportLog("请输入文本后再整理。");
    return;
  }
  appendImportLog("正在调用后端整理文本...");
  try {
    const data = await apiPost("/api/import/text", {
      text,
      direction: els.direction.value,
      export: false,
    });
    appendImportLog(JSON.stringify(data, null, 2));
    if (data.markdown) {
      updateResultView(text, data.markdown);
      switchTab("result");
    }
    refreshNotes();
  } catch (err) {
    appendImportLog(`整理失败：${err.message}`);
  }
}

async function importFiles() {
  const files = Array.from(els.fileInput.files || []);
  if (files.length === 0) {
    appendImportLog("请先选择 txt/md 文件。");
    return;
  }
  appendImportLog(`正在处理 ${files.length} 个上传文件...`);
  try {
    const payloadFiles = [];
    for (const file of files) {
      const content = await file.text();
      payloadFiles.push({ name: file.name, content });
    }
    const data = await apiPost("/api/import/files", {
      files: payloadFiles,
      direction: els.direction.value,
      export: false,
    });
    appendImportLog(JSON.stringify(data, null, 2));
    const lastSuccess = (data.results || []).find((item) => item.markdown);
    if (lastSuccess && lastSuccess.markdown) {
      updateResultView("", lastSuccess.markdown);
      switchTab("result");
    }
    refreshNotes();
  } catch (err) {
    appendImportLog(`文件导入失败：${err.message}`);
  }
}

async function importFileByPath() {
  const path = els.filePath.value.trim();
  if (!path) {
    appendImportLog("请输入文件路径。");
    return;
  }
  appendImportLog(`正在导入文件：${path}`);
  try {
    const data = await apiPost("/api/import/file-path", {
      path,
      direction: els.direction.value,
      export: false,
    });
    appendImportLog(JSON.stringify(data, null, 2));
    if (data.markdown) {
      updateResultView("", data.markdown);
      switchTab("result");
    }
    refreshNotes();
  } catch (err) {
    appendImportLog(`路径导入失败：${err.message}`);
  }
}

async function importDirectory() {
  const path = els.dirPath.value.trim();
  if (!path) {
    appendImportLog("请输入目录路径。");
    return;
  }
  appendImportLog(`正在增量扫描目录：${path}`);
  try {
    const data = await apiPost("/api/import/dir", {
      path,
      direction: els.direction.value,
      recursive: true,
    });
    appendImportLog(JSON.stringify(data, null, 2));
    refreshNotes();
  } catch (err) {
    appendImportLog(`目录增量失败：${err.message}`);
  }
}

async function startOrganizeLibrary() {
  try {
    els.organizeStatus.textContent = "正在启动知识库整理任务...";
    const data = await apiPost("/api/library/organize/start", {});
    state.organizeRunId = data.run_id || "";
    els.organizeStatus.textContent = `整理任务已启动: ${state.organizeRunId}`;
    startOrganizePolling();
  } catch (err) {
    els.organizeStatus.textContent = `启动失败：${err.message}`;
  }
}

function startOrganizePolling() {
  if (!state.organizeRunId) return;
  if (state.organizePollTimer) {
    clearInterval(state.organizePollTimer);
  }
  state.organizePollTimer = setInterval(async () => {
    try {
      const data = await apiGet(`/api/library/organize/status?run_id=${encodeURIComponent(state.organizeRunId)}`);
      const run = data.run || {};
      const live = run.live || {};
      const status = live.status || run.status || "running";
      const percent = Number(live.percent || 0);
      const msg = live.message || "";
      els.organizeStatus.textContent = `整理中(${status}) ${percent}% ${msg}`;
      if (status === "completed" || status === "failed") {
        clearInterval(state.organizePollTimer);
        state.organizePollTimer = null;
        await loadLatestOrganizeRun();
        await refreshNotes();
      }
    } catch (err) {
      els.organizeStatus.textContent = `整理状态轮询失败：${err.message}`;
      clearInterval(state.organizePollTimer);
      state.organizePollTimer = null;
    }
  }, 2500);
}

async function refreshNotes() {
  const params = new URLSearchParams();
  params.set("limit", `${Math.max(1, Number(els.filterLimit.value || "30"))}`);
  if (els.filterTheme.value.trim()) {
    params.set("theme", els.filterTheme.value.trim());
  }
  if (els.filterType.value.trim()) {
    params.set("note_type", els.filterType.value.trim());
  }
  try {
    const data = await apiGet(`/api/notes?${params.toString()}`);
    renderNoteList(data.notes || []);
  } catch (err) {
    els.noteList.innerHTML = `<li class="note-item">加载失败：${escapeHtml(err.message)}</li>`;
  }
}

function renderNoteList(notes) {
  if (!notes.length) {
    els.noteList.innerHTML = `<li class="note-item">暂无条目。</li>`;
    clearNoteEditor();
    els.noteDetail.textContent = "点击左侧条目查看详情并编辑";
    return;
  }
  const html = notes
    .map(
      (note) => `
      <li class="note-item" data-note-id="${escapeAttr(note.note_id)}">
        <strong>${escapeHtml(note.title || "未命名")}</strong>
        <small>${escapeHtml(note.note_type || "未分类")} · ${escapeHtml(note.updated_at || "")}</small>
      </li>
    `,
    )
    .join("");
  els.noteList.innerHTML = html;
  els.noteList.querySelectorAll(".note-item").forEach((item) => {
    item.addEventListener("click", () => loadNoteDetail(item.dataset.noteId));
  });
}

async function loadNoteDetail(noteId) {
  if (!noteId) return;
  state.selectedNoteId = noteId;
  try {
    const data = await apiGet(`/api/notes/${encodeURIComponent(noteId)}`);
    const note = data.note;
    state.selectedNote = note;
    populateNoteEditor(note);
    els.noteDetail.innerHTML = markdownToHtml(note.markdown_content || "", {
      hideSourceInfo: true,
      removeEmptySections: true,
    });
    updateResultView("", note.markdown_content || "");
  } catch (err) {
    els.noteDetail.textContent = `加载失败：${err.message}`;
  }
}

async function saveSelectedNote() {
  if (!state.selectedNoteId) {
    alert("请先从左侧列表选择一条知识。");
    return;
  }
  try {
    const payload = collectEditorPayload();
    const data = await apiPost("/api/notes/update", payload);
    const note = data.note;
    state.selectedNote = note;
    state.currentMarkdown = note.markdown_content || "";
    els.noteDetail.innerHTML = markdownToHtml(note.markdown_content || "", {
      hideSourceInfo: true,
      removeEmptySections: true,
    });
    renderPreview(state.currentMarkdown);
    refreshNotes();
    alert("保存成功。");
  } catch (err) {
    alert(`保存失败：${err.message}`);
  }
}

function resetSelectedNoteEditor() {
  if (!state.selectedNote) {
    clearNoteEditor();
    return;
  }
  populateNoteEditor(state.selectedNote);
  els.noteDetail.innerHTML = markdownToHtml(state.selectedNote.markdown_content || "", {
    hideSourceInfo: true,
    removeEmptySections: true,
  });
}

async function keywordSearch() {
  const query = els.keywordQuery.value.trim();
  if (!query) {
    els.answerView.textContent = "请输入关键词。";
    return;
  }
  els.answerView.textContent = "正在检索...";
  try {
    const data = await apiGet(`/api/search?query=${encodeURIComponent(query)}&limit=12`);
    renderSearchResults(data.results || []);
    els.answerView.textContent = `命中 ${data.results.length} 条。`;
  } catch (err) {
    els.answerView.textContent = `检索失败：${err.message}`;
  }
}

async function askQuestion() {
  const question = els.questionQuery.value.trim();
  if (!question) {
    els.answerView.textContent = "请输入问题。";
    return;
  }
  els.answerView.textContent = "正在生成结构化回答...";
  try {
    const data = await apiPost("/api/ask", {
      question,
      limit: 15,
      export: false,
    });
    renderSearchResults(data.results || []);
    state.latestAnswer = data.answer || "";
    els.answerView.innerHTML = markdownToHtml(state.latestAnswer, { hideSourceInfo: false, removeEmptySections: false });
    switchTab("qa");
  } catch (err) {
    els.answerView.textContent = `问答失败：${err.message}`;
  }
}

async function exportCurrentMarkdown() {
  const content = (state.currentMarkdown || "").trim();
  if (!content) {
    alert("当前没有可导出的 Markdown。");
    return;
  }
  const filename = prompt("导出文件名（不带 .md 也可以）", "export_note");
  if (!filename) return;
  try {
    const data = await apiPost("/api/export/markdown", {
      content,
      filename,
      overwrite: false,
    });
    alert(`导出成功：${data.exported_path}`);
  } catch (err) {
    alert(`导出失败：${err.message}`);
  }
}

function renderSearchResults(results) {
  if (!results.length) {
    els.searchResultList.innerHTML = `<li class="note-item">暂无命中条目。</li>`;
    return;
  }
  els.searchResultList.innerHTML = results
    .map(
      (item) => `
      <li class="note-item">
        <strong>${escapeHtml(item.title || "未命名")}</strong>
        <small>${escapeHtml(item.note_type || "未分类")} · score=${Number(item.score || 0).toFixed(4)}</small>
        <div>${escapeHtml(item.snippet || "")}</div>
      </li>
    `,
    )
    .join("");
}

function switchTab(tabName) {
  els.tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === tabName));
  els.panels.forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${tabName}`));
}

function updateResultView(raw, markdown) {
  if (raw) {
    state.currentRawText = raw;
    els.resultRaw.value = raw;
  }
  state.currentMarkdown = markdown || "";
  renderPreview(state.currentMarkdown);
}

function renderPreview(markdown) {
  els.markdownPreview.innerHTML = markdownToHtml(markdown, {
    hideSourceInfo: true,
    removeEmptySections: true,
  });
}

function populateNoteEditor(note) {
  els.editTitle.value = note.title || "";
  els.editNoteType.value = note.note_type || "";
  els.editThemes.value = (note.themes || []).join(", ");
  els.editKeywords.value = (note.keywords || []).join(", ");
  els.editSummary.value = note.summary || "";
  els.editKeyPoints.value = (note.key_points || []).join("\n");
  els.editUsageScenarios.value = (note.usage_scenarios || []).join("\n");
  els.editUserInsights.value = note.user_insights || "";
  els.editSourceExcerpt.value = note.source_excerpt || "";
}

function clearNoteEditor() {
  els.editTitle.value = "";
  els.editNoteType.value = "";
  els.editThemes.value = "";
  els.editKeywords.value = "";
  els.editSummary.value = "";
  els.editKeyPoints.value = "";
  els.editUsageScenarios.value = "";
  els.editUserInsights.value = "";
  els.editSourceExcerpt.value = "";
}

function collectEditorPayload() {
  return {
    note_id: state.selectedNoteId,
    title: els.editTitle.value.trim(),
    note_type: els.editNoteType.value.trim(),
    themes: splitCsv(els.editThemes.value),
    keywords: splitCsv(els.editKeywords.value),
    summary: els.editSummary.value.trim(),
    key_points: splitLines(els.editKeyPoints.value),
    usage_scenarios: splitLines(els.editUsageScenarios.value),
    user_insights: els.editUserInsights.value.trim(),
    source_excerpt: els.editSourceExcerpt.value.trim(),
  };
}

function buildMarkdownFromEditor() {
  const payload = collectEditorPayload();
  const sections = [`# ${payload.title || "未命名条目"}`];

  if (payload.note_type) {
    sections.push(`## 内容类型\n${payload.note_type}`);
  }
  if (payload.themes.length) {
    sections.push(`## 核心主题\n${payload.themes.map((item) => `- ${item}`).join("\n")}`);
  }
  if (payload.summary) {
    sections.push(`## 摘要\n${payload.summary}`);
  }
  if (payload.key_points.length) {
    sections.push(`## 关键要点\n${payload.key_points.map((item, idx) => `${idx + 1}. ${item}`).join("\n")}`);
  }
  if (payload.usage_scenarios.length) {
    sections.push(`## 可用场景\n${payload.usage_scenarios.map((item) => `- ${item}`).join("\n")}`);
  }
  if (payload.user_insights) {
    sections.push(`## 我的进一步想法\n${payload.user_insights}`);
  }
  if (payload.keywords.length) {
    sections.push(`## 关键词\n${payload.keywords.map((item) => `- ${item}`).join("\n")}`);
  }
  if (payload.source_excerpt) {
    sections.push(`## 来源摘录\n${payload.source_excerpt}`);
  }

  return `${sections.join("\n\n")}\n`;
}

function appendImportLog(text) {
  const time = new Date().toLocaleTimeString();
  els.importLog.textContent += `[${time}] ${text}\n`;
  els.importLog.scrollTop = els.importLog.scrollHeight;
}

async function apiGet(url) {
  const res = await fetch(url, { method: "GET" });
  const data = await parseResponse(res);
  if (!res.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

async function apiPost(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await parseResponse(res);
  if (!res.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

async function parseResponse(res) {
  const text = await res.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { error: text };
  }
}

function markdownToHtml(markdown, options = {}) {
  const prepared = preprocessMarkdown(markdown || "", options);
  const lines = prepared.split(/\r?\n/);
  let html = "";
  let inUl = false;
  let inOl = false;

  const closeLists = () => {
    if (inUl) {
      html += "</ul>";
      inUl = false;
    }
    if (inOl) {
      html += "</ol>";
      inOl = false;
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      closeLists();
      html += "<br />";
      continue;
    }
    if (line.startsWith("# ")) {
      closeLists();
      html += `<h1>${escapeHtml(line.slice(2))}</h1>`;
      continue;
    }
    if (line.startsWith("## ")) {
      closeLists();
      html += `<h2>${escapeHtml(line.slice(3))}</h2>`;
      continue;
    }
    if (line.startsWith("### ")) {
      closeLists();
      html += `<h3>${escapeHtml(line.slice(4))}</h3>`;
      continue;
    }
    if (line.startsWith("- ")) {
      if (inOl) {
        html += "</ol>";
        inOl = false;
      }
      if (!inUl) {
        html += "<ul>";
        inUl = true;
      }
      html += `<li>${escapeHtml(line.slice(2))}</li>`;
      continue;
    }
    if (/^\d+\.\s/.test(line)) {
      if (inUl) {
        html += "</ul>";
        inUl = false;
      }
      if (!inOl) {
        html += "<ol>";
        inOl = true;
      }
      html += `<li>${escapeHtml(line.replace(/^\d+\.\s/, ""))}</li>`;
      continue;
    }
    closeLists();
    html += `<p>${escapeHtml(line)}</p>`;
  }
  closeLists();
  return html;
}

function preprocessMarkdown(markdown, options) {
  const hideSourceInfo = Boolean(options.hideSourceInfo);
  const removeEmptySections = Boolean(options.removeEmptySections);
  const lines = markdown.split(/\r?\n/);
  const output = [];
  let pendingHeading = null;
  let pendingContent = [];

  const flushPending = () => {
    if (!pendingHeading) {
      return;
    }
    const heading = pendingHeading;
    const contentLines = pendingContent.slice();
    pendingHeading = null;
    pendingContent = [];

    if (hideSourceInfo && heading === "来源信息") {
      return;
    }
    if (removeEmptySections && isEmptySectionContent(contentLines)) {
      return;
    }
    output.push(`## ${heading}`);
    output.push(...contentLines);
  };

  for (const rawLine of lines) {
    const trimmed = rawLine.trim();
    if (trimmed.startsWith("## ")) {
      flushPending();
      pendingHeading = trimmed.slice(3).trim();
      pendingContent = [];
      continue;
    }
    if (pendingHeading) {
      pendingContent.push(rawLine);
    } else {
      output.push(rawLine);
    }
  }
  flushPending();
  return output.join("\n");
}

function isEmptySectionContent(lines) {
  const cleaned = lines
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => line.replace(/^\d+\.\s*/, "").replace(/^-\s*/, "").trim());
  if (cleaned.length === 0) return true;
  return cleaned.every((line) => line === "（空）");
}

function splitCsv(text) {
  return String(text || "")
    .replaceAll("，", ",")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function splitLines(text) {
  return String(text || "")
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttr(text) {
  return escapeHtml(text).replaceAll("`", "&#96;");
}
