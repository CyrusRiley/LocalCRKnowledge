const state = {
  currentRawText: "",
  currentMarkdown: "",
  latestAnswer: "",
  organizeRunId: "",
  organizePollTimer: null,
  notes: [],
  relations: [],
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
  organizeStatus: document.getElementById("organize-status"),
  networkLimit: document.getElementById("network-limit"),
  networkSummary: document.getElementById("network-summary"),
  graph: document.getElementById("knowledge-graph"),
  relationList: document.getElementById("relation-list"),
  keywordQuery: document.getElementById("keyword-query"),
  questionQuery: document.getElementById("question-query"),
  searchResultList: document.getElementById("search-result-list"),
  answerView: document.getElementById("answer-view"),
};

init();

function init() {
  bindEvents();
  refreshHealth();
  refreshNetwork();
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
  document.getElementById("btn-refresh-network").addEventListener("click", refreshNetwork);
  document.getElementById("btn-organize-library").addEventListener("click", startOrganizeLibrary);
  document.getElementById("btn-keyword-search").addEventListener("click", keywordSearch);
  document.getElementById("btn-ask").addEventListener("click", askQuestion);
  document.getElementById("btn-export-current").addEventListener("click", exportCurrentMarkdown);
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
    refreshNetwork();
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
    refreshNetwork();
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
    refreshNetwork();
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
    refreshNetwork();
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
        await refreshNetwork();
      }
    } catch (err) {
      els.organizeStatus.textContent = `整理状态轮询失败：${err.message}`;
      clearInterval(state.organizePollTimer);
      state.organizePollTimer = null;
    }
  }, 2500);
}

async function refreshNetwork() {
  const limit = Math.max(10, Math.min(200, Number(els.networkLimit.value || "80")));
  try {
    const [notesData, organizeData] = await Promise.all([
      apiGet(`/api/notes?limit=${limit}`),
      apiGet("/api/library/organize/latest"),
    ]);
    state.notes = notesData.notes || [];
    state.relations = organizeData.relations || [];
    renderKnowledgeNetwork(state.notes, state.relations, organizeData.run || null);
  } catch (err) {
    els.networkSummary.textContent = `网络加载失败：${err.message}`;
    els.graph.innerHTML = "";
    els.relationList.innerHTML = "";
  }
}

function renderKnowledgeNetwork(notes, relations, run) {
  const noteMap = new Map(notes.map((note) => [note.note_id, note]));
  const relationNodes = new Set();
  relations.forEach((rel) => {
    relationNodes.add(rel.from_note_id);
    relationNodes.add(rel.to_note_id);
  });
  const nodes = notes
    .filter((note) => relationNodes.size === 0 || relationNodes.has(note.note_id))
    .slice(0, 48);
  const visible = new Set(nodes.map((note) => note.note_id));
  const edges = relations
    .filter((rel) => visible.has(rel.from_note_id) && visible.has(rel.to_note_id))
    .slice(0, 140);

  const stats = run && run.stats ? run.stats : {};
  els.networkSummary.innerHTML = [
    `<strong>${nodes.length}</strong> 个节点`,
    `<strong>${edges.length}</strong> 条关系`,
    stats.duplicate_candidates !== undefined ? `<strong>${stats.duplicate_candidates}</strong> 个重复候选` : "",
    run ? `最近整理：${escapeHtml(run.status || "unknown")}` : "尚未整理",
  ]
    .filter(Boolean)
    .map((item) => `<span>${item}</span>`)
    .join("");

  renderGraph(nodes, edges, noteMap);
  renderRelationList(edges, noteMap);
}

function renderGraph(nodes, edges, noteMap) {
  if (!nodes.length) {
    els.graph.innerHTML = `<text x="36" y="60" class="graph-empty">暂无网络。导入内容后点击“整理知识库”。</text>`;
    return;
  }

  const width = 900;
  const height = 520;
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.min(width, height) * 0.34;
  const positions = new Map();

  nodes.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / nodes.length - Math.PI / 2;
    positions.set(node.note_id, {
      x: cx + Math.cos(angle) * radius,
      y: cy + Math.sin(angle) * radius,
    });
  });

  const edgeHtml = edges
    .map((edge) => {
      const a = positions.get(edge.from_note_id);
      const b = positions.get(edge.to_note_id);
      if (!a || !b) return "";
      return `<line class="edge edge-${escapeAttr(edge.relation_type || "related")}" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"><title>${escapeHtml(edge.relation_type || "")}</title></line>`;
    })
    .join("");

  const nodeHtml = nodes
    .map((node) => {
      const pos = positions.get(node.note_id);
      const title = compactTitle(node.title || "未命名", 12);
      return `
        <g class="graph-node">
          <circle cx="${pos.x}" cy="${pos.y}" r="20"></circle>
          <text x="${pos.x}" y="${pos.y + 36}" text-anchor="middle">${escapeHtml(title)}</text>
          <title>${escapeHtml(node.title || "未命名")}</title>
        </g>`;
    })
    .join("");

  const legend = `
    <g class="graph-legend">
      <text x="24" y="32">same_group / sequence_next / shared_keyword / duplicate_of / related_to</text>
    </g>`;

  els.graph.innerHTML = `${legend}${edgeHtml}${nodeHtml}`;
}

function renderRelationList(relations, noteMap) {
  if (!relations.length) {
    els.relationList.innerHTML = `<li class="relation-item">暂无关系。点击“整理知识库”生成关系网络。</li>`;
    return;
  }
  els.relationList.innerHTML = relations
    .slice(0, 80)
    .map((rel) => {
      const from = noteMap.get(rel.from_note_id);
      const to = noteMap.get(rel.to_note_id);
      return `
        <li class="relation-item">
          <strong>${escapeHtml(rel.relation_type || "related")}</strong>
          <span>${escapeHtml((from && from.title) || rel.from_note_id)} → ${escapeHtml((to && to.title) || rel.to_note_id)}</span>
          <small>score=${Number(rel.score || 0).toFixed(3)} ${escapeHtml(rel.reason || "")}</small>
        </li>`;
    })
    .join("");
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

function compactTitle(title, maxLength) {
  const clean = String(title || "").trim();
  if (clean.length <= maxLength) return clean;
  return `${clean.slice(0, maxLength - 1)}…`;
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
