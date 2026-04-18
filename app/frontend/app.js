const state = {
  currentRawText: "",
  currentMarkdown: "",
  latestAnswer: "",
  importJobId: "",
  importPollTimer: null,
  organizeRunId: "",
  organizePollTimer: null,
  notes: [],
  relations: [],
  graphNodes: [],
  graphEdges: [],
  history: [],
};

const els = {
  tabs: document.querySelectorAll(".tab-button"),
  panels: document.querySelectorAll(".tab-panel"),
  healthBadge: document.getElementById("health-badge"),
  direction: document.getElementById("direction-select"),
  rawText: document.getElementById("raw-text"),
  importLog: document.getElementById("import-log"),
  importProgress: document.getElementById("import-progress"),
  importProgressText: document.getElementById("import-progress-text"),
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
  historySummary: document.getElementById("history-summary"),
  historyList: document.getElementById("history-list"),
  keywordQuery: document.getElementById("keyword-query"),
  questionQuery: document.getElementById("question-query"),
  searchResultList: document.getElementById("search-result-list"),
  answerModeBadge: document.getElementById("answer-mode-badge"),
  answerView: document.getElementById("answer-view"),
};

init();

function init() {
  bindEvents();
  refreshHealth();
  refreshNetwork();
  refreshKnowledgeHistory();
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
  document.getElementById("btn-refresh-history").addEventListener("click", refreshKnowledgeHistory);
  document.getElementById("btn-organize-quick").addEventListener("click", () => startOrganizeLibrary("quick"));
  document.getElementById("btn-organize-full").addEventListener("click", () => startOrganizeLibrary("full"));
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
      stats.mode ? `模式: ${stats.mode}` : "",
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
  setImportProgress(true, "正在启动导入任务...", 2);
  let completed = false;
  try {
    const data = await runImportJob({
      kind: "text",
      text,
      direction: els.direction.value,
      export: false,
    });
    setImportProgress(true, "整理完成，正在刷新知识网络...", 96);
    appendImportLog(JSON.stringify(data, null, 2));
    if (data.markdown) {
      updateResultView(text, data.markdown);
      switchTab("result");
    }
    refreshNetwork();
    refreshKnowledgeHistory();
    completed = true;
  } catch (err) {
    appendImportLog(`整理失败：${err.message}`);
  } finally {
    setImportProgress(false, completed ? "整理完成" : "整理失败", completed ? 100 : 0);
  }
}

async function importFiles() {
  const files = Array.from(els.fileInput.files || []);
  if (files.length === 0) {
    appendImportLog("请先选择 txt/md 文件。");
    return;
  }
  appendImportLog(`正在处理 ${files.length} 个上传文件...`);
  setImportProgress(true, `正在读取 ${files.length} 个文件...`, 8);
  let completed = false;
  try {
    const payloadFiles = [];
    for (const file of files) {
      const content = await file.text();
      payloadFiles.push({ name: file.name, content });
      setImportProgress(true, `已读取 ${payloadFiles.length}/${files.length} 个文件...`, Math.round((payloadFiles.length / files.length) * 35));
    }
    setImportProgress(true, `正在启动 ${files.length} 个文件的导入任务...`, 40);
    const data = await runImportJob({
      kind: "files",
      files: payloadFiles,
      direction: els.direction.value,
      export: false,
    });
    appendImportLog(JSON.stringify(data, null, 2));
    setImportProgress(true, "导入完成，正在刷新知识网络...", 96);
    const lastSuccess = (data.results || []).find((item) => item.markdown);
    if (lastSuccess && lastSuccess.markdown) {
      updateResultView("", lastSuccess.markdown);
      switchTab("result");
    }
    refreshNetwork();
    refreshKnowledgeHistory();
    completed = true;
  } catch (err) {
    appendImportLog(`文件导入失败：${err.message}`);
  } finally {
    setImportProgress(false, completed ? "导入完成" : "导入失败", completed ? 100 : 0);
  }
}

async function importFileByPath() {
  const path = els.filePath.value.trim();
  if (!path) {
    appendImportLog("请输入文件路径。");
    return;
  }
  appendImportLog(`正在导入文件：${path}`);
  setImportProgress(true, "正在启动本地文件导入任务...", 2);
  let completed = false;
  try {
    const data = await runImportJob({
      kind: "file_path",
      path,
      direction: els.direction.value,
      export: false,
    });
    setImportProgress(true, "导入完成，正在刷新知识网络...", 96);
    appendImportLog(JSON.stringify(data, null, 2));
    if (data.markdown) {
      updateResultView("", data.markdown);
      switchTab("result");
    }
    refreshNetwork();
    refreshKnowledgeHistory();
    completed = true;
  } catch (err) {
    appendImportLog(`路径导入失败：${err.message}`);
  } finally {
    setImportProgress(false, completed ? "导入完成" : "导入失败", completed ? 100 : 0);
  }
}

async function importDirectory() {
  const path = els.dirPath.value.trim();
  if (!path) {
    appendImportLog("请输入目录路径。");
    return;
  }
  appendImportLog(`正在增量扫描目录：${path}`);
  setImportProgress(true, "正在启动目录增量任务...", 2);
  let completed = false;
  try {
    const data = await runImportJob({
      kind: "dir",
      path,
      direction: els.direction.value,
      recursive: true,
    });
    setImportProgress(true, "增量更新完成，正在刷新知识网络...", 96);
    appendImportLog(JSON.stringify(data, null, 2));
    refreshNetwork();
    refreshKnowledgeHistory();
    completed = true;
  } catch (err) {
    appendImportLog(`目录增量失败：${err.message}`);
  } finally {
    setImportProgress(false, completed ? "增量更新完成" : "增量更新失败", completed ? 100 : 0);
  }
}

async function runImportJob(payload) {
  const started = await apiPost("/api/import/start", payload);
  state.importJobId = started.job_id || "";
  if (!state.importJobId) {
    throw new Error("后端没有返回导入任务 ID");
  }
  appendImportLog(`导入任务已启动：${state.importJobId}`);
  return waitForImportJob(state.importJobId);
}

function waitForImportJob(jobId) {
  if (state.importPollTimer) {
    clearInterval(state.importPollTimer);
    state.importPollTimer = null;
  }

  return new Promise((resolve, reject) => {
    state.importPollTimer = setInterval(async () => {
      try {
        const data = await apiGet(`/api/import/status?job_id=${encodeURIComponent(jobId)}`);
        const job = data.job || {};
        const status = job.status || "running";
        const percent = Number(job.percent || 0);
        setImportProgress(true, formatImportProgressMessage(job), percent);

        if (status === "completed") {
          clearInterval(state.importPollTimer);
          state.importPollTimer = null;
          state.importJobId = "";
          resolve(job.result || {});
        } else if (status === "failed") {
          clearInterval(state.importPollTimer);
          state.importPollTimer = null;
          state.importJobId = "";
          reject(new Error(job.error || job.message || "导入任务失败"));
        }
      } catch (err) {
        clearInterval(state.importPollTimer);
        state.importPollTimer = null;
        state.importJobId = "";
        reject(err);
      }
    }, 1000);
  });
}

function formatImportProgressMessage(job) {
  const pieces = [];
  const message = job.message || "正在处理...";
  pieces.push(message);
  if (job.current_file && job.total_files) {
    pieces.push(`文件 ${job.current_file}/${job.total_files}`);
  }
  if (job.current_chunk && job.total_chunks) {
    pieces.push(`知识单元 ${job.current_chunk}/${job.total_chunks}`);
  }
  return pieces.join(" · ");
}

async function startOrganizeLibrary(mode = "quick") {
  const normalizedMode = mode === "full" ? "full" : "quick";
  if (normalizedMode === "full") {
    const ok = confirm("全局整理会从头重建全部知识关系，耗时会更久。确定要继续吗？");
    if (!ok) return;
  }
  try {
    els.organizeStatus.textContent = normalizedMode === "full" ? "正在启动全局整理任务..." : "正在启动快速整理任务...";
    const data = await apiPost("/api/library/organize/start", { mode: normalizedMode });
    state.organizeRunId = data.run_id || "";
    els.organizeStatus.textContent = `${normalizedMode === "full" ? "全局整理" : "快速整理"}任务已启动: ${state.organizeRunId}`;
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
      const mode = live.mode || (run.stats && run.stats.mode) || "";
      els.organizeStatus.textContent = `整理中(${status}${mode ? `/${mode}` : ""}) ${percent}% ${msg}`;
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
  const limit = Math.max(20, Math.min(300, Number(els.networkLimit.value || "120")));
  try {
    const data = await apiGet(`/api/library/graph?limit=${limit}`);
    const graph = data.graph || {};
    state.graphNodes = graph.nodes || [];
    state.graphEdges = graph.edges || [];
    renderKnowledgeNetwork(graph, data.run || null);
  } catch (err) {
    els.networkSummary.textContent = `网络加载失败：${err.message}`;
    els.graph.innerHTML = "";
    els.relationList.innerHTML = "";
  }
}

async function refreshKnowledgeHistory() {
  if (!els.historyList || !els.historySummary) return;
  try {
    const data = await apiGet("/api/library/history?limit=80");
    state.history = data.history || [];
    renderKnowledgeHistory(state.history, data.summary || {});
  } catch (err) {
    els.historySummary.textContent = `入库历史加载失败：${err.message}`;
    els.historyList.innerHTML = "";
  }
}

function renderKnowledgeHistory(history, summary) {
  const sourceCount = Number(summary.sources || history.length || 0);
  const noteCount = Number(summary.notes || 0);
  els.historySummary.innerHTML = [
    `<span><strong>${sourceCount}</strong> 次来源记录</span>`,
    `<span><strong>${noteCount}</strong> 条知识入库</span>`,
  ].join("");

  if (!history.length) {
    els.historyList.innerHTML = `<li class="history-item">暂无入库历史。</li>`;
    return;
  }

  els.historyList.innerHTML = history
    .map((item) => {
      const titles = (item.note_titles || []).slice(0, 4);
      const titleText = titles.length ? titles.join("；") : "未生成知识条";
      return `
        <li class="history-item">
          <strong>${escapeHtml(item.imported_at || item.created_at || "未知时间")}</strong>
          <span>${escapeHtml(item.source_type || "unknown")} · 新知识 ${Number(item.note_count || 0)} 条 · ${escapeHtml(item.status || "")}</span>
          <small>${escapeHtml(item.file_path || item.source_id || "")}</small>
          <div>${escapeHtml(titleText)}</div>
        </li>`;
    })
    .join("");
}

function renderKnowledgeNetwork(graph, run) {
  const nodes = (graph.nodes || []).slice(0, 140);
  const edges = (graph.edges || []).slice(0, 360);
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const stats = run && run.stats ? run.stats : {};
  const summary = graph.summary || {};
  els.networkSummary.innerHTML = [
    `<strong>${summary.unit_nodes || 0}</strong> 条知识`,
    `<strong>${summary.concept_nodes || 0}</strong> 个概念`,
    `<strong>${summary.structure_edges || 0}</strong> 条结构边`,
    `<strong>${summary.semantic_edges || 0}</strong> 条语义边`,
    stats.duplicate_candidates !== undefined ? `<strong>${stats.duplicate_candidates}</strong> 个重复候选` : "",
    run ? `最近整理：${escapeHtml(run.status || "unknown")}` : "尚未整理",
  ]
    .filter(Boolean)
    .map((item) => `<span>${item}</span>`)
    .join("");

  renderGraph(nodes, edges, nodeMap);
  renderRelationList(edges, nodeMap);
}

function renderGraph(nodes, edges, noteMap) {
  if (!nodes.length) {
    els.graph.innerHTML = `<text x="36" y="60" class="graph-empty">暂无网络。导入内容后点击“整理知识库”。</text>`;
    return;
  }

  const width = 900;
  const height = 520;
  const positions = new Map();
  const lanes = {
    topic: { y: 70, nodes: [] },
    concept: { y: 180, nodes: [] },
    group: { y: 300, nodes: [] },
    unit: { y: 430, nodes: [] },
  };

  nodes.forEach((node) => {
    const lane = lanes[node.node_type] ? node.node_type : "unit";
    lanes[lane].nodes.push(node);
  });

  Object.values(lanes).forEach((lane) => {
    const count = Math.max(1, lane.nodes.length);
    lane.nodes.forEach((node, index) => {
      const margin = node.node_type === "unit" ? 48 : 70;
      const x = margin + ((width - margin * 2) * (index + 0.5)) / count;
      positions.set(node.id, { x, y: lane.y });
    });
  });

  const edgeHtml = edges
    .map((edge) => {
      const a = positions.get(edge.from_id);
      const b = positions.get(edge.to_id);
      if (!a || !b) return "";
      const cls = `edge edge-${escapeAttr(edge.relation_layer || "semantic")} edge-${escapeAttr(edge.relation_strength || "medium")} edge-${escapeAttr(edge.relation_type || "related")}`;
      return `<line class="${cls}" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"><title>${escapeHtml(edge.relation_type || "")}</title></line>`;
    })
    .join("");

  const nodeHtml = nodes
    .map((node) => {
      const pos = positions.get(node.id);
      if (!pos) return "";
      const title = compactTitle(node.label || "未命名", node.node_type === "unit" ? 10 : 12);
      const radius = node.node_type === "unit" ? 16 : node.node_type === "concept" ? 18 : 20;
      return `
        <g class="graph-node graph-node-${escapeAttr(node.node_type || "unit")}">
          <circle cx="${pos.x}" cy="${pos.y}" r="${radius}"></circle>
          <text x="${pos.x}" y="${pos.y + 36}" text-anchor="middle">${escapeHtml(title)}</text>
          <title>${escapeHtml(node.label || "未命名")}</title>
        </g>`;
    })
    .join("");

  const legend = `
    <g class="graph-legend">
      <text x="24" y="32">主题 -> 概念 -> 知识组 -> 知识条；强语义边用于检索扩展</text>
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
      const from = noteMap.get(rel.from_id);
      const to = noteMap.get(rel.to_id);
      return `
        <li class="relation-item">
          <strong>${escapeHtml(rel.relation_type || "related")}</strong>
          <span>${escapeHtml((from && from.label) || rel.from_id)} → ${escapeHtml((to && to.label) || rel.to_id)}</span>
          <small>${escapeHtml(rel.relation_layer || "")} · ${escapeHtml(rel.relation_strength || "")} · score=${Number(rel.score || 0).toFixed(3)} ${escapeHtml(rel.reason || "")}</small>
        </li>`;
    })
    .join("");
}

async function keywordSearch() {
  const query = els.keywordQuery.value.trim();
  if (!query) {
    els.answerView.textContent = "请输入关键词。";
    setAnswerMode("");
    return;
  }
  els.answerView.textContent = "正在检索...";
  setAnswerMode("");
  try {
    const data = await apiGet(`/api/search?query=${encodeURIComponent(query)}&limit=12`);
    renderSearchResults(data.results || []);
    els.answerView.textContent = `命中 ${data.results.length} 条。`;
  } catch (err) {
    els.answerView.textContent = `检索失败：${err.message}`;
    setAnswerMode("");
  }
}

async function askQuestion() {
  const question = els.questionQuery.value.trim();
  if (!question) {
    els.answerView.textContent = "请输入问题。";
    setAnswerMode("");
    return;
  }
  els.answerView.textContent = "正在生成结构化回答...";
  setAnswerMode("");
  try {
    const data = await apiPost("/api/ask", {
      question,
      limit: 15,
      export: false,
    });
    renderSearchResults(data.results || []);
    state.latestAnswer = data.answer || "";
    setAnswerMode(data.answer_mode || "extract");
    els.answerView.innerHTML = markdownToHtml(state.latestAnswer, { hideSourceInfo: false, removeEmptySections: false });
    switchTab("qa");
  } catch (err) {
    els.answerView.textContent = `问答失败：${err.message}`;
    setAnswerMode("");
  }
}

function setAnswerMode(mode) {
  if (!els.answerModeBadge) return;
  if (!mode) {
    els.answerModeBadge.className = "mode-badge hidden";
    els.answerModeBadge.textContent = "";
    return;
  }
  const normalized = mode === "model" ? "model" : "extract";
  els.answerModeBadge.textContent = normalized === "model" ? "模型" : "摘录";
  els.answerModeBadge.className = `mode-badge mode-${normalized}`;
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
        <small>
          ${escapeHtml(item.note_type || "未分类")}
          · ${escapeHtml(item.relevance_level || "相关")}
          · 相关度=${Number(item.relevance_score || 0).toFixed(3)}
          ${item.relation_type ? ` · 关系=${escapeHtml(item.relation_type)}` : ""}
          ${item.match_source === "vector" ? " · 向量语义" : ""}
        </small>
        <small>${escapeHtml(item.relevance_reason || "")}</small>
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

function setImportProgress(active, text, percent = 0) {
  if (!els.importProgress || !els.importProgressText) return;
  els.importProgress.classList.toggle("hidden", !active);
  els.importProgress.classList.toggle("active", Boolean(active));
  els.importProgressText.textContent = text || (active ? "正在处理..." : "等待开始");
  const bar = els.importProgress.querySelector(".progress-bar");
  if (bar) {
    bar.style.setProperty("--progress-width", `${Math.max(0, Math.min(100, Number(percent || 0)))}%`);
  }
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
