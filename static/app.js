const STYLE_PRESETS = [
  { id: "cinematic", name: "好莱坞电影", description: "青橙色调 (Teal & Orange)，高对比度，深邃阴影，极具戏剧感。" },
  { id: "vintage", name: "经典胶片", description: "Kodak 暖黄色调，柔和的高光溢出，低饱和度，怀旧质感。" },
  { id: "minimal", name: "清新日系", description: "高调照明 (High-key)，低对比度，淡蓝色或偏白影调，干净明亮。" },
  { id: "noir", name: "暗黑悬疑", description: "低色温，强调阴影细节，冷峻的青蓝色系，压抑且迷人。" },
  { id: "commercial", name: "时尚商业", description: "高饱和，色彩还原准确且明亮，光影分布均匀，质感通透。" },
  { id: "cyber", name: "赛博都市", description: "霓虹冷暖色差，强烈的紫色与青色碰撞，极具现代冲击力。" }
];

const state = {
  file: null,
  dataUrl: "",
  results: [],
  analysis: "",
  runId: ""
};

const elements = {
  fileInput: document.getElementById("file-input"),
  previewImage: document.getElementById("preview-image"),
  uploadArea: document.getElementById("upload-area"),
  uploadPlaceholder: document.querySelector(".upload-placeholder"),
  resetButton: document.getElementById("reset-button"),
  lutToggle: document.getElementById("lut-toggle"),
  lutSpace: document.getElementById("lut-space"),
  debugToggle: document.getElementById("debug-toggle"),
  generateButton: document.getElementById("generate-button"),
  regenerateButton: document.getElementById("regenerate-button"),
  statusPanel: document.getElementById("status-panel"),
  statusText: document.getElementById("status-text"),
  analysisCard: document.getElementById("analysis-card"),
  analysisText: document.getElementById("analysis-text"),
  copyAnalysisButton: document.getElementById("copy-analysis"),
  results: document.getElementById("results"),
  errorPanel: document.getElementById("error-panel"),
  errorMessage: document.getElementById("error-message")
};

elements.fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) {
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    state.file = file;
    state.dataUrl = e.target.result;
    elements.previewImage.src = state.dataUrl;
    elements.previewImage.classList.remove("hidden");
    elements.resetButton.classList.remove("hidden");
    elements.uploadArea.classList.add("has-image");
    elements.uploadPlaceholder.classList.add("hidden");
    clearResults();
  };
  reader.readAsDataURL(file);
});

elements.resetButton.addEventListener("click", () => {
  state.file = null;
  state.dataUrl = "";
  elements.fileInput.value = "";
  elements.previewImage.src = "";
  elements.previewImage.classList.add("hidden");
  elements.resetButton.classList.add("hidden");
  elements.uploadPlaceholder.classList.remove("hidden");
  clearResults();
});

elements.copyAnalysisButton.addEventListener("click", async () => {
  const text = elements.analysisText.textContent.trim();
  if (!text) {
    return;
  }
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
    }
    elements.copyAnalysisButton.textContent = "已复制";
    elements.copyAnalysisButton.classList.add("copied");
    setTimeout(() => {
      elements.copyAnalysisButton.textContent = "复制";
      elements.copyAnalysisButton.classList.remove("copied");
    }, 1500);
  } catch (error) {
    showError("复制失败，请手动选择文本。");
  }
});

elements.generateButton.addEventListener("click", () => {
  generateStyles();
});

elements.regenerateButton.addEventListener("click", () => {
  generateStyles();
});

const params = new URLSearchParams(window.location.search);
const historyRunId = params.get("run_id");
if (historyRunId) {
  loadHistoryRecord(historyRunId);
}

function setStatus(visible, text) {
  if (visible) {
    elements.statusPanel.classList.remove("hidden");
    elements.statusText.textContent = text;
    elements.generateButton.disabled = true;
    elements.regenerateButton.classList.add("hidden");
  } else {
    elements.statusPanel.classList.add("hidden");
    elements.generateButton.disabled = false;
  }
}

function showError(message) {
  elements.errorMessage.textContent = message;
  elements.errorPanel.classList.remove("hidden");
}

function clearError() {
  elements.errorPanel.classList.add("hidden");
  elements.errorMessage.textContent = "";
}

function clearResults() {
  state.results = [];
  state.analysis = "";
  state.runId = "";
  elements.analysisCard.classList.add("hidden");
  elements.regenerateButton.classList.add("hidden");
  renderEmptyState();
}

function renderEmptyState() {
  elements.results.className = "results empty-state";
  elements.results.innerHTML = `
    <div class="empty-card">
      <div class="empty-icon">🎬</div>
      <h3>准备就绪</h3>
      <p>上传静帧后，AI 将基于场景分析生成 6 种调色参考。</p>
    </div>
  `;
}

function renderResults() {
  elements.results.className = "results";
  elements.results.innerHTML = "";
  state.results.forEach((item) => {
    const card = document.createElement("div");
    card.className = "result-card";

    const img = document.createElement("img");
    img.className = "result-image";
    img.src = item.image || item.image_url || "";
    img.alt = item.name;

    const body = document.createElement("div");
    body.className = "result-body";

    const title = document.createElement("h4");
    title.textContent = item.name;

    const desc = document.createElement("p");
    desc.textContent = item.description;

    const actions = document.createElement("div");
    actions.className = "result-actions";

    const lutLink = document.createElement("a");
    lutLink.className = "action-button action-primary";
    lutLink.textContent = "下载 3D LUT";
    if (item.lut_url) {
      lutLink.href = item.lut_url;
      lutLink.setAttribute("download", "");
    } else {
      lutLink.classList.add("disabled");
      lutLink.href = "#";
    }

    const imgLink = document.createElement("a");
    imgLink.className = "action-button action-secondary";
    imgLink.textContent = "保存图";
    imgLink.href = item.image || item.image_url || "#";
    imgLink.setAttribute("download", `ref_${item.id}.png`);

    actions.appendChild(lutLink);
    actions.appendChild(imgLink);

    body.appendChild(title);
    body.appendChild(desc);
    body.appendChild(actions);

    card.appendChild(img);
    card.appendChild(body);

  elements.results.appendChild(card);
  });
}

async function readJsonResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return await response.json();
  }
  const text = await response.text();
  if (response.status === 401) {
    throw new Error("登录已过期，请刷新页面。");
  }
  if (text && text.trim().startsWith("<!doctype")) {
    throw new Error("服务端返回了 HTML，可能是登录失效或服务异常。");
  }
  throw new Error(text ? text.trim() : "服务端返回非 JSON 响应。");
}

async function generateStyle(style, analysis) {
  const formData = new FormData();
  formData.append("image", state.file);
  formData.append("analysis", analysis);
  formData.append("generate_lut", elements.lutToggle.checked ? "1" : "0");
  if (elements.lutSpace) {
    formData.append("lut_space", elements.lutSpace.value);
  }
  formData.append("debug_requests", elements.debugToggle.checked ? "1" : "0");
  formData.append("styles", style.id);

  const response = await fetch("/api/generate", {
    method: "POST",
    body: formData,
  });
  const data = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(data.error || "生成失败，请重试。");
  }
  const result = (data.results || [])[0];
  if (!result) {
    throw new Error("未获取到生成结果。");
  }
  return { result, analysis: data.analysis || analysis };
}

async function loadHistoryRecord(runId) {
  if (!runId) {
    return;
  }
  clearError();
  setStatus(false, "");
  try {
    const response = await fetch(`/api/history/${encodeURIComponent(runId)}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "加载记录失败，请重试。");
    }
    clearResults();
    state.results = data.results || [];
    state.analysis = data.analysis || "";
    state.runId = data.run_id || runId;

    if (data.source_url) {
      elements.previewImage.src = data.source_url;
      elements.previewImage.classList.remove("hidden");
      elements.uploadArea.classList.add("has-image");
      elements.uploadPlaceholder.classList.add("hidden");
    }

    if (state.analysis) {
      elements.analysisText.textContent = state.analysis;
      elements.analysisCard.classList.remove("hidden");
    }

    if (state.results.length) {
      renderResults();
      elements.regenerateButton.classList.add("hidden");
    } else {
      renderEmptyState();
    }
  } catch (error) {
    showError(error.message || "加载记录失败，请重试。");
  }
}

async function streamAnalysis() {
  const formData = new FormData();
  formData.append("image", state.file);
  formData.append("debug_requests", elements.debugToggle.checked ? "1" : "0");

  const response = await fetch("/api/analyze_stream", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = "解析失败，请重试。";
    const text = await response.text();
    try {
      const data = JSON.parse(text);
      message = data.error || message;
    } catch (error) {
      if (text) {
        message = text;
      }
    }
    throw new Error(message);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let analysis = "";

  elements.analysisText.textContent = "";
  elements.analysisCard.classList.remove("hidden");

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    const chunk = decoder.decode(value, { stream: true });
    if (chunk) {
      analysis += chunk;
      elements.analysisText.textContent = analysis;
    }
  }

  analysis += decoder.decode();
  elements.analysisText.textContent = analysis;
  return analysis.trim();
}

async function generateStyles() {
  if (!state.file) {
    showError("请先上传静帧。");
    return;
  }
  clearError();
  setStatus(true, "正在解析场景...");

  try {
    const analysis = await streamAnalysis();
    state.analysis = analysis;
    if (!analysis) {
      elements.analysisCard.classList.add("hidden");
    }

    state.results = [];
    elements.results.className = "results";
    elements.results.innerHTML = "";

    for (let index = 0; index < STYLE_PRESETS.length; index += 1) {
      const style = STYLE_PRESETS[index];
      setStatus(true, `正在生成调色参考 (${index + 1}/${STYLE_PRESETS.length})...`);
      const { result, analysis: mergedAnalysis } = await generateStyle(style, analysis);
      state.analysis = mergedAnalysis;
      state.results.push(result);
      renderResults();
      elements.regenerateButton.classList.remove("hidden");
    }

    if (state.analysis) {
      elements.analysisText.textContent = state.analysis;
      elements.analysisCard.classList.remove("hidden");
    }
  } catch (error) {
    showError(error.message || "生成失败，请重试。");
    if (!state.results.length) {
      renderEmptyState();
    }
  } finally {
    setStatus(false, "");
  }
}

renderEmptyState();
