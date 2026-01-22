const STYLE_PRESETS = [
  // --- Landscape ---
  {
    id: "blue_gold",
    name: "蓝金色调",
    description: "风光主流。主色蓝、辅色金黄，冷天暖光，通透壮阔，适合日出日落。",
    category: "landscape"
  },
  {
    id: "teal_orange",
    name: "青橙色调",
    description: "电影感。主色青、辅色橙，冷暖强对冲，戏剧化冲击，适合大场景。",
    category: "landscape"
  },
  {
    id: "blue_cyan",
    name: "蓝青冷色调",
    description: "极简克制。整体偏冷，孤独冷静，适合雪山、极地、清晨。",
    category: "landscape"
  },
  {
    id: "warm_golden",
    name: "暖橙金色调",
    description: "暖色主导。温暖厚重，适合秋季森林、沙漠、丹霞。",
    category: "landscape"
  },
  {
    id: "blue_green",
    name: "蓝绿色调",
    description: "自然生态。蓝绿主导，清新自然，适合草原、湖泊、夏季山地。",
    category: "landscape"
  },
  {
    id: "muted_nordic",
    name: "灰蓝低饱和",
    description: "高级感。灰蓝/灰青，安静克制，适合阴天、雾景、北欧风光。",
    category: "landscape"
  },
  {
    id: "monotone",
    name: "单色倾向",
    description: "色彩极简。单一色相主导，强情绪，适合雾、雪、剪影。",
    category: "landscape"
  },
  {
    id: "black_white",
    name: "黑白风光",
    description: "结构力量。脱离色彩，强调纹理，适合高反差地形。",
    category: "landscape"
  },
  // --- Portrait ---
  {
    id: "teal_orange_portrait",
    name: "青橙色调 (人像)",
    description: "人像首选。青色背景+橙色皮肤，强对比立体感，适合商业、街拍。",
    category: "portrait"
  },
  {
    id: "warm_skin_cool_bg",
    name: "暖肤冷背景",
    description: "干净耐看。暖肤色+冷灰背景，青橙的自然版，适合肖像。",
    category: "portrait"
  },
  {
    id: "soft_warm_pastel",
    name: "日系清透",
    description: "温柔空气感。浅暖主色+低饱和绿蓝，适合日常、校园。",
    category: "portrait"
  },
  {
    id: "creamy_beige",
    name: "奶油色调",
    description: "高级轻奢。米白/奶油黄，柔和高级，适合棚拍、女性肖像。",
    category: "portrait"
  },
  {
    id: "cool_cinematic",
    name: "冷灰电影",
    description: "克制硬朗。冷灰/蓝灰+少量暖肤，适合男性、街头、剧情。",
    category: "portrait"
  },
  {
    id: "vintage_brown",
    name: "暖棕复古",
    description: "怀旧胶片。棕色/橙棕，弱对比，适合复古穿搭。",
    category: "portrait"
  },
  {
    id: "bw_contrast_portrait",
    name: "高对比黑白",
    description: "结构戏剧性。强调明暗力量，适合男性、纪实。",
    category: "portrait"
  },
  {
    id: "monotone_portrait",
    name: "单色人像",
    description: "实验情绪。单一色相，强风格化，适合概念人像。",
    category: "portrait"
  }
];

const state = {
  file: null,
  dataUrl: "",
  results: [],
  analysis: "",
  runId: "",
  generatedStyleIds: new Set(),
  availableStyles: [],
  detectedCategory: null
};

const elements = {
  fileInput: document.getElementById("file-input"),
  previewImage: document.getElementById("preview-image"),
  uploadArea: document.getElementById("upload-area"),
  uploadPlaceholder: document.querySelector(".upload-placeholder"),
  resetButton: document.getElementById("reset-button"),
  lutToggle: document.getElementById("lut-toggle"),
  lutSpace: document.getElementById("lut-space"),
  sceneType: document.getElementById("scene-type"),
  styleStrength: document.getElementById("style-strength"),
  styleStrengthValue: document.getElementById("style-strength-value"),
  debugToggle: document.getElementById("debug-toggle"),
  generateButton: document.getElementById("generate-button"),
  regenerateButton: document.getElementById("regenerate-button"),
  continueButton: document.getElementById("continue-button"),
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
    // Reset state for new file
    state.generatedStyleIds.clear();
    state.availableStyles = [];
    state.detectedCategory = null;
    elements.continueButton.classList.add("hidden");
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
  // Reset state
  state.generatedStyleIds.clear();
  state.availableStyles = [];
  state.detectedCategory = null;
  elements.continueButton.classList.add("hidden");
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
  startGeneration();
});

elements.regenerateButton.addEventListener("click", () => {
  startGeneration();
});

elements.continueButton.addEventListener("click", () => {
  generateNextBatch();
});

function syncStyleStrength() {
  if (!elements.styleStrength || !elements.styleStrengthValue) {
    return;
  }
  const value = Number(elements.styleStrength.value || 0);
  elements.styleStrengthValue.textContent = `${value}%`;
}

if (elements.styleStrength) {
  syncStyleStrength();
  elements.styleStrength.addEventListener("input", syncStyleStrength);
}

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
      <p>上传静帧后，AI 将基于场景分析生成 8 种调色参考。</p>
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
  if (elements.sceneType) {
    formData.append("scene_type", elements.sceneType.value);
  }
  if (elements.styleStrength) {
    formData.append("style_strength", elements.styleStrength.value);
  }
  formData.append("debug_requests", elements.debugToggle.checked ? "1" : "0");
  formData.append("styles", style.id);
  if (state.runId) {
    formData.append("run_id", state.runId);
  }

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
  return { result, analysis: data.analysis || analysis, runId: data.run_id || "" };
}

async function loadHistoryRecord(runId) {
  if (!runId) {
    return;
  }
  clearError();
  setStatus(false, "");
  try {
    const response = await fetch(`/api/history/${encodeURIComponent(runId)}`);
    const data = await readJsonResponse(response);
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

async function startGeneration() {
  if (!state.file) {
    showError("请先上传静帧。");
    return;
  }
  clearError();
  setStatus(true, "正在解析场景...");
  
  // Reset for fresh start
  state.generatedStyleIds.clear();
  state.results = [];
  elements.results.className = "results";
  elements.results.innerHTML = "";
  elements.regenerateButton.classList.add("hidden");
  elements.continueButton.classList.add("hidden");

  try {
    const analysis = await streamAnalysis();
    state.analysis = analysis;
    if (!analysis) {
      elements.analysisCard.classList.add("hidden");
    }

    // Determine category from analysis
    state.detectedCategory = "landscape"; // Default
    if (analysis) {
      if (/SCENE_CATEGORY:\s*portrait/i.test(analysis)) {
        state.detectedCategory = "portrait";
      } else if (/SCENE_CATEGORY:\s*landscape/i.test(analysis)) {
        state.detectedCategory = "landscape";
      }
    }

    // Filter available styles based on category
    state.availableStyles = STYLE_PRESETS.filter(style => style.category === state.detectedCategory);

    if (state.analysis) {
      elements.analysisText.textContent = state.analysis;
      elements.analysisCard.classList.remove("hidden");
    }
    
    // Start first batch
    await generateNextBatch();

  } catch (error) {
    showError(error.message || "生成失败，请重试。");
    if (!state.results.length) {
      renderEmptyState();
    }
    setStatus(false, "");
  }
}

async function generateNextBatch() {
  const BATCH_SIZE = 3;
  
  // Filter out already generated styles
  const remainingStyles = state.availableStyles.filter(style => !state.generatedStyleIds.has(style.id));
  
  if (remainingStyles.length === 0) {
      setStatus(false, "");
      elements.regenerateButton.classList.remove("hidden");
      elements.continueButton.classList.add("hidden");
      return;
  }
  
  const batch = remainingStyles.slice(0, BATCH_SIZE);
  
  try {
    for (let i = 0; i < batch.length; i++) {
      const style = batch[i];
      setStatus(true, `正在生成调色参考 (${i + 1}/${batch.length}): ${style.name}...`);
      
      const { result, analysis: mergedAnalysis, runId } = await generateStyle(style, state.analysis);
      
      if (!state.runId && runId) {
        state.runId = runId;
      }
      state.analysis = mergedAnalysis;
      
      // Add result
      state.results.push(result);
      // Mark as generated
      state.generatedStyleIds.add(style.id);
      
      renderResults();
    }
  } catch (error) {
      showError(error.message || "生成中断，请重试。");
  } finally {
      setStatus(false, "");
      
      // Check if there are more styles available
      const stillRemaining = state.availableStyles.filter(style => !state.generatedStyleIds.has(style.id));
      if (stillRemaining.length > 0) {
          elements.continueButton.classList.remove("hidden");
      } else {
          elements.continueButton.classList.add("hidden");
      }
      elements.regenerateButton.classList.remove("hidden");
  }
}

// Legacy function removed/replaced by startGeneration + generateNextBatch
// async function generateStyles() { ... }

renderEmptyState();
