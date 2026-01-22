document.addEventListener('DOMContentLoaded', () => {
  const generateBtn = document.getElementById('generate-button');
  const promptInput = document.getElementById('prompt');
  const sizeInput = document.getElementById('size');
  const variantsInput = document.getElementById('variants');
  const urlsInput = document.getElementById('ref-urls');
  const statusPanel = document.getElementById('status-panel');
  const statusText = document.getElementById('status-text');
  const errorPanel = document.getElementById('error-panel');
  const errorMessage = document.getElementById('error-message');
  const resultsContainer = document.getElementById('results');

  generateBtn.addEventListener('click', async () => {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      showError('请输入提示词');
      return;
    }

    // Reset UI
    hideError();
    resultsContainer.innerHTML = ''; // Clear previous results or keep them? Maybe clear for now.
    resultsContainer.classList.add('empty-state');
    resultsContainer.innerHTML = `
      <div class="empty-card">
        <div class="empty-icon">⏳</div>
        <h3>正在生成</h3>
        <p>AI 正在挥洒创意，请稍候...</p>
      </div>
    `;
    
    generateBtn.disabled = true;
    statusPanel.classList.remove('hidden');
    statusText.textContent = '正在连接生成服务...';

    const formData = new FormData();
    formData.append('prompt', prompt);
    formData.append('size', sizeInput.value);
    formData.append('variants', variantsInput.value);
    
    const urls = urlsInput.value.trim().split('\n').filter(u => u.trim());
    urls.forEach(url => formData.append('urls', url.trim()));

    try {
      const response = await fetch('/api/generate_image', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || '请求失败');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      resultsContainer.classList.remove('empty-state');
      resultsContainer.innerHTML = ''; // Clear "Waiting" state

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        
        // Process buffer line by line or object by object
        // The backend yields raw chunks from GRSAI or JSON error
        // GRSAI SSE format usually starts with "data: " if strictly SSE, 
        // but our backend yields raw chunks.
        // Assuming backend yields JSON strings or parts of it.
        // Wait, generate_image_stream yields:
        // if chunk.startswith("data:"): chunk = chunk[len("data:") :].strip()
        // So we are getting JSON strings.
        
        // We might get multiple JSON objects in one chunk or split across chunks.
        // Simple approach: try to parse the buffer if it looks complete, or split by newline if backend sends newlines.
        // The backend loop: `for line in response.iter_lines... yield chunk`
        // So each yield is likely a complete JSON string (because iter_lines splits by \n).
        // But Flask stream might buffer.
        
        // Let's assume lines.
        const lines = buffer.split('\n');
        // Keep the last part if it's not empty (incomplete line)
        buffer = lines.pop() || ''; 
        
        for (const line of lines) {
            if (!line.trim()) continue;
            try {
                const data = JSON.parse(line);
                handleStreamData(data);
            } catch (e) {
                console.log('JSON Parse error for line:', line, e);
            }
        }
      }
      
      // Process remaining buffer
      if (buffer.trim()) {
          try {
              const data = JSON.parse(buffer);
              handleStreamData(data);
          } catch (e) {
              // Ignore incomplete json at end
          }
      }

    } catch (err) {
      showError(err.message);
      resultsContainer.innerHTML = `
        <div class="empty-card">
            <div class="empty-icon">❌</div>
            <h3>生成失败</h3>
            <p>${err.message}</p>
        </div>
      `;
    } finally {
      generateBtn.disabled = false;
      statusPanel.classList.add('hidden');
    }
  });

  function handleStreamData(data) {
    if (data.error) {
        showError(data.error);
        return;
    }

    if (data.status === 'running' || (data.progress && data.progress < 100)) {
        statusText.textContent = `生成中... ${data.progress || 0}%`;
    }

    if (data.status === 'succeeded' || (data.results && data.results.length > 0)) {
        statusText.textContent = '生成完成！';
        renderResults(data.results);
    }
    
    // Also handle top-level url/width/height if results array is missing (old format?)
    if (!data.results && data.url) {
         renderResults([{url: data.url, width: data.width, height: data.height}]);
    }
  }

  function renderResults(results) {
    // We append results or replace? 
    // Since stream might send final result multiple times or partials, 
    // usually the last one is the complete set.
    // Let's clear and render.
    resultsContainer.innerHTML = '';
    
    if (!results || results.length === 0) return;

    results.forEach(item => {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
            <img src="${item.url}" class="result-image" alt="Generated Image" loading="lazy">
            <div class="result-body">
                <p>尺寸: ${item.width} x ${item.height}</p>
                <div class="result-actions">
                    <a href="${item.url}" target="_blank" class="action-button action-primary">查看大图</a>
                </div>
            </div>
        `;
        resultsContainer.appendChild(card);
    });
  }

  function showError(msg) {
    errorPanel.classList.remove('hidden');
    errorMessage.textContent = msg;
  }

  function hideError() {
    errorPanel.classList.add('hidden');
    errorMessage.textContent = '';
  }
});
