/* ==========================================
   MediScan AI - Main Application JavaScript
   ========================================== */

(function () {
  'use strict';

  // ─── State ────────────────────────────────────────────────────────────────
  let selectedModel = null;
  let selectedFile = null;
  let isAnalyzing = false;

  // ─── DOM refs ─────────────────────────────────────────────────────────────
  const modelCards      = document.querySelectorAll('.model-card[data-model]');
  const fileInput       = document.getElementById('fileInput');
  const uploadZone      = document.getElementById('uploadZone');
  const previewContainer = document.getElementById('previewContainer');
  const previewImg      = document.getElementById('previewImg');
  const previewName     = document.getElementById('previewName');
  const previewSize     = document.getElementById('previewSize');
  const previewRemove   = document.getElementById('previewRemove');
  const analyzeBtn      = document.getElementById('analyzeBtn');
  const btnText         = document.getElementById('btnText');
  const btnSpinner      = document.getElementById('btnSpinner');
  const resultPanel     = document.getElementById('resultPanel');
  const errorPanel      = document.getElementById('errorPanel');
  const errorText       = document.getElementById('errorText');

  // ─── Model selection ──────────────────────────────────────────────────────
  modelCards.forEach(card => {
    card.addEventListener('click', () => {
      modelCards.forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      selectedModel = parseInt(card.dataset.model);
      updateAnalyzeBtn();
      hideResults();
    });
  });

  function updateAnalyzeBtn() {
    const ready = selectedModel && selectedFile;
    analyzeBtn.disabled = !ready;
    if (ready) {
      btnText.textContent = 'Analyze Image';
    } else if (!selectedModel) {
      btnText.textContent = 'Select a scan type first';
    } else {
      btnText.textContent = 'Upload an image first';
    }
  }

  // ─── File upload ──────────────────────────────────────────────────────────
  uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
  });

  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    const allowed = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!allowed.includes(file.type)) {
      showToast('⚠️ Please upload a valid image file (JPG, PNG, GIF)', 'warning');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showToast('⚠️ File size must be less than 10 MB', 'warning');
      return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      previewName.textContent = file.name;
      previewSize.textContent = formatBytes(file.size);
      previewContainer.classList.add('visible');
      hideResults();
    };
    reader.readAsDataURL(file);
    updateAnalyzeBtn();
  }

  previewRemove.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewContainer.classList.remove('visible');
    previewImg.src = '';
    updateAnalyzeBtn();
    hideResults();
  });

  // ─── Analyze ──────────────────────────────────────────────────────────────
  analyzeBtn.addEventListener('click', async () => {
    if (!selectedModel || !selectedFile || isAnalyzing) return;
    await runAnalysis();
  });

  async function runAnalysis() {
    isAnalyzing = true;
    setLoading(true);
    hideResults();
    hideError();

    try {
      const formData = new FormData();
      formData.append('choice', selectedModel);
      formData.append('data', selectedFile);

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Prediction failed. Please try again.');
      }

      renderResult(data);
      showToast('✅ Analysis complete!', 'success');

    } catch (err) {
      console.error(err);
      showError(err.message || 'An unexpected error occurred. Please try again.');
    } finally {
      isAnalyzing = false;
      setLoading(false);
    }
  }

  function setLoading(state) {
    analyzeBtn.disabled = state;
    analyzeBtn.classList.toggle('loading', state);
    if (state) {
      btnText.textContent = 'Analyzing...';
      btnSpinner.style.display = 'block';
    } else {
      btnText.textContent = 'Analyze Image';
      btnSpinner.style.display = 'none';
      if (!selectedModel || !selectedFile) updateAnalyzeBtn();
    }
  }

  // ─── Render result ────────────────────────────────────────────────────────
  function renderResult(data) {
    const color = data.color || 'success';
    const panel = resultPanel;

    // Card class
    const card = document.getElementById('resultCard');
    card.className = `result-card ${color}`;

    // Icon
    const icons = { success: '✅', danger: '🔴', warning: '🟡' };
    document.getElementById('resultIconEmoji').textContent = icons[color] || '🔵';

    // Labels
    document.getElementById('resultSeverityLabel').textContent =
      `${data.severity?.toUpperCase()} RISK DETECTED`;
    document.getElementById('resultDiagnosis').textContent = data.display_result || data.result;
    document.getElementById('resultModel').textContent = `Detected via ${data.model_info?.name || 'AI Model'}`;

    // Confidence bar
    const confFill = document.getElementById('confidenceFill');
    document.getElementById('confidenceValue').textContent = `${data.confidence?.toFixed(1)}%`;
    setTimeout(() => {
      confFill.style.width = `${Math.min(data.confidence, 100)}%`;
    }, 200);

    // All predictions breakdown
    const breakdownContainer = document.getElementById('predictionsBreakdown');
    breakdownContainer.innerHTML = '';
    if (data.all_predictions && data.all_predictions.length > 0) {
      data.all_predictions.forEach(pred => {
        const isActive = pred.label.toLowerCase() === (data.display_result || '').toLowerCase()
          || pred.label === data.result;
        const item = document.createElement('div');
        item.className = `breakdown-item ${isActive ? 'prediction-active' : ''}`;
        item.innerHTML = `
          <div class="breakdown-row">
            <span class="label">${pred.label}</span>
            <span class="value">${pred.confidence.toFixed(1)}%</span>
          </div>
          <div class="breakdown-track">
            <div class="breakdown-fill" data-width="${pred.confidence}"></div>
          </div>
        `;
        breakdownContainer.appendChild(item);
      });
      // Animate breakdown bars
      setTimeout(() => {
        document.querySelectorAll('.breakdown-fill').forEach(el => {
          el.style.width = `${el.dataset.width}%`;
        });
      }, 400);
    }

    // Description & recommendation
    document.getElementById('resultDescription').textContent = data.description || '';
    document.getElementById('resultRecommendation').textContent = data.recommendation || '';

    // Show panel
    panel.classList.add('visible');
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function hideResults() {
    if (resultPanel) resultPanel.classList.remove('visible');
  }

  // ─── Error ────────────────────────────────────────────────────────────────
  function showError(message) {
    if (errorPanel && errorText) {
      errorText.textContent = message;
      errorPanel.classList.add('visible');
      errorPanel.scrollIntoView({ behavior: 'smooth' });
    }
  }

  function hideError() {
    if (errorPanel) errorPanel.classList.remove('visible');
  }

  // ─── Toast ────────────────────────────────────────────────────────────────
  function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3500);
  }

  // ─── Utils ────────────────────────────────────────────────────────────────
  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  // ─── Animated number counters ─────────────────────────────────────────────
  function animateCounters() {
    document.querySelectorAll('[data-count]').forEach(el => {
      const target = parseFloat(el.dataset.count);
      const suffix = el.dataset.suffix || '';
      const duration = 1800;
      const start = performance.now();

      function tick(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3);
        const current = target * ease;
        el.textContent = Number.isInteger(target)
          ? Math.round(current) + suffix
          : current.toFixed(1) + suffix;
        if (progress < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    });
  }

  // Run counters on load
  window.addEventListener('load', () => {
    setTimeout(animateCounters, 300);
  });

  // Initial state
  updateAnalyzeBtn();

})();
