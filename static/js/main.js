/* main.js — AI Image Classifier · Frontend Logic */

document.addEventListener('DOMContentLoaded', () => {

  // ── Image Upload Preview ─────────────────────────────────
  const fileInput   = document.getElementById('imageInput');
  const uploadBox   = document.getElementById('uploadBox');
  const uploadContent = document.getElementById('uploadContent');
  const uploadPreview = document.getElementById('uploadPreview');
  const previewImg  = document.getElementById('previewImg');

  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file && file.type.startsWith('image/')) {
        showPreview(file);
      }
    });
  }

  function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImg) previewImg.src = e.target.result;
      if (uploadContent) uploadContent.style.display = 'none';
      if (uploadPreview) uploadPreview.style.display = 'block';
    };
    reader.readAsDataURL(file);
  }

  // ── Drag & Drop ──────────────────────────────────────────
  if (uploadBox) {
    uploadBox.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadBox.classList.add('drag-over');
    });

    uploadBox.addEventListener('dragleave', () => {
      uploadBox.classList.remove('drag-over');
    });

    uploadBox.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadBox.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {
        // Assign to input
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
        showPreview(file);
      }
    });
  }

  // ── Model Card Radio Selection ───────────────────────────
  const modelCards = document.querySelectorAll('.model-card input[type="radio"]');
  const modelDescs = document.querySelectorAll('.model-desc');

  modelCards.forEach((radio) => {
    radio.addEventListener('change', () => {
      // Update description display
      modelDescs.forEach(d => d.style.display = 'none');
      const desc = document.getElementById('desc_' + radio.value);
      if (desc) desc.style.display = 'flex';
    });
  });

  // ── Form Submit → Loading State ──────────────────────────
  const form       = document.getElementById('classifyForm');
  const predictBtn = document.getElementById('predictBtn');
  const btnInner   = predictBtn ? predictBtn.querySelector('.predict-btn-inner') : null;
  const btnLoading = predictBtn ? predictBtn.querySelector('.predict-loading') : null;

  if (form) {
    form.addEventListener('submit', (e) => {
      // Validate file selected
      if (!fileInput || !fileInput.files.length) {
        e.preventDefault();
        showToast('Please select an image before classifying.');
        return;
      }
      // Show loading
      if (btnInner)   btnInner.style.display   = 'none';
      if (btnLoading) btnLoading.style.display  = 'flex';
      if (predictBtn) predictBtn.disabled = true;
    });
  }

  // ── Toast Notification ───────────────────────────────────
  function showToast(msg) {
    let toast = document.getElementById('_toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = '_toast';
      toast.style.cssText = `
        position: fixed;
        bottom: 2rem; left: 50%;
        transform: translateX(-50%);
        background: var(--card);
        border: 1px solid rgba(244,63,94,0.35);
        color: var(--red);
        font-family: var(--mono);
        font-size: 0.72rem;
        padding: 0.75rem 1.4rem;
        border-radius: 8px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
        z-index: 9999;
        transition: opacity 0.3s;
        pointer-events: none;
      `;
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.style.opacity = '1';
    setTimeout(() => { toast.style.opacity = '0'; }, 3000);
  }

  // ── Smooth Scroll for Anchor Links ──────────────────────
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', (e) => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // ── Score Bar Animation (result page) ───────────────────
  document.querySelectorAll('.score-fill').forEach(el => {
    const targetW = el.style.width;
    el.style.width = '0%';
    requestAnimationFrame(() => {
      setTimeout(() => {
        el.style.transition = 'width 0.9s cubic-bezier(0.16,1,0.3,1)';
        el.style.width = targetW;
      }, 200);
    });
  });

  // ── Animate verdict card on result page ─────────────────
  const verdictCard = document.querySelector('.verdict-card');
  if (verdictCard) {
    verdictCard.style.opacity = '0';
    verdictCard.style.transform = 'translateY(16px)';
    requestAnimationFrame(() => {
      setTimeout(() => {
        verdictCard.style.transition = 'opacity 0.5s ease, transform 0.5s cubic-bezier(0.16,1,0.3,1)';
        verdictCard.style.opacity = '1';
        verdictCard.style.transform = 'translateY(0)';
      }, 100);
    });
  }

});
