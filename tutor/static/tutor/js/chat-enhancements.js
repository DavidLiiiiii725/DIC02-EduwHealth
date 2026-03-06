// chat-enhancements.js
// 前端增强：风险监控、干预建议、警报

let currentInterventions = [];
let currentInterventionId = null;

// ── 显示干预建议面板 ────────────────────────────────────
function showInterventionPanel(interventions) {
  if (!interventions || interventions.length === 0) return;

  const panel = document.getElementById('interventionPanel');
  const list = document.getElementById('interventionList');

  currentInterventions = interventions;

  // 清空现有内容
  list.innerHTML = '';

  // 填充干预建议
  interventions.forEach(function(intervention, idx) {
    const card = createInterventionCard(intervention, idx);
    list.appendChild(card);
  });

  // 显示面板
  panel.classList.add('open');
}

// ── 创建干预卡片 ────────────────────────────────────────
function createInterventionCard(intervention, idx) {
  const card = document.createElement('div');
  card.className = 'intervention-card';
  card.dataset.interventionId = idx;

  let stepsHtml = '';
  if (intervention.steps && intervention.steps.length > 0) {
    stepsHtml = '<ul class="intervention-steps">' +
      intervention.steps.map(function(step) { return '<li>' + escHtmlSafe(step) + '</li>'; }).join('') +
      '</ul>';
  }

  let toolsHtml = '';
  if (intervention.tools && intervention.tools.length > 0) {
    toolsHtml = '<div class="intervention-tools" style="margin-top:10px; font-size:0.72rem; color:var(--muted);">' +
      '<strong>工具:</strong> ' + intervention.tools.map(escHtmlSafe).join(', ') +
      '</div>';
  }

  card.innerHTML =
    '<div class="intervention-type">' + escHtmlSafe(intervention.type || '策略') + '</div>' +
    '<div class="intervention-strategy">' + escHtmlSafe(intervention.strategy || intervention.description || '个性化支持策略') + '</div>' +
    stepsHtml +
    toolsHtml;

  // 点击卡片选中
  card.addEventListener('click', function() {
    document.querySelectorAll('.intervention-card').forEach(function(c) { c.classList.remove('selected'); });
    card.classList.add('selected');
    currentInterventionId = idx;
  });

  return card;
}

// ── 关闭干预面板 ────────────────────────────────────────
function closeInterventionPanel() {
  const panel = document.getElementById('interventionPanel');
  if (panel) panel.classList.remove('open');
}

// ── 应用干预策略 ────────────────────────────────────────
function applyIntervention() {
  if (currentInterventionId === null) {
    showNotification('请先选择一个策略', 'info');
    return;
  }

  const intervention = currentInterventions[currentInterventionId];

  // 记录用户选择（发送到后端）
  fetch('/api/interventions/apply/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCookie('csrftoken')
    },
    body: JSON.stringify({
      intervention_id: currentInterventionId,
      intervention_type: intervention.type
    })
  })
  .then(function(res) { return res.json(); })
  .then(function(data) {
    console.log('[INFO] Intervention applied:', data);
  })
  .catch(function(err) {
    console.warn('[WARN] Failed to record intervention:', err);
  });

  closeInterventionPanel();
  showNotification('策略已应用！如果有帮助请告诉我。', 'success');
}

// ── 显示高风险警报 ──────────────────────────────────────
function showRiskAlert() {
  const modal = document.getElementById('riskAlertModal');
  if (modal) {
    modal.style.display = 'flex';
  }
}

function closeRiskAlert() {
  const modal = document.getElementById('riskAlertModal');
  if (modal) {
    modal.style.display = 'none';
  }
}

function showInterventions() {
  closeRiskAlert();
  if (currentInterventions.length > 0) {
    showInterventionPanel(currentInterventions);
  }
}

// ── 监控风险变化 ────────────────────────────────────────
function monitorRiskChanges(newRiskLevel, interventions) {
  const riskIndicator = document.querySelector('.risk-indicator');
  if (!riskIndicator) return;

  const previousLevel = riskIndicator.dataset.riskLevel;

  // 更新风险等级显示
  riskIndicator.dataset.riskLevel = newRiskLevel;
  const riskText = riskIndicator.querySelector('.risk-text');
  if (riskText) riskText.textContent = newRiskLevel.toUpperCase();

  // 如果风险等级升高到 high 或 severe，显示警报
  if ((newRiskLevel === 'high' || newRiskLevel === 'severe') &&
      (previousLevel !== 'high' && previousLevel !== 'severe')) {
    currentInterventions = interventions || [];
    setTimeout(showRiskAlert, 1000);
  }
}

// ── 更新风险指示器 ──────────────────────────────────────
function updateRiskIndicator(riskAssessment) {
  const indicator = document.querySelector('.risk-indicator');
  if (!indicator) return;
  indicator.dataset.riskLevel = riskAssessment.risk_level;
  const riskText = indicator.querySelector('.risk-text');
  if (riskText) riskText.textContent = riskAssessment.risk_level.toUpperCase();
  const riskConf = indicator.querySelector('.risk-confidence');
  if (riskConf) riskConf.textContent = Math.round((riskAssessment.confidence || 0) * 100) + '%';
}

// ── 显示通知 ────────────────────────────────────────────
function showNotification(message, type) {
  type = type || 'info';
  const notification = document.createElement('div');
  notification.className = 'notification notification-' + type;
  notification.textContent = message;
  notification.style.cssText =
    'position:fixed;top:20px;right:20px;padding:12px 20px;border-radius:8px;' +
    'background:var(--surface);border:1px solid var(--border);box-shadow:var(--shadow);' +
    'z-index:300;animation:slideInRight 0.3s ease;font-size:0.82rem;color:var(--text);';

  document.body.appendChild(notification);

  setTimeout(function() {
    notification.style.opacity = '0';
    notification.style.transform = 'translateX(100px)';
    notification.style.transition = 'opacity 0.3s,transform 0.3s';
    setTimeout(function() { notification.remove(); }, 300);
  }, 3000);
}

// ── 安全转义 HTML ───────────────────────────────────────
function escHtmlSafe(s) {
  if (!s) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── 获取 CSRF Token ─────────────────────────────────────
function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    const cookies = document.cookie.split(';');
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      if (cookie.substring(0, name.length + 1) === (name + '=')) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}

// ── 初始化 ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  console.log('[INFO] Chat enhancements loaded');
});
