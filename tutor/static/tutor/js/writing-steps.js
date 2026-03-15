// writing-steps.js - 分步卡片式任务展示
let taskSteps = [];
let currentStepIndex = 0;

function parseTaskSteps(promptText) {
  const stepPattern = /\*\*第?(\d+|[一二三四五六七八九十]+)步[：:](.*?)\*\*/g;
  const steps = [];
  let lastIndex = 0;
  let match;
  
  while ((match = stepPattern.exec(promptText)) !== null) {
    if (lastIndex > 0) {
      const content = promptText.substring(lastIndex, match.index).trim();
      if (content) {
        steps[steps.length - 1].content = content;
      }
    }
    steps.push({
      title: match[2].trim(),
      content: ''
    });
    lastIndex = match.index + match[0].length;
  }
  
  if (steps.length > 0 && lastIndex < promptText.length) {
    const content = promptText.substring(lastIndex).trim();
    steps[steps.length - 1].content = content;
  }
  
  if (steps.length === 0) {
    return [{
      title: '写作任务',
      content: promptText
    }];
  }
  
  return steps;
}

function showStep(index) {
  if (index < 0 || index >= taskSteps.length) return;
  
  currentStepIndex = index;
  const step = taskSteps[index];
  
  const contentEl = document.getElementById('step-content');
  if (contentEl) {
    contentEl.innerHTML = 
      '<div style="font-size:1.05rem;font-weight:600;color:var(--blue);margin-bottom:12px;">' + 
      step.title + 
      '</div><div style="white-space:pre-wrap;line-height:1.8;">' + 
      step.content + 
      '</div>';
  }
  
  const progressText = document.getElementById('step-progress');
  if (progressText) {
    progressText.textContent = '步骤 ' + (index + 1) + ' / ' + taskSteps.length;
  }
  
  const progressBar = document.getElementById('step-progress-bar');
  if (progressBar) {
    const percent = ((index + 1) / taskSteps.length) * 100;
    progressBar.style.width = percent + '%';
  }
  
  const prevBtn = document.getElementById('btn-prev-step');
  const nextBtn = document.getElementById('btn-next-step');
  
  if (prevBtn) {
    prevBtn.style.display = index > 0 ? 'inline-flex' : 'none';
  }
  
  if (nextBtn) {
    if (index >= taskSteps.length - 1) {
      nextBtn.textContent = '完成 ✓';
      nextBtn.style.background = 'var(--green)';
    } else {
      nextBtn.textContent = '下一步 →';
      nextBtn.style.background = 'var(--blue)';
    }
  }
}

function nextStep() {
  if (currentStepIndex >= taskSteps.length - 1) {
    const contentEl = document.getElementById('step-content');
    if (contentEl) {
      contentEl.innerHTML = 
        '<div style="text-align:center;padding:40px 20px;">' +
        '<div style="font-size:2rem;margin-bottom:12px;">🎉</div>' +
        '<div style="font-size:1.1rem;font-weight:600;margin-bottom:8px;">太棒了！</div>' +
        '<div style="color:var(--muted);margin-bottom:20px;">你已经完成了所有步骤指导。现在可以开始在右侧作答区写作了。</div>' +
        '<button class="btn" onclick="resetSteps()">重新查看步骤</button>' +
        '</div>';
    }
    const nextBtn = document.getElementById('btn-next-step');
    if (nextBtn) nextBtn.style.display = 'none';
    return;
  }
  showStep(currentStepIndex + 1);
}

function prevStep() {
  if (currentStepIndex > 0) {
    showStep(currentStepIndex - 1);
  }
}

function resetSteps() {
  showStep(0);
  const nextBtn = document.getElementById('btn-next-step');
  if (nextBtn) nextBtn.style.display = 'inline-flex';
}

function toggleFullPrompt() {
  const promptEl = document.getElementById('out-prompt');
  const toggleText = document.getElementById('toggle-prompt-text');
  if (!promptEl || !toggleText) return;
  
  if (promptEl.style.display === 'none') {
    promptEl.style.display = 'block';
    toggleText.textContent = '收起';
  } else {
    promptEl.style.display = 'none';
    toggleText.textContent = '展开';
  }
}

// 在生成任务后调用此函数来初始化步骤卡片
function initStepCards(promptText) {
  if (!promptText) return;
  
  taskSteps = parseTaskSteps(promptText);
  if (taskSteps.length > 0) {
    const stepContainer = document.getElementById('step-card-container');
    if (stepContainer) {
      stepContainer.style.display = 'block';
      showStep(0);
    }
  }
}
