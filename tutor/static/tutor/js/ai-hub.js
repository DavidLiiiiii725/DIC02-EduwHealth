// ai-hub.js
// 全局 AI Hub 悬浮球（Reading / Writing 共用）

(function () {
  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function escapeHtml(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // 将任意长文本压缩为不超过 6 行的要点式摘要（前端近似实现）
  function summarizeToSixLines(raw) {
    if (!raw) return "";
    const text = String(raw).trim();

    // 优先按换行拆分（保留原有分段语义）
    let lines = text.split(/\r?\n+/).map(function (l) { return l.trim(); }).filter(Boolean);

    // 如果本身行数就 <= 6，直接返回
    if (lines.length <= 6) {
      return lines.join("\n");
    }

    // 否则按句号拆分成“句子”，再组装成最多 6 行
    const sentenceSplitRe = /(?<=[。！？!?\.])\s+/g;
    const sentences = text.split(sentenceSplitRe).map(function (s) { return s.trim(); }).filter(Boolean);

    const picked = sentences.slice(0, 6);
    return picked.join("\n");
  }

  // 仿生阅读：对每个单词前半部分加粗，保持原有换行
  function toBionicHtml(text) {
    if (!text) return "";
    const escaped = escapeHtml(text);
    return escaped
      .split(/\r?\n/)
      .map(function (line) {
        const parts = line.split(/(\s+)/); // 保留空白
        const transformed = parts.map(function (token) {
          if (!token || /^\s+$/.test(token)) return token;
          // 对纯字母或中日韩字符序列做处理，其它符号保持原样
          const pureWord = token.replace(/[^A-Za-z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/g, "");
          if (pureWord.length < 4) return token;
          const splitIndex = Math.ceil(pureWord.length * 0.5);
          let seen = 0;
          let head = "";
          let tail = "";
          for (let i = 0; i < token.length; i++) {
            const ch = token[i];
            if (/[A-Za-z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/.test(ch)) {
              if (seen < splitIndex) {
                head += ch;
              } else {
                tail += ch;
              }
              seen++;
            } else {
              tail += ch;
            }
          }
          return "<strong>" + head + "</strong>" + tail;
        });
        return transformed.join("");
      })
      .join("<br>");
  }

  function createHubInstance(config) {
    const ball = document.getElementById(config.ballId || "ai-hub-ball");
    const panel = document.getElementById(config.panelId || "ai-hub-panel");
    const messagesBox = document.getElementById(config.messagesId || "ai-hub-messages");
    const inputEl = document.getElementById(config.inputId || "ai-hub-input");
    const sendBtn = document.getElementById(config.sendBtnId || "ai-hub-send");

    if (!ball || !panel || !messagesBox || !inputEl) {
      console.warn("[AIHub] Missing DOM nodes, skipping init.");
      return null;
    }

    const storageKey = config.storageKey || "aiHubPosition";

    // ── 拖拽逻辑 ─────────────────────────────────────────────
    let dragging = false;
    let offsetX = 0;
    let offsetY = 0;

    function restorePosition() {
      try {
        const stored = window.localStorage.getItem(storageKey);
        if (!stored) return;
        const pos = JSON.parse(stored);
        if (typeof pos.x === "number" && typeof pos.y === "number") {
          ball.style.left = pos.x + "px";
          ball.style.top = pos.y + "px";
          ball.style.right = "auto";
          ball.style.bottom = "auto";
        }
      } catch (e) {
        // ignore
      }
    }

    function persistPosition() {
      try {
        const rect = ball.getBoundingClientRect();
        const pos = { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
        window.localStorage.setItem(storageKey, JSON.stringify(pos));
      } catch (e) {
        // ignore
      }
    }

    function pointerDown(ev) {
      const e = ev.touches ? ev.touches[0] : ev;
      dragging = true;
      const rect = ball.getBoundingClientRect();
      offsetX = e.clientX - rect.left - rect.width / 2;
      offsetY = e.clientY - rect.top - rect.height / 2;
      ev.preventDefault();
    }

    function pointerMove(ev) {
      if (!dragging) return;
      const e = ev.touches ? ev.touches[0] : ev;
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const radius = ball.offsetWidth / 2 || 24;
      let x = e.clientX - radius - offsetX;
      let y = e.clientY - radius - offsetY;
      x = clamp(x, 8, vw - radius * 2 - 8);
      y = clamp(y, 8, vh - radius * 2 - 8);
      ball.style.left = x + "px";
      ball.style.top = y + "px";
      ball.style.right = "auto";
      ball.style.bottom = "auto";
      ev.preventDefault();
    }

    function pointerUp() {
      if (!dragging) return;
      dragging = false;
      persistPosition();
    }

    ball.style.position = "fixed";
    ball.style.cursor = "grab";
    restorePosition();

    ball.addEventListener("mousedown", pointerDown);
    ball.addEventListener("touchstart", pointerDown, { passive: false });
    window.addEventListener("mousemove", pointerMove);
    window.addEventListener("touchmove", pointerMove, { passive: false });
    window.addEventListener("mouseup", pointerUp);
    window.addEventListener("touchend", pointerUp);

    // ── 面板 & 消息渲染 ─────────────────────────────────────
    function togglePanel() {
      const visible = panel.style.display === "block";
      panel.style.display = visible ? "none" : "block";
      if (!visible) {
        inputEl.focus();
      }
    }

    ball.addEventListener("click", function (e) {
      if (dragging) return; // 拖拽结束时不要触发展开
      togglePanel();
    });

    function appendMessage(html, role) {
      const wrapper = document.createElement("div");
      wrapper.className = "ai-hub-msg ai-hub-msg-" + role;
      wrapper.innerHTML = html;
      messagesBox.appendChild(wrapper);
      messagesBox.scrollTop = messagesBox.scrollHeight;
      return wrapper;
    }

    function appendTyping() {
      const dotHtml =
        '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
      const el = appendMessage(dotHtml, "assistant");
      el.classList.add("ai-hub-typing");
      return el;
    }

    function renderAssistantMessage(rawText) {
      const summary = summarizeToSixLines(rawText || "");
      const bionicHtml = toBionicHtml(summary);
      return bionicHtml;
    }

    async function send() {
      const text = (inputEl.value || "").trim();
      if (!text) return;
      appendMessage(toBionicHtml(text), "user");
      inputEl.value = "";

      const typingEl = appendTyping();

      try {
        const payload =
          typeof config.buildPayload === "function"
            ? config.buildPayload(text)
            : { message: text };

        const csrf =
          document.querySelector('input[name="csrfmiddlewaretoken"]')?.value ||
          (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] ||
          "";

        const resp = await fetch("/api/chat/stream/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": csrf,
          },
          body: JSON.stringify(payload),
        });

        if (!resp.ok) throw new Error('Request failed');

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();

        let metadata = null;
        let assistantMsgEl = null;
        let accumulatedContent = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const dataStr = line.slice(6);
            try {
              const event = JSON.parse(dataStr);

              if (event.type === 'metadata') {
                metadata = event.data;
              } else if (event.type === 'chunk') {
                // First chunk: create message element
                if (!assistantMsgEl) {
                  typingEl.remove();
                  assistantMsgEl = createAssistantMessageElement();
                  messagesBox.appendChild(assistantMsgEl);
                }
                // Accumulate content
                accumulatedContent += event.data;
                // Apply bionic reading and update
                const html = renderAssistantMessage(accumulatedContent);
                assistantMsgEl.innerHTML = html;
                messagesBox.scrollTop = messagesBox.scrollHeight;
              } else if (event.type === 'done') {
                // Handle UI action if present
                if (metadata && metadata.ui_action && metadata.ui_action.type === "focus_sentence" && typeof window.showFocusOverlay === "function") {
                  window.showFocusOverlay(
                    metadata.ui_action.sentence_text || "",
                    metadata.ui_action.analysis || accumulatedContent || "",
                    "AGENT · 句子精讲（第一句）"
                  );
                }
              } else if (event.type === 'error') {
                throw new Error(event.data);
              }
            } catch (e) {
              // Skip malformed JSON
            }
          }
        }
      } catch (e) {
        typingEl.remove();
        appendMessage(toBionicHtml("Connection issue — try again in a moment."), "assistant");
      }
    }

    function createAssistantMessageElement() {
      const wrapper = document.createElement("div");
      wrapper.className = "ai-hub-msg ai-hub-msg-assistant";
      return wrapper;
    }

    if (sendBtn) {
      sendBtn.addEventListener("click", send);
    }
    inputEl.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });

    // 暴露全局函数，兼容模板里的 onclick
    window.toggleFloatHub = togglePanel;
    window.sendFloatMessage = send;

    return {
      togglePanel: togglePanel,
      send: send,
    };
  }

  window.AIHub = {
    init: function (config) {
      return createHubInstance(config || {});
    },
  };
})();

