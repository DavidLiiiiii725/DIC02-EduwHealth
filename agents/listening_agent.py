from __future__ import annotations

from typing import Any, Dict, Optional


def _safe_str(x: Any) -> str:
    return ("" if x is None else str(x)).strip()


def generate_adhd_listening_strategy(
    *,
    scenario: str,
    environment: str,
    goal: str,
    ld_profile: Optional[Dict[str, Any]] = None,
    llm=None,
) -> str:
    """
    生成 ADHD / 焦虑友好的听力/听讲策略说明（Markdown，中文）。

    输入：
      - scenario: 听的场景（上课 / 会议 / 线上课程 / 听力练习 等）
      - environment: 环境特点（安静 / 有噪音 / 公共场所 等）
      - goal: 这次听的核心目标（例如“抓住关键信息”“提升听力理解”）
      - ld_profile: 学习者的 LD 配置（confirmed / suspected）
      - llm: LLMClient 实例；为空时返回降级版固定文案
    """
    scenario = _safe_str(scenario) or "课堂听讲"
    environment = _safe_str(environment) or "普通教室，有一定背景噪音"
    goal = _safe_str(goal) or "在有限时间内抓住讲话的关键信息"

    confirmed = (ld_profile or {}).get("confirmed") or []
    suspected = (ld_profile or {}).get("suspected") or []

    if llm is None:
        # 降级版：不依赖外部 LLM 时的简易策略
        return (
            f"## 0) 开始前：你没有做错\n"
            "ADHD/焦虑会让“听”更耗能，这是神经系统差异，不是懒。\n\n"
            f"## 1) 任务概览（先抓大，再抓细）\n"
            f"- 场景：{scenario}\n"
            f"- 环境：{environment}\n"
            f"- 目标：{goal}\n"
            "- 这次只需要：抓住 3 个关键信息 + 1 个问题\n"
            "- 这次不需要：逐字记完、每句话都懂\n\n"
            "## 2) 听之前：预热 3 分钟\n"
            "- 写下 2–3 个你“可能会听到的关键词”。\n"
            "- 设一个小目标：**我只找‘转折词/结论句’**。\n"
            "- 如果环境吵：换座位/戴耳机/把手机静音。\n\n"
            "## 3) 听的过程中：分段 + 标记\n"
            "- 每 5–10 分钟为一段：听完立刻停 30 秒。\n"
            "- 只记关键词：名词、数字、因果箭头（`因 → 果`）。\n"
            "- 没听懂先放过：画 `?`，继续往下听。\n\n"
            "## 4) 听完之后：3 分钟回顾\n"
            "- 写 3 个关键信息（短语即可）。\n"
            "- 写 1 个还模糊的问题（之后回放/提问用）。\n"
            "- 如果开始焦虑：先写一句“我在担心什么？”把担忧从脑子里倒出来。\n\n"
            "## 5) 自我倡导小句子（可直接照抄）\n"
            "- Could you repeat the key point more slowly?\n"
            "- Sorry, the environment is a bit noisy. Could you say that again?\n"
            "- 老师，可以把刚才的 3 个要点再重复一遍吗？我想确认自己记对了。\n"
            "- 我可能需要你说慢一点点，或者给一个简短的大纲。\n"
        )

    system = (
        "你是一名专门为 ADHD / 焦虑学习者设计**听力与听讲策略**的教练。\n"
        "请用**中文**输出，结构清晰、条目化，语气温和但不啰嗦。\n"
        "目标：让学习者在真实环境中更好地完成“听”的任务，而不是责备自己。\n"
        "重要约束：\n"
        "- 避免长大段文字；多用小标题 + 列表。\n"
        "- 每一步都要“可执行”，最好是 2–5 分钟能做完的小动作。\n"
        "- 尽量把注意力 / 焦虑 / 工作记忆等认知机制讲清楚，但不要用太学术的语气。\n"
        "- 帮助学习者练习元认知监控（比如自问：我刚才听懂了吗？）和自我倡导（比如如何礼貌地请老师重复）。\n"
        "- 要体现对 ADHD 与焦虑的理解：问题来自神经发展差异，而不是懒惰。\n"
    )

    user = f"""
学习者信息（可为空）：
- LD confirmed: {confirmed}
- LD suspected: {suspected}

本次听力/听讲场景：
- 场景：{scenario}
- 环境：{environment}
- 本次目标：{goal}

请输出一份 **“ADHD / 焦虑友好听力策略卡片（可分页）”**（Markdown）。

关键要求（必须严格遵守，便于前端拆分成卡片）：
- 只允许使用二级标题 `##` 来分模块。
- 必须且只能包含以下模块标题（顺序一致、标题文字一致）：
  1) `## 0) 开始前：你没有做错`
  2) `## 1) 任务概览（先抓大，再抓细）`
  3) `## 2) 听之前：预热 3 分钟`
  4) `## 3) 听的过程中：分段 + 标记`
  5) `## 4) 听完之后：3 分钟回顾`
  6) `## 5) 自我倡导小句子（可直接照抄）`
  7) `## 6) 结束语：把注意力当作资源保护`
- 每个模块最多 5 条要点；每条尽量 1 句话，避免长段落。
- 必须体现：注意力选择困难、工作记忆受限、冲动打断、焦虑占用工作记忆/信噪比下降。
- 必须包含“没听懂先标记 ? 继续往下听”的策略。
- 自我倡导模块给 3–5 句中英混合可照抄句。

只输出 Markdown 正文，不要输出其他解释。
"""

    text = llm.chat(system=system, user=user, temperature=0.6)
    return text.strip()


def generate_sample_listening_passage(
    *,
    scenario: str,
    llm=None,
) -> str:
    """
    根据听力场景生成一段雅思真题风格的英文示例文本（对话或独白），供 TTS 朗读练习。
    """
    scenario = _safe_str(scenario) or "IELTS listening"
    if llm is None:
        # 降级：雅思真题风格固定示例（Section 1–4 风格轮换）
        s = scenario.lower()
        # Section 1 风格：日常对话（预订/咨询）
        if any(k in s for k in ["英语", "ielts", "english", "听力"]):
            return (
                "Good morning, Riverside Sports Centre. How can I help you? "
                "— Hi, I'd like to join the gym. What's the monthly fee? "
                "— Our standard membership is forty-five pounds per month, or ninety for three months if you pay in advance. "
                "— And what are the opening hours? "
                "— We're open from six in the morning until ten at night on weekdays, and eight until eight on weekends."
            )
        # Section 2 风格：独白（设施介绍）
        if any(k in s for k in ["大课", "大学", "lecture", "课", "介绍"]):
            return (
                "Welcome to the campus library. Let me explain the layout. On the ground floor you'll find the main desk and the reference section. "
                "The first floor has study rooms which can be booked online. "
                "The second floor is for group projects; please keep noise to a minimum. "
                "The cafeteria is in the basement, open from eight until six."
            )
        # Section 3 风格：师生讨论
        if any(k in s for k in ["会议", "meeting", "讨论"]):
            return (
                "So for your project, have you decided on the topic? "
                "— We were thinking of comparing renewable energy policies in two countries. "
                "— Good idea. Which countries? "
                "— Germany and Japan. We've found some useful data already. "
                "— Remember to include both advantages and limitations. The deadline is the fifteenth."
            )
        # Section 4 风格：学术讲座
        return (
            "Today we'll look at how bees communicate. When a forager finds a good food source, she returns to the hive and performs a waggle dance. "
            "The angle of the dance indicates direction relative to the sun. The duration indicates distance. "
            "Other bees then fly directly to the source. This system is remarkably efficient."
        )

    system = (
        "You write IELTS Listening-style scripts. Output ONLY a short passage (60-100 words) that could appear in the real test.\n"
        "Choose ONE style: Section 1 (everyday conversation, e.g. booking, enquiry), Section 2 (monologue, e.g. facility intro, radio), "
        "Section 3 (student-tutor discussion), or Section 4 (academic lecture excerpt).\n"
        "Write realistic dialogue or monologue. Include numbers, names, or dates where natural. No preamble, no Chinese, no markdown."
    )
    user = (
        f"Scenario hint: {scenario}. Generate an IELTS-style passage (dialogue or monologue) that fits. Output the passage only."
    )
    text = llm.chat(system=system, user=user, temperature=0.7)
    out = (text or "").strip()
    if len(out) > 600:
        out = out[:600].rsplit(".", 1)[0] + "."
    return out if out else (
        "Good morning, Riverside Sports Centre. How can I help you? "
        "— I'd like to book a swimming lesson. "
        "— We have lessons on Tuesday and Thursday at ten or three. "
        "— Thursday at three, please. My name is Sarah Mitchell. "
        "— That's M-I-T-C-H-E-L-L. Your lesson is confirmed."
    )


def extract_logic_chain(passage: str, llm=None) -> str:
    """
    从听力示例文本中提取内容逻辑链：起因 → 经过 → 结果，便于 ADHD 学习者把握音频逻辑。
    """
    passage = _safe_str(passage)
    if not passage:
        return "暂无示例音频，请先生成听力策略卡片。"

    if llm is None:
        # 降级：按句粗略分成起因/经过/结果
        sentences = [s.strip().rstrip(".") for s in passage.replace(". ", ".|").split("|") if s.strip()]
        sentences = [s[:50] + "..." if len(s) > 50 else s for s in sentences[:6]]
        a = sentences[0] if sentences else "（未识别）"
        b = "\n".join(f"  - {s}" for s in sentences[1:-1]) if len(sentences) > 2 else "（未识别）"
        c = sentences[-1] if len(sentences) > 1 else "（未识别）"
        return "【起因】背景/原因：\n  " + a + "\n\n【经过】过程/发展：\n" + b + "\n\n【结果】结论/要点：\n  " + c

    system = (
        "你是听力学习助手。根据给定的英文听力文本，分析其**内容逻辑**，按「起因 → 经过 → 结果」输出。\n\n"
        "输出格式（必须严格遵守，用中文）：\n"
        "【起因】背景/原因是什么？\n"
        "- 用 1–2 条简短概括（每条 15 字以内）\n\n"
        "【经过】过程/发展讲了什么？\n"
        "- 用 2–4 条列出关键步骤或要点（每条 15 字以内）\n\n"
        "【结果】结论/要点是什么？\n"
        "- 用 1–2 条概括结论或行动建议（每条 15 字以内）\n\n"
        "只输出以上三块内容，不要其他解释，不要重复原文。"
    )
    user = f"听力文本：\n{passage}\n\n请按「起因 → 经过 → 结果」分析内容逻辑："
    try:
        text = llm.chat(system=system, user=user, temperature=0.3)
        return (text or "").strip() or "无法提取逻辑链。"
    except Exception:
        return extract_logic_chain(passage, llm=None)

