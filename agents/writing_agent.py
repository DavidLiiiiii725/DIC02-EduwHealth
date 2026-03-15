from __future__ import annotations

from typing import Any, Dict, Optional


def _safe_str(x: Any) -> str:
    return ("" if x is None else str(x)).strip()


def generate_step_by_step_guide(
    *,
    ielts_question: str,
    genre: str,
    audience: str,
    target_words: int,
    difficulty: str = "normal",
    task_size: str = "big",
    ld_profile: Optional[Dict[str, Any]] = None,
    llm=None,
) -> Dict[str, str]:
    """
    基于雅思题目生成详细的分步写作指导（4-6步）。
    
    返回 dict：
      - prompt: 分步写作指导（包含4-6个步骤，每步有明确的时间和任务）
      - outline: 空字符串（不再需要）
      - checklist: 提交前检查清单
    """
    ielts_question = _safe_str(ielts_question)
    genre = _safe_str(genre) or "议论文"
    audience = _safe_str(audience) or "普通大众"
    target_words = int(target_words or 600)
    target_words = max(150, min(2000, target_words))
    difficulty = _safe_str(difficulty) or "normal"
    task_size = _safe_str(task_size) or "big"

    confirmed = (ld_profile or {}).get("confirmed") or []
    suspected = (ld_profile or {}).get("suspected") or []

    # 根据字数和任务大小调整步骤数量
    if task_size == "small" or target_words < 300:
        num_steps = "4-5"
        time_budget = "20-30"
    else:
        num_steps = "5-6"
        time_budget = "30-40"

    system = (
        "你是一名擅长 ADHD/Dysgraphia 支持的雅思写作教练。\n"
        "你要为学习者生成低启动成本、步骤清晰、可以立刻开始写的分步写作指导。\n"
        "输出必须是中文，结构清晰，避免长段落。\n"
        "要求：\n"
        f"- 将任务拆成 {num_steps} 个小步骤（每步 5-15 分钟可完成）。\n"
        "- 每个步骤必须用 **第X步：标题（时间）** 格式标注。\n"
        "- 每个步骤下面给出具体的操作指导（2-4句话）。\n"
        "- 语气温和但明确，减少羞耻/责备语气。\n"
        "- 重点关注 Dysgraphia 友好：先口述/列要点，再扩写成句子。\n"
    )

    user = f"""
学习者信息（可能为空，仅供适配）：
- LD confirmed: {confirmed}
- LD suspected: {suspected}

雅思写作题目：
{ielts_question}

写作要求：
- 文体：{genre}
- 目标读者：{audience}
- 目标字数：约 {target_words} 字
- 难度偏好：{difficulty}（easy/normal/hard）
- 总时间预算：约 {time_budget} 分钟

请生成一个友好的分步写作指导，帮助学习者完成上面的雅思题目。

格式要求：
1. 开头用一句话欢迎并说明任务（例如：你好！这次我们来写一篇议论文，主题是XXX，字数大约XXX字。这个任务有点挑战性，但别担心，我会把它拆成几个小步骤，每一步你只需要专注5-10分钟。准备好了吗？我们开始吧。）

2. 然后列出 {num_steps} 个步骤，每个步骤格式如下：
**第一步：头脑风暴（5分钟）**
请打开一个空白文档或拿出一张纸。设置一个5分钟的计时器。在这5分钟内，快速写下所有当你想到主题关键词时，脑海中浮现的任何想法、问题或关联。不要评判，只管写下来。时间到就停。

**第二步：确定核心论点（10分钟）**
...

3. 最后一句话鼓励学习者：完成以上X步，你就已经成功启动了这篇有挑战的文章！后续步骤可以参照上面的节奏，一次只攻克一小段。

注意：
- 每个步骤的时间加起来应该接近 {time_budget} 分钟
- 步骤要循序渐进：头脑风暴 → 确定论点 → 搭建大纲 → 逐段写作
- 针对 Dysgraphia：强调先说/列要点，再写完整句子
"""

    if llm is None:
        # 降级版本
        prompt = (
            f"你好！这次我们来完成一篇{genre}，字数大约{target_words}字。\n\n"
            f"雅思题目：{ielts_question}\n\n"
            "我会把它拆成几个小步骤：\n\n"
            "**第一步：头脑风暴（5分钟）**\n"
            "快速写下所有相关的想法和例子。\n\n"
            "**第二步：确定论点（10分钟）**\n"
            "从头脑风暴中提炼出你的核心观点。\n\n"
            "**第三步：搭建大纲（10分钟）**\n"
            "列出每段的主题句。\n\n"
            "**第四步：逐段写作（15-20分钟）**\n"
            "一次只写一段，写完休息一下。"
        )
        checklist = (
            "□ 每段都有一句本段要点\n"
            "□ 总结处给出 1 个可执行的小行动\n"
            "□ 段落不超过 6 行\n"
            "□ 用列表替代大段解释\n"
            "□ 删除重复句子"
        )
        return {"prompt": prompt, "outline": "", "checklist": checklist}

    text = llm.chat(system=system, user=user, temperature=0.6)
    
    # 简单提取 checklist（如果 LLM 生成了）
    checklist = (
        "□ 每段都有明确的主题句\n"
        "□ 论点有具体例子支持\n"
        "□ 段落长度适中（不超过6行）\n"
        "□ 结尾有总结和建议\n"
        "□ 检查拼写和语法"
    )
    
    return {"prompt": text, "outline": "", "checklist": checklist}


def generate_adhd_writing_task(
    *,
    topic: str,
    genre: str,
    audience: str,
    target_words: int,
    difficulty: str = "normal",
    ld_profile: Optional[Dict[str, Any]] = None,
    llm=None,
) -> Dict[str, str]:
    """
    生成 ADHD 友好的写作任务：清晰目标、分步产出、低摩擦启动。

    返回 dict：
      - prompt: 写作任务说明（可直接展示给用户）
      - outline: 建议大纲（可复制粘贴）
      - checklist: 提交前检查清单（短、可勾选）
    """
    topic = _safe_str(topic) or "ADHD 的学习支持"
    genre = _safe_str(genre) or "科普短文"
    audience = _safe_str(audience) or "普通大众"
    target_words = int(target_words or 600)
    target_words = max(150, min(2000, target_words))
    difficulty = _safe_str(difficulty) or "normal"

    confirmed = (ld_profile or {}).get("confirmed") or []
    suspected = (ld_profile or {}).get("suspected") or []

    system = (
        "你是一名擅长 ADHD 支持的写作教练。\n"
        "你要为学习者生成“低启动成本、步骤清晰、可以立刻开始写”的写作任务。\n"
        "输出必须是中文，结构清晰，避免长段落。\n"
        "要求：\n"
        "- 将任务拆成小步骤（每步 5-10 分钟可完成）。\n"
        "- 给出一个可直接照抄的大纲（含每段要点与建议句式）。\n"
        "- 给一个简短检查清单（不超过 10 条）。\n"
        "- 语气温和但明确，减少羞耻/责备语气。\n"
    )

    user = f"""
学习者信息（可能为空，仅供适配）：
- LD confirmed: {confirmed}
- LD suspected: {suspected}

请为以下写作场景生成任务：
- 主题：{topic}
- 文体：{genre}
- 目标读者：{audience}
- 目标字数：约 {target_words} 字
- 难度偏好：{difficulty}（easy/normal/hard）

请严格按以下格式输出三段（用清晰标题标注）：
【写作任务】
【建议大纲】
【检查清单】
"""

    if llm is None:
        # 允许在无 LLM 的情况下给一个可用的降级版本
        prompt = (
            f"请写一篇《{topic}》的{genre}，面向“{audience}”，字数约 {target_words} 字。\n"
            "写作方式：先写 3 句“你想让读者记住的要点”，再按大纲逐段扩写。"
        )
        outline = (
            "1) 开头（2-3 句）：用一个常见误解/场景引入主题\n"
            "2) 解释概念：用一句话定义 + 1 个例子\n"
            "3) 关键点 1：问题表现/影响（用小标题）\n"
            "4) 关键点 2：支持策略（用 3-5 条清单）\n"
            "5) 结尾（2-3 句）：总结 + 给读者一个可行动的建议"
        )
        checklist = (
            "□ 每段都有一句“本段要点”\n"
            "□ 总结处给出 1 个可执行的小行动\n"
            "□ 段落不超过 6 行\n"
            "□ 用列表替代大段解释\n"
            "□ 删除重复句子"
        )
        return {"prompt": prompt, "outline": outline, "checklist": checklist}

    text = llm.chat(system=system, user=user, temperature=0.6)
    return _split_three_sections(text)


def generate_adhd_writing_feedback(
    *,
    prompt: str,
    draft: str,
    genre: str,
    ld_profile: Optional[Dict[str, Any]] = None,
    llm=None,
) -> str:
    """
    ADHD 友好的写作反馈：先肯定有效点，再给 3-5 个最关键可执行修改。
    """
    prompt = _safe_str(prompt)
    draft = _safe_str(draft)
    genre = _safe_str(genre) or "未指定文体"

    confirmed = (ld_profile or {}).get("confirmed") or []
    suspected = (ld_profile or {}).get("suspected") or []

    if not draft:
        return "还没有检测到正文内容。你可以先随便写 5 句要点（不用完整），我再帮你把它扩成段落。"

    if llm is None:
        return (
            "我已经看到你的草稿了。为了更快变好：\n"
            "1) 先在开头加一句“这篇文章要解决什么问题”。\n"
            "2) 把最长的段落拆成 2 段。\n"
            "3) 每个小标题下只保留 3 个最重要的要点。\n"
            "4) 结尾补上 1 个可执行行动（例如：今天就做的 10 分钟练习）。"
        )

    system = (
        "你是一名写作教练，擅长 ADHD 友好反馈。\n"
        "输出必须是中文。\n"
        "规则：\n"
        "- 先给 2 条具体优点（必须引用草稿中的具体内容/表达方式）。\n"
        "- 再给 3-5 条“最关键、最少改动、收益最大”的修改建议，每条建议都要：说明原因 + 给一个可直接替换的示例句/示例段落。\n"
        "- 最后给一个 10 分钟可完成的下一步任务。\n"
        "- 避免一次性给太多建议，避免羞辱/责备语气。\n"
    )

    user = f"""
学习者信息（可能为空，仅供适配）：
- LD confirmed: {confirmed}
- LD suspected: {suspected}

写作任务（如有）：
{prompt or "(未提供任务说明)"}

文体：{genre}

草稿如下（原文照评）：
{draft}
"""
    return llm.chat(system=system, user=user, temperature=0.55)


def _split_three_sections(text: str) -> Dict[str, str]:
    """
    尝试把 LLM 输出按三段结构拆开，前端展示更稳定。
    允许模型输出略有偏差时做容错。
    """
    raw = _safe_str(text)
    if not raw:
        return {"prompt": "", "outline": "", "checklist": ""}

    markers = ["【写作任务】", "【建议大纲】", "【检查清单】"]
    idx = {m: raw.find(m) for m in markers}
    if all(v != -1 for v in idx.values()):
        p0 = idx["【写作任务】"]
        p1 = idx["【建议大纲】"]
        p2 = idx["【检查清单】"]
        prompt = raw[p0 + len("【写作任务】") : p1].strip()
        outline = raw[p1 + len("【建议大纲】") : p2].strip()
        checklist = raw[p2 + len("【检查清单】") :].strip()
        return {"prompt": prompt, "outline": outline, "checklist": checklist}

    # 兜底：不拆分，全部放到 prompt
    return {"prompt": raw, "outline": "", "checklist": ""}

