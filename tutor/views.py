import json
import logging
import time
import threading
import re
from pathlib import Path
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.utils import timezone
from django.db.models import Avg, Sum
from django.conf import settings

from .models import (
    LearnerProfile,
    ChatSession,
    ChatMessage,
    IELTSPassage,
    IELTSSection,
    IELTSQuestion,
    ReadingAttempt,
    StrategyPerformance,
    ReadingStrategyExperiment,
    SpeakingPractice,
)
from agents.reading_agent import generate_paragraph_guidance, generate_passage_prompt
from agents.reading_agent import reading_agent_explain_sentence

from core.llm_client import LLMClient
from agents.study_plan_agent import generate_study_plan, study_plan_chat_reply
from analytics.feature_extractor import FeatureExtractorLLM
from agents.speaking_agent import generate_adhd_speaking_pack, speaking_coach_reply
from agents.writing_agent import generate_adhd_writing_task, generate_adhd_writing_feedback, generate_step_by_step_guide
from agents.listening_agent import (
    generate_adhd_listening_strategy,
    generate_sample_listening_passage,
    extract_logic_chain,
)

logger = logging.getLogger(__name__)

# ── LD option definitions ─────────────────────────────────────────
LD_OPTIONS = {
    'executive_function':  {'name': 'Executive Function', 'icon': '🧩', 'desc': 'Planning, organising, starting tasks, working memory'},
    'adhd':                {'name': 'ADHD',               'icon': '⚡', 'desc': 'Attention, hyperactivity, impulsivity, focus'},
    'anxiety':             {'name': 'Anxiety',             'icon': '🌀', 'desc': 'Worry occupying cognitive resources during learning'},
    'motivation_disorder': {'name': 'Motivation',          'icon': '🪫', 'desc': 'Learned helplessness, avoidance, low self-efficacy'},
}

# ── Orchestrator singleton per learner ────────────────────────────
_orchestrators = {}
_orch_lock = threading.Lock()


def _get_orchestrator(learner_id: str):
    with _orch_lock:
        if learner_id not in _orchestrators:
            try:
                from core.orchestrator import TutorOrchestrator
                _orchestrators[learner_id] = TutorOrchestrator(learner_id=learner_id)
            except Exception as e:
                print(f"[WARN] Orchestrator unavailable: {e}")
                _orchestrators[learner_id] = None
        return _orchestrators[learner_id]


def _get_or_create_learner(request) -> LearnerProfile:
    learner_id = request.session.get('learner_id', 'default')
    learner, _ = LearnerProfile.objects.get_or_create(
        learner_id=learner_id,
        defaults={'display_name': 'Learner'}
    )
    return learner


def _mock_response(user_input: str) -> dict:
    return {
        'response': (
            "⚠️ Agent system not connected (demo mode).\n\n"
            f"You said: \"{user_input}\"\n\n"
            "Set EDUWHEALTH_PATH in settings.py and make sure the vector KB is built."
        ),
        'active_agent': 'demo',
        'cognitive_state': {'working_memory_load': 0.3, 'motivation_level': 0.7, 'affect_valence': 0.1, 'cognitive_fatigue': 0.1},
        'intervention_flags': {}, 'trajectory_flags': {},
        'risk': 0.0, 'risk_level': 'low', 'escalation': 'OK',
    }


# ══════════════════════════════════════════════════════════════════
#  PAGE VIEWS
# ══════════════════════════════════════════════════════════════════

def onboarding(request):
    if request.session.get('onboarded'):
        return redirect('chat')
    return render(request, 'tutor/onboarding.html')


def chat(request):
    learner = _get_or_create_learner(request)
    session = ChatSession.objects.filter(learner=learner, ended_at__isnull=True).last()
    if not session:
        session = ChatSession.objects.create(learner=learner)
    messages = session.messages.all()
    return render(request, 'tutor/chat.html', {
        'learner': learner,
        'session': session,
        'messages': messages,
    })


def profile(request):
    learner = _get_or_create_learner(request)
    return render(request, 'tutor/profile.html', {
        'learner': learner,
        'ld_options': LD_OPTIONS,
    })


def dashboard(request):
    learner  = _get_or_create_learner(request)
    sessions = ChatSession.objects.filter(learner=learner).order_by('-started_at')[:20]

    session_data = []
    for s in sessions:
        msgs = s.messages.filter(role='assistant').exclude(wm_load__isnull=True)
        if msgs.exists():
            session_data.append({
                'id':      s.id,
                'date':    s.started_at.strftime('%m/%d %H:%M'),
                'turns':   s.total_turns,
                'avg_wm':  round(msgs.aggregate(v=Avg('wm_load'))['v'] or 0, 2),
                'avg_mot': round(msgs.aggregate(v=Avg('motivation'))['v'] or 0, 2),
                'avg_aff': round(msgs.aggregate(v=Avg('affect'))['v'] or 0, 2),
            })

    agent_counts = {}
    for m in ChatMessage.objects.filter(session__learner=learner, role='assistant'):
        k = m.active_agent or 'unknown'
        agent_counts[k] = agent_counts.get(k, 0) + 1

    all_msgs      = ChatMessage.objects.filter(session__learner=learner, role='assistant')
    total_turns   = sessions.aggregate(t=Sum('total_turns'))['t'] or 0
    avg_motivation = round(all_msgs.aggregate(v=Avg('motivation'))['v'] or 0, 2)
    avg_wm         = round(all_msgs.aggregate(v=Avg('wm_load'))['v'] or 0, 2)

    return render(request, 'tutor/dashboard.html', {
        'learner':           learner,
        'sessions':          sessions,
        'session_data_json': json.dumps(session_data),
        'agent_counts_json': json.dumps(agent_counts),
        'total_turns':       total_turns,
        'avg_motivation':    avg_motivation,
        'avg_wm':            avg_wm,
    })


def _compute_study_stats(learner: LearnerProfile) -> dict:
    """
    Aggregate simple reading/quiz statistics for the study planner.

    This intentionally keeps the schema small and English-named so it can
    be passed directly into the LLM prompt.
    """
    # Recent reading attempts (last 14 days)
    since = timezone.now() - timezone.timedelta(days=14)
    attempts_qs = ReadingAttempt.objects.filter(learner=learner, started_at__gte=since)

    attempts_total = attempts_qs.count()
    completed = attempts_qs.filter(completed=True)
    completed_count = completed.count()

    # Average score across completed attempts
    avg_score = completed.aggregate(v=Avg('score'))['v'] if completed_count else None

    # Approximate total reading minutes (using started_at / updated_at)
    total_minutes = 0.0
    for att in attempts_qs.only('started_at', 'updated_at'):
        if att.started_at and att.updated_at:
            delta = att.updated_at - att.started_at
            total_minutes += max(0.0, delta.total_seconds() / 60.0)

    # Hint usage as a rough proxy for scaffolding need
    total_hints = attempts_qs.aggregate(v=Sum('hints_used'))['v'] or 0

    # Fallback budget: if user has little data, keep it gentle
    if total_minutes == 0 and attempts_total == 0:
        budget = 45
    else:
        # Start from recent average daily minutes, clipped to a sane range
        avg_per_attempt = total_minutes / max(attempts_total, 1)
        budget = int(min(150, max(30, avg_per_attempt * 2)))

    return {
        "learner_id": learner.learner_id,
        "display_name": learner.display_name,
        "recent_attempts_count": attempts_total,
        "recent_completed_attempts": completed_count,
        "recent_average_score": float(avg_score) if avg_score is not None else None,
        "recent_total_reading_minutes": round(total_minutes, 1),
        "recent_total_hints_used": int(total_hints),
        "total_minutes_budget": int(budget),
    }


def study_plan(request):
    """
    Simple page that shows today's AI-generated study plan and allows the
    learner to ask follow-up questions.
    """
    learner = _get_or_create_learner(request)
    stats = _compute_study_stats(learner)
    # Speed: do NOT block initial page render on LLM.
    # We persist stats immediately and let the frontend fetch/generate the plan
    # via a dedicated endpoint (cached in session).
    request.session['study_plan_stats'] = stats
    # Invalidate any previous plan if the budget/stats changed significantly.
    # Keep it simple: always clear on fresh page load to avoid stale UI.
    request.session.pop('study_plan_markdown', None)

    return render(request, 'tutor/study_plan.html', {
        'learner': learner,
        'stats': stats,
        'plan_markdown': "",
    })


@csrf_exempt
@require_POST
def api_study_plan_generate(request):
    """
    Generate (or return cached) study plan markdown for today.

    This endpoint exists to keep the study-plan page fast: the initial page
    load returns immediately, then the frontend calls here to generate the plan.
    """
    learner = _get_or_create_learner(request)
    # Load or compute stats
    stats = request.session.get('study_plan_stats')
    if not stats:
        stats = _compute_study_stats(learner)
        request.session['study_plan_stats'] = stats

    # Return cached plan if present
    cached = request.session.get('study_plan_markdown')
    if cached:
        return JsonResponse({'ok': True, 'plan_markdown': cached, 'cached': True})

    llm = LLMClient()
    plan_markdown = generate_study_plan(stats, llm, cognitive_snapshot=None)
    request.session['study_plan_markdown'] = plan_markdown
    return JsonResponse({'ok': True, 'plan_markdown': plan_markdown, 'cached': False})


@csrf_exempt
@require_POST
def api_dashboard_feedback(request):
    """
    Collect free-form dashboard feedback so the AI tutor can adapt.

    For now we append it to a simple JSONL log file under the project root.
    This keeps the schema flexible and avoids touching the DB schema during prototyping.
    """
    learner = _get_or_create_learner(request)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'ok': False, 'error': 'Invalid JSON'}, status=400)

    message = (data.get('message') or '').strip()
    if not message:
        return JsonResponse({'ok': False, 'error': 'Empty message'}, status=400)

    payload = {
        'learner_id': learner.learner_id,
        'display_name': learner.display_name,
        'message': message,
        'source': 'dashboard_manual_feedback',
        'timestamp': timezone.now().isoformat(),
    }

    try:
        log_path = Path(settings.BASE_DIR) / 'dashboard_feedback.jsonl'
        with log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')
    except Exception:
        # Swallow logging errors to avoid breaking UX
        return JsonResponse({'ok': False, 'error': 'Failed to persist feedback'}, status=500)

    return JsonResponse({'ok': True})


def writing(request):
    """ADHD 写作模块：写作任务生成 + 草稿反馈。"""
    learner = _get_or_create_learner(request)
    return render(request, 'tutor/writing.html', {
        'learner': learner,
    })


def speaking(request):
    """ADHD 口语模块：训练包 + 陪练反馈。"""
    learner = _get_or_create_learner(request)
    latest = SpeakingPractice.objects.filter(learner=learner).first()
    return render(request, 'tutor/speaking.html', {
        'learner': learner,
        'practice': latest,
    })


def listening(request):
    """ADHD / 焦虑友好的听力策略页面。"""
    learner = _get_or_create_learner(request)
    return render(request, 'tutor/listening.html', {
        'learner': learner,
    })


def agent_workflow(request):
    """Standalone agent workflow visualization (no auth required)."""
    path = Path(settings.BASE_DIR) / 'tutor' / 'static' / 'tutor' / 'agent-workflow.html'
    with open(path, 'r', encoding='utf-8') as f:
        return HttpResponse(f.read(), content_type='text/html; charset=utf-8')


# ══════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@csrf_exempt
@require_POST
def api_onboard(request):
    data = json.loads(request.body)
    learner_id = f"learner_{int(time.time())}"
    request.session['learner_id'] = learner_id
    request.session['onboarded']  = True

    learner = LearnerProfile.objects.create(
        learner_id=learner_id,
        display_name=data.get('name', 'Learner'),
        ld_confirmed=data.get('confirmed', []),
        ld_suspected=data.get('suspected', []),
        ld_severity=data.get('severity', {}),
        attention_min=float(data.get('attention_min', 15)),
        frustration_thresh=float(data.get('frustration_thresh', 0.55)),
    )

    try:
        orch = _get_orchestrator(learner_id)
        if orch:
            orch.set_learner_ld_profile(
                confirmed=learner.ld_confirmed,
                suspected=learner.ld_suspected,
                severity=learner.ld_severity,
            )
    except Exception:
        pass

    return JsonResponse({'ok': True, 'learner_id': learner_id})


@csrf_exempt
@require_POST
def api_chat(request):
    data       = json.loads(request.body)
    user_input = data.get('message', '').strip()
    paragraph_context = data.get('context', '').strip()
    sentence_list = data.get('sentence_list') or []
    channel = (data.get('channel', '') or '').strip().lower()
    if not user_input:
        return JsonResponse({'error': 'Empty message'}, status=400)

    learner = _get_or_create_learner(request)
    session = ChatSession.objects.filter(learner=learner, ended_at__isnull=True).last()
    if not session:
        session = ChatSession.objects.create(learner=learner)

    ChatMessage.objects.create(session=session, role='user', content=user_input)
    session.total_turns += 1
    session.save()

    try:
        # AI Hub 特殊能力：当用户明确问“第一句”，自动触发句子精讲 agent，并让前端聚焦该句
        if channel == "hub" and paragraph_context:
            # 简单意图识别：覆盖中英常见问法
            wants_first_sentence = any(
                k in user_input.lower()
                for k in ["第一句", "首句", "first sentence", "sentence 1", "第一句话"]
            )
            if wants_first_sentence:
                # 抽取文章第一句（英文优先，按 . ! ? 分句）
                first_sentence = ""
                if isinstance(sentence_list, list) and sentence_list:
                    # 前端已做纯文本切句，优先使用（更稳定）
                    first_sentence = str(sentence_list[0]).strip()
                if not first_sentence:
                    clean = re.sub(r"\s+", " ", paragraph_context.strip())
                    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
                    first_sentence = parts[0] if parts else clean[:240]

                ld_profile = {
                    'confirmed': learner.ld_confirmed,
                    'suspected': learner.ld_suspected,
                }
                analysis = reading_agent_explain_sentence(
                    sentence=first_sentence,
                    passage_context=paragraph_context,
                    ld_profile=ld_profile,
                )

                # 记录到会话消息（统一走既有 ChatMessage 结构）
                cs = {'working_memory_load': 0.25, 'motivation_level': 0.65, 'affect_valence': 0.1, 'cognitive_fatigue': 0.1}
                ChatMessage.objects.create(
                    session=session,
                    role='assistant',
                    content=analysis,
                    active_agent="sentence_explainer",
                    wm_load=cs.get('working_memory_load'),
                    motivation=cs.get('motivation_level'),
                    affect=cs.get('affect_valence'),
                    fatigue=cs.get('cognitive_fatigue'),
                    risk_score=0.0,
                    risk_level='low',
                )

                return JsonResponse({
                    'response': analysis,
                    'active_agent': 'sentence_explainer',
                    'cognitive_state': cs,
                    'intervention_flags': {},
                    'trajectory_flags': {},
                    'risk': 0.0,
                    'risk_level': 'low',
                    'escalation': 'OK',
                    # 给前端一个明确的 UI 指令：聚焦第 0 句并展示解析
                    'ui_action': {
                        'type': 'focus_sentence',
                        'sentence_index': 0,
                        'sentence_text': first_sentence,
                        'analysis': analysis,
                    },
                })

        orch = _get_orchestrator(learner.learner_id)
        # Optionally enrich RAG memory with the current reading paragraph/passage.
        if orch and paragraph_context:
            try:
                orch.add_paragraph_to_memory(
                    paragraph_context,
                    meta={"source": "reading_ai_hub"},
                )
            except Exception:
                # If dynamic memory write fails, fall back silently.
                pass

        result = orch.handle(user_input, hub_mode=(channel == "hub")) if orch else _mock_response(user_input)
    except Exception as e:
        result = _mock_response(user_input)
        result['response'] = f"[Error: {e}]"

    cs = result.get('cognitive_state', {})
    ChatMessage.objects.create(
        session=session,
        role='assistant',
        content=result['response'],
        active_agent=result.get('active_agent', ''),
        wm_load=cs.get('working_memory_load'),
        motivation=cs.get('motivation_level'),
        affect=cs.get('affect_valence'),
        fatigue=cs.get('cognitive_fatigue'),
        risk_score=result.get('risk'),
        risk_level=result.get('risk_level', ''),
    )

    return JsonResponse({
        'response':              result['response'],
        'active_agent':          result.get('active_agent', 'tutor'),
        'cognitive_state':       cs,
        'intervention_flags':    result.get('intervention_flags', {}),
        'trajectory_flags':      result.get('trajectory_flags', {}),
        'risk':                  result.get('risk', 0.0),
        'risk_level':            result.get('risk_level', 'low'),
        'escalation':            result.get('escalation', 'OK'),

        # Risk assessment object for frontend risk indicator
        'risk_assessment': {
            'risk_level':  result.get('risk_level', 'low'),
            'risk_score':  result.get('risk', 0.0),
            'confidence':  result.get('risk_confidence', 0.85),
        },

        # Intervention recommendations
        'interventions':          result.get('interventions', []),
        'intervention_priority':  result.get('intervention_priority', 'routine'),

        # Key affective indicators
        'key_indicators': {
            'anxiety':         result.get('anxiety_index', 0.0),
            'depression':      result.get('depression_index', 0.0),
            'positive_affect': result.get('positive_affect', 0.0),
        },
    })


def api_chat_stream(request):
    """Streaming chat endpoint using SSE (Server-Sent Events) - TRUE streaming."""
    import re
    from core.llm_client import LLMClient

    data = json.loads(request.body)
    user_input = data.get('message', '').strip()
    paragraph_context = data.get('context', '').strip()
    sentence_list = data.get('sentence_list') or []
    channel = (data.get('channel', '') or '').strip().lower()

    if not user_input:
        return JsonResponse({'error': 'Empty message'}, status=400)

    learner = _get_or_create_learner(request)
    session = ChatSession.objects.filter(learner=learner, ended_at__isnull=True).last()
    if not session:
        session = ChatSession.objects.create(learner=learner)

    ChatMessage.objects.create(session=session, role='user', content=user_input)
    session.total_turns += 1
    session.save()

    def generate():
        try:
            # AI Hub 特殊能力：当用户明确问"第一句"
            if channel == "hub" and paragraph_context:
                wants_first_sentence = any(
                    k in user_input.lower()
                    for k in ["第一句", "首句", "first sentence", "sentence 1", "第一句话"]
                )
                if wants_first_sentence:
                    first_sentence = ""
                    if isinstance(sentence_list, list) and sentence_list:
                        first_sentence = str(sentence_list[0]).strip()
                    if not first_sentence:
                        clean = re.sub(r"\s+", " ", paragraph_context.strip())
                        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
                        first_sentence = parts[0] if parts else clean[:240]

                    ld_profile = {
                        'confirmed': list(learner.ld_confirmed) if learner.ld_confirmed else [],
                        'suspected': list(learner.ld_suspected) if learner.ld_suspected else [],
                    }
                    analysis = reading_agent_explain_sentence(
                        sentence=first_sentence,
                        passage_context=paragraph_context,
                        ld_profile=ld_profile,
                    )

                    cs = {'working_memory_load': 0.25, 'motivation_level': 0.65, 'affect_valence': 0.1, 'cognitive_fatigue': 0.1}
                    risk = 0.0
                    risk_level = 'low'

                    metadata = {
                        'active_agent': 'sentence_explainer',
                        'cognitive_state': cs,
                        'intervention_flags': {},
                        'trajectory_flags': {},
                        'risk': risk,
                        'risk_level': risk_level,
                        'escalation': 'OK',
                        'risk_assessment': {'risk_level': risk_level, 'risk_score': risk, 'confidence': 0.85},
                        'interventions': [],
                        'intervention_priority': 'routine',
                        'key_indicators': {'anxiety': 0.0, 'depression': 0.0, 'positive_affect': 0.0},
                        'ui_action': {
                            'type': 'focus_sentence',
                            'sentence_index': 0,
                            'sentence_text': first_sentence,
                            'analysis': analysis,
                        },
                    }
                    yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"

                    # Stream each character for real-time effect
                    for char in analysis:
                        escaped = char.replace('\n', '\\n').replace('\r', '')
                        yield f"data: {json.dumps({'type': 'chunk', 'data': escaped})}\n\n"
                        # Small delay for visual effect
                        import time
                        time.sleep(0.01)

                    yield f"data: {json.dumps({'type': 'done'})}\n\n"

                    ChatMessage.objects.create(
                        session=session,
                        role='assistant',
                        content=analysis,
                        active_agent="sentence_explainer",
                        wm_load=cs.get('working_memory_load'),
                        motivation=cs.get('motivation_level'),
                        affect=cs.get('affect_valence'),
                        fatigue=cs.get('cognitive_fatigue'),
                        risk_score=risk,
                        risk_level=risk_level,
                    )
                    return

            # For normal chat: get cognitive state first, then stream LLM response
            orch = _get_orchestrator(learner.learner_id)
            if orch and paragraph_context:
                try:
                    orch.add_paragraph_to_memory(paragraph_context, meta={"source": "reading_ai_hub"})
                except Exception:
                    pass

            # Get initial state from orchestrator (for cognitive state)
            if orch:
                result = orch.handle(user_input, hub_mode=(channel == "hub")) if orch else _mock_response(user_input)
            else:
                result = _mock_response(user_input)

            cs = result.get('cognitive_state', {})
            risk = result.get('risk', 0.0)
            risk_level = result.get('risk_level', 'low')
            escalation = "OK" if risk < 0.8 else "ESCALATE"

            # Build system prompt based on cognitive state
            all_ld = set(list(learner.ld_confirmed or []) + list(learner.ld_suspected or []))
            int_flags = result.get('intervention_flags', {})

            adaptive_instructions = []
            if channel == "hub":
                adaptive_instructions.append("AI HUB MODE: Answer the user's question using the retrieved context. Be direct and concise (max ~70 words). Prefer 2–4 short bullets. No long preambles.")
            if int_flags.get("wm_overload"):
                adaptive_instructions.append("IMPORTANT: Working memory is overloaded. Give MAXIMUM 3 bullet points. No nested lists. Bold the single most important term. One concept only.")
            if int_flags.get("fatigue_high"):
                adaptive_instructions.append("IMPORTANT: The student is fatigued. Be extremely brief (under 80 words). End with: 'Want to take a 5-minute break before continuing?'")
            if "adhd" in all_ld:
                adaptive_instructions.append("Use ⚡ before any key point so the student knows where to focus. Use short sentences. Add a transition word before every new idea (First / Then / Finally / Most importantly).")
            if int_flags.get("affect_negative"):
                adaptive_instructions.append("The student is upset. Acknowledge this with one short empathetic sentence BEFORE giving any content.")

            adaptive_block = "\n".join(adaptive_instructions)
            if adaptive_block:
                adaptive_block = "\n[ADAPTIVE INSTRUCTIONS]\n" + adaptive_block + "\n"

            # Get RAG context
            rag_context = ""
            if orch:
                try:
                    from agents.rag_node import RagNode
                    rag = RagNode(orch.learner_store, orch.vector_store)
                    rag_context = rag.retrieve(user_input, top_k=3)
                except Exception:
                    pass

            system_prompt = f"You are an academic tutor. Be precise, structured, and grounded in retrieved context.{adaptive_block}"

            user_prompt = f"""
You MUST use the following retrieved knowledge as your primary grounding.
If the knowledge is insufficient, say what is missing and ask one clarifying question.
{rag_context}

Student question:
{user_input}
""".strip()

            # Send metadata first
            metadata = {
                'active_agent': result.get('active_agent', 'tutor'),
                'cognitive_state': cs,
                'intervention_flags': result.get('intervention_flags', {}),
                'trajectory_flags': result.get('trajectory_flags', {}),
                'risk': risk,
                'risk_level': risk_level,
                'escalation': escalation,
                'risk_assessment': {'risk_level': risk_level, 'risk_score': risk, 'confidence': 0.85},
                'interventions': result.get('interventions', []),
                'intervention_priority': result.get('intervention_priority', 'routine'),
                'key_indicators': {
                    'anxiety': result.get('anxiety_index', 0.0),
                    'depression': result.get('depression_index', 0.0),
                    'positive_affect': result.get('positive_affect', 0.0),
                },
            }
            yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"

            # Stream LLM response in REAL-TIME - character by character
            llm = LLMClient()
            accumulated = ""

            for chunk in llm.stream_chat(system_prompt, user_prompt, temperature=0.4):
                accumulated += chunk
                # Escape for JSON
                escaped = accumulated.replace('\n', '\\n').replace('\r', '')
                yield f"data: {json.dumps({'type': 'chunk', 'data': escaped})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            # Format and save the final response
            if orch:
                formatted_response = orch._format_for_adhd_chat(accumulated)
            else:
                formatted_response = accumulated

            ChatMessage.objects.create(
                session=session,
                role='assistant',
                content=formatted_response,
                active_agent=result.get('active_agent', ''),
                wm_load=cs.get('working_memory_load'),
                motivation=cs.get('motivation_level'),
                affect=cs.get('affect_valence'),
                fatigue=cs.get('cognitive_fatigue'),
                risk_score=risk,
                risk_level=risk_level,
            )

        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'data': error_msg})}\n\n"

    return StreamingHttpResponse(generate(), content_type='text/event-stream')


@csrf_exempt
@require_POST
def api_profile_save(request):
    data    = json.loads(request.body)
    learner = _get_or_create_learner(request)
    learner.display_name       = data.get('display_name', learner.display_name)
    learner.ld_confirmed       = data.get('ld_confirmed', learner.ld_confirmed)
    learner.ld_suspected       = data.get('ld_suspected', learner.ld_suspected)
    learner.ld_severity        = data.get('ld_severity', learner.ld_severity)
    learner.attention_min      = float(data.get('attention_min', learner.attention_min))
    learner.frustration_thresh = float(data.get('frustration_thresh', learner.frustration_thresh))
    learner.save()

    try:
        orch = _get_orchestrator(learner.learner_id)
        if orch:
            orch.set_learner_ld_profile(
                confirmed=learner.ld_confirmed,
                suspected=learner.ld_suspected,
                severity=learner.ld_severity,
            )
    except Exception:
        pass

    return JsonResponse({'ok': True})


@require_GET
def api_session_history(request, session_id):
    learner  = _get_or_create_learner(request)
    session  = get_object_or_404(ChatSession, id=session_id, learner=learner)
    messages = list(session.messages.values(
        'role', 'content', 'active_agent', 'timestamp',
        'wm_load', 'motivation', 'affect', 'fatigue', 'risk_score', 'risk_level'
    ))
    for m in messages:
        if m['timestamp']:
            m['timestamp'] = m['timestamp'].strftime('%H:%M')
    return JsonResponse({'messages': messages})


@csrf_exempt
@require_POST
def api_session_end(request):
    learner = _get_or_create_learner(request)
    session = ChatSession.objects.filter(learner=learner, ended_at__isnull=True).last()
    if session:
        session.ended_at = timezone.now()
        session.save()
    orch = _get_orchestrator(learner.learner_id)
    if orch:
        try:
            orch.end_session()
        except Exception:
            pass
    return JsonResponse({'ok': True})


@csrf_exempt
@require_POST
def api_study_plan_chat(request):
    """
    Chat-style endpoint to ask follow-up questions about today's plan.

    Body: { "message": str }
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    user_msg = (data.get('message') or '').strip()
    if not user_msg:
        return JsonResponse({'error': 'Empty message'}, status=400)

    learner = _get_or_create_learner(request)

    # Load or (re)compute stats + base plan if missing, so this endpoint is robust
    stats = request.session.get('study_plan_stats')
    plan_markdown = request.session.get('study_plan_markdown')
    llm = LLMClient()

    if not stats or not plan_markdown:
        stats = _compute_study_stats(learner)
        plan_markdown = generate_study_plan(stats, llm, cognitive_snapshot=None)
        request.session['study_plan_stats'] = stats
        request.session['study_plan_markdown'] = plan_markdown

    # Use the same LLM backend to extract a lightweight cognitive snapshot
    # from the learner's free-text comment. This gives the planner real-time
    # signals such as motivation, fatigue, and WM load.
    fx = FeatureExtractorLLM(llm_client=llm, max_retries=1)
    feats = fx.extract({"user_input": user_msg, "rag_context": ""})
    cognitive_snapshot = {
        "wm_load_estimate": feats.wm_load_estimate,
        "motivation_estimate": feats.motivation_estimate,
        "affect_estimate": feats.affect_estimate,
        "fatigue_estimate": feats.fatigue_estimate,
        "negative_attribution": feats.negative_attribution,
        "topic_shift": feats.topic_shift,
        "task_avoidance": feats.task_avoidance,
    }

    reply = study_plan_chat_reply(
        stats=stats,
        current_plan_markdown=plan_markdown,
        user_question=user_msg,
        cognitive_snapshot=cognitive_snapshot,
        llm=llm,
    )

    # Optionally refresh the stored plan if the assistant proposes edits.
    # For now we just keep the original plan and treat replies as commentary.

    return JsonResponse({
        'reply': reply,
        'stats': stats,
        'cognitive': cognitive_snapshot,
    })


@csrf_exempt
@require_POST
def api_speaking_generate(request):
    """
    生成一份 ADHD-friendly 口语训练包，并保存到数据库。

    Body:
      {
        "topic": str,
        "scenario": str,
        "minutes": int,
        "english_level": str
      }
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    learner = _get_or_create_learner(request)
    topic = (data.get('topic') or '').strip()
    scenario = (data.get('scenario') or '').strip()
    english_level = (data.get('english_level') or '').strip() or 'A2-B1'
    minutes = data.get('minutes', 8)

    llm = LLMClient()
    ld_profile = {'confirmed': learner.ld_confirmed, 'suspected': learner.ld_suspected}

    try:
        pack = generate_adhd_speaking_pack(
            llm=llm,
            learner_name=learner.display_name,
            english_level=english_level,
            topic=topic,
            scenario=scenario,
            minutes=int(minutes) if str(minutes).isdigit() else 8,
            ld_profile=ld_profile,
        )
    except Exception as e:
        return JsonResponse({'error': f'LLM error: {e}'}, status=400)

    practice = SpeakingPractice.objects.create(
        learner=learner,
        topic=topic,
        scenario=scenario,
        english_level=english_level,
        minutes_budget=max(3, min(30, int(minutes) if str(minutes).isdigit() else 8)),
        pack_markdown=pack,
        history=[],
    )

    return JsonResponse({
        'ok': True,
        'practice_id': practice.id,
        'pack_markdown': practice.pack_markdown,
    })


@csrf_exempt
@require_POST
def api_speaking_chat(request):
    """
    口语陪练聊天：根据训练包给短反馈 + 下一步提示，并写入 history。

    Body:
      { "practice_id": int, "message": str }
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    learner = _get_or_create_learner(request)
    practice_id = data.get('practice_id')
    msg = (data.get('message') or '').strip()
    if not practice_id:
        return JsonResponse({'error': 'practice_id required'}, status=400)
    if not msg:
        return JsonResponse({'error': 'Empty message'}, status=400)

    practice = get_object_or_404(SpeakingPractice, id=practice_id, learner=learner)

    llm = LLMClient()
    history = list(practice.history or [])
    history.append({'role': 'user', 'content': msg})

    try:
        reply = speaking_coach_reply(
            llm=llm,
            practice_pack_markdown=practice.pack_markdown,
            learner_msg=msg,
            history=history,
        )
    except Exception as e:
        return JsonResponse({'error': f'LLM error: {e}'}, status=400)

    history.append({'role': 'assistant', 'content': reply})
    practice.history = history[-50:]  # 防止无限增长
    practice.save(update_fields=['history', 'updated_at'])

    return JsonResponse({'ok': True, 'reply': reply, 'history': practice.history})


@csrf_exempt
@require_POST
def api_listening_strategy(request):
    """
    根据当前听力/听讲场景，生成 ADHD / 焦虑友好的听力策略卡片（Markdown）。

    Body:
      {
        "scenario": str,    # 例如：大课听讲 / 线上课 / 英语听力真题
        "environment": str, # 例如：有背景噪音 / 家里比较安静
        "goal": str         # 例如：抓住关键信息 / 训练听力理解
      }
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    scenario = (data.get('scenario') or '').strip()
    environment = (data.get('environment') or '').strip()
    goal = (data.get('goal') or '').strip()

    learner = _get_or_create_learner(request)
    ld_profile = {'confirmed': learner.ld_confirmed, 'suspected': learner.ld_suspected}

    llm = LLMClient()
    try:
        markdown = generate_adhd_listening_strategy(
            scenario=scenario,
            environment=environment,
            goal=goal,
            ld_profile=ld_profile,
            llm=llm,
        )
    except Exception:
        markdown = generate_adhd_listening_strategy(
            scenario=scenario,
            environment=environment,
            goal=goal,
            ld_profile=ld_profile,
            llm=None,
        )

    try:
        sample_passage = generate_sample_listening_passage(scenario=scenario, llm=llm)
    except Exception:
        sample_passage = generate_sample_listening_passage(scenario=scenario, llm=None)

    return JsonResponse({
        'ok': True,
        'strategy_markdown': markdown,
        'sample_passage': sample_passage,
    })


@csrf_exempt
@require_POST
def api_listening_logic_chain(request):
    """
    根据案例音频文本提取逻辑链，简要展示其逻辑结构。
    Body: { "passage": str }
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'ok': False, 'error': 'Invalid JSON'}, status=400)

    passage = (data.get('passage') or '').strip()
    if not passage:
        return JsonResponse({'ok': True, 'logic_chain': '暂无示例音频，请先生成听力策略卡片。'})

    try:
        llm = LLMClient()
        try:
            logic_chain = extract_logic_chain(passage=passage, llm=llm)
        except Exception:
            logic_chain = extract_logic_chain(passage=passage, llm=None)
    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)

    return JsonResponse({'ok': True, 'logic_chain': logic_chain})


# ── ADHD / Dysgraphia 写作模块 ───────────────────────────────────

# 轻量雅思写作题库（示例：可后续扩展为数据库或独立 JSON）
IELTS_WRITING_BANK = [
    {
        "id": "c15_t1_task2_city_traffic",
        "task_type": "2",
        "topics": ["traffic", "city", "transport"],
        "genre": "议论文",
        "prompt": (
            "In many cities, traffic congestion is becoming a severe problem. "
            "Some people think governments should increase the cost of fuel, "
            "while others believe other measures would be more effective.\n\n"
            "Discuss both views and give your own opinion."
        ),
        "source": "Cambridge IELTS 15 · Test 1 · Task 2",
    },
    {
        "id": "c14_t3_task2_health_food",
        "task_type": "2",
        "topics": ["health", "food", "diet"],
        "genre": "议论文",
        "prompt": (
            "Some people think that governments should make laws about "
            "nutrition and healthy food choices, while others believe it is "
            "a matter of personal responsibility.\n\n"
            "Discuss both views and give your own opinion."
        ),
        "source": "Cambridge IELTS 14 · Test 3 · Task 2",
    },
    {
        "id": "c13_t2_task2_environment_plastic",
        "task_type": "2",
        "topics": ["environment", "plastic", "pollution"],
        "genre": "议论文",
        "prompt": (
            "Plastic containers have become more common than ever and are "
            "used by many food and drink companies.\n\n"
            "Do the advantages of this trend outweigh the disadvantages?"
        ),
        "source": "Cambridge IELTS 13 · Test 2 · Task 2",
    },
]


def _match_ielts_writing_question(topic: str, genre: str) -> dict | None:
    """Best-effort keyword match between user topic and local IELTS bank."""
    if not IELTS_WRITING_BANK:
        return None
    topic_l = (topic or "").lower()
    genre_l = (genre or "").lower()
    best = None
    best_score = -1
    for q in IELTS_WRITING_BANK:
        score = 0
        # 题型/文体匹配加权
        if q.get("genre") and q["genre"].lower() in genre_l:
            score += 2
        # 主题关键词简单匹配
        for kw in q.get("topics", []):
            if kw.lower() in topic_l:
                score += 3
        if score > best_score:
            best_score = score
            best = q
    # 如果完全没有匹配到关键词，则返回 None，走“改写生成”流程
    return best if best_score > 0 else None


@csrf_exempt
@require_POST
def api_writing_generate(request):
    """生成 ADHD / Dysgraphia 友好的写作任务（基于雅思题目生成分步指导）。"""
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    topic = (data.get('topic') or '').strip()
    genre = (data.get('genre') or '').strip()
    audience = (data.get('audience') or '').strip()
    target_words = data.get('target_words') or 600
    difficulty = (data.get('difficulty') or 'normal').strip().lower()
    task_size = (data.get('task_size') or 'big').strip().lower()

    learner = _get_or_create_learner(request)
    ld_profile = {'confirmed': learner.ld_confirmed, 'suspected': learner.ld_suspected}

    # 1. 根据用户主题直接生成雅思风格题目
    ielts_question = ""
    ielts_source = "AI Generated IELTS-style Task"
    
    llm_temp = LLMClient()
    try:
        # 构建生成雅思题目的 prompt
        ielts_prompt = f"""Generate an IELTS Task 2 writing question based on the following requirements:

Topic: {topic or 'a general social issue'}
Genre: {genre or 'argumentative essay'}
Target audience: {audience or 'general public'}
Word count: approximately {target_words} words

Requirements:
1. The question should follow standard IELTS Task 2 format
2. It should be clear, specific, and relevant to the topic
3. For argumentative essays, use phrases like "Discuss both views and give your own opinion" or "To what extent do you agree or disagree?"
4. For other genres, adjust the question format accordingly
5. The question should be challenging but achievable
6. Output ONLY the question in English, no additional explanation

Example format:
"In many countries, [topic context]. Some people believe [view A], while others think [view B]. Discuss both views and give your own opinion."
"""
        
        ielts_question = llm_temp.chat(
            system="You are an expert IELTS examiner who creates high-quality Task 2 writing questions.",
            user=ielts_prompt,
            temperature=0.7
        ).strip()
        
        # 如果生成失败或太短，使用备用方案
        if not ielts_question or len(ielts_question) < 50:
            ielts_question = f"Write an essay discussing the topic of {topic or 'modern society'}. Consider different perspectives and provide your own opinion with relevant examples. Your essay should be approximately {target_words} words."
            
    except Exception as e:
        # 降级：使用简单模板
        ielts_question = f"Write an essay about {topic or 'a current social issue'}. Discuss different viewpoints and give your own opinion with supporting examples. Aim for approximately {target_words} words."

    # 2. 基于雅思题目生成分步指导
    llm = LLMClient()
    try:
        result = generate_step_by_step_guide(
            ielts_question=ielts_question,
            genre=genre,
            audience=audience,
            target_words=int(target_words),
            difficulty=difficulty,
            task_size=task_size,
            ld_profile=ld_profile,
            llm=llm,
        )
    except Exception as e:
        # 降级：返回可用结构
        result = {
            'prompt': f"请完成以下雅思写作题目：\n\n{ielts_question}\n\n建议分步完成：\n1. 头脑风暴（5分钟）\n2. 确定论点（10分钟）\n3. 搭建大纲（10分钟）\n4. 逐段写作（20-30分钟）",
            'outline': '',
            'checklist': ''
        }

    # 添加雅思题目信息到返回结果
    result['ielts_question'] = ielts_question
    result['ielts_source'] = ielts_source

    return JsonResponse({'ok': True, **result})


@csrf_exempt
@require_POST
def api_writing_ielts_topic(request):
    """
    根据用户配置自动匹配雅思写作题目：
      1. 优先返回本地题库中的真题；
      2. 若无明显匹配，则基于真题改写为当前主题。

    Body:
      { "topic": str, "genre": str, "audience": str,
        "target_words": int, "difficulty": str, "task_size": "big"|"small" }
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    topic = (data.get('topic') or '').strip()
    genre = (data.get('genre') or '').strip()

    # IMPORTANT: 用户要求“直接让 LLM 生成符合雅思大作文类型的题目”，不做题库匹配、不做真题改写。
    llm = LLMClient()
    try:
        system_prompt = (
            "You are an expert IELTS examiner. You write high-quality IELTS Writing Task 2 prompts.\n"
            "Output ONLY the final Task 2 question in English (no title, no bullets, no explanations)."
        )
        # Keep it stable by forcing a small set of classic Task 2 prompt styles.
        user_prompt = f"""
Create ONE IELTS Writing Task 2 question.

Constraints:
- Must be a standard IELTS Task 2 question (single prompt).
- Must be clearly related to the learner's theme.
- Use one of these styles ONLY:
  A) Discuss both views and give your own opinion.
  B) To what extent do you agree or disagree?
  C) What are the causes of this problem and what measures could be taken to solve it?
  D) Do the advantages outweigh the disadvantages?
- Avoid niche local references; keep it globally accessible.
- Keep it 1 short paragraph + the instruction sentence (typical IELTS format).

Learner theme (Chinese or English): {topic or 'a general social issue'}
Requested genre (may be Chinese): {genre or 'argumentative essay'}
""".strip()
        question = llm.chat(system=system_prompt, user=user_prompt, temperature=0.7).strip()
        if not question or len(question) < 60:
            raise ValueError("Generated question too short")
    except Exception:
        # Robust fallback template (still Task 2 style)
        theme = topic or "modern society"
        question = (
            f"In many countries, {theme} is becoming an increasingly important issue. "
            f"Some people believe that individuals should take responsibility for addressing this, "
            f"while others think governments should play the main role.\n\n"
            f"Discuss both views and give your own opinion."
        )

    return JsonResponse({
        'ok': True,
        'is_real_past_paper': False,
        'source': "LLM Generated (Task 2)",
        'question': question,
    })


@csrf_exempt
@require_POST
def api_writing_stt(request):
    """
    简单 STT 后端：接受音频文件并尝试转写为文本。

    为了避免强绑定具体厂商，这里假定在 settings 或环境变量中配置了
    OPENAI_API_KEY，并使用 Whisper 模型进行语音转文字。
    """
    audio = request.FILES.get('audio')
    if not audio:
        return JsonResponse({'ok': False, 'error': 'audio file (field name "audio") is required'}, status=400)

    api_key = getattr(settings, "OPENAI_API_KEY", None)
    if not api_key:
        return JsonResponse({'ok': False, 'error': 'STT backend not configured (missing OPENAI_API_KEY)'}, status=500)

    try:
        import openai
    except ImportError:
        return JsonResponse({'ok': False, 'error': 'openai package not installed on server'}, status=500)

    openai.api_key = api_key

    try:
        # Whisper STT 调用：将上传的音频转成英文文本（也支持中英混合）
        resp = openai.Audio.transcribe(
            model="whisper-1",
            file=audio,
        )
        text = resp.get("text", "").strip()
        return JsonResponse({'ok': True, 'text': text})
    except Exception as e:
        return JsonResponse({'ok': False, 'error': f'STT provider error: {e}'}, status=500)


@csrf_exempt
@require_POST
def api_writing_feedback(request):
    """对草稿给 ADHD 友好的可执行反馈。"""
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    prompt = (data.get('prompt') or '').strip()
    draft = (data.get('draft') or '').strip()
    genre = (data.get('genre') or '').strip()

    learner = _get_or_create_learner(request)
    ld_profile = {'confirmed': learner.ld_confirmed, 'suspected': learner.ld_suspected}

    llm = LLMClient()
    try:
        feedback = generate_adhd_writing_feedback(
            prompt=prompt,
            draft=draft,
            genre=genre,
            ld_profile=ld_profile,
            llm=llm,
        )
    except Exception as e:
        feedback = generate_adhd_writing_feedback(
            prompt=prompt,
            draft=draft,
            genre=genre,
            ld_profile=ld_profile,
            llm=None,
        )
        feedback = f"{feedback}\n\n（提示：当前反馈失败：{e}）"

    return JsonResponse({'ok': True, 'feedback': feedback})


@csrf_exempt
@require_POST
def api_interventions_apply(request):
    """Record the learner's choice of an intervention strategy."""
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    learner = _get_or_create_learner(request)
    intervention_id = data.get('intervention_id')
    intervention_type = data.get('intervention_type', '')

    # Attempt to forward the choice to the orchestrator if it supports it
    orch = _get_orchestrator(learner.learner_id)
    if orch:
        try:
            if hasattr(orch, 'record_intervention_applied'):
                orch.record_intervention_applied(
                    intervention_id=intervention_id,
                    intervention_type=intervention_type,
                )
        except Exception:
            pass

    return JsonResponse({'ok': True, 'intervention_id': intervention_id})


# ══════════════════════════════════════════════════════════════════
#  IELTS READING MODULE
# ══════════════════════════════════════════════════════════════════

def reading(request):
    """IELTS reading page — SPARK ADHD design."""
    learner = _get_or_create_learner(request)
    return render(request, 'tutor/reading.html', {
        'learner': learner,
    })


@csrf_exempt
@require_POST
def api_reading_paragraph(request):
    """Returns AI-generated guidance for a single paragraph."""
    try:
        data = json.loads(request.body)
        paragraph_text = data.get('text', '')
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    if not paragraph_text:
        return JsonResponse({'error': 'Paragraph text is required'}, status=400)

    learner = _get_or_create_learner(request)
    ld_profile = {
        'confirmed': learner.ld_confirmed,
        'suspected': learner.ld_suspected,
    }

    guidance = generate_paragraph_guidance(paragraph_text, ld_profile)
    
    return JsonResponse({'guidance': guidance})


@csrf_exempt
@require_POST
def api_reading_upload(request):
    """Process an uploaded IELTS PDF: convert to paragraph images, extract questions.

    Request: multipart/form-data
        pdf   (file, required) — PDF file containing the IELTS passage + questions
        title (str, optional)  — passage title; defaults to the filename stem

    The passage body is represented as PNG images (one per labelled paragraph A–I).
    Questions are extracted from the PDF's native text structure (no OCR).
    """
    pdf_file = request.FILES.get('pdf')
    if not pdf_file:
        return JsonResponse({'error': 'No PDF file uploaded. Please attach a file with field name "pdf".'}, status=400)

    if not pdf_file.name.lower().endswith('.pdf'):
        return JsonResponse({'error': 'Only PDF files are supported.'}, status=400)

    pdf_bytes = pdf_file.read()

    # ── Title: prefer explicit param, then filename stem ──────────
    title = request.POST.get('title', '').strip()
    if not title:
        import os
        title = os.path.splitext(pdf_file.name)[0].replace('_', ' ').replace('-', ' ').strip() or 'IELTS Reading Passage'

    learner = _get_or_create_learner(request)

    # Deactivate previous passages
    IELTSPassage.objects.filter(learner=learner, is_active=True).update(is_active=False)

    # Cancel any open attempts
    ReadingAttempt.objects.filter(learner=learner, completed=False).update(completed=True)

    # ── Create passage record first (need ID for image naming) ────
    passage = IELTSPassage.objects.create(
        learner=learner,
        title=title,
        raw_text='',  # passage text is in images; questions extracted separately
    )

    # ── Convert PDF to paragraph images ──────────────────────────
    from django.conf import settings
    from agents.reading_agent import (
        extract_paragraph_images,
        extract_question_groups_from_pdf,
        map_questions_to_paragraphs,
    )

    images_dir = str(settings.MEDIA_ROOT / 'passage_images')
    try:
        paragraphs = extract_paragraph_images(pdf_bytes, images_dir, passage.id)
    except Exception as e:
        passage.delete()
        return JsonResponse({'error': f'Could not convert PDF to images: {e}'}, status=400)

    if not paragraphs:
        passage.delete()
        return JsonResponse({'error': 'No paragraphs found in the PDF. Please upload a standard IELTS reading passage.'}, status=400)

    # ── Extract text for evaluation/hint context (not displayed to user) ──
    # We store the full passage text in raw_text so the hint/evaluation backend
    # has textual context even though the passage is displayed as images.
    try:
        import fitz as _fitz
        _doc = _fitz.open(stream=pdf_bytes, filetype="pdf")
        raw_text = '\n'.join(p.get_text() for p in _doc).strip()
        _doc.close()
    except Exception:
        raw_text = ''

    # Update passage with extracted raw text (for evaluation context)
    passage.raw_text = raw_text
    passage.save()

    # ── Extract questions with group labels from native PDF text ─────────
    try:
        question_groups = extract_question_groups_from_pdf(pdf_bytes)
    except Exception:
        question_groups = []

    # Fallback: if group extraction returned nothing, try the simpler extractor
    if not question_groups:
        logger.warning('[WARN] extract_question_groups_from_pdf returned 0 results; trying extract_questions_from_pdf fallback')
        try:
            from agents.reading_agent import extract_questions_from_pdf
            plain_questions = extract_questions_from_pdf(pdf_bytes)
            question_groups = [
                {'order': i + 1, 'text': t, 'group_label': '', 'group_instruction': ''}
                for i, t in enumerate(plain_questions)
            ]
        except Exception:
            question_groups = []

    # ── Persist paragraph sections ────────────────────────────────
    section_objs = []
    for para in paragraphs:
        s_obj = IELTSSection.objects.create(
            passage=passage,
            order=para['order'],
            heading=para['heading'],
            body=para['body'],
            image_path=para['image_path'],
        )
        section_objs.append(s_obj)

    # ── Persist questions (section assignment done after mapping) ─
    q_objs = []
    for qg in question_groups:
        q_obj = IELTSQuestion.objects.create(
            passage=passage,
            section=None,       # assigned below after mapping
            order=qg['order'],
            text=qg['text'],
            group_label=qg.get('group_label', ''),
            group_instruction=qg.get('group_instruction', ''),
        )
        q_objs.append(q_obj)

    # ── Map questions to sections and update section FK ───────────
    if q_objs and section_objs:
        sections_for_map = [
            {'id': s.id, 'order': s.order, 'heading': s.heading, 'body': s.body}
            for s in section_objs
        ]
        questions_for_map = [
            {'id': q.id, 'order': q.order, 'text': q.text, 'group_label': q.group_label}
            for q in q_objs
        ]
        q_mapping = map_questions_to_paragraphs(sections_for_map, questions_for_map)
        # q_mapping: {section_id: [question_id, …]}
        for sec_id, q_ids in q_mapping.items():
            IELTSQuestion.objects.filter(id__in=q_ids).update(section_id=sec_id)

    # ── Create attempt starting at paragraph 1 ────────────────────
    attempt = ReadingAttempt.objects.create(
        learner=learner,
        passage=passage,
        current_section_order=1,
    )

    # ── Build LD profile (used by tips and guidance) ──────────────
    ld_profile = {
        'confirmed': learner.ld_confirmed,
        'suspected': learner.ld_suspected,
    }

    # ── Generate preflight reading tips (AI pre-solves questions) ────
    try:
        from agents.reading_agent import reading_agent_preflight_tips
        sections_for_tips = [
            {'id': s.id, 'order': s.order, 'heading': s.heading, 'body': s.body}
            for s in section_objs
        ]
        questions_for_tips = [
            {'id': q.id, 'order': q.order, 'text': q.text, 'group_label': q.group_label}
            for q in q_objs
        ]
        tips_map = reading_agent_preflight_tips(sections_for_tips, questions_for_tips, ld_profile)
        # Persist tips into each section
        for sec_obj in section_objs:
            sec_tips = tips_map.get(sec_obj.id, [])
            if sec_tips:
                sec_obj.reading_tips = sec_tips
                sec_obj.save(update_fields=['reading_tips'])
    except Exception as e:
        print(f"[WARN] Preflight tips generation failed: {e}")

    # ── Build first paragraph guidance ────────────────────────────
    first_section = section_objs[0] if section_objs else None
    from agents.reading_agent import reading_agent_guide_section
    guidance = ''
    if first_section:
        guidance = reading_agent_guide_section(
            section={'heading': first_section.heading, 'body': first_section.body},
            section_num=1,
            total_sections=len(section_objs),
            ld_profile=ld_profile,
        )

    section_data = None
    if first_section:
        # Re-fetch questions from DB after mapping update to ensure section FK is committed
        first_section_qs = list(
            IELTSQuestion.objects.filter(
                passage=passage, section=first_section
            ).values('id', 'order', 'text', 'group_label', 'group_instruction').order_by('order')
        )
        section_data = {
            'id': first_section.id,
            'order': first_section.order,
            'heading': first_section.heading,
            'body': first_section.body,
            'image_url': request.build_absolute_uri(settings.MEDIA_URL + first_section.image_path) if first_section.image_path else '',
            'questions': first_section_qs,
            'tips': first_section.reading_tips or [],
        }

    all_questions = list(
        IELTSQuestion.objects.filter(passage=passage)
        .values('id', 'order', 'text', 'group_label', 'group_instruction', 'section_id')
        .order_by('order')
    )

    return JsonResponse({
        'ok': True,
        'passage_id': passage.id,
        'attempt_id': attempt.id,
        'total_sections': len(section_objs),
        'total_questions': len(q_objs),
        'current_section': section_data,
        'guidance': guidance,
        'title': title,
        'all_questions': all_questions,
    })


@csrf_exempt
@require_POST
def api_reading_next_section(request):
    """Advance to the next section (or mark as complete).

    Body: { "attempt_id": int }
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    learner = _get_or_create_learner(request)
    attempt_id = data.get('attempt_id')
    attempt = get_object_or_404(ReadingAttempt, id=attempt_id, learner=learner)

    if attempt.completed:
        return JsonResponse({'done': True, 'message': 'Passage already completed.'})

    next_order = attempt.current_section_order + 1
    total_sections = attempt.passage.sections.count()

    if next_order > total_sections:
        attempt.completed = True
        # Compute final score
        answers = attempt.answers
        if answers:
            scores = [v.get('score', 0) for v in answers.values() if isinstance(v, dict)]
            attempt.score = round(sum(scores) / len(scores), 2) if scores else None
        attempt.save()

        # Record experiment completion for A/B tracking
        if attempt.score is not None:
            try:
                from analytics.strategy_tracker import record_experiment_completion
                record_experiment_completion(attempt.pk, attempt.score)
            except Exception as exc:
                logger.exception("Failed to record experiment completion: %s", exc)

        from agents.reading_agent import reading_agent_strategy
        ld_profile = {'confirmed': learner.ld_confirmed, 'suspected': learner.ld_suspected}
        strategy = reading_agent_strategy(
            {'answers': attempt.answers, 'hints_used': attempt.hints_used},
            ld_profile,
        )
        return JsonResponse({'done': True, 'score': attempt.score, 'strategy': strategy})

    attempt.current_section_order = next_order
    attempt.save()

    section = get_object_or_404(IELTSSection, passage=attempt.passage, order=next_order)
    # Return only the questions assigned to this section
    section_questions = list(section.questions.values('id', 'order', 'text', 'group_label', 'group_instruction').order_by('order'))

    from agents.reading_agent import reading_agent_guide_section
    from django.conf import settings
    ld_profile = {'confirmed': learner.ld_confirmed, 'suspected': learner.ld_suspected}

    # Compute recent score to pass to guidance
    recent_score = None
    if attempt.answers:
        scores = [v.get('score', 0) for v in attempt.answers.values() if isinstance(v, dict)]
        if scores:
            recent_score = sum(scores) / len(scores)

    guidance = reading_agent_guide_section(
        section={'heading': section.heading, 'body': section.body},
        section_num=next_order,
        total_sections=total_sections,
        ld_profile=ld_profile,
        attempt_score=recent_score,
    )

    image_url = ''
    if section.image_path:
        image_url = request.build_absolute_uri(settings.MEDIA_URL + section.image_path)

    all_questions = list(
        IELTSQuestion.objects.filter(passage=attempt.passage)
        .values('id', 'order', 'text', 'group_label', 'group_instruction', 'section_id')
        .order_by('order')
    )

    return JsonResponse({
        'done': False,
        'current_section': {
            'id': section.id,
            'order': section.order,
            'heading': section.heading,
            'body': section.body,
            'image_url': image_url,
            'questions': section_questions,
            'tips': list(section.reading_tips or []),
        },
        'guidance': guidance,
        'all_questions': all_questions,
    })


@csrf_exempt
@require_POST
def api_reading_answer(request):
    """Submit an answer for a question.

    Body: { "attempt_id": int, "question_id": int, "answer": str }
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    learner = _get_or_create_learner(request)
    attempt = get_object_or_404(ReadingAttempt, id=data.get('attempt_id'), learner=learner)
    question = get_object_or_404(IELTSQuestion, id=data.get('question_id'), passage=attempt.passage)
    user_answer = data.get('answer', '').strip()

    # Use raw_text from passage for evaluation context (body may be empty in image mode)
    section_body = question.section.body if (question.section and question.section.body) else attempt.passage.raw_text

    from agents.reading_agent import reading_agent_evaluate
    result = reading_agent_evaluate(user_answer, question.text, section_body)

    answers = attempt.answers or {}
    answers[str(question.id)] = {
        'question': question.text,
        'answer': user_answer,
        'score': result['score'],
        'correct': result['correct'],
    }
    attempt.answers = answers
    attempt.save()

    return JsonResponse({
        'ok': True,
        'correct': result['correct'],
        'score': result['score'],
        'feedback': result['feedback'],
    })


@csrf_exempt
@require_POST
def api_reading_hint(request):
    """Get a hint for a specific question.

    Body: { "attempt_id": int, "question_id": int }
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    learner = _get_or_create_learner(request)
    attempt = get_object_or_404(ReadingAttempt, id=data.get('attempt_id'), learner=learner)
    question = get_object_or_404(IELTSQuestion, id=data.get('question_id'), passage=attempt.passage)
    # Use raw_text for hint context (body may be empty in image-only mode)
    section_body = question.section.body if (question.section and question.section.body) else attempt.passage.raw_text

    ld_profile = {'confirmed': learner.ld_confirmed, 'suspected': learner.ld_suspected}

    from agents.reading_agent import reading_agent_hint
    hint = reading_agent_hint(question.text, section_body, ld_profile, attempt.hints_used)

    attempt.hints_used += 1
    attempt.save()

    return JsonResponse({'hint': hint})


@require_GET
def api_reading_strategy(request):
    """Return the current learning strategy for the active attempt."""
    learner = _get_or_create_learner(request)
    attempt = ReadingAttempt.objects.filter(learner=learner).last()
    if not attempt:
        return JsonResponse({'strategy': 'No reading attempt found. Upload a passage to begin.'})

    ld_profile = {'confirmed': learner.ld_confirmed, 'suspected': learner.ld_suspected}
    from agents.reading_agent import reading_agent_strategy
    strategy = reading_agent_strategy(
        {'answers': attempt.answers, 'hints_used': attempt.hints_used},
        ld_profile,
    )
    return JsonResponse({'strategy': strategy})


@require_POST
def api_reading_assistant(request):
    """Return a proactive assistant tip for the current attempt state.

    Body: { "attempt_id": int }

    The assistant analyses current progress (paragraph position, answer scores,
    hint usage, LD profile) and returns adaptive guidance including keywords
    extracted from the current paragraph.  The selected assistant mode
    (stored in the session) is injected into the tip generation.
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    learner = _get_or_create_learner(request)
    attempt_id = data.get('attempt_id')
    attempt = get_object_or_404(ReadingAttempt, id=attempt_id, learner=learner)

    current_order = attempt.current_section_order
    total_sections = attempt.passage.sections.count()

    heading = ''
    section_body = ''
    try:
        current_section = IELTSSection.objects.get(
            passage=attempt.passage, order=current_order
        )
        heading = current_section.heading or ''
        # Prefer section body; fall back to raw passage text for image-only mode
        section_body = current_section.body or attempt.passage.raw_text or ''
    except IELTSSection.DoesNotExist:
        section_body = attempt.passage.raw_text or ''

    ld_profile = {'confirmed': learner.ld_confirmed, 'suspected': learner.ld_suspected}
    # Read the session-stored assistant mode (default: 'auto')
    assistant_mode = request.session.get('assistant_mode', 'auto')

    from agents.reading_agent import reading_agent_assistant_tip
    tip = reading_agent_assistant_tip(
        para_order=current_order,
        total_sections=total_sections,
        answers=attempt.answers or {},
        hints_used=attempt.hints_used,
        ld_profile=ld_profile,
        current_section_heading=heading,
        section_body=section_body,
        mode=assistant_mode,
    )

    return JsonResponse({'tip': tip})


@csrf_exempt
@require_POST
def api_reading_set_assistant_mode(request):
    """Set the assistant mode for the current learner session.

    Body: { "mode": "auto" | "focus" | "calm" | "speed" }

    The mode is persisted in the Django session so that subsequent calls
    to ``api_reading_assistant`` use the chosen mode automatically.
    Records an anonymous experiment entry for A/B tracking if an active
    reading attempt exists.
    """
    _VALID_MODES = {'auto', 'focus', 'calm', 'speed'}

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    mode = data.get('mode', 'auto')
    if mode not in _VALID_MODES:
        return JsonResponse(
            {'error': f"Invalid mode. Choose from: {', '.join(sorted(_VALID_MODES))}"},
            status=400,
        )

    request.session['assistant_mode'] = mode

    # Optionally log the mode switch against the active attempt for the
    # strategy optimizer A/B tracking system.
    learner = _get_or_create_learner(request)
    active_attempt = ReadingAttempt.objects.filter(
        learner=learner, completed=False
    ).last()

    if active_attempt:
        from agents.strategy_optimizer import assign_strategy_variant
        from analytics.strategy_tracker import create_experiment
        ld_profile = {'confirmed': learner.ld_confirmed, 'suspected': learner.ld_suspected}
        variant = assign_strategy_variant(ld_profile, mode=mode)
        # Only create if no experiment exists yet for this attempt
        from tutor.models import ReadingStrategyExperiment
        if not ReadingStrategyExperiment.objects.filter(attempt=active_attempt).exists():
            create_experiment(
                learner_id=learner.pk,
                attempt_id=active_attempt.pk,
                variant=variant,
            )

    return JsonResponse({'ok': True, 'mode': mode})


@csrf_exempt
@require_POST
def api_reading_paragraph_strategy(request):
    """Return an inline reading strategy for a specific passage section.

    Body: { "attempt_id": int, "section_order": int }

    Returns:
        strategy:               Markdown-formatted inline strategy (may be empty
                                string if no questions map to this paragraph).
        related_question_orders: List of question order numbers for this paragraph.
        related_question_ids:   List of question IDs for this paragraph.
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    learner = _get_or_create_learner(request)
    attempt_id = data.get('attempt_id')
    section_order = int(data.get('section_order', 1))

    attempt = get_object_or_404(
        ReadingAttempt.objects.select_related('passage'),
        id=attempt_id,
        learner=learner,
    )
    section = get_object_or_404(IELTSSection, passage=attempt.passage, order=section_order)

    all_questions = list(attempt.passage.questions.values('id', 'order', 'text', 'group_label', 'group_instruction').order_by('order'))
    all_sections = list(attempt.passage.sections.values('id', 'order', 'heading', 'body').order_by('order'))

    from agents.reading_agent import map_questions_to_paragraphs, reading_agent_paragraph_strategy

    # Use the stored section FK first; fall back to the dynamic mapping if needed
    related_q_ids = list(
        IELTSQuestion.objects.filter(section=section).values_list('id', flat=True)
    )
    if related_q_ids:
        related_questions = [q for q in all_questions if q['id'] in related_q_ids]
    else:
        mapping = map_questions_to_paragraphs(all_sections, all_questions)
        related_q_ids = mapping.get(section.id, [])
        related_questions = [q for q in all_questions if q['id'] in related_q_ids]

    ld_profile = {'confirmed': learner.ld_confirmed, 'suspected': learner.ld_suspected}

    section_dict = {
        'id': section.id,
        'order': section.order,
        'heading': section.heading or '',
        'body': section.body or attempt.passage.raw_text or '',
    }

    strategy = reading_agent_paragraph_strategy(section_dict, related_questions, ld_profile)
    related_q_orders = sorted(q['order'] for q in related_questions)

    return JsonResponse({
        'strategy': strategy,
        'related_question_orders': related_q_orders,
        'related_question_ids': related_q_ids,
    })


@require_GET
def api_reading_section_tips(request):
    """Return the pre-generated reading tips for a specific section.

    Query params: attempt_id=<int>, section_order=<int>

    Returns:
        tips: list of tip strings for this section
    """
    learner = _get_or_create_learner(request)
    attempt_id = request.GET.get('attempt_id')
    section_order = int(request.GET.get('section_order', 1))

    attempt = get_object_or_404(
        ReadingAttempt.objects.select_related('passage'),
        id=attempt_id,
        learner=learner,
    )
    section = get_object_or_404(IELTSSection, passage=attempt.passage, order=section_order)
    return JsonResponse({'tips': list(section.reading_tips or [])})


@require_GET
def api_strategy_performance(request):
    """Admin endpoint: return anonymised strategy performance data.

    Query params (optional):
        ld_type=<str>   Filter by LD profile type (e.g. "adhd")

    Returns JSON with performance summaries for the admin dashboard.
    Restricted to Django staff users (is_staff=True) to prevent
    unauthorised access to aggregate performance metrics.
    """
    if not (request.user.is_authenticated and request.user.is_staff):
        return JsonResponse({'error': 'Permission denied.'}, status=403)

    from analytics.strategy_tracker import get_performance_summary
    ld_type = request.GET.get('ld_type')

    summary = get_performance_summary()
    if ld_type:
        summary = [r for r in summary if r['ld_profile_type'] == ld_type]

    return JsonResponse({'performance': summary})
