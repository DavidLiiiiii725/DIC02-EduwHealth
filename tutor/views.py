import json
import time
import threading
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.utils import timezone
from django.db.models import Avg, Sum

from .models import LearnerProfile, ChatSession, ChatMessage, IELTSPassage, IELTSSection, IELTSQuestion, ReadingAttempt

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
    if not user_input:
        return JsonResponse({'error': 'Empty message'}, status=400)

    learner = _get_or_create_learner(request)
    session = ChatSession.objects.filter(learner=learner, ended_at__isnull=True).last()
    if not session:
        session = ChatSession.objects.create(learner=learner)

    ChatMessage.objects.create(session=session, role='user', content=user_input)
    session.total_turns += 1
    session.save()

    orch = _get_orchestrator(learner.learner_id)
    try:
        result = orch.handle(user_input) if orch else _mock_response(user_input)
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
    """IELTS reading assistant page."""
    learner = _get_or_create_learner(request)
    # Find any active (non-completed) attempt
    attempt = ReadingAttempt.objects.filter(learner=learner, completed=False).last()
    passage = attempt.passage if attempt else None
    return render(request, 'tutor/reading.html', {
        'learner': learner,
        'attempt': attempt,
        'passage': passage,
    })


@csrf_exempt
@require_POST
def api_reading_upload(request):
    """Parse and store a new IELTS reading passage.

    Body: { "title": str, "text": str, "num_sections": int (optional) }
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    raw_text = data.get('text', '').strip()
    if not raw_text:
        return JsonResponse({'error': 'text is required'}, status=400)

    learner = _get_or_create_learner(request)
    num_sections = int(data.get('num_sections', 3))

    # Deactivate previous passages
    IELTSPassage.objects.filter(learner=learner, is_active=True).update(is_active=False)

    # Cancel any open attempts
    ReadingAttempt.objects.filter(learner=learner, completed=False).update(completed=True)

    from agents.reading_agent import parse_passage
    parsed = parse_passage(raw_text, num_sections=num_sections)

    passage = IELTSPassage.objects.create(
        learner=learner,
        title=data.get('title', '').strip() or 'IELTS Reading Passage',
        raw_text=raw_text,
    )

    # Persist sections and questions
    section_objs = []
    for i, sec in enumerate(parsed['sections'], start=1):
        s_obj = IELTSSection.objects.create(
            passage=passage,
            order=i,
            heading=sec['heading'],
            body=sec['body'],
        )
        section_objs.append(s_obj)

    q_objs = []
    for q_i, q_text in enumerate(parsed['questions'], start=1):
        # Find which section this question belongs to
        section_obj = None
        for s_i, s_obj in enumerate(section_objs):
            if q_i - 1 in parsed['sections'][s_i].get('question_indices', []):
                section_obj = s_obj
                break
        q_obj = IELTSQuestion.objects.create(
            passage=passage,
            section=section_obj,
            order=q_i,
            text=q_text,
        )
        q_objs.append(q_obj)

    # Create attempt starting at section 1
    attempt = ReadingAttempt.objects.create(
        learner=learner,
        passage=passage,
        current_section_order=1,
    )

    # Build first section response
    first_section = section_objs[0] if section_objs else None
    from agents.reading_agent import reading_agent_guide_section
    ld_profile = {
        'confirmed': learner.ld_confirmed,
        'suspected': learner.ld_suspected,
    }
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
        questions_for_section = list(first_section.questions.values('id', 'order', 'text'))
        section_data = {
            'id': first_section.id,
            'order': first_section.order,
            'heading': first_section.heading,
            'body': first_section.body,
            'questions': questions_for_section,
        }

    return JsonResponse({
        'ok': True,
        'passage_id': passage.id,
        'attempt_id': attempt.id,
        'total_sections': len(section_objs),
        'total_questions': len(q_objs),
        'current_section': section_data,
        'guidance': guidance,
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
    questions_for_section = list(section.questions.values('id', 'order', 'text'))

    from agents.reading_agent import reading_agent_guide_section
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

    return JsonResponse({
        'done': False,
        'current_section': {
            'id': section.id,
            'order': section.order,
            'heading': section.heading,
            'body': section.body,
            'questions': questions_for_section,
        },
        'guidance': guidance,
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

    section_body = question.section.body if question.section else ''

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
    section_body = question.section.body if question.section else ''

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
