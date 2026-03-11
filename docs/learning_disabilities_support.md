# Learning Disabilities Support

## Overview

EduwHealth 2.0 provides evidence-based detection, monitoring, and intervention support for learners with the following learning disabilities and related conditions:

| Condition | Code in system | Severity tracking |
|---|---|---|
| ADHD (Attention Deficit Hyperactivity Disorder) | `ADHD` | Yes |
| Executive Function Deficit | `executive_function_deficit` | Yes |
| Anxiety Disorder (GAD, Social Anxiety, Panic) | `anxiety_disorder` | Yes |
| Learned Helplessness | `learned_helplessness` | No |
| Motivational Disorder / Academic Burnout | `motivation_disorder` | No |
| Social Anxiety Disorder | `social_anxiety` | No |

---

## How Conditions Are Identified

The system uses a multi-signal approach to identify and flag learning difficulties.  
No single signal triggers an intervention; the system looks for converging evidence.

### Signals used

| Signal | Source | Example |
|---|---|---|
| Explicit disclosure | Learner profile (JSON) | `"confirmed": ["ADHD"]` |
| Cognitive state indicators | `CognitiveStateMachine` | `wm_overload = True` |
| Emotion detection | `EmotionDetector` | `fear = 0.72` |
| Mental health risk | `MentalHealthRiskDetector` | `risk_level = "high"` |
| Behavioural patterns | Feature extractor | `task_avoidance`, `negative_attribution` |
| Learner-reported difficulty | Free text | "I can't focus at all today" |

### Updating a learner's profile

Learner profiles are stored as JSON files in the `learner_profiles/` directory.  
To mark a confirmed or suspected disability, edit the relevant profile:

```json
{
  "learner_id": "student123",
  "ld_profile": {
    "confirmed": ["ADHD", "anxiety_disorder"],
    "suspected": ["executive_function_deficit"],
    "severity": {
      "ADHD": 0.7,
      "anxiety_disorder": 0.5
    }
  }
}
```

---

## Intervention Protocols

### ADHD

**Goal**: Reduce attentional and working-memory demands while maintaining academic engagement.

Tier 1 (all ADHD learners):
- Chunked task delivery (≤10 minute segments)
- Written agenda before each session
- Pomodoro timer with mandatory breaks

Tier 2 (moderate severity or `wm_overload` flag):
- Task decomposition scaffold: Step 1 → Step 2 → Step 3 (no more than 3 at a time)
- Noise-cancelling headphone or FM system recommendation
- Visual anchors for key points

Tier 3 (severe, or combined with high risk):
- Shift to ADHD-specific support mode in `SupportAgent`
- Reduce session duration and complexity
- Human educator notification

### Executive Function Deficit (EFD)

**Goal**: Externalise planning and working memory functions that the learner cannot hold internally.

- Pre-task planning template (mandatory for writing tasks)
- Graphic organiser or mind-map tool
- Structured writing scaffold matched to `scaffold_density` (high / medium / low)
- AI writing assistance tools activated

### Anxiety Disorder

**Goal**: Reduce uncertainty and perceived cognitive demand; build self-advocacy skills.

- Information chunked into clearly signposted segments ("First… Then… Finally…")
- Advance content preview before lectures or tasks
- Breathing exercise offered when anxiety indicator > 0.6
- Graded exposure to feared academic tasks
- Scripts for requesting accommodation

### Motivational Disorder / Learned Helplessness

**Goal**: Rebuild self-efficacy through micro-successes and accurate attribution.

- Micro-success architecture: first step is guaranteed to succeed
- Explicit effort-based feedback after each completion
- Attribution retraining when failure is attributed to stable internal causes
- Growth-mindset framing in all feedback

---

## Risk Detection System

### Components

```
User input
    │
    ▼
EmotionDetector (j-hartmann/emotion-english-distilroberta-base)
    │  → emotion scores: joy, sadness, fear, anger, surprise, disgust, neutral
    │
    ▼
MentalHealthRiskDetector (mental/mental-bert-base-uncased or heuristic fallback)
    │  → risk_level: low | moderate | high | severe
    │  → score: 0..1
    │  → confidence: 0..1
    │
    ▼
EnsembleAffectiveDetector (fuses both signals)
    │  → key_indicators: anxiety, depression, positive_affect
    │  → intervention_priority: low | medium | high | immediate
    │  → summary: human-readable description
    │
    ▼
InterventionAgent
    │  → prioritised list of recommended interventions
    │
    ▼
SupportAgent / LD Specialist Agent
       → adaptive response in appropriate mode
```

### Risk levels and responses

| Level | Score range | System response |
|---|---|---|
| `low` | 0.0 – 0.3 | Standard support; growth-mindset reinforcement |
| `moderate` | 0.3 – 0.6 | Increased check-in frequency; Tier 2 scaffolding |
| `high` | 0.6 – 0.8 | Immediate acknowledgement; task-demand reduction; counsellor alert |
| `severe` | 0.8 – 1.0 | Crisis support resources; mandatory human escalation |

---

## Intervention Recommendation Engine

The `InterventionAgent` (see `agents/intervention_agent.py`) generates a prioritised list of interventions based on:

1. **Risk level**: protocol from `_build_risk_protocols()`
2. **Disability type**: disability-specific intervention banks
3. **Emotion indicators**: threshold-based strategies when anxiety > 0.6 or depression > 0.6
4. **Past successes**: strategies that previously succeeded are promoted in priority

### Calling the agent

```python
from agents.intervention_agent import InterventionAgent

agent = InterventionAgent(kb_retriever=None)

learner_state = {
    "risk":           {"risk_level": "moderate"},
    "emotions":       {"fear": 0.55, "sadness": 0.40},
    "key_indicators": {"anxiety": 0.60, "depression": 0.42, "positive_affect": 0.20},
    "disabilities":   ["ADHD", "anxiety_disorder"],
    "successful_strategies": ["task_decomposition"],
}

interventions = agent.recommend_interventions(learner_state)
for iv in interventions:
    print(f"[{iv['priority'].upper()}] {iv['type']}: {iv['strategy']}")
```

---

## Real-time Support Agent

The `SupportAgent` (see `agents/support_agent.py`) selects a conversational mode based on the learner's profile:

| Mode | Trigger | System prompt style |
|---|---|---|
| `crisis_support` | `risk_level` is `high` or `severe` | Calm, safe, non-academic |
| `task_decomposition` | ADHD/EFD with `wm_overload` | Direct, energetic, numbered steps |
| `anxiety_support` | Anxiety disorder or `anxiety > 0.6` | Gentle, signposted, grounding offered |
| `motivation_support` | Motivational disorder or `motivation_low` | Encouraging, micro-task focused |
| `general_support` | Default | Warm, strategic suggestion |

---

## Monitoring and Analytics

### Risk Dashboard

```python
from analytics.risk_dashboard import RiskDashboard

dashboard = RiskDashboard()

# Timeline for one learner
timeline = dashboard.get_learner_risk_timeline("student123")

# All high-risk learners in the last 24 hours
high_risk = dashboard.get_high_risk_learners(lookback_hours=24)

# Aggregate risk statistics
summary = dashboard.get_risk_summary("student123")
```

### Model Evaluator

```python
from analytics.model_evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate risk model on labelled data
test_data = [
    {"text": "I feel hopeless", "expected_risk_level": "high"},
    {"text": "What is gradient descent?", "expected_risk_level": "low"},
]
results = evaluator.evaluate_risk_model(test_data)
print(f"Accuracy: {results['accuracy']}, FN rate: {results['fn_rate']}")
```

---

## Knowledge Base

Three knowledge base files are indexed at build time:

| File | Content | Domain tag |
|---|---|---|
| `kb.txt` | IELTS preparation (all four skills, vocabulary, grammar, strategies) | `general` |
| `kb_learning_disabilities.txt` | EFD, ADHD, anxiety, motivational disorders, 2024-2026 advances | `learning_disabilities` |
| `kb_interventions.txt` | Tiered intervention protocols, risk-response procedures, technology tools | `interventions` |

### Rebuilding the vector index

```bash
python build_vector_kb.py
```

The script reads all files listed in `KB_FILES` (see `config.py`), chunks them, embeds them with `all-MiniLM-L6-v2`, and stores the FAISS index in `kb_store/`.

Each stored chunk is tagged with a `kb_domain` field (`general`, `learning_disabilities`, or `interventions`) to enable domain-filtered retrieval in future extensions.

---

## Empirical Research References

- Mayes, S. D., & Calhoun, S. L. (2007). Learning, attention, writing, and processing speed in typical children and children with ADHD, autism, anxiety, depression, and oppositional-defiant disorder. *Child Neuropsychology*, 13(6), 469–493.
- Altemeier, L., Abbott, R., & Berninger, V. (2008). Executive functions for reading and writing in typical literacy development and dyslexia. *Journal of Clinical and Experimental Neuropsychology*, 30(5), 588–606.
- Eysenck, M. W., Derakshan, N., Santos, R., & Calvo, M. G. (2007). Anxiety and cognitive performance: Attentional control theory. *Emotion*, 7(2), 336–353.
- Abramson, L. Y., Seligman, M. E., & Teasdale, J. D. (1978). Learned helplessness in humans: Critique and reformulation. *Journal of Abnormal Psychology*, 87(1), 49–74.
- Graham, S., & Harris, K. R. (2020). Common Core State Standards and writing: Introduction to the special issue. *The Elementary School Journal*, 120(4), 530–541.
- Chermak, G. D., & Musiek, F. E. (2011). Handbook of (central) auditory processing disorder. Plural Publishing.
- The Next Chapter in ADHD Treatment: What to Expect in 2025 and 2026. (2024). ADDitude Magazine.
