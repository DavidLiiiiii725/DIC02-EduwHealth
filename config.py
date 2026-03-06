# config.py
# ─────────────────────────────────────────────
# EduwHealth 2.0  –  Central Configuration
# ─────────────────────────────────────────────

# ── LLM ──────────────────────────────────────
# 切换这里: "ollama" 或 "gemini" or deepseek
LLM_BACKEND = "ollama"

OLLAMA_MODEL = "phi3:mini"
OLLAMA_HOST  = "http://localhost:11434"

GEMINI_API_KEY = ""   # 粘贴你的 key
GEMINI_MODEL   = "gemini-2.0-flash"  # 免费额度最高，速度最快

DEEPSEEK_API_KEY = ""
DEEPSEEK_MODEL   = "deepseek-chat"
# ── Legacy thresholds (kept for backward compat) ──
EMOTION_THRESHOLD = 0.4
RISK_THRESHOLD    = 0.8

# ── Cognitive State Machine ───────────────────
# EWMA smoothing factor  (0 = all history,  1 = only current)
CSM_ALPHA = 0.35

# Thresholds that trigger adaptive behaviour
CSM_WM_OVERLOAD_THRESHOLD    = 0.70   # working_memory_load  → reduce density
CSM_MOTIVATION_LOW_THRESHOLD = 0.30   # motivation_level     → activate Coach early
CSM_FATIGUE_HIGH_THRESHOLD   = 0.80   # cognitive_fatigue    → propose break
CSM_AFFECT_NEG_THRESHOLD     = -0.50  # affect_valence       → affective support mode

# Trajectory: how many consecutive turns to watch before proactive intervention
CSM_TRAJECTORY_WINDOW = 3
CSM_MOTIVATION_SLOPE_TRIGGER = -0.15  # drop per turn that triggers Coach

# ── Learner Model ─────────────────────────────
LEARNER_MODEL_PATH = "learner_profiles"   # directory for JSON profiles

# ── LD Specialist / Scaffold ──────────────────
# ef_severity bands for Writing Scaffold density
SCAFFOLD_MINIMAL_MAX   = 0.40   # below → guiding questions only
SCAFFOLD_MODERATE_MAX  = 0.70   # below → paragraph frame + key terms
                                 # above → sentence starters + fill-in-blank

# ── Agent priority weights (Orchestrator) ─────
AGENT_PRIORITY = {
    "ld_specialist":  1.0,
    "coach":          0.9,
    "meta_cognition": 0.7,
    "tutor":          0.6,
    "critic":         0.5,
}
AGENT_TIE_THRESHOLD = 0.05   # abs difference below which higher-priority wins

# ── Knowledge Base ────────────────────────────
KB_STORE_DIR = "kb_store"
# Domain partition sub-directories inside KB_STORE_DIR
KB_DOMAINS = {
    "writing":     "kb_writing",
    "auditory":    "kb_auditory",
    "motivation":  "kb_motivation",
    "general":     "kb_general",
}

# Source files ingested by build_vector_kb.py
KB_FILES = [
    "kb.txt",                        # ML fundamentals (original)
    "kb_learning_disabilities.txt",  # Learning disabilities knowledge base
    "kb_interventions.txt",          # Intervention strategies library
]

# ── Mental Health Configuration ───────────────────────────────
# Recommended model: mental/mental-bert-base-uncased
# Fallback options: mrm8488/mental-bert  or  distilbert-base-uncased (fine-tuned)
MENTAL_HEALTH_MODEL = "mental/mental-bert-base-uncased"

# Risk level thresholds: (lower_bound_inclusive, upper_bound_exclusive)
RISK_THRESHOLDS = {
    "low":      (0.0, 0.3),
    "moderate": (0.3, 0.6),
    "high":     (0.6, 0.8),
    "severe":   (0.8, 1.01),   # 1.01 so that score==1.0 maps to "severe"
}

# ── Intervention Configuration ────────────────────────────────
ENABLE_AUTO_INTERVENTION = True
HIGH_RISK_ALERT_EMAIL    = "counselor@nyush.edu.cn"

# ── Learning Disabilities Support ─────────────────────────────
SUPPORTED_DISABILITIES = [
    "ADHD",
    "executive_function_deficit",
    "anxiety_disorder",
    "learned_helplessness",
    "motivation_disorder",
    "academic_burnout",
    "social_anxiety",
]
