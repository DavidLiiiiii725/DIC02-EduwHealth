# config.py
# ─────────────────────────────────────────────
# EduwHealth 2.0  –  Central Configuration
# ─────────────────────────────────────────────

# ── LLM ──────────────────────────────────────
# 切换这里: "ollama" 或 "gemini" or deepseek
LLM_BACKEND = "ollama"

OLLAMA_MODEL = "phi3:mini"
OLLAMA_HOST  = "http://localhost:11434"

GEMINI_API_KEY = "AIzaSyCODuW0BGhlZpBQapN1yT5H4PkA0ehoLpE"   # 粘贴你的 key
GEMINI_MODEL   = "gemini-2.0-flash"  # 免费额度最高，速度最快

DEEPSEEK_API_KEY = "sk-f1c313070a9a44e99f9260f1e8d72e1d"
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
