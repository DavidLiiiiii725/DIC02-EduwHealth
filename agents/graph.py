# agents/graph.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  LangGraph Pipeline
#
# New execution order:
#   rag → affect → csm_update →
#     [tutor || coach || critic || ld_specialist || metacog (conditional)]
#   → parliament → risk
#
# The csm_update node:
#   - calls FeatureExtractor to get cognitive signals
#   - updates CognitiveStateMachine
#   - injects cognitive_state, intervention_flags, trajectory_flags into state
#   - loads LearnerProfile and injects ld_profile, scaffold_density, etc.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from langgraph.graph import StateGraph

from agents.state          import TutorState
from agents.rag_node       import rag_retrieve_node
from agents.tutor_agent    import tutor_agent
from agents.coach_agent    import coach_agent
from agents.critic_agent   import critic_agent
from agents.ld_agent       import ld_specialist_agent
from agents.metacog_agent  import metacog_agent, should_activate_metacog
from agents.parliament     import parliament_node

from core.llm_client       import LLMClient
from core.cognitive_state  import CognitiveStateMachine

from affect.emotion_model  import EmotionDetector

from analytics.feature_extractor import FeatureExtractorLLM
from analytics.risk_model        import RiskModelLLM

from memory.learner_model  import LearnerModelStore


def build_graph(memory):
    llm              = LLMClient()
    emotion_detector = EmotionDetector()
    fx               = FeatureExtractorLLM(llm_client=llm, max_retries=2)
    risk_model       = RiskModelLLM(feature_extractor=fx)
    csm              = CognitiveStateMachine()
    learner_store    = LearnerModelStore()

    # turn counter (shared across calls in this session)
    _turn = {"n": 0}

    # ── Node definitions ──────────────────────────────────────────

    def affective_node(state):
        emotion = emotion_detector.detect(state["user_input"])
        return {"emotion": emotion}

    def csm_update_node(state):
        """
        1. Extract cognitive signals via LLM feature extractor.
        2. Update CSM with EWMA.
        3. Load learner profile and inject profile data into state.
        """
        _turn["n"] += 1
        turn = _turn["n"]

        # Extract full feature set (risk + cognitive signals)
        feats  = fx.extract(state)
        cog_signals = feats.cognitive_signals()

        # Load learner profile
        learner_id = state.get("learner_id", "default")
        profile    = learner_store.load(learner_id)

        # Sync CSM baseline with learner's known attention span
        csm.set_baseline(profile.baseline.avg_session_attention_min)

        # Update CSM
        new_state = csm.update(cog_signals, turn=turn)
        int_flags  = csm.get_intervention_flags()
        traj_flags = csm.get_trajectory_flags()

        return {
            "turn_number":         turn,
            "cognitive_state":     new_state.to_dict(),
            "intervention_flags":  int_flags,
            "trajectory_flags":    traj_flags,
            "cognitive_signals":   cog_signals,
            # Learner profile snapshot for agents
            "ld_profile":              {
                "confirmed": profile.ld_profile.confirmed,
                "suspected": profile.ld_profile.suspected,
                "severity":  profile.ld_profile.severity,
            },
            "scaffold_density":        profile.scaffold_density(),
            "successful_strategies":   profile.successful_strategies(),
        }

    def risk_node(state):
        res = risk_model.predict(state)
        return {
            "risk_score":   res.score,
            "risk_level":   res.level,
            "risk_reasons": res.reasons,
        }

    def metacog_node_conditional(state):
        """Only runs the MetaCog agent when conditions are met."""
        if should_activate_metacog(state):
            return metacog_agent(state, llm)
        return {"metacog_response": ""}

    # ── Build graph ───────────────────────────────────────────────

    graph = StateGraph(TutorState)

    graph.add_node("rag",           lambda s: rag_retrieve_node(s, memory, k=6, depth=2))
    graph.add_node("affect",        affective_node)
    graph.add_node("csm_update",    csm_update_node)

    graph.add_node("tutor",         lambda s: tutor_agent(s, llm))
    graph.add_node("coach",         lambda s: coach_agent(s, llm))
    graph.add_node("critic",        lambda s: critic_agent(s, llm))
    graph.add_node("ld_specialist", lambda s: ld_specialist_agent(s, llm))
    graph.add_node("metacog",       metacog_node_conditional)

    graph.add_node("parliament",    parliament_node)
    graph.add_node("risk",          risk_node)

    # ── Edges ─────────────────────────────────────────────────────

    graph.set_entry_point("rag")

    graph.add_edge("rag",        "affect")
    graph.add_edge("affect",     "csm_update")

    # All agents run in parallel after CSM update
    graph.add_edge("csm_update", "tutor")
    graph.add_edge("csm_update", "coach")
    graph.add_edge("csm_update", "critic")
    graph.add_edge("csm_update", "ld_specialist")
    graph.add_edge("csm_update", "metacog")

    # All agents feed parliament
    graph.add_edge("tutor",         "parliament")
    graph.add_edge("coach",         "parliament")
    graph.add_edge("critic",        "parliament")
    graph.add_edge("ld_specialist", "parliament")
    graph.add_edge("metacog",       "parliament")

    graph.add_edge("parliament", "risk")

    return graph.compile()
