# main.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Entry Point
#
# Usage:
#   python main.py
#   python main.py --learner alice --ld executive_function,adhd
# ─────────────────────────────────────────────────────────────────
import argparse
import time

from core.orchestrator import TutorOrchestrator


def parse_args():
    p = argparse.ArgumentParser(description="EduwHealth 2.0 Tutor")
    p.add_argument("--learner", default="default",
                   help="Learner ID (used for persistent profile)")
    p.add_argument("--ld", default="",
                   help="Comma-separated confirmed LD types, e.g. 'executive_function,adhd'")
    p.add_argument("--severity", default="",
                   help="Comma-separated ld_type=float pairs, e.g. 'executive_function=0.7'")
    return p.parse_args()


def main():
    args = parse_args()

    tutor = TutorOrchestrator(learner_id=args.learner)

    # ── Optionally set LD profile from CLI ────────────────────────
    if args.ld:
        confirmed = [x.strip() for x in args.ld.split(",") if x.strip()]
        severity  = {}
        if args.severity:
            for pair in args.severity.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    try:
                        severity[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
        tutor.set_learner_ld_profile(confirmed=confirmed, severity=severity)
        print(f"[Profile] Learner '{args.learner}' | LD: {confirmed} | Severity: {severity}")

    print(f"\n{'='*60}")
    print("  EduwHealth 2.0 — Cognition-Aware Tutor")
    print(f"  Learner: {args.learner}")
    print(f"{'='*60}\n")

    session_start = time.time()

    try:
        while True:
            try:
                user_input = input("Student > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                break

            output = tutor.handle(user_input)

            # ── Primary response ──────────────────────────────────
            print(f"\n[{output['active_agent'].upper()}]")
            print(output["response"])

            # ── Cognitive state summary ───────────────────────────
            cs = output.get("cognitive_state", {})
            if cs:
                wm  = cs.get("working_memory_load", 0)
                mot = cs.get("motivation_level",    0)
                aff = cs.get("affect_valence",      0)
                fat = cs.get("cognitive_fatigue",   0)
                print(f"\n  [Cognitive State]  "
                      f"WM:{wm:.2f}  Motivation:{mot:.2f}  "
                      f"Affect:{aff:+.2f}  Fatigue:{fat:.2f}")

            # ── Flags ─────────────────────────────────────────────
            flags = output.get("intervention_flags", {})
            active = [k for k, v in flags.items() if v]
            if active:
                print(f"  [Flags]  {' | '.join(active)}")

            # ── Risk ──────────────────────────────────────────────
            print(f"  [Risk]  {output['risk']:.3f} ({output['risk_level']})"
                  f"  Escalation: {output['escalation']}")
            print("-" * 60)

    finally:
        # ── End-of-session profile update ─────────────────────────
        session_min = (time.time() - session_start) / 60.0
        tutor.end_session(session_attention_min=session_min)
        print(f"\n[Session saved] Duration: {session_min:.1f} min  Learner: {args.learner}")


if __name__ == "__main__":
    main()
