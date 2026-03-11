# main.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Entry Point
#
# Usage:
#   python main.py
#   python main.py --learner alice --ld executive_function,adhd
#
# Special commands:
#   !reading - Generates an ADHD-friendly HTML view for the IELTS reading passage.
# ─────────────────────────────────────────────────────────────────
import argparse
import time
import json
from pathlib import Path

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


def generate_reading_html_view():
    """Reads passage.txt and generates a two-column HTML view."""
    print("[Generator] Creating ADHD-friendly reading view...")
    try:
        passage_path = Path(__file__).parent / "passage.txt"
        output_path = Path(__file__).parent / "ielts_reading_view.html"

        with open(passage_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # De-duplicate questions
        seen_prompts = set()
        unique_questions = []
        for q in data['questions']:
            if q['prompt'] not in seen_prompts:
                unique_questions.append(q)
                seen_prompts.add(q['prompt'])
        data['questions'] = unique_questions

        # Build HTML
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; margin: 0; padding: 1rem; background-color: #f0f2f5; color: #1c1e21; }}
        .container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; max-width: 1400px; margin: auto; }}
        .passage, .questions {{ background-color: #ffffff; padding: 1.5rem; border-radius: 8px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1); height: fit-content; }}
        h1, h2 {{ color: #333; border-bottom: 1px solid #ddd; padding-bottom: 0.5rem; margin-top: 0; }}
        h2 {{ font-size: 1.2rem; color: #555; }}
        .paragraph {{ margin-bottom: 1rem; display: flex; }}
        .paragraph-id {{ font-weight: bold; margin-right: 1rem; color: #0056b3; }}
        .paragraph-text {{ text-align: justify; }}
        .question {{ margin-bottom: 2rem; padding: 1rem; border: 1px solid #e0e0e0; border-radius: 6px; }}
        .question p {{ margin-top: 0; }}
        .options {{ list-style-type: none; padding: 0; }}
        .options li {{ margin-bottom: 0.5em; }}
        select {{ width: 100%; padding: 8px; border-radius: 4px; border: 1px solid #ccc; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="passage">
            <h1>{title}</h1>
            <h2>{subtitle}</h2>
            {passages_html}
        </div>
        <div class="questions">
            <h2>Questions</h2>
            {questions_html}
        </div>
    </div>
</body>
</html>
        """

        # Populate Passages
        passages_html = ""
        for p in data['paragraphs']:
            passages_html += f'<div class="paragraph"><span class="paragraph-id">{p["id"]}</span><p class="paragraph-text">{p["text"]}</p></div>'

        # Populate Questions
        questions_html = ""
        for q in data['questions']:
            question_content = f'<div class="question" id="q{q["id"]}"><p><strong>Question {q["id"]}</strong></p><p>{q["instruction"]}</p><p>{q["prompt"]}</p>'
            if q['type'] == 'paragraph_matching':
                options = "".join([f'<option value="{opt}">{opt}</option>' for opt in q['options']])
                question_content += f'<select><option value="">Select Paragraph</option>{options}</select>'
            elif q['type'] == 'person_matching':
                options_html = ""
                for person_key in q['options']:
                    if person_key in data['people']:
                        options_html += f'<option value="{person_key}">{data["people"][person_key]}</option>'
                question_content += f'<select><option value="">Select Person</option>{options_html}</select>'
            elif q['type'] == 'multiple_choice':
                options = "".join([f'<li><label><input type="checkbox" name="q{q["id"]}" value="{key}"> {key}: {value}</label></li>' for key, value in q['options'].items()])
                question_content += f'<ul class="options">{options}</ul>'
                if q.get('select'):
                    question_content += f'<p><em>(Choose {q["select"]})</em></p>'
            question_content += '</div>'
            questions_html += question_content

        # Final render
        final_html = html.format(
            title=data['title'],
            subtitle=data['subtitle'],
            passages_html=passages_html,
            questions_html=questions_html
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
        
        print(f"[Success] View generated: {output_path.resolve()}")

    except FileNotFoundError:
        print(f"[Error] Could not find passage.txt at {passage_path.resolve()}")
    except json.JSONDecodeError:
        print(f"[Error] Could not parse passage.txt. Please ensure it is valid JSON.")
    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")


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
            
            if user_input.lower() == "!reading":
                generate_reading_html_view()
                print("-" * 60)
                continue

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
