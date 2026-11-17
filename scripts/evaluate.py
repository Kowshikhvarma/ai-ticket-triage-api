"""Measure triage accuracy against the hand-labelled sample emails.

Usage:
    python -m scripts.evaluate            # uses the LLM if OPENAI_API_KEY is set, otherwise rules
    python -m scripts.evaluate --rules    # force the rule-based engine
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.classifier import RuleBasedClassifier, build_classifier
from app.config import get_settings
from app.schemas import EmailIn

SAMPLES = Path(__file__).resolve().parents[1] / "samples" / "emails.json"


def evaluate(classifier) -> dict:
    rows = json.loads(SAMPLES.read_text())
    results = []
    for row in rows:
        expected = row.pop("expected")
        triage = classifier.classify(EmailIn(**row))
        results.append(
            {
                "subject": row["subject"],
                "priority": (triage.priority.value, expected["priority"]),
                "category": (triage.category.value, expected["category"]),
            }
        )
    n = len(results)
    return {
        "engine": results and triage.engine,
        "n": n,
        "priority_accuracy": sum(p == e for p, e in (r["priority"] for r in results)) / n,
        "category_accuracy": sum(p == e for p, e in (r["category"] for r in results)) / n,
        "rows": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", action="store_true", help="force rule-based engine")
    args = parser.parse_args()
    clf = RuleBasedClassifier() if args.rules else build_classifier(get_settings())
    report = evaluate(clf)

    print(f"{'Subject':<48} {'Priority (got/exp)':<20} Category (got/exp)")
    print("-" * 100)
    for r in report["rows"]:
        p_ok = "✓" if r["priority"][0] == r["priority"][1] else "✗"
        c_ok = "✓" if r["category"][0] == r["category"][1] else "✗"
        print(f"{r['subject'][:46]:<48} {p_ok} {r['priority'][0]:>6}/{r['priority'][1]:<10} "
              f"{c_ok} {r['category'][0]}/{r['category'][1]}")
    print("-" * 100)
    print(f"Engine: {report['engine']} | emails: {report['n']} | "
          f"priority accuracy: {report['priority_accuracy']:.0%} | "
          f"category accuracy: {report['category_accuracy']:.0%}")


if __name__ == "__main__":
    main()
