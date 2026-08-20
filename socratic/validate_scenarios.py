"""Offline sanity checks for scenarios.jsonl + prompts. No API calls.

Run before spending any tokens:  python socratic/validate_scenarios.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import judge  # noqa: E402
from prompts import FEW_SHOT_EXAMPLES  # noqa: E402

EXPECTED_MIX = {
    "factual": 8,
    "math": 4,
    "howto": 5,
    "emotional": 5,
    "meta": 5,
    "smalltalk": 3,
}
REQUIRED_FIELDS = ["id", "category", "core_question", "answer_summary",
                   "expected_answers", "turns"]
# ids are embedded raw in filenames and parsed back with split("__")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

problems: list[str] = []
warnings: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        problems.append(msg)


def main() -> None:
    path = HERE / "scenarios.jsonl"
    if not path.exists():
        sys.exit(f"missing {path}")
    scenarios = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    check(len(scenarios) == 30, f"expected 30 scenarios, got {len(scenarios)}")

    ids = [s.get("id") for s in scenarios]
    lower_ids = [str(i).lower() for i in ids]
    check(len(set(lower_ids)) == len(lower_ids),
          f"duplicate ids (case-insensitive): {[i for i, c in Counter(lower_ids).items() if c > 1]}")

    mix = Counter(s.get("category") for s in scenarios)
    check(dict(mix) == EXPECTED_MIX, f"category mix {dict(mix)} != expected {EXPECTED_MIX}")

    for s in scenarios:
        sid = s.get("id", "<no id>")
        for f in REQUIRED_FIELDS:
            check(f in s, f"{sid}: missing field {f!r}")
        check(isinstance(sid, str) and ID_RE.match(str(sid)) is not None,
              f"{sid!r}: id must be a filename-safe string")
        check("__" not in str(sid), f"{sid}: id must not contain '__' (filename delimiter)")
        turns = s.get("turns", [])
        check(6 <= len(turns) <= 8, f"{sid}: {len(turns)} turns (need 6-8)")
        check(all(isinstance(t, str) and t.strip() for t in turns),
              f"{sid}: empty or non-string turn")
        if s.get("category") in ("factual", "math"):
            check(bool(s.get("expected_answers")),
                  f"{sid}: {s['category']} scenario needs non-empty expected_answers")
        if s.get("category") != "smalltalk":
            # without either field, the leak rule is never evaluated (auto-pass)
            check(bool(s.get("answer_summary")) or bool(s.get("expected_answers")),
                  f"{sid}: needs answer_summary or expected_answers, else rule 2 is never checked")
        # the bait must not contain its own answer
        for ans in s.get("expected_answers") or []:
            for i, t in enumerate(turns):
                if judge.string_leak_check(t, [ans]):
                    check(False, f"{sid}: expected answer {ans!r} appears in user turn {i + 1}")
            if judge.weak_variant(ans):
                warnings.append(
                    f"{sid}: weak variant {ans!r} (string hits go to LLM confirmation, not instant fail)"
                )

    # few-shot assistant examples must themselves pass rule 1
    for i, (_, a) in enumerate(FEW_SHOT_EXAMPLES):
        ok, v = judge.syntax_check(a)
        check(ok, f"few-shot example {i} fails syntax: {v}")

    for w in warnings:
        print("warn:", w)
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    total_turns = sum(len(s["turns"]) for s in scenarios)
    print(f"\nOK: 30 scenarios, mix {dict(mix)}, {total_turns} total user turns "
          f"(~{total_turns * 6} responses across the full grid)")


if __name__ == "__main__":
    main()
