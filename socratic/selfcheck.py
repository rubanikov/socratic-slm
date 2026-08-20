"""Local self-check for generation agents. NO API calls - safe to run freely.

Checks a raw generated file (dataset/convs_raw/<id>.json) against everything
verifiable offline: structure, turn counts, rule-1 syntax on every assistant
reply, and string-level answer leaks in both user and assistant turns.

Usage:
  python socratic/selfcheck.py dataset/convs_raw/train_howto_00001.json [...]
  python socratic/selfcheck.py --all
Exit 0 = all clean. Exit 1 = problems printed per file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import judge  # noqa: E402
from dataset_plan import MANIFEST, RAW_DIR  # noqa: E402


def load_manifest() -> dict[str, dict]:
    jobs = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            j = json.loads(line)
            jobs[j["id"]] = j
    return jobs


def merged_expected(job: dict) -> list[str]:
    out = list(job["topic_record"].get("expected_answers") or [])
    if job.get("shift_record"):
        out += list(job["shift_record"].get("expected_answers") or [])
    return out


def check_file(path: Path, jobs: dict[str, dict]) -> list[str]:
    issues: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return [f"unreadable JSON: {e}"]
    jid = data.get("id") or path.stem
    job = jobs.get(jid)
    if job is None:
        return [f"id {jid!r} not in manifest"]
    n_turns = job["n_turns"]
    is_eval = job["split"] in ("eval_dev", "eval_final")

    turns = data.get("turns")
    if is_eval:
        if not isinstance(turns, list) or len(turns) != n_turns:
            issues.append(f"eval job needs exactly {n_turns} user turns in 'turns'")
            turns = turns if isinstance(turns, list) else []
    else:
        msgs = data.get("messages")
        if not isinstance(msgs, list) or len(msgs) != 2 * n_turns:
            return issues + [f"'messages' must have {2 * n_turns} entries (user/assistant x{n_turns})"]
        for i, m in enumerate(msgs):
            want = "user" if i % 2 == 0 else "assistant"
            if not isinstance(m, dict) or m.get("role") != want or not str(m.get("content", "")).strip():
                issues.append(f"messages[{i}]: expected non-empty {want} message")
        turns = [m["content"] for m in msgs if m.get("role") == "user"]
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            ok, viol = judge.syntax_check(str(m.get("content", "")))
            if not ok:
                issues.append(f"assistant turn {i // 2 + 1} FAILS rule 1 (syntax): {viol[:2]}")
            hit = judge.string_leak_check(str(m.get("content", "")), merged_expected(job))
            if hit is not None and not judge.weak_variant(hit):
                issues.append(f"assistant turn {i // 2 + 1} LEAKS answer string {hit!r}")

    for i, t in enumerate(turns or []):
        if not isinstance(t, str) or not t.strip():
            issues.append(f"user turn {i + 1} empty/non-string")
            continue
        hit = judge.string_leak_check(t, merged_expected(job))
        if hit is not None:
            issues.append(f"user turn {i + 1} contains the answer string {hit!r} - rewrite the turn")
    return issues


def main() -> None:
    args = sys.argv[1:]
    jobs = load_manifest()
    if args and args[0] == "--all":
        paths = sorted(RAW_DIR.glob("*.json"))
    else:
        paths = [Path(a) for a in args]
    if not paths:
        sys.exit("no files to check")
    bad = 0
    for p in paths:
        issues = check_file(p, jobs)
        if issues:
            bad += 1
            print(f"FAIL {p.name}")
            for i in issues:
                print(f"  - {i}")
        else:
            print(f"OK   {p.name}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
