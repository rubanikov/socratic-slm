"""The quality gate: judge every raw generated file with the SAME judge.py the
eval uses, then promote or ticket it.

- eval jobs   raw -> validated + enriched from the manifest -> blueprints/<id>.json
- train/test  raw -> every assistant turn judged (syntax local, leak via the
               Gemini judge) -> convs/<id>.json on full pass, or a repair ticket
               (dataset/repair/<id>.json) listing failed turns. After MAX_ATTEMPTS
               failed filter rounds the conversation is dropped (<id>.dropped).

Costs pennies (leak judge only, cached). Usage:
  python socratic/filter_convs.py [--workers 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import judge as judge_mod  # noqa: E402
from dataset_plan import MANIFEST, RAW_DIR, REPAIR_DIR  # noqa: E402
from generate_dataset import BP_DIR, CONV_DIR, make_client  # noqa: E402
from selfcheck import check_file, load_manifest, merged_expected  # noqa: E402

MAX_ATTEMPTS = 3


def job_meta(job: dict) -> dict:
    rec, shift = job["topic_record"], job.get("shift_record")
    core_q = rec["core_question"]
    ans_sum = rec.get("answer_summary")
    if shift:
        core_q += f" (secondary: {shift['core_question']})"
        second = shift.get("answer_summary") or ""
        ans_sum = f"{ans_sum or ''}; after the topic shift: {second}".strip("; ")
    return {
        "category": job["category"],
        "topic": rec["topic"],
        "core_question": core_q,
        "answer_summary": ans_sum or None,
        "expected_answers": merged_expected(job),
        "persona": f"{job['mood']}; {job['context']}",
    }


def process(client, path: Path, jobs: dict[str, dict]) -> str:
    jid = path.stem
    job = jobs.get(jid)
    if job is None:
        return "unknown-id"
    meta = job_meta(job)
    ticket_path = REPAIR_DIR / f"{jid}.json"
    raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    attempts = 0
    prior_sha = None
    if ticket_path.exists():
        ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
        attempts = ticket.get("attempts", 0)
        prior_sha = ticket.get("raw_sha")
        if prior_sha == raw_sha:
            # unchanged since it was ticketed - awaiting repair, don't re-judge
            # and don't burn an attempt
            return "awaiting-repair"

    issues = check_file(path, jobs)
    failed_turns: list[dict] = []
    data: dict = {}
    if issues:
        failed_turns.append({"turn": 0, "reason": "structural: " + " | ".join(issues[:4])})
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not issues and job["split"] in ("eval_dev", "eval_final"):
        out = {"id": jid, "split": job["split"], **meta, "turns": data["turns"]}
        (BP_DIR / f"{jid}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        path.unlink()
        ticket_path.unlink(missing_ok=True)
        return "eval-accepted"
    elif not issues:
        for i, m in enumerate(data["messages"]):
            if m["role"] != "assistant":
                continue
            v = judge_mod.judge_response(
                client, m["content"], meta["core_question"],
                meta["expected_answers"], meta["answer_summary"])
            if v.get("judge_error"):
                return "judge-error"  # leave raw untouched; retry next run
            if not v["passed"]:
                reason = ("; ".join(v["syntax_violations"])[:200]
                          if not v["syntax_pass"] else str(v["leak_reason"])[:200])
                failed_turns.append({"turn": i // 2 + 1, "reason": reason})

    if not failed_turns:
        out = {"id": jid, "split": job["split"], **meta,
               "regens": attempts, "status": "ok", "messages": data["messages"]}
        (CONV_DIR / f"{jid}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        path.unlink()
        ticket_path.unlink(missing_ok=True)
        return "accepted"

    # count an attempt only when a repair actually changed the file (or first ticket)
    if prior_sha is None or prior_sha != raw_sha:
        attempts += 1
    if attempts >= MAX_ATTEMPTS:
        (REPAIR_DIR / f"{jid}.dropped").write_text(
            json.dumps({"id": jid, "failed_turns": failed_turns}, ensure_ascii=False),
            encoding="utf-8")
        path.unlink(missing_ok=True)
        ticket_path.unlink(missing_ok=True)
        return "dropped"
    ticket_path.write_text(json.dumps({
        "id": jid, "attempts": attempts, "raw_sha": raw_sha,
        "failed_turns": failed_turns,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return "ticketed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    jobs = load_manifest()
    paths = sorted(RAW_DIR.glob("*.json"))
    if not paths:
        print("nothing in convs_raw/")
        return
    client = make_client()
    tally: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, client, p, jobs): p for p in paths}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="filtering"):
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                r = f"error: {e}"
                print(f"\n{futs[fut].name}: {e}")
            tally[r.split(":")[0]] = tally.get(r.split(":")[0], 0) + 1
    print(json.dumps(tally, indent=1))


if __name__ == "__main__":
    main()
