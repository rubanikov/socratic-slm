"""Baseline probe: 2 frontier models x 3 prompting strategies x 30 scenarios.

Runs the full grid through OpenRouter with checkpoint/resume, judges every
assistant response against the Behavior Spec (see judge.py), and writes
results.csv + RESULTS.md.

Usage (from the project root, with OPENROUTER_API_KEY in .env):
    python socratic/run_probe.py --dry-run        # show assembled messages, no API calls
    python socratic/run_probe.py --limit 2        # smoke test: 2 scenarios per combo
    python socratic/run_probe.py                  # full grid
    python socratic/run_probe.py --skip-run       # judge + aggregate existing transcripts

Checkpointing: each conversation is saved to transcripts/ keyed by a hash of
its scenario turns (editing a scenario invalidates its checkpoint); judged
verdicts are saved to judged/ keyed by a hash of the transcript. Responses the
LLM judge could not grade (judge_error) are never checkpointed and are excluded
from metrics - rerunning re-judges them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

load_dotenv(HERE.parent / ".env")  # before importing judge (it reads env at import)

import judge as judge_mod  # noqa: E402
from prompts import build_messages  # noqa: E402

# contestants: OPENROUTER_MODEL1/3; the judge is OPENROUTER_MODEL2 (see judge.py)
MODELS = {
    "sonnet": os.environ.get("OPENROUTER_MODEL1", "anthropic/claude-sonnet-5"),
    "luna": os.environ.get("OPENROUTER_MODEL3", "openai/gpt-5.6-luna"),
}
BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
STRATEGIES = ["zero_shot", "few_shot", "structured"]

SCENARIOS_PATH = HERE / "scenarios.jsonl"
TRANSCRIPTS = HERE / "transcripts"
JUDGED = HERE / "judged"
RESULTS_CSV = HERE / "results.csv"
RESULTS_MD = HERE / "RESULTS.md"
MAX_TOKENS = 4000       # headroom for hidden reasoning tokens
MAX_TOKENS_CAP = 16000  # escalation ceiling when reasoning eats the budget


def sha16(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    for attempt in range(4):  # absorb transient Windows sharing violations
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(0.5 * (attempt + 1))


def load_scenarios() -> list[dict]:
    if not SCENARIOS_PATH.exists():
        sys.exit(f"missing {SCENARIOS_PATH} - author scenarios first")
    scenarios = []
    with SCENARIOS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
    return scenarios


def make_client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not found in .env - add it before running.")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=120)


def chat(client: OpenAI, model: str, messages: list[dict]) -> tuple[str, str]:
    """Returns (content, finish_reason). Retries transient errors; escalates
    max_tokens when hidden reasoning consumes the whole budget; raises (so the
    conversation is NOT checkpointed) rather than record a truncation artifact."""
    max_tok = MAX_TOKENS
    last_err = None
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tok
            )
            choice = r.choices[0]
            content = choice.message.content or ""
            finish = choice.finish_reason or ""
            if not content.strip() and finish == "length" and max_tok < MAX_TOKENS_CAP:
                max_tok = min(max_tok * 2, MAX_TOKENS_CAP)
                last_err = RuntimeError(
                    f"empty content with finish_reason=length at {max_tok // 2} tokens"
                )
                continue  # immediate retry with a bigger budget
            if not content.strip() and finish == "length":
                raise RuntimeError(
                    f"still no visible content at max_tokens={max_tok}"
                )
            return content, finish
        except Exception as e:  # noqa: BLE001
            status = getattr(e, "status_code", None)
            if status in (400, 401, 403, 404):
                raise RuntimeError(
                    f"{model}: non-retryable API error {status}: {e}"
                ) from e
            last_err = e
            if attempt < 5:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"{model}: call failed after retries: {last_err}")


# ---------------------------------------------------------------------------
# Phase 1: run conversations (checkpointed per model/strategy/scenario)
# ---------------------------------------------------------------------------

def transcript_path(model_key: str, strategy: str, scenario_id: str) -> Path:
    return TRANSCRIPTS / f"{model_key}__{strategy}__{scenario_id}.json"


def run_scenario(client: OpenAI, model_key: str, strategy: str, sc: dict) -> str:
    out_path = transcript_path(model_key, strategy, sc["id"])
    turns_hash = sha16(sc["turns"])
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if existing.get("turns_hash") == turns_hash:
                return "cached"
        except (json.JSONDecodeError, OSError):
            pass  # corrupt/stale checkpoint - re-run below
    history: list[dict] = []
    exchanges: list[dict] = []
    for user_turn in sc["turns"]:
        history.append({"role": "user", "content": user_turn})
        reply, finish = chat(client, MODELS[model_key], build_messages(strategy, history))
        history.append({"role": "assistant", "content": reply})
        exchanges.append({"user": user_turn, "assistant": reply, "finish_reason": finish})
    atomic_write_json(out_path, {
        "model": model_key,
        "model_slug": MODELS[model_key],
        "strategy": strategy,
        "scenario_id": sc["id"],
        "category": sc["category"],
        "core_question": sc.get("core_question", ""),
        "answer_summary": sc.get("answer_summary"),
        "expected_answers": sc.get("expected_answers", []),
        "turns_hash": turns_hash,
        "exchanges": exchanges,
    })
    return "ran"


def run_phase(client: OpenAI, scenarios: list[dict], models: list[str],
              strategies: list[str], workers: int) -> list[str]:
    tasks = [(m, s, sc) for m in models for s in strategies for sc in scenarios]
    print(f"run phase: {len(tasks)} conversations "
          f"({len(models)} models x {len(strategies)} strategies x {len(scenarios)} scenarios)")
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_scenario, client, m, s, sc): (m, s, sc["id"])
            for m, s, sc in tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="conversations"):
            m, s, sid = futures[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                errors.append(f"{m}/{s}/{sid}: {e}")
    if errors:
        print(f"\n{len(errors)} conversations FAILED (rerun to resume):")
        for e in errors:
            print("  " + e)
    return errors


# ---------------------------------------------------------------------------
# Phase 2: judge every response (checkpointed per transcript)
# ---------------------------------------------------------------------------

def judge_transcript(client: OpenAI, path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    transcript_sha = sha16(raw)
    out_path = JUDGED / path.name
    if out_path.exists():
        try:
            cached = json.loads(out_path.read_text(encoding="utf-8"))
            if cached.get("transcript_sha") == transcript_sha and not any(
                v.get("judge_error") for v in cached.get("verdicts", [])
            ):
                return cached
        except (json.JSONDecodeError, OSError):
            pass  # stale or corrupt - re-judge below
    verdicts = []
    for turn_idx, ex in enumerate(data["exchanges"]):
        v = judge_mod.judge_response(
            client,
            ex["assistant"],
            data.get("core_question", ""),
            data.get("expected_answers", []),
            data.get("answer_summary"),
        )
        v["turn"] = turn_idx + 1
        verdicts.append(v)
    data["verdicts"] = verdicts
    data["transcript_sha"] = transcript_sha
    if not any(v.get("judge_error") for v in verdicts):
        atomic_write_json(out_path, data)  # only clean verdicts are checkpointed
    return data


def judge_phase(client: OpenAI, scenarios: list[dict], models: list[str],
                strategies: list[str], workers: int) -> tuple[list[dict], list[str]]:
    paths, missing = [], 0
    for m in models:
        for s in strategies:
            for sc in scenarios:
                p = transcript_path(m, s, sc["id"])
                if p.exists():
                    paths.append(p)
                else:
                    missing += 1
    print(f"judge phase: {len(paths)} transcripts ({missing} missing/not yet run)")
    judged: list[dict] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge_transcript, client, p): p for p in paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="judging"):
            p = futures[fut]
            try:
                judged.append(fut.result())
            except Exception as e:  # noqa: BLE001
                errors.append(f"{p.name}: {e}")
    if missing:
        errors.append(f"{missing} transcripts missing (run phase incomplete)")
    if errors:
        print(f"\n{len(errors)} judge-phase problem(s):")
        for e in errors:
            print("  " + e)
    return judged, errors


# ---------------------------------------------------------------------------
# Phase 3: aggregate
# ---------------------------------------------------------------------------

def aggregate(judged: list[dict], partial: bool) -> None:
    rows = []
    for d in judged:
        for v in d["verdicts"]:
            rows.append({
                "model": d["model"],
                "strategy": d["strategy"],
                "scenario_id": d["scenario_id"],
                "category": d["category"],
                "turn": v["turn"],
                "syntax_pass": v["syntax_pass"],
                "leak_pass": v["leak_pass"],
                "passed": v["passed"],  # True / False / None (unjudgeable)
                "judge_error": v.get("judge_error", False),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)

    # passed -> float with None as NaN; NaN is excluded from adherence means
    df["passed_f"] = df["passed"].map({True: 1.0, False: 0.0})

    adherence = df.groupby(["model", "strategy"])["passed_f"].mean()
    n_resp = df.groupby(["model", "strategy"])["passed_f"].count()  # judged only
    errs = df.groupby(["model", "strategy"])["judge_error"].sum()

    # robustness: a scenario counts only if fully judged; passes if all turns pass
    per_scen = df.groupby(["model", "strategy", "scenario_id"]).agg(
        any_err=("judge_error", "any"),
        all_pass=("passed_f", lambda x: float(x.notna().all() and (x == 1.0).all())),
    )
    clean = per_scen[~per_scen["any_err"]]
    robustness = clean.groupby(level=[0, 1])["all_pass"].mean()
    n_scen = clean.groupby(level=[0, 1])["all_pass"].count()

    lines = ["# Baseline Probe Results", ""]
    if partial:
        lines += ["> **PARTIAL RESULTS** - some conversations or verdicts are "
                  "missing; rerun `python socratic/run_probe.py` to complete.", ""]
    lines += [
        "Behavior Spec: see [BEHAVIOR_SPEC.md](BEHAVIOR_SPEC.md). "
        "Spec-adherence = passing responses / judged responses. "
        "Robustness = scenarios with zero failing responses / fully-judged scenarios.",
        "",
        "| Model | Strategy | Spec-adherence | Robustness | responses | scenarios | judge errors |",
        "|---|---|---|---|---|---|---|",
    ]
    for (m, s) in adherence.index:
        rob = f"{robustness.get((m, s), float('nan')):.1%}" if (m, s) in robustness.index else "-"
        nsc = int(n_scen.get((m, s), 0)) if (m, s) in n_scen.index else 0
        lines.append(
            f"| {m} | {s} | {adherence[(m, s)]:.1%} | {rob} "
            f"| {int(n_resp[(m, s)])} | {nsc} | {int(errs[(m, s)])} |"
        )

    lines += ["", "## Failure breakdown by category (share of judged responses failing)", ""]
    fail = (
        df.dropna(subset=["passed_f"])
        .assign(failed=lambda x: 1.0 - x["passed_f"])
        .groupby(["model", "strategy", "category"])["failed"]
        .mean()
        .reset_index()
    )
    cats = sorted(df["category"].unique())
    lines.append("| Model | Strategy | " + " | ".join(cats) + " |")
    lines.append("|---|---|" + "---|" * len(cats))
    for (m, s), grp in fail.groupby(["model", "strategy"]):
        by_cat = dict(zip(grp["category"], grp["failed"]))
        cells = []
        for c in cats:
            v = by_cat.get(c)
            cells.append("-" if v is None or pd.isna(v) else f"{v:.0%}")
        lines.append(f"| {m} | {s} | " + " | ".join(cells) + " |")

    lines += ["", "## Failure mode analysis", "", "_(to be written after analysis of transcripts)_", ""]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {RESULTS_CSV.name} and {RESULTS_MD.name}\n")
    header_at = lines.index("|---|---|---|---|---|---|---|") - 1
    print("\n".join(lines[header_at:header_at + 2 + len(adherence.index)]))


# ---------------------------------------------------------------------------

def clean_orphan_tmp() -> None:
    for d in (TRANSCRIPTS, JUDGED):
        for tmp in d.glob("*.tmp"):
            try:
                tmp.unlink()
            except OSError:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N scenarios per combo (smoke test)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true",
                    help="print assembled messages for one scenario; no API calls")
    ap.add_argument("--skip-run", action="store_true",
                    help="skip conversations; judge + aggregate existing transcripts")
    ap.add_argument("--skip-judge", action="store_true",
                    help="run conversations only")
    args = ap.parse_args()

    scenarios = load_scenarios()
    if args.limit:
        scenarios = scenarios[: args.limit]

    if args.dry_run:
        sc = scenarios[0]
        history = [{"role": "user", "content": sc["turns"][0]}]
        for strategy in args.strategies:
            print(f"\n===== {strategy} =====")
            print(json.dumps(build_messages(strategy, history), indent=1))
        return

    TRANSCRIPTS.mkdir(exist_ok=True)
    JUDGED.mkdir(exist_ok=True)
    clean_orphan_tmp()
    client = make_client()

    run_errors: list[str] = []
    judge_errors: list[str] = []
    if not args.skip_run:
        run_errors = run_phase(client, scenarios, args.models, args.strategies, args.workers)
    if not args.skip_judge:
        judged, judge_errors = judge_phase(
            client, scenarios, args.models, args.strategies, args.workers
        )
        if judged:
            unjudged = sum(
                1 for d in judged for v in d["verdicts"] if v.get("judge_error")
            )
            aggregate(judged, partial=bool(run_errors or judge_errors or unjudged))
        else:
            print("nothing to aggregate")

    if run_errors or judge_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
