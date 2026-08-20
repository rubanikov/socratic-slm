"""Evaluate a local model (base or fine-tuned checkpoint) against the Behavior Spec.

Plays every scenario's scripted user turns against the model (NO system prompt -
the behavior must come from the weights), judges every reply with the FULL
two-rule judge (regex syntax + string match + Gemini leak judge - the exact
grader used for the frontier baseline), and reports the same two metrics:

  spec-adherence = passing replies / judged replies
  robustness     = scenarios with zero failing replies / fully-judged scenarios

Defaults to eval_dev.jsonl (the development eval). eval_final.jsonl stays
untouched until the end - point --scenarios at it exactly once, for the report.

Usage:
  python socratic/run_eval.py --name base                          # untrained baseline
  python socratic/run_eval.py --adapter socratic/checkpoints/qwen3-1.7b-socratic-125
  python socratic/run_eval.py --adapter ... --scenarios socratic/dataset/final/eval_final.jsonl
Results -> socratic/eval_results/<name>__<scenario-set>.{json,md}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
load_dotenv(HERE.parent / ".env")

import judge as judge_mod  # noqa: E402
from generate_dataset import make_client  # noqa: E402

DEFAULT_BASE = "Qwen/Qwen3-1.7B"
RESULTS_DIR = HERE / "eval_results"


def load_model(base: str, adapter: str | None):
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def play_scenarios(tokenizer, model, scenarios: list[dict], batch_size: int,
                   max_new_tokens: int) -> dict[str, list[str]]:
    """Multi-turn: every scenario's user turns are played in order; replies are
    generated in batches across scenarios that share the same turn index."""
    tokenizer.padding_side = "left"
    histories: dict[str, list[dict]] = {s["id"]: [] for s in scenarios}
    replies: dict[str, list[str]] = {s["id"]: [] for s in scenarios}
    max_turns = max(len(s["turns"]) for s in scenarios)
    for turn_idx in range(max_turns):
        active = [s for s in scenarios if turn_idx < len(s["turns"])]
        for start in range(0, len(active), batch_size):
            batch = active[start:start + batch_size]
            prompts = []
            for s in batch:
                histories[s["id"]].append({"role": "user", "content": s["turns"][turn_idx]})
                prompts.append(tokenizer.apply_chat_template(
                    histories[s["id"]], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                ))
            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=4096).to(model.device)
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen = out[:, inputs["input_ids"].shape[1]:]
            for s, text in zip(batch, tokenizer.batch_decode(gen, skip_special_tokens=True)):
                text = text.strip()
                histories[s["id"]].append({"role": "assistant", "content": text})
                replies[s["id"]].append(text)
        print(f"turn {turn_idx + 1}/{max_turns}: generated for {len(active)} scenarios")
    return replies


def judge_all(client, scenarios: list[dict], replies: dict[str, list[str]]) -> list[dict]:
    rows = []
    for s in scenarios:
        for turn_idx, reply in enumerate(replies[s["id"]]):
            v = judge_mod.judge_response(
                client, reply, s.get("core_question", ""),
                s.get("expected_answers", []), s.get("answer_summary"),
            )
            rows.append({
                "scenario_id": s["id"], "category": s["category"], "turn": turn_idx + 1,
                "reply": reply, "passed": v["passed"], "syntax_pass": v["syntax_pass"],
                "leak_pass": v["leak_pass"], "leak_reason": v["leak_reason"],
                "syntax_violations": v["syntax_violations"],
                "judge_error": v.get("judge_error", False),
            })
    return rows


def aggregate(rows: list[dict]) -> dict:
    judged = [r for r in rows if not r["judge_error"]]
    by_scenario: dict[str, list] = defaultdict(list)
    err_scenarios = {r["scenario_id"] for r in rows if r["judge_error"]}
    for r in judged:
        by_scenario[r["scenario_id"]].append(r["passed"])
    clean = {sid: v for sid, v in by_scenario.items() if sid not in err_scenarios}
    adherence = sum(r["passed"] for r in judged) / max(1, len(judged))
    robustness = sum(all(v) for v in clean.values()) / max(1, len(clean))
    scen_cat = {r["scenario_id"]: r["category"] for r in judged}
    by_cat: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in judged}):
        cr = [r for r in judged if r["category"] == cat]
        cscen = [sid for sid in clean if scen_cat.get(sid) == cat]
        by_cat[cat] = {
            "adherence": sum(r["passed"] for r in cr) / max(1, len(cr)),
            "robustness": sum(all(clean[sid]) for sid in cscen) / max(1, len(cscen)),
            "n_responses": len(cr),
        }
    syntax_fails = sum(1 for r in judged if not r["syntax_pass"])
    leak_fails = sum(1 for r in judged if r["syntax_pass"] and r["leak_pass"] is False)
    return {
        "spec_adherence": adherence,
        "robustness": robustness,
        "n_responses": len(judged),
        "n_scenarios": len(clean),
        "judge_errors": len(rows) - len(judged),
        "fail_split": {"syntax": syntax_fails, "leak": leak_fails},
        "by_category": by_cat,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=None, help="checkpoint dir (omit for base model)")
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--name", default=None, help="label for result files")
    ap.add_argument("--scenarios", type=Path,
                    default=HERE / "dataset" / "final" / "eval_dev.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    args = ap.parse_args()

    name = args.name or (Path(args.adapter).name if args.adapter else "base")
    scen_set = args.scenarios.stem
    scenarios = [json.loads(l) for l in args.scenarios.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        scenarios = scenarios[: args.limit]
    print(f"eval '{name}' on {len(scenarios)} scenarios from {args.scenarios.name}")

    tokenizer, model = load_model(args.base_model, args.adapter)
    replies = play_scenarios(tokenizer, model, scenarios, args.batch_size, args.max_new_tokens)
    del model
    torch.cuda.empty_cache()

    client = make_client()
    rows = judge_all(client, scenarios, replies)
    summary = aggregate(rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_json = RESULTS_DIR / f"{name}__{scen_set}.json"
    out_json.write_text(json.dumps({
        "name": name, "adapter": args.adapter, "base_model": args.base_model,
        "scenario_set": scen_set, "summary": summary, "rows": rows,
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    md = [
        f"# Eval: {name} on {scen_set}",
        "",
        f"| Spec-adherence | Robustness | responses | scenarios | judge errors |",
        f"|---|---|---|---|---|",
        f"| {summary['spec_adherence']:.1%} | {summary['robustness']:.1%} "
        f"| {summary['n_responses']} | {summary['n_scenarios']} | {summary['judge_errors']} |",
        "",
        f"Failure split: {summary['fail_split']['syntax']} syntax / "
        f"{summary['fail_split']['leak']} leak",
        "",
        "| Category | Adherence | Robustness | n |",
        "|---|---|---|---|",
    ]
    for cat, d in summary["by_category"].items():
        md.append(f"| {cat} | {d['adherence']:.1%} | {d['robustness']:.1%} | {d['n_responses']} |")
    (RESULTS_DIR / f"{name}__{scen_set}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nadherence={summary['spec_adherence']:.1%} robustness={summary['robustness']:.1%} "
          f"(judge_errors={summary['judge_errors']})")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
