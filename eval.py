"""One-command eval against the Behavior Spec.

    python eval.py --model <model> --eval-set <path>     # one model
    python eval.py --eval-set <path>                     # ALL models -> full results table

--model accepts any of:
    base                                   the untrained base model (Qwen/Qwen3-1.7B)
    a local checkpoint dir                 e.g. socratic/checkpoints/qwen3-1.7b-socratic-500
    an HF repo id of a LoRA adapter        e.g. user/qwen3-1.7b-socratic-500
    an HF repo id of a base model          evaluated as-is, no adapter

Omitting --model evaluates the base model plus every checkpoint under
socratic/checkpoints/qwen3-1.7b-socratic-* and regenerates the full results
table (printed + written to socratic/eval_results/RESULTS_TABLE__<set>.md).

Every reply is graded by the project judge (regex syntax + string-match +
google/gemini-3.7-flash leak judge) - the exact grader used for the frontier
baseline. Requires OPENROUTER_API_KEY in .env and a CUDA GPU.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "socratic"))

import os  # noqa: E402

import torch  # noqa: E402
from run_eval import (  # noqa: E402
    DEFAULT_BASE, RESULTS_DIR, aggregate, judge_all, load_model, play_scenarios,
)
from generate_dataset import make_client  # noqa: E402

# private HF repos authenticate via HF_TOKEN; mirror the project's key name
if os.environ.get("HUGGINGFACE_API_KEY") and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_API_KEY"]


def resolve(model_arg: str) -> tuple[str, str | None, str]:
    """-> (base_model, adapter_or_None, display_name)"""
    if model_arg == "base":
        return DEFAULT_BASE, None, "base"
    p = Path(model_arg)
    if p.exists():
        if (p / "adapter_config.json").exists():
            return DEFAULT_BASE, str(p), p.name
        return str(p), None, p.name  # local full model
    # HF repo id: adapter repos carry adapter_config.json, base repos don't
    try:
        from huggingface_hub import file_exists
        if file_exists(model_arg, "adapter_config.json", token=os.environ.get("HF_TOKEN")):
            return DEFAULT_BASE, model_arg, model_arg.split("/")[-1]
    except Exception:  # noqa: BLE001 - offline/no-auth: assume base repo
        pass
    return model_arg, None, model_arg.split("/")[-1]


def eval_one(model_arg: str, scenarios: list[dict], scen_set: str,
             batch_size: int, max_new_tokens: int) -> dict:
    base, adapter, name = resolve(model_arg)
    print(f"\n=== {name} (base={base}, adapter={adapter or '-'}) on {scen_set} ===")
    tokenizer, model = load_model(base, adapter)
    replies = play_scenarios(tokenizer, model, scenarios, batch_size, max_new_tokens)
    del model
    torch.cuda.empty_cache()
    rows = judge_all(make_client(), scenarios, replies)
    summary = aggregate(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{name}__{scen_set}.json").write_text(json.dumps({
        "name": name, "base_model": base, "adapter": adapter,
        "scenario_set": scen_set, "summary": summary, "rows": rows,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    # full per-model results table - the literal `--model X --eval-set Y` command
    # regenerates this from nothing
    md = [
        f"# Results — {name} on {scen_set} ({summary['n_scenarios']} scenarios)",
        "",
        "| Spec-adherence | Robustness | responses | scenarios | syntax fails | leak fails | judge errors |",
        "|---|---|---|---|---|---|---|",
        f"| {summary['spec_adherence']:.1%} | {summary['robustness']:.1%} "
        f"| {summary['n_responses']} | {summary['n_scenarios']} "
        f"| {summary['fail_split']['syntax']} | {summary['fail_split']['leak']} "
        f"| {summary['judge_errors']} |",
        "",
        "| Category | Adherence | Robustness | n |",
        "|---|---|---|---|",
    ]
    for cat, d in summary["by_category"].items():
        md.append(f"| {cat} | {d['adherence']:.1%} | {d['robustness']:.1%} | {d['n_responses']} |")
    table = "\n".join(md) + "\n"
    (RESULTS_DIR / f"{name}__{scen_set}.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    return {"name": name, **summary}


def size_key(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1  # base sorts first


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="base | local checkpoint dir | HF repo id (omit = all models)")
    ap.add_argument("--eval-set", type=Path,
                    default=ROOT / "socratic" / "dataset" / "final" / "eval_dev.jsonl")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    args = ap.parse_args()

    scenarios = [json.loads(l) for l in args.eval_set.read_text(encoding="utf-8").splitlines() if l.strip()]
    scen_set = args.eval_set.stem
    print(f"eval set: {args.eval_set} ({len(scenarios)} scenarios)")

    if args.model:
        eval_one(args.model, scenarios, scen_set, args.batch_size, args.max_new_tokens)
        return

    # full table: base + every checkpoint, ascending dataset size
    targets = ["base"] + sorted(
        (str(d) for d in (ROOT / "socratic" / "checkpoints").glob("qwen3-1.7b-socratic-*")
         if (d / "adapter_model.safetensors").exists() and d.name.split("-")[-1].isdigit()),
        key=lambda d: size_key(Path(d).name),
    )
    results = [eval_one(t, scenarios, scen_set, args.batch_size, args.max_new_tokens)
               for t in targets]

    lines = [
        f"# Results table — {scen_set} ({len(scenarios)} scenarios)",
        "",
        "| Model | Spec-adherence | Robustness | responses | scenarios | syntax fails | leak fails | judge errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['spec_adherence']:.1%} | {r['robustness']:.1%} "
            f"| {r['n_responses']} | {r['n_scenarios']} "
            f"| {r['fail_split']['syntax']} | {r['fail_split']['leak']} | {r['judge_errors']} |"
        )
    table = "\n".join(lines) + "\n"
    out = RESULTS_DIR / f"RESULTS_TABLE__{scen_set}.md"
    out.write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
