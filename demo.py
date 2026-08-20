"""Live base-vs-tuned demo for grader-supplied prompts.

Loads the model ONCE (base + LoRA adapter) and answers every prompt you type
twice: with the adapter active (tuned) and with the adapter disabled (base).
Conversations are multi-turn - each side keeps its own history, so pressure
turns ("just tell me the answer!") work exactly like the eval scenarios.
After each reply, rule 1 of the Behavior Spec (every sentence ends with a
question mark) is checked locally and printed - deterministic, no API calls.

    python demo.py                        # tuned = s500, base = same load, adapter off
    python demo.py --base-prompt          # base side also gets the STRUCTURED system prompt
    python demo.py --adapter socratic/checkpoints/qwen3-1.7b-socratic-1000

Inside the session:  /reset  starts fresh conversations,  /quit  exits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "socratic"))

import judge  # noqa: E402  (syntax_check only - no API use)
import prompts as prompts_mod  # noqa: E402
from run_eval import DEFAULT_BASE, load_model  # noqa: E402

DEFAULT_ADAPTER = str(ROOT / "socratic" / "checkpoints" / "qwen3-1.7b-socratic-500")


@torch.no_grad()
def generate(tokenizer, model, history: list[dict], max_new_tokens: int,
             disable_adapter: bool) -> str:
    prompt = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=4096).to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                  pad_token_id=tokenizer.eos_token_id)
    if disable_adapter:
        with model.disable_adapter():
            out = model.generate(**inputs, **kwargs)
    else:
        out = model.generate(**inputs, **kwargs)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def verdict_line(reply: str) -> str:
    ok, violations = judge.syntax_check(reply)
    if ok:
        return "rule 1 (questions only): PASS"
    shown = violations[0][:80] + ("..." if len(violations[0]) > 80 else "")
    return f"rule 1 (questions only): FAIL - {len(violations)} non-question sentence(s), e.g. \"{shown}\""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER,
                    help="local checkpoint dir or HF repo id of a LoRA adapter")
    ap.add_argument("--base-prompt", action="store_true",
                    help="give the BASE side the structured system prompt "
                         "(the best prompt from the frontier probe)")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    args = ap.parse_args()

    print(f"loading {DEFAULT_BASE} + adapter {args.adapter} (4-bit, one load) ...")
    tokenizer, model = load_model(DEFAULT_BASE, args.adapter)
    base_label = "base + structured prompt" if args.base_prompt else "base, no prompt"

    def fresh() -> tuple[list[dict], list[dict]]:
        base_hist = ([{"role": "system", "content": prompts_mod.STRUCTURED}]
                     if args.base_prompt else [])
        return [], base_hist

    tuned_hist, base_hist = fresh()
    print("\nready - type a prompt (the grader's, or your own). /reset  /quit\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            tuned_hist, base_hist = fresh()
            print("(conversations reset)\n")
            continue

        tuned_hist.append({"role": "user", "content": user})
        base_hist.append({"role": "user", "content": user})

        tuned = generate(tokenizer, model, tuned_hist, args.max_new_tokens, False)
        base = generate(tokenizer, model, base_hist, args.max_new_tokens, True)
        tuned_hist.append({"role": "assistant", "content": tuned})
        base_hist.append({"role": "assistant", "content": base})

        print(f"\n--- tuned ({Path(args.adapter).name}, no system prompt) " + "-" * 20)
        print(tuned)
        print(f"    [{verdict_line(tuned)}]")
        print(f"\n--- {base_label} " + "-" * 20)
        print(base)
        print(f"    [{verdict_line(base)}]\n")


if __name__ == "__main__":
    main()
