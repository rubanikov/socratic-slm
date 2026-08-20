"""QLoRA fine-tune of Qwen3-1.7B on the Socratic-only SFT dataset.

Adapted from the proven SLLM/finetune/train_qlora.py (same machine, same stack):
4-bit nf4 quantization, LoRA r16/a32 on all attention+MLP projections,
assistant-only loss, bf16, gradient checkpointing. Verified settings for an
RTX 3070 Laptop 8GB.

Differences from the MedQA version:
- data = socratic/dataset/final/train_<N>.jsonl (chat messages, NO system
  prompt - the behavior must live in the weights)
- eval loss on the held-out test.jsonl (teacher-forced, assistant-only)
- live behavioral probe each eval: generate replies to eval_dev scenario
  openers and score rule-1 (syntax) pass rate plus the deterministic rule-2
  string-leak check with the project judge - deterministic and free (the
  full two-rule eval incl. the LLM leak judge runs after training)
- metrics stream to trackio (local, no network): one project across all
  ladder rungs, so runs are comparable side by side via `trackio show
  --project socratic-qlora`; alerts fire on non-finite loss and probe
  regressions
- keeps the best-eval-loss checkpoint (load_best_model_at_end), not the
  last-step one

Ladder usage (4 checkpoints at different dataset sizes):
  python socratic/train_qlora.py --train-size 500
  python socratic/train_qlora.py --train-size 1000
  python socratic/train_qlora.py --train-size 2500
  python socratic/train_qlora.py --train-size 5000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import trackio
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge  # noqa: E402  (rule-1 checker only; no API use in this script)

FINAL = HERE / "dataset" / "final"
DEFAULT_BASE = "Qwen/Qwen3-1.7B"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class BehaviorProbeCallback(TrainerCallback):
    """At every eval point, generate replies to eval_dev scenario openers and
    score them with the free half of the project judge: rule 1 (every sentence
    ends with '?') and the deterministic rule-2 string-leak check against each
    scenario's known answer variants. leak_hit_rate is an upper bound - weak
    variants (short/numeric) that the full judge would send to the LLM count
    as hits here. clean_rate = syntax pass AND no string hit. The full
    two-rule eval runs after training via the eval script."""

    def __init__(self, tokenizer, scenarios: list[dict], batch_size: int = 8):
        self.tokenizer = tokenizer
        self.scenarios = scenarios
        self.batch_size = batch_size
        self.records: list[dict] = []
        self.best_clean: float | None = None

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        was_training = model.training
        model.eval()
        side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        syntax_passes, leak_hits, clean, rows = 0, 0, 0, []
        try:
            for start in range(0, len(self.scenarios), self.batch_size):
                batch = self.scenarios[start:start + self.batch_size]
                prompts = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": s["turns"][0]}],  # NO system prompt
                        tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for s in batch
                ]
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=160, do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                gen = out[:, inputs["input_ids"].shape[1]:]
                for sc, text in zip(batch, self.tokenizer.batch_decode(gen, skip_special_tokens=True)):
                    ok, viol = judge.syntax_check(text)
                    hit = judge.string_leak_check(text, sc.get("expected_answers"))
                    syntax_passes += int(ok)
                    leak_hits += int(hit is not None)
                    clean += int(ok and hit is None)
                    rows.append({"opener": sc["turns"][0][:80], "reply": text[:200],
                                 "syntax_pass": ok, "leak_hit": hit})
        finally:
            self.tokenizer.padding_side = side
            if was_training:
                model.train()
        n = max(1, len(self.scenarios))
        syntax_rate, leak_rate, clean_rate = syntax_passes / n, leak_hits / n, clean / n
        eval_loss = (metrics or {}).get("eval_loss")
        self.records.append({"step": state.global_step, "epoch": state.epoch,
                             "eval_loss": eval_loss, "syntax_pass_rate": syntax_rate,
                             "leak_hit_rate": leak_rate, "clean_rate": clean_rate,
                             "n": len(self.scenarios), "samples": rows[:4]})
        trackio.log({"probe/syntax_pass_rate": syntax_rate,
                     "probe/leak_hit_rate": leak_rate,
                     "probe/clean_rate": clean_rate}, step=state.global_step)
        if self.best_clean is not None and clean_rate < self.best_clean - 0.2:
            trackio.alert(
                title="Probe regression",
                text=f"clean_rate {clean_rate:.1%} at step {state.global_step}, "
                     f"down from best {self.best_clean:.1%}",
                level=trackio.AlertLevel.WARN,
            )
        if self.best_clean is None or clean_rate > self.best_clean:
            self.best_clean = clean_rate
        loss_str = f"{eval_loss:.4f}" if eval_loss is not None else "n/a"
        print(f"[probe@step {state.global_step}] eval_loss={loss_str} "
              f"rule1_pass={syntax_rate:.1%} leak_hits={leak_rate:.1%} "
              f"clean={clean_rate:.1%} (n={len(self.scenarios)})")


class StepLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"[train@step {state.global_step} epoch {logs.get('epoch', 0):.3f}] "
                  f"loss={logs['loss']:.4f} lr={logs.get('learning_rate', 0):.2e}")
            if not math.isfinite(logs["loss"]):
                trackio.alert(
                    title="Non-finite training loss",
                    text=f"loss={logs['loss']} at step {state.global_step} - run is broken",
                    level=trackio.AlertLevel.ERROR,
                )


def total_steps(train_size: int, epochs: float, effective_batch: int) -> int:
    steps_per_epoch = math.ceil(train_size / effective_batch)  # Trainer rounds up per epoch
    return max(1, round(steps_per_epoch * epochs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-size", type=int, required=True,
                    help="ladder rung: reads dataset/final/train_<N>.jsonl")
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--val-size", type=int, default=48, help="test.jsonl slice for eval loss")
    ap.add_argument("--probe-size", type=int, default=24, help="eval_dev openers for the rule-1 probe")
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir or (HERE / "checkpoints" / f"qwen3-1.7b-socratic-{args.train_size}")
    train_rows = load_jsonl(FINAL / f"train_{args.train_size}.jsonl")
    test_rows = load_jsonl(FINAL / "test.jsonl")
    dev_scenarios = load_jsonl(FINAL / "eval_dev.jsonl")

    rng = random.Random(12345)
    rng.shuffle(test_rows)
    val_rows = test_rows[: args.val_size]
    probe_scenarios = dev_scenarios[: args.probe_size]

    train_ds = Dataset.from_list([{"messages": r["messages"]} for r in train_rows])
    val_ds = Dataset.from_list([{"messages": r["messages"]} for r in val_rows])
    print(f"train={len(train_ds)} val={len(val_ds)} probes={len(probe_scenarios)} -> {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    effective_batch = 2 * 8
    steps = total_steps(len(train_ds), args.epochs, effective_batch)
    cadence = max(3, steps // 5)  # ~5 eval/save points per run
    # warmup_ratio no longer exists in transformers 5.x TrainingArguments;
    # ~3% of steps, the ratio the removed default implemented
    warmup_steps = max(2, round(0.03 * steps))
    print(f"{steps} total steps, eval/save every {cadence}, warmup {warmup_steps}")

    # one project across all ladder rungs -> side-by-side dashboard comparison;
    # local SQLite only (no space_id), so nothing leaves the machine.
    # project/run_name are repeated in SFTConfig so the TrackioCallback's own
    # init(resume="allow") attaches to this run instead of opening a second one.
    project = "socratic-qlora"
    run_name = f"qwen3-1.7b-s{args.train_size}-seed{args.seed}"
    trackio.init(
        project=project,
        name=run_name,
        config={
            "base_model": args.base_model,
            "train_size": args.train_size,
            "epochs": args.epochs,
            "lora": "r16-a32-d0.05-all-linear",
            "quant": "nf4-double-bf16",
            "learning_rate": 2e-4,
            "effective_batch": effective_batch,
            "max_length": args.max_length,
            "seed": args.seed,
        },
    )

    probe = BehaviorProbeCallback(tokenizer, probe_scenarios)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=cadence,
            save_strategy="steps",
            save_steps=cadence,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["trackio"],
            project=project,
            run_name=run_name,
            max_length=args.max_length,
            assistant_only_loss=True,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            seed=args.seed,
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[probe, StepLogger()],
    )
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))  # best-eval-loss weights (load_best_model_at_end)
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "training_log.json").write_text(json.dumps({
        "base_model": args.base_model,
        "train_size": args.train_size,
        "epochs": args.epochs,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "trainer_log": trainer.state.log_history,
        "behavior_probe": probe.records,
    }, indent=1), encoding="utf-8")
    try:
        trackio.finish()
    except RuntimeError:
        pass  # TRL's TrackioCallback already finished the shared run at train end
    print(f"saved adapter + training_log.json to {out_dir}")
    print("dashboard: trackio show --project socratic-qlora")


if __name__ == "__main__":
    main()
