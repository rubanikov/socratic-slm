# Socratic-Only SLM

Instill one falsifiable behavior into a small open model — and prove it, end to end.

**Behavior Spec:** every sentence the model outputs ends with a question mark, and it
never reveals the answer — not stated, not embedded inside a question, not via a
uniquely-identifying hint. A 1.7B model fine-tuned on ~500 judge-filtered conversations
holds this constraint (with **no system prompt**) better than Claude Sonnet 5 or
GPT-5.6-Luna running their best prompts. All project code lives in [`socratic/`](socratic/).

## Results at a glance

| Model | Prompt | Spec-adherence | Robustness |
|---|---|---|---|
| Qwen3-1.7B-socratic-1000 | none | **96.8%** | **93.3%** |
| Qwen3-1.7B-socratic-500 | none | 93.1% | 83.3% |
| gpt-5.6-luna (best of 3 prompts) | structured | 89.0% | 66.7% |
| claude-sonnet-5 (best of 3 prompts) | structured | 76.1% | 43.3% |
| Qwen3-1.7B base | none | 0.0% | 0.0% |

(Head-to-head on identical 30 adversarial conversations, identical judge.)
Full story: [`socratic/FINAL_RESULTS.md`](socratic/FINAL_RESULTS.md) ·
baseline probe: [`socratic/RESULTS.md`](socratic/RESULTS.md) ·
dataset v1→v2 iteration: [`socratic/DATASET_ITERATION.md`](socratic/DATASET_ITERATION.md) ·
spec + judge policy: [`socratic/BEHAVIOR_SPEC.md`](socratic/BEHAVIOR_SPEC.md)

**Data-efficiency curves** (spec-adherence and robustness vs training-set size,
with the best-prompted-frontier reference line): plotted in
[`socratic/FINAL_RESULTS.md` §4](socratic/FINAL_RESULTS.md), PNGs + plotting
script in [`socratic/figures/`](socratic/figures/). The behavior locks in at
~250 examples and saturates at ~500.

## Setup

```bash
pip install -r requirements.txt        # pinned training stack (torch 2.13/cu126, trl 1.10, peft, bitsandbytes)
cp .env.example .env                   # then fill in:
#   OPENROUTER_API_KEY=...             # judge + frontier baseline calls
#   OPENROUTER_MODEL1=anthropic/claude-sonnet-5    # baseline contestant
#   OPENROUTER_MODEL2=google/gemini-3.7-flash      # the LLM leak judge
#   OPENROUTER_MODEL3=openai/gpt-5.6-luna          # baseline contestant
```

Hardware: any 8GB+ NVIDIA GPU (developed on an RTX 3070 Laptop 8GB; each training
rung takes ~20-30 min).

## How to run (all commands from the repo root)

### 1. Frontier baseline probe (2 models x 3 prompt strategies x 30 scenarios)

```bash
python socratic/run_probe.py --limit 2     # smoke test
python socratic/run_probe.py               # full grid -> socratic/RESULTS.md + results.csv
```

### 2. Dataset

The shipped dataset is in `socratic/dataset/final/`:
nested train ladder `train_{125,250,500,1000}.jsonl` · `test.jsonl` (100) ·
`eval_dev.jsonl` (120) · `eval_final.jsonl` (80, frozen for final grading).
All conversations passed the same judge that grades the evals.

To inspect/rebuild: `python socratic/dataset_plan.py status`,
`python socratic/generate_dataset.py --phase splits --train-size 2000 --test-size 100 --eval-dev 120 --eval-final 80`.
(The eval files are curated supersets and are never overwritten by assembly.)
(Generation itself ran via Claude-agent waves — `socratic/gen_wave.js` + `filter_convs.py`;
see `DATASET_ITERATION.md`.)

### 3. Train (QLoRA, 4 checkpoints at different dataset sizes)

```bash
python socratic/train_qlora.py --train-size 125
python socratic/train_qlora.py --train-size 250
python socratic/train_qlora.py --train-size 500
python socratic/train_qlora.py --train-size 1000
python socratic/train_qlora.py --train-size 2000   # extension rung (once train_2000.jsonl is assembled)
```

Adapters + per-step checkpoints + `training_log.json` land in
`socratic/checkpoints/qwen3-1.7b-socratic-<N>/`. Live metrics: local trackio project
`socratic-qlora` (`trackio show --project socratic-qlora`; on Windows the CLI is at
`%LOCALAPPDATA%\..\Local\Python\pythoncore-3.14-64\Scripts\trackio.exe`).

### 4. Evaluate — one command (same judge as the baseline)

The graders' one-command form — regenerates the full results table from nothing
(downloads the adapter from the Hub, plays every scenario, judges every reply,
writes the markdown table):

```bash
python eval.py --model rubanikov/qwen3-1.7b-socratic-500 --eval-set socratic/dataset/final/eval_dev.jsonl
```

Published checkpoints (LoRA adapters on `Qwen/Qwen3-1.7B`), all **public**:
`rubanikov/qwen3-1.7b-socratic-125` · `-250` · `-500` · `-1000`
(exact Hub revisions pinned in [`socratic/MODEL_HASHES.json`](socratic/MODEL_HASHES.json)).

Other forms:

```bash
# local checkpoint dir instead of a Hub id:
python eval.py --model socratic/checkpoints/qwen3-1.7b-socratic-500 --eval-set socratic/dataset/final/eval_dev.jsonl
python eval.py --model base --eval-set socratic/scenarios.jsonl                # untrained baseline, head-to-head set

# regenerate the FULL results table from nothing (base + every checkpoint):
python eval.py --eval-set socratic/dataset/final/eval_dev.jsonl

# THE final one-shot on the frozen set (run once, when ready to lock the number):
python eval.py --eval-set socratic/dataset/final/eval_final.jsonl
```

The four final ladder adapters are committed in `socratic/checkpoints/`, so the
all-models form above works on a fresh clone with no Hub download; `--model
<hf-repo-id>` pulls the identical weights from the Hub instead (byte-identical —
same SHA-256, see provenance below).

Table lands in `socratic/eval_results/RESULTS_TABLE__<set>.md`; per-model
transcripts + verdicts in `socratic/eval_results/<name>__<set>.json`.
(The frontier-baseline rows regenerate separately with their own single command:
`python socratic/run_probe.py`.)

Results -> `socratic/eval_results/<name>__<set>.{json,md}`.
Metrics: **spec-adherence** = passing replies / all replies; **robustness** =
conversations with zero failing replies / all conversations.

### 5. Live demo — grader-supplied prompt, base vs tuned

```bash
python demo.py                 # tuned s500 vs base, side by side
python demo.py --base-prompt   # harder comparison: base also gets its best (structured) prompt
```

One model load, then type anything: each prompt is answered twice — adapter
active (tuned) and adapter disabled (base) — with independent multi-turn
histories, so pressure turns ("just tell me!") work like the eval scenarios.
After every reply, rule 1 of the spec (questions only) is checked locally and
printed as PASS/FAIL. No API calls, runs offline on the 8GB GPU.
`/reset` starts fresh conversations, `/quit` exits.

## Models used

| Role | Model | Access |
|---|---|---|
| Student (fine-tuned) | `Qwen/Qwen3-1.7B` | local, 4-bit QLoRA |
| Judge (rule 2 leak) | `google/gemini-3.7-flash`, temp 0 | OpenRouter |
| Baseline contestants | `anthropic/claude-sonnet-5`, `openai/gpt-5.6-luna` | OpenRouter |
| Dataset teachers | Claude Haiku 4.5 (generation) + Claude Sonnet (repairs), via Claude-Code agent waves | subscription |

## Model provenance & pinned versions

Every reported number is reproducible against a frozen state, pinned by two hashes:

- **Weights** — each adapter's exact HF Hub revision (commit hash) is recorded in
  [`socratic/MODEL_HASHES.json`](socratic/MODEL_HASHES.json) (`hub_revisions`),
  alongside SHA-256 digests of the adapter files themselves.
- **Eval code** — the git commit hash of this repository, cited in the submission.

Machine-readable copy of all hashes: [`socratic/MODEL_HASHES.json`](socratic/MODEL_HASHES.json).

**Base model** — `Qwen/Qwen3-1.7B`, HF revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`:

| File | SHA-256 |
|---|---|
| model-00001-of-00002.safetensors | `169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5` |
| model-00002-of-00002.safetensors | `912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9` |

**Fine-tuned adapters** (`adapter_model.safetensors` in each checkpoint dir):

| Checkpoint | SHA-256 |
|---|---|
| qwen3-1.7b-socratic-125 | `5115d61143e91f7a6c708430b32b7fd8841f266833ca6aeb61b3c7c8035a555a` |
| qwen3-1.7b-socratic-250 | `f027bad677e4e815aa5f5112d346fa7550c6e93c86191a04af60ffbe47df6907` |
| qwen3-1.7b-socratic-500 | `f0801cd9bdec227e6900511b1e33779c642e21a62ad2006b329ffe35dc340a1d` |
| qwen3-1.7b-socratic-1000 | `4ae9eeb4a306d68b60f491dc4a1ef2d8b582c4739489f549fd8f98d182d692aa` |

## Repo layout

```
socratic/            all project code, data, results (see file map in FINAL_RESULTS.md)
socratic/checkpoints/  final ladder adapters + training_log.json per rung
                       (committed; mid-training resume state + smoke runs gitignored)
requirements.txt     pinned python deps
archive_pre_socratic/  earlier, unrelated work (gitignored)
build_debris/          scratch files from generation agents (gitignored)
```
