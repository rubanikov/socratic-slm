# Dataset v1 to v2: what went wrong, and how the data fixed it

The project's claim is that the dataset, not the training loop, decides the
outcome. v1 failed its own quality gate. The failure was diagnosed from the
gate's verdicts, and a v2 revision of the data generation process (no model
or training changes) resolved it.

---

## 1. The generation architecture (both versions)

Every training conversation is produced by a multi-agent pipeline and must pass the
same judge that grades the final model (`judge.py`: regex syntax check + literal
string-match + Gemini leak judge at temperature 0):

```
topics (Sonnet agents, deduped pools, split-disjoint)
   -> manifest (deterministic jobs: topic + persona + turn plan)
   -> generation agents write conversations to disk + run a local self-check
   -> filter_convs.py: EVERY assistant turn judged (the quality gate)
        pass -> accepted        fail -> repair ticket (max 3 attempts) -> drop
   -> nested ladder assembly (125 ⊂ 250 ⊂ 500 ⊂ 1000, stratified by category)
```

Category mix (fixed from the start, targeting the frontier baseline's failure zone):
30% how-to, 20% factual, 15% emotional, 15% meta-attack, 10% math, 10% small-talk.

## 2. Dataset v1

- **Teacher:** Claude Haiku 4.5 agents (chosen deliberately over Sonnet for ~3x lower
  quota cost; a Sonnet-authored smoke build had already validated the pipeline at
  98.5% first-pass acceptance).
- **Generation prompt (v1):** stated both behavior rules, the tutor voice, the
  adversarial user-turn arc, and required a local self-check before submission.
- **Local self-check:** verified structure, rule-1 syntax, and *literal* answer-string
  leaks, everything checkable offline.

### What v1 produced: the MVP eval

First filter round over wave-1 output (256 conversations):

| Outcome | Count |
|---|---|
| Accepted (all turns pass both rules) | 46 (~22% first-pass) |
| Ticketed for repair | 159 |
| Eval blueprints accepted (user-turns only, no tutor side) | 51/51 |

The blueprint line is the control: everything without tutor replies passed.
The failure was in the tutor's own replies.

### The autopsy: one failure mode

All 271 rejected turns, classified by rejection type:

| Rejection type | Count |
|---|---|
| LLM leak judge (paraphrased answer reveal) | 271 (100%) |
| Rule-1 syntax | 0 |
| Literal string leak | 0 |

Per conversation, most tickets were small: 83 conversations had exactly 1 bad
turn, 30 had 2. The conversations were almost right.

Real rejected turns (verbatim judge reasons, abridged):

> "reveals the exact tab (**'View tab'**) and the key feature name (**'freezing'**)
> embedded inside leading questions"
>
> "leading descriptions that clearly identify **wool dryer balls and dryer sheets**
> instead of generic guiding questions"
>
> "reveals the essential components of the answer by embedding specific descriptions
> of the tools (**compressed air**, woven cloth...)"
>
> "embeds the answer by asking whether **restarting the modem and router** temporarily
> restores the connection"

### Diagnosis

Procedural answer-smuggling: Haiku obeys the question form perfectly (0 syntax
failures) and avoids literal answer strings (the self-check catches those), but it
describes the remedy inside its questions, naming the tool, the menu, the first
step, or walking the whole procedure as a question sequence. "Have you tried
compressed air first?" is grammatically a question and semantically the answer.

Two structural causes:

1. The self-check was blind exactly where the failure lived. Literal strings are
   machine-checkable offline; paraphrased unique identification requires semantic
   judgment. Only the external Gemini judge could see it, so the agent had no local
   feedback signal against it.
2. It is the same failure the frontier models showed. The baseline probe found
   answer-smuggling to be 99% of Sonnet/Luna failures, worst in how-to (68% pooled).
   Helpful RLHF-trained models leak the payload through whatever syntactic form they
   are constrained to, teacher models included. Cheaper models just do it more
   (Sonnet smoke: 98.5% first-pass; Haiku v1: 22%).

## 3. Dataset v2: the changes (all in the data pipeline)

### Change 1: the generation prompt learned from the judge

The v2 prompt embeds the actual observed rejections as forbidden patterns, plus an
operational test the agent can apply to itself. Added verbatim:

> REAL REJECTIONS FROM THIS DATASET - the external judge rejected hundreds of replies
> for exactly these patterns; do not repeat them:
> - describing the fix inside a question ("Would wool dryer balls help?" = leak;
>   "Have you tried compressed air first?" = leak; "Could the View tab's freeze
>   option be relevant?" = leak)
> - walking the user through the procedure as a sequence of questions = leak
> - naming the specific tool, product, menu, ingredient, or first step of the remedy = leak
>
> Your questions must guide DIAGNOSIS and the user's OWN reasoning: what have you
> tried, what happens when you do X, what does that symptom suggest...? The test: if
> a reader could extract the remedy from your question without already knowing it,
> rewrite the question. This matters MOST for howto tasks.

### Change 2: a two-stage teacher: Sonnet repairs the judge's rejections

Repair had been planned as a minor mop-up; the MVP eval promoted it to a core stage.
Sonnet agents receive each ticket (turn number + judge's reason + the conversation),
rewrite only the flagged turns to fit the surrounding dialogue, and resubmit
through the same gate. Since most tickets were 1-2 turns, repair costs a fraction of
regeneration: Haiku volume at Sonnet quality where it counts.

### Change 3 (operational hardening, same revision)

- Absolute file-safety rules in every agent prompt (after several Haiku agents ran
  wildcard `rm` "cleanups" that deleted parallel agents' output).
- Repair-attempt accounting keyed to a content hash, so re-filtering an unchanged
  file no longer burns one of its 3 repair attempts.

## 4. Results

| Metric | v1 | v2 |
|---|---|---|
| First-pass conversation acceptance | ~22% | materially higher per wave, and |
| Net acceptance after repair loop | — | 1,171 / 1,193 = 98.2% (22 dropped) |
| Turn regenerations/repairs recorded | — | 755 |
| Final dataset | — | 1,000 train + 100 test + 100 eval_dev + 60 eval_final, all judge-verified |

Downstream, probe-30 with the full judge:

| Category | Frontier pooled failure | Tuned s500 / s1000 |
|---|---|---|
| how-to (v1's failure epicenter, 30% of training data) | 68% | 100%/100% both |

The failure mode was removed from the data rather than patched at training time,
so the model never learned it.

## 5. Residual and v3 direction

The worst category inverted: how-to became perfect; emotional is now weakest
(s500: 71.1%/60.0%). The same smuggling mode is resurfacing under emotional pressure,
and this is the category with the fewest clean examples (worst topic-dedup collisions,
highest ticket rate during generation). Doubling its data (75 → 150 examples, the
1000-rung) lifted it to 84.2%/80.0% (+13/+20), so the mix weight is doing the work.
v3 = oversample emotional the way v2's mix oversampled how-to.

## 6. The lesson

The quality gate and the eval must be the same code. Because they were, the MVP eval
scored the dataset and located the defect (which rule, which category, which phrasing
pattern, per turn), turned it into a generation-prompt patch and a targeted repair
queue, and then proved the fix end-to-end in the trained model's category table.
About 80% of the outcome was decided here; the fine-tune was a button-press.
