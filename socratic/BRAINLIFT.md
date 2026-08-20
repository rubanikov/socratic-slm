# BrainLift — Socratic SLM: Behavior in the Weights

*The thinking behind `socratic/`: a falsifiable behavior spec (every sentence a question, never reveal the answer), instilled into Qwen3-1.7B from judge-filtered teacher data, proven against the measured prompt ceiling of frontier models.*

---

## Purpose

Prove — with a judge a stranger can rerun — that one falsifiable behavior (respond only in questions, never reveal the answer) which collides head-on with trained helpfulness can be instilled in a 1.7B model's weights from a few hundred judge-filtered examples, more reliably than the best-prompted frontier model. The dataset, not the training run, is the artifact.

**The core tension:** the constraint competes with RLHF helpfulness — the payload wants out. The assignment's gate ("a well-prompted base model can't already do it reliably") is only passable by a behavior that collides with what the model is trained to do; this project's collision surfaces as **answer-smuggling**: the answer leaking *inside* the tutor's questions, from the very first benign turn.

## North Star

> **"Does this make the data→behavior claim harder to fake and easier to falsify?"**

This question resolves every design dispute: judge strictness (fixed for all runs, applied identically to base and tuned), no system prompt in training data, one judge for data gate and eval alike, frozen `eval_final`, fix-failures-in-the-data-never-in-the-config.

## Scope

**In scope:** the behavior spec + judge design; the prompt-ceiling probe; the teacher-data pipeline (generation prompt, judge gate, repair loop); the data-efficiency ladder; the failure-mode story (answer-smuggling; the how-to→emotional inversion).

**Out of scope:** actual student learning outcomes (no pedagogy-efficacy claims are made); tutoring product design; capability benchmarks; all pre-socratic work outside `socratic/`.

---

## DOK 4 — Spiky Points of View

### SPOV 1: A constraint that must ship bare lives in weights — prompting is a measured ceiling, not a path, and anything the model leans on at inference is a dependency, not a behavior.

The prompt-ceiling probe put the best frontier combo (gpt-5.6-luna + structured self-check prompt) at 89.0% adherence / 66.7% robustness — and the failure rate was *flat from turn 1*, meaning the leak isn't pressure-drift that better pressure-handling prompts could fix; it's RLHF helpfulness delivering the payload on benign turns. Qi et al. (2024) explain why prompted and shallowly-trained constraints collapse; LIMA explains why weights-level instillation is cheap (format/style is what fine-tuning teaches best). The scaling curve confirms both: 500 judge-filtered conversations took a bare 1.7B from 0%/0% to 97.2%/93.0%, with *no system prompt anywhere* — and the bare tuned model beat every prompted frontier combo on the identical 30 conversations.

The concession that keeps this honest: an inference-time judge-and-resample wrapper could exceed these numbers *where it's affordable* — but that imports an external LLM call per turn, with cost, latency, and failure modes of its own, and it's scaffolding around a model that still wants to leak. When deployment is bare — edge, consumer GPU, no network judge — weights are the only place the behavior can live. The probe is the arbitration mechanism between the prompting and tuning camps: measure the ceiling first; tune only when it sits below the reliability bar.

**Named opponents:** "a strong system prompt is enough" practitioners (the grader's likely prior); the inference-scaffolding camp (judge-and-resample loops).

**Design rule:** Zero system prompts in training data; the tuned model is evaluated bare; every reliability claim ships with the measured prompt ceiling beside it.

### SPOV 2: The teacher cannot verify itself — prompt engineering is the primary quality lever, but no example enters the dataset except through an external judge, the same judge that grades the final eval.

v1's raw teacher generations passed the judge only ~22% first-time, and the autopsy of the 271 rejected turns found 100% semantic answer-smuggling — of which the teacher-side self-check caught *zero* (while catching every literal string-leak). The leak is invisible from inside: the same helpfulness that produces "Would wool dryer balls help?" makes it feel like good tutoring to its author. This is the family-trait insight operationalized — the teacher shares the contestants' RLHF flaw, so no teacher swap avoids the gate.

The v2 generation-prompt rewrite (concrete forbidden patterns + the extraction test: "if a reader could extract the remedy from your question without already knowing it, rewrite the question") did the heavy lifting, raising net acceptance to 98.2% — and the external judge still caught the residue: 755 turn-regenerations, 22 conversations dropped rather than patched. The downstream proof that the gate transfers to behavior: how-to, the frontier's worst category (68% pooled failure), scored 100%/100% in the tuned model — the failure mode was removed from the data, so the model never learned it. Using the *same* judge for the data gate and the eval keeps the whole claim honest: what's filtered is exactly what's measured.

**Named opponents:** the Alpaca/Vicuna-era "just generate from a frontier model" practice; curation-by-feel (LIMA-style hand selection); "the teacher's own self-check suffices."

**Design rule:** The judge exists before any data is generated; generation prompts iterate against observed judge rejections; every example passes the eval judge to enter the dataset; unrepairable conversations are dropped, never hand-patched.

---

## Experts

Curated list — whose work informs (or most sharply challenges) this design:

| Expert | Camp | Why on this list |
|---|---|---|
| **Kenneth Koedinger** (CMU) | Calibrated assistance | The assistance dilemma (Koedinger & Aleven 2007) is the sharpest *opponent* of the spec: withholding-vs-giving is an optimization problem, not a doctrine. Naming him keeps the SPOVs honest — the spec trades pedagogical contingency for falsifiability, knowingly. |
| **Manu Kapur** (ETH Zurich) | Struggle-first | Productive Failure gives struggle-first its meta-analytic backing (g≈0.36; ≈0.8 high-fidelity) — the half-ally who would still demand the consolidation phase this tutor never gives. |
| **Paul Kirschner & John Sweller** | Direct/guided instruction | The guidance-first canon (2006 minimal-guidance critique; Cognitive Load Theory). They would call the spec pedagogically harmful for novices — the strongest external critique the document must survive. |
| **Chunting Zhou et al.** (Meta, LIMA) | Distillation optimists | The Superficial Alignment Hypothesis predicted this project's 500-example saturation almost on the nose: style/format is cheap to instill; knowledge comes from pretraining. |
| **Arnav Gudibande & Eric Wallace** (Berkeley) | Distillation skeptics | "The False Promise of Imitating Proprietary LLMs" is the result this project must be reconciled with: imitation transfers style, not capability. The reconciliation (a constraint is neither) is DOK 3 Insight 3. |
| **ETH RL-tutor group (Sachan et al.) / Google LearnLM** | Pedagogical fine-tuning works | The published cousins: LearnLM proves tutor behavior trains into weights at frontier scale; the ETH group's "solution leakage" metric is the nearest named relative of answer-smuggling — and their SFT-baseline capability losses mark a hazard this project did not measure. |

---

## Grader Cognitive Map

The user whose head is mapped: the **ML peer / grader**, arriving with the prior *"just use a strong system prompt"* (or *"tiny-model imitation doesn't really work"*), walking the artifact trail from spec to live test.

### Phase 1 — The Spec (MEDIUM load)

| Dimension | Analysis |
|---|---|
| **User sees** | `BEHAVIOR_SPEC.md`: two pass/fail rules, two metrics, the judge-policy bullets |
| **User thinks** | "Questions-only, no reveals — cute. Is this *hard*, or does a system prompt solve it? Can a judge really call 'uniquely-identifying hint' pass/fail?" |
| **Decision point** | Cognitive: do I accept this spec as falsifiable and non-trivial enough to keep reading? |
| **Cognitive load** | MEDIUM — 4 items: Rule 1 (syntax), Rule 2 (no leak), adherence definition, robustness definition |
| **Confusion risk** | The strictness edge cases (`?!` fails, line-break = sentence boundary, lone initials) read as arbitrary — possibly rigged against contestants |
| **Design response** | Every policy decision is stated as fixed-for-all-runs and applied identically to base and tuned; the hint clause carries a worked example ("Isn't it Paris?" = FAIL) |

### Phase 2 — The Ceiling (HIGH load)

| Dimension | Analysis |
|---|---|
| **User sees** | `RESULTS.md`: the 6-combo grid, per-category breakdown, failure-mode paragraph |
| **User thinks** | *"Were these 30 scenarios written to make frontier models fail?* Pressure turns, topic shifts — is this a fair test or a rigged one?" |
| **Decision point** | Cognitive: is the plateau real evidence of a prompt ceiling, or an artifact of scenario design? |
| **Cognitive load** | HIGH — 2 models × 3 strategies, 2 metrics, 6 categories, plus the failure-mode claim (4+ items) |
| **Confusion risk** | Cherry-picking suspicion — scenario provenance is invisible inside a results table |
| **Design response** | `scenarios.jsonl` is committed and readable; user turns identical across all runs; the *same* 30 scenarios later grade the tuned model head-to-head, so any rigging would handicap both sides equally; `validate_scenarios.py` gives offline checkability |

### Phase 3 — The Pipeline (HIGH load)

| Dimension | Analysis |
|---|---|
| **User sees** | `FINAL_RESULTS.md` §2/§6: teacher, judge gate, repair loop, splits, the v1→v2 story |
| **User thinks** | *"The same judge filters the training data AND grades the eval — isn't the model just optimized to one judge's quirks? And are the eval scenarios paraphrases of the training conversations?"* |
| **Decision point** | Cognitive: is the headline number honest, or optimized to its own grader? |
| **Cognitive load** | HIGH — teacher identity, judge identity, gate mechanics, four split sizes, disjointness claims (5+ items) |
| **Confusion risk** | Judge circularity and train/eval leakage — both have answers, but the answers are scattered across documents |
| **Design response** | Three-family separation stated in one place (Claude teaches, Gemini judges, Sonnet/Luna contest); splits topic-disjoint and prompt-only; `eval_final` frozen and never run during dev; the staff held-out set is the external answer to circularity — the harness runs unmodified on scenarios the author never saw |

### Phase 4 — The Replay (MEDIUM load)

| Dimension | Analysis |
|---|---|
| **User sees** | The reproduction block: one command per artifact; checkpoint paths; `.env` key requirements |
| **User thinks** | "Will this run on *my* machine? What do I need — GPU, OpenRouter key, HF cache?" |
| **Decision point** | First spatial/procedural (which command, where keys go), then cognitive: do the regenerated numbers match the table? |
| **Cognitive load** | MEDIUM — command, adapter path, eval-set path, required keys (4 items) |
| **Confusion risk** | Environment friction (8GB-GPU assumption, judge needs an OpenRouter key) — a failed env gets misread as a failed claim |
| **Design response** | One-command eval (`run_eval.py --adapter …`); judge verdicts cached at temperature 0 so re-judging is cheap and deterministic; judge errors get their own column and are never silent passes |

### Phase 5 — The Live Test (MEDIUM load)

| Dimension | Analysis |
|---|---|
| **User sees** | A live chat with the tuned model (base beside it), free to type anything |
| **User thinks** | *"I'll make it leak"* — emotional appeal, "my exam is tomorrow, just tell me" — *"and let me try something off-distribution: code? another language?"* |
| **Decision point** | Cognitive: does the behavior hold on input I chose — and does it generalize past the trained pocket? |
| **Cognitive load** | MEDIUM — their attack plan plus the two rules they're checking against (3 items) |
| **Confusion risk** | An off-distribution wobble or an emotional-category leak gets read as disproof of the whole claim rather than as a mapped boundary |
| **Design response** | The failure map is *published*, not hidden — emotional disclosed as weakest with numbers (71.1% at s500, 84.2% at s1000); the spec scopes the claim (six categories), so off-distribution probing tests a stated boundary; the unicode-aware judge policy shows language tricks were anticipated |

## Cognitive Load Analysis

| Phase | Load | Working-memory items |
|---|---|---|
| 1 — The Spec | MEDIUM | 2 rules + 2 metric definitions |
| 2 — The Ceiling | HIGH | 6 combos + 2 metrics + 6 categories + failure claim |
| 3 — The Pipeline | HIGH | teacher, judge, gate mechanics, 4 splits, disjointness |
| 4 — The Replay | MEDIUM | command, adapter path, eval-set path, keys |
| 5 — The Live Test | MEDIUM | attack plan + 2 rules |

**Load principles applied:**
- **One rubric everywhere** — spec = data gate = eval criterion = the project's POV, and one judge grades all of it. The grader holds a single judge in mind, not three.
- **Metrics defined once** (in the spec), referenced by name everywhere else.
- **Mental-model reuse** — the same 30 scenarios from Phase 2 reappear in the head-to-head, so Phase 5's comparison rides on a structure the grader already built.
- **Procedural collapse** — one-command reproduction turns Phase 4 from a setup project into a single decision (do the numbers match?).

**Confusion points → mitigations:**

| Confusion point | Mitigation |
|---|---|
| Strictness looks rigged | Judge policy decisions stated as fixed-for-all-runs, applied identically to base and tuned |
| Scenarios cherry-picked | `scenarios.jsonl` committed; identical user turns across runs; same scenarios grade the tuned model |
| Judge circularity | Third-model-family judge; staff held-out set runs the unmodified harness |
| Train/eval leakage | Topic-disjoint splits; prompt-only evals; frozen `eval_final` |
| Env failure ≠ claim failure | One-command eval; cached deterministic judge; judge errors reported, never silent passes |
| Off-distribution wobble read as disproof | Published failure map (emotional weakest, with numbers); claim scoped to six categories |

---

## DOK 3 — Insights

### Insight 1: Answer-leaking is a family trait of RLHF training, not a weakness of any particular model — so the filter gate, not model choice, is the only lever.

The probe and the data pipeline failed identically: 98–99% of frontier contestant failures were Rule-2 leaks, and 100% of the teacher's first-pass rejections were Rule-2 leaks — the same answer-smuggled-inside-a-question mode, concentrated in the same category (how-to), across different vendors and different roles (contestant vs. teacher). The shared cause is RLHF helpfulness, not architecture or scale. The strong version: you cannot procure your way around it — a non-RLHF base model can't produce coherent multi-turn Socratic dialogue at all, so any *usable* teacher carries the leak gene; the helpfulness that makes generation possible is the same training that makes it leak. This goes beyond Gudibande's warning that imitation copies teacher style: here the teacher's flaw is the very thing being distilled *against*, which is why the judge gate is structurally unavoidable rather than a quality nicety.

### Insight 2: Empathy is a second helpfulness gravity — and the gravity causes the data scarcity.

Emotional pressure reframes withholding as cruelty ("I hear you, and I want to help…" — the tuned model's own drift opener in `emo_02`), recruiting empathy training as a second pull alongside helpfulness. The mechanism shows at *both ends* of the pipeline: in generation (emotional had the highest repair-ticket rate and the worst topic-dedup collisions → fewest clean examples) and at inference (emotional is the tuned model's weakest category: 71.1% at s500). Data scarcity isn't the rival explanation — it's the transmission mechanism of the gravity: empathy makes clean emotional examples hardest to *generate*, which makes the behavior weakest to *hold*. The +75-example intervention (+13 adherence / +20 robustness) proves the lever without closing the gap. And no literature covers this: pedagogy handles affect via human tutors, ML safety handles pressure as adversarial jailbreaks — a sincere emotional appeal is neither. This is the unmapped frontier and the v3 target: oversample emotional exactly the way v2 oversampled how-to.

### Insight 3: Every surface metric flatters a constraint; the hard half is semantic depth under pressure — robustness is the only honest number.

Question-syntax was mastered instantly by everyone: frontier models across all six prompt combos (5 of 297 failures were syntax), the raw teacher (0 syntax rejections), even the unstable s125 checkpoint. All real difficulty lives in Rule 2, the semantic half. s125 is the flagship: 56.4% adherence, 6.7% robustness — imitating the form often, surviving a conversation almost never; adherence without robustness is the signature of an under-trained constraint. This reframes LIMA-vs-Gudibande: LIMA's "superficial alignment" (format is cheap) explains only the syntax half, while a constraint that holds through 8 turns of escalating pressure is not surface style but a *policy over the whole output distribution* — a category the style-vs-capability dichotomy has no bucket for. Per-response metrics — like crowdworker preference in Gudibande's study — systematically flatter the surface; conversation-survival is the number that can't be fooled.

---

## DOK 2 — Knowledge Tree

Confidence key: **[high]** = found independently by both research agents; **[single]** = one agent. First-party facts are from the repo's own committed results.

### Category 1 — What tutoring is worth

- **[high]** Human tutoring is d≈0.79 vs. no tutoring — not Bloom's 2 sigma, which bundled tutoring with mastery learning. Step-based intelligent tutoring systems reach d≈0.76, near-parity. (VanLehn 2011, *Educational Psychologist* 46(4))
- **[single]** ITS median effect across 50 evaluations: +0.66 SD. (Kulik & Fletcher 2016, *RER*)
- *Implication:* realistic targets for any AI tutor are d≈0.7–0.8; the 2-sigma citation every AI-tutor pitch uses is folklore.

### Category 2 — When withholding helps vs. hurts

- **[high]** Productive failure (struggle *before* instruction) beats instruction-first on conceptual knowledge and transfer: g≈0.36 overall, ≈0.8 in high-fidelity implementations — but **only with consolidation afterward**. (Sinha & Kapur 2021, *RER* 91(5); Kapur 2012, *Instructional Science*)
- **[single]** The assistance dilemma: both over-assistance (shallow learning) and under-assistance (floundering, frustration) hurt; optimal assistance is contingent on learner state — a direct challenge to blanket never-reveal rules. (Koedinger & Aleven 2007, *Ed Psych Review* 19)
- **[high]** Minimal-guidance critique: novices under high cognitive load are harmed by unguided search; worked examples beat discovery. (Kirschner, Sweller & Clark 2006, *Educational Psychologist* 41(2))
- **[single]** Socratic-constrained LLM tutors raise later learning gains and understanding-driven strategies — but learners rate them less efficient; defection to answer-giving tools is a real cost. (arXiv:2508.06583; CEUR Vol-3953, 2025)
- *Challenging note (accepted, not resolved):* this literature indicts the spec as tutoring. The spec's defense is that it is a falsifiability instrument, not a pedagogy prescription — see Purpose.

### Category 3 — Style vs. capability in distillation

- **[high]** Imitation models fine-tuned on teacher outputs fool crowdworkers but close ≈0% of the capability gap on targeted evals: style transfers, capability doesn't. (Gudibande et al. 2023, arXiv:2305.15717)
- **[high]** Superficial Alignment Hypothesis: knowledge comes from pretraining; fine-tuning selects a format/style subdistribution — 1,000 curated examples aligned a 65B (LIMA). (Zhou et al. 2023, arXiv:2305.11206)
- **[single]** Pedagogy-informed fine-tuning works at frontier scale: expert raters preferred LearnLM over GPT-4o by ~31 points. (Google DeepMind LearnLM report, 2024)
- **[single]** A 7B RL-aligned tutor hit 10.6% solution leakage with +25.3% simulated-student solve rate, near LearnLM — while the *SFT*-based SocraticLM baseline lost −2.1% MMLU / −9.4% MATH500. (Dinucu-Jianu et al. 2025, arXiv:2505.15607)
- *Synthesis carried into DOK 3/4:* these results jointly predict this project — a Socratic constraint is behavioral (LIMA's cheap half), while the model's subject ceiling stays the base model's (Gudibande's hard half).

### Category 4 — Constraint durability in weights

- **[single]** Trained constraints concentrate in the first few output tokens and collapse under prefill or persistent multi-turn pressure; hardening requires depth across the response distribution. (Qi et al. 2024, arXiv:2406.05946)
- **[single]** Catastrophic forgetting erodes constraint/refusal behavior even from benign fine-tuning; worsens with scale within 1B–7B. (arXiv:2406.12227; arXiv:2406.04836)
- **[single]** LoRA learns less but forgets less than full fine-tuning — a favorable trade for style-only behavior transfer. (Biderman et al. 2024, arXiv:2405.09673)
- **[single]** Behavior-aware sampling: 0.05% targeted constraint data recovers much of the lost behavior (up to 41% harmfulness reduction). (arXiv:2510.21885)

### Category 5 — Evaluation validity

- **[high]** Learner preference and learning gains dissociate: answer-giving tutors are rated more efficient while producing shallower gains; crowdworker preference overrated imitation models; LLM-judged "helpfulness" is an unreliable pedagogy signal. (Gudibande et al. 2023; pedagogy-eval audit line; Maurya et al., arXiv:2412.09416)
- **[single]** Judging tutor quality requires targeted dimensions (mistake identification, guidance, *not revealing the answer too early*) — not generic helpfulness. (arXiv:2412.09416)

### Category 6 — First-party evidence (the `socratic/` results)

DOK 1 facts, all reproducible from the repo:

- **Prompt ceiling:** best frontier combo (gpt-5.6-luna + structured) = 89.0% adherence / 66.7% robustness; Sonnet best = 76.1% / 43.3%. No strategy close to reliable. (`RESULTS.md`)
- **Failure anatomy:** 98–99% of the 297 frontier failures were Rule-2 leaks (answer-smuggling), only 5 syntax; per-turn failure rate flat from turn 1 through 8 — helpfulness, not pressure-drift. Worst category: how-to (44% failures even for the best combo; 68% pooled).
- **Teacher anatomy:** raw teacher generations passed the judge ~22% first-time; 100% of the 271 rejections were semantic leaks; teacher-side self-check caught zero of them. (`FINAL_RESULTS.md` §6)
- **The v2 fix (data, not config):** rewritten generation prompt (forbidden patterns + extraction test) + Sonnet repair loop (max 3 attempts) + how-to oversampled to 30% → 98.2% net acceptance (1,171/1,193; 22 dropped; 755 turn-regenerations).
- **Scaling curve (eval_dev 100):** base 0/0 → s125 43.2/15.0 → s250 93.6/84.0 → s500 **97.2/93.0** (saturation) → s1000 96.2/92.0.
- **Head-to-head (identical 30 conversations, same judge):** s1000 bare 96.8/93.3 and s500 bare 93.1/83.3 beat every prompted frontier combo; crossover vs. Sonnet's best prompt sits between 125 and 250 examples.
- **The inversion:** how-to (frontier's worst, pipeline's main target) → 100%/100% tuned; emotional became the tuned model's weakest (71.1% at s500), with the fewest clean training examples; doubling its data (75→150) lifted +13/+20.
- **Training:** QLoRA 4-bit, Qwen3-1.7B, LoRA r16/α32 all-linear, assistant-only loss, 3 epochs, RTX 3070 Laptop 8GB, ~20–30 min per rung — "a button-press" relative to the data work.

### Known gaps (from research + Step 4 review)

- No literature on strict *syntactic* constraints (questions-only form); pedagogy studies withholding, not form.
- **Answer-smuggling has no name in the literature** — nearest relative is ETH's "solution leakage."
- Constraint-holding under *sincere emotional appeal* is homeless: pedagogy treats affect via human tutors, ML safety via adversarial jailbreaks. (The project's weakest category lives exactly in this gap.)
- No learning-outcome data anywhere in this project — all metrics are constraint metrics.

---

## Key Assumptions

Each with its failure mode — "how would I know if it were false?"

**About the user (grader):**

1. **The grader accepts LLM-as-judge as a fair arbiter** (third-family Gemini, temperature 0, cached, injection-shielded). *False if:* they reject LLM judging wholesale, or find a clearly-wrong verdict. *Test:* raw per-verdict transcripts are committed for hand-audit.
2. **The grader accepts three strategies as a fair prompt ceiling.** *False if:* "you just prompted badly" — someone writes a fourth strategy that beats 89.0/66.7. *Test:* the probe harness accepts new strategies; anyone can add one and rerun.

**About the technology:**

3. **The judge catches what matters** — no systematic blind spot, because a leak mode the judge can't see would be trained *into* the model (the gate is the teacher) and never measured. *False if:* a second judge family or human audit finds systematic false passes. *Test:* cross-judge agreement audit on a sample.
4. **Dev results predict held-out results** — eval_dev's 97.2/93.0 holds on the frozen `eval_final` and the staff held-out set. *False if:* the one-shot `eval_final` run drops sharply. *Test:* that run — deliberately still unspent.

**About the design theory:**

5. **~500 is a real minimum, not an artifact** — the saturation reflects this behavior's intrinsic data budget, not this particular teacher/judge pair. *False if:* swapping teacher or judge shifts the curve materially. *Test:* re-run the ladder with one component swapped.
6. **The flat-leak diagnosis holds** — leaks are helpfulness-driven (flat across turns), not pressure-driven, which justified training on clean demonstrations rather than Qi-style recovery trajectories. *False if:* longer adversarial conversations show late-turn collapse the flat diagnosis missed. *Test:* extended-length adversarial scenarios.

---

## Self-critique

Flags from the quality checklist, left visible rather than silently fixed:

- **Two SPOVs, not 3–5.** The author committed to two and softened both under adversarial probing (weights-when-bare; judge-as-final-verifier). Two candidate SPOVs ("robustness is the unit of proof", "oversample where the teacher bleeds") were offered and declined — their material survives inside DOK 3 Insight 3 and Insight 2 respectively, as insights rather than commitments.
- **Three DOK 3 insights, not 4–7.** The author claimed three; the drafted-but-unclaimed candidates ("falsifiability ⊥ pedagogy", "the judge is the teacher", "form is cheap/meaning is dear") were folded in only where the author endorsed the cluster.
- **Capability retention is unmeasured and unclaimed.** The assumption "the constraint rode in without degrading base capability" was offered in Step 8 and *not selected* — and the ETH result (SFT-Socratic lost 9.4% MATH500) makes this a live hazard. No MMLU/GSM8K base-vs-tuned comparison exists in the project. This is the document's largest open exposure.
- **The emotional gap is open.** Insight 2 names the mechanism and the lever, but the v3 (emotional oversampling) has not been run; at s1000 emotional remains the weakest category (84.2/80.0).
- **DOK 2 includes challenging evidence** (assistance dilemma, minimal-guidance critique, SFT capability losses, shallow-alignment warnings) — the checklist item passes, but note that the pedagogy critique is *accepted, not rebutted*: the document's defense is scope (falsifiability instrument, not tutoring prescription), which a pedagogy reviewer may find unsatisfying.
