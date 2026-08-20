# Baseline Probe Results

Behavior Spec: see [BEHAVIOR_SPEC.md](BEHAVIOR_SPEC.md). Spec-adherence = passing responses / judged responses. Robustness = scenarios with zero failing responses / fully-judged scenarios.

| Model | Strategy | Spec-adherence | Robustness | responses | scenarios | judge errors |
|---|---|---|---|---|---|---|
| luna | few_shot | 79.4% | 46.7% | 218 | 30 | 0 |
| luna | structured | 89.0% | 66.7% | 218 | 30 | 0 |
| luna | zero_shot | 83.9% | 46.7% | 218 | 30 | 0 |
| sonnet | few_shot | 69.7% | 40.0% | 218 | 30 | 0 |
| sonnet | structured | 76.1% | 43.3% | 218 | 30 | 0 |
| sonnet | zero_shot | 65.6% | 43.3% | 218 | 30 | 0 |

## Failure breakdown by category (share of judged responses failing)

| Model | Strategy | emotional | factual | howto | math | meta | smalltalk |
|---|---|---|---|---|---|---|---|
| luna | few_shot | 18% | 12% | 64% | 4% | 19% | 0% |
| luna | structured | 3% | 5% | 44% | 0% | 11% | 0% |
| luna | zero_shot | 16% | 14% | 44% | 4% | 11% | 0% |
| sonnet | few_shot | 29% | 24% | 92% | 25% | 3% | 0% |
| sonnet | structured | 21% | 22% | 72% | 0% | 11% | 5% |
| sonnet | zero_shot | 32% | 26% | 92% | 0% | 42% | 0% |

## Failure mode analysis

**The failure mode that survives the best prompting attempt is answer-smuggling:
both frontier models master the question-only surface form almost immediately, but
cannot stop the answer from leaking *inside* their questions.** Of 297 failing
responses across the grid, 98–99% fail Rule 2 (answer leak) and only 5 fail Rule 1
(syntax) — the models say "tungsten"/"wolfram" outright, or embed a
uniquely-identifying hint ("the composer with progressive deafness", "that closed
six-sided shape", "have you tried restarting the modem and router?"). The leak is
**not** pressure-induced drift: the per-turn failure rate is flat (19–27% from turn 1
through turn 8), meaning RLHF-trained helpfulness leaks the payload from the very
first, benign turn. It is concentrated where helpfulness and the constraint collide
hardest — **how-to advice fails 44% of the time even for the best combo** (68%
pooled), because guiding someone toward a fix without naming the fix is exactly
what a helpful assistant is trained not to do. Even the overall best combo,
gpt-5.6-luna with the structured self-check prompt (89.0% adherence), still leaked
24 times and let a third of conversations fail (66.7% robustness); Sonnet's best was
76.1% / 43.3%. No prompting strategy came close to reliable — which is the gate this
probe was built to test: the behavior is coherent, partially promptable, and
reliably held by no one. The fine-tune's job is therefore specific: **teach the
model to withhold the answer, especially in advice-shaped conversations, while
keeping the interrogative form it already finds easy.**

*(Measurement notes: judge = google/gemini-3.7-flash at temperature 0 with cached
verdicts, identical for every combo; 0 judge errors across 1,308 responses. The
handful of syntax fails include 2–3 borderline splits on mid-sentence quoted
exclamations and "Symphony No. 9" — immaterial at this scale. Some leak verdicts
count repeating the user's own identifying words (e.g. "the filament metal") as
leaks; this strictness is applied symmetrically to both models and will apply
identically to the tuned model.)*
