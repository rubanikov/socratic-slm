"""SFT dataset generation pipeline for the Socratic-only behavior.

Phases (all checkpointed under socratic/dataset/, resumable, threaded):
  1. topics      - teacher generates topic records; pools are split FIRST into
                   train/test/eval_dev/eval_final so splits are topic-disjoint.
  2. blueprints  - teacher scripts the adversarial USER turns per topic.
  3. convs       - teacher roleplays the tutor turn-by-turn (structured prompt);
                   EVERY turn is judge-checked (same judge.py as the eval);
                   failed turns regenerate up to 3x with feedback; conversations
                   with an unrescuable turn are dropped.
  4. splits      - assembles nested train ladder (500/1000/2500/5000), test set,
                   and the two prompt-only eval sets + DATASET_CARD.md.

Training examples store ONLY clean user/assistant messages - no system prompt.
The behavior must live in the weights; that is the thesis.

Usage (from project root):
  python socratic/generate_dataset.py --smoke          # tiny end-to-end run
  python socratic/generate_dataset.py                  # full build
  python socratic/generate_dataset.py --phase splits   # re-assemble only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
load_dotenv(HERE.parent / ".env")

import judge as judge_mod  # noqa: E402
from prompts import STRUCTURED  # noqa: E402
import gen_prompts as gp  # noqa: E402

TEACHER = os.environ.get("OPENROUTER_MODEL3", "openai/gpt-5.6-luna")
BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

DATASET = HERE / "dataset"
TOPICS_DIR = DATASET / "topics"
BP_DIR = DATASET / "blueprints"
CONV_DIR = DATASET / "convs"
FINAL = DATASET / "final"

# $/1M tokens (input, output) for cost tracking
PRICES = {
    "openai/gpt-5.6-luna": (1.0, 6.0),
    "anthropic/claude-sonnet-5": (2.0, 10.0),
    "google/gemini-3.7-flash": (0.375, 1.875),
}

SHIFT_CATS = {"factual", "math", "emotional", "howto"}
SHIFT_PROB = 0.5
MAX_REGEN = 3
RNG_SEED = 42

# ---------------------------------------------------------------------------
# counting client + chat helper
# ---------------------------------------------------------------------------

class Counter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.usage: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.calls = 0

    def add(self, model: str, pt: int, ct: int) -> None:
        with self.lock:
            u = self.usage[model]
            u[0] += pt
            u[1] += ct
            self.calls += 1

    def cost(self) -> float:
        with self.lock:
            total = 0.0
            for m, (pt, ct) in self.usage.items():
                pin, pout = PRICES.get(m, (2.0, 10.0))
                total += pt * pin / 1e6 + ct * pout / 1e6
            return total

    def report(self) -> str:
        with self.lock:
            parts = [
                f"{m}: {pt/1000:.0f}k in / {ct/1000:.0f}k out"
                for m, (pt, ct) in self.usage.items()
            ]
        return f"{self.calls} calls | " + "; ".join(parts) + f" | est ${self.cost():.2f}"


COUNTER = Counter()
MAX_COST = 150.0


class CountingCompletions:
    def __init__(self, inner):
        self._inner = inner

    def create(self, **kw):
        if COUNTER.cost() > MAX_COST:
            raise RuntimeError(f"cost guard tripped: estimate exceeds ${MAX_COST}")
        r = self._inner.create(**kw)
        try:
            COUNTER.add(kw.get("model", "?"), r.usage.prompt_tokens, r.usage.completion_tokens)
        except Exception:  # noqa: BLE001 - usage may be absent
            pass
        return r


class CountingClient:
    """Duck-types the bit of the OpenAI client that judge.py and we use."""

    def __init__(self, inner: OpenAI):
        self.chat = type("C", (), {})()
        self.chat.completions = CountingCompletions(inner.chat.completions)


def make_client() -> CountingClient:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not found in .env")
    return CountingClient(OpenAI(base_url=BASE_URL, api_key=key, timeout=120))


def chat(client, model: str, messages: list[dict], max_tokens: int = 2000,
         minimal_reasoning: bool = True) -> str:
    """Teacher call: retries transient errors, escalates max_tokens if hidden
    reasoning eats the budget, drops the reasoning param if the provider 400s it."""
    max_tok = max_tokens
    use_reasoning = minimal_reasoning
    last_err = None
    for attempt in range(6):
        try:
            kw = dict(model=model, messages=messages, max_tokens=max_tok)
            if use_reasoning:
                kw["extra_body"] = {"reasoning": {"effort": "minimal"}}
            r = client.chat.completions.create(**kw)
            choice = r.choices[0]
            content = choice.message.content or ""
            if not content.strip() and choice.finish_reason == "length" and max_tok < 16000:
                max_tok = min(max_tok * 2, 16000)
                continue
            if not content.strip():
                raise RuntimeError(f"empty content (finish={choice.finish_reason})")
            return content
        except Exception as e:  # noqa: BLE001
            status = getattr(e, "status_code", None)
            if status == 400 and use_reasoning:
                use_reasoning = False  # provider rejects the reasoning param
                continue
            if status in (400, 401, 403, 404):
                raise RuntimeError(f"{model}: non-retryable {status}: {e}") from e
            last_err = e
            if attempt < 5:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"{model}: failed after retries: {last_err}")


def parse_json_array(text: str) -> list:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        idx = t.find("[")
        if idx < 0:
            raise ValueError(f"no JSON array: {t[:150]!r}")
        obj, _ = json.JSONDecoder().raw_decode(t[idx:])
    if not isinstance(obj, list):
        raise ValueError("not a list")
    return obj


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


# ---------------------------------------------------------------------------
# phase 1: topics
# ---------------------------------------------------------------------------

def topic_targets(sizes: dict[str, int]) -> dict[str, dict[str, int]]:
    """{category: {split: n_topics}} - includes headroom for blueprint and
    conversation drops. Shift secondaries RECYCLE already-used topics from the
    same split pool, so they cost no extra topics."""
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for cat, frac in gp.CATEGORY_MIX.items():
        for split, size in sizes.items():
            need = math.ceil(size * frac)
            headroom = 1.30 if split in ("train", "test") else 1.15
            out[cat][split] = math.ceil(need * headroom) + 2
    return out


def gen_topics(client, sizes: dict[str, int], workers: int) -> None:
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    targets = topic_targets(sizes)
    rng = random.Random(RNG_SEED)

    def build_category(cat: str) -> None:
        path = TOPICS_DIR / f"{cat}.jsonl"
        seen: set[str] = set()
        records: list[dict] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    records.append(r)
                    seen.add(norm(r.get("topic", "")))
        need_total = sum(targets[cat].values())
        domains = gp.TOPIC_DOMAINS[cat]
        di = 0
        stall = 0
        with path.open("a", encoding="utf-8") as f:
            while len(records) < need_total and stall < 40:
                domain = domains[di % len(domains)]
                di += 1
                avoid = [r.get("answer") or r.get("topic", "") for r in records[-80:]]
                prompt = gp.TOPIC_PROMPT.format(
                    category=cat, domain=domain, n=25,
                    category_block=gp.TOPIC_BLOCKS[cat],
                    avoid=json.dumps(avoid[-60:], ensure_ascii=False),
                )
                try:
                    batch = parse_json_array(
                        chat(client, TEACHER, [{"role": "user", "content": prompt}], 6000)
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"topics {cat}: batch failed ({e}); continuing")
                    stall += 1
                    continue
                added = 0
                for r in batch:
                    if not isinstance(r, dict) or "topic" not in r or "core_question" not in r:
                        continue
                    key = norm(r["topic"])
                    akey = norm(r.get("answer") or "")
                    if key in seen or (akey and akey in seen):
                        continue
                    seen.add(key)
                    if akey:
                        seen.add(akey)
                    r["category"] = cat
                    r.setdefault("expected_answers", [])
                    r.setdefault("answer_summary", None)
                    records.append(r)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    added += 1
                f.flush()
                stall = 0 if added else stall + 1
        if len(records) < need_total:
            print(f"WARNING topics {cat}: only {len(records)}/{need_total}")
        # assign splits deterministically, disjoint, and persist the assignment
        rng.shuffle(records)
        assign_path = TOPICS_DIR / f"{cat}.splits.json"
        if not assign_path.exists():
            assign: dict[str, list[dict]] = {}
            i = 0
            for split in ("eval_final", "eval_dev", "test", "train"):
                n = targets[cat][split]
                assign[split] = records[i:i + n]
                i += n
            assign_path.write_text(json.dumps(assign, ensure_ascii=False), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=min(workers, len(gp.CATEGORY_MIX))) as pool:
        futs = {pool.submit(build_category, c): c for c in gp.CATEGORY_MIX}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="topic pools"):
            fut.result()


# ---------------------------------------------------------------------------
# phase 2: blueprints
# ---------------------------------------------------------------------------

def bp_targets(sizes: dict[str, int]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for cat, frac in gp.CATEGORY_MIX.items():
        for split, size in sizes.items():
            need = math.ceil(size * frac)
            headroom = 1.15 if split in ("train", "test") else 1.05
            out[cat][split] = math.ceil(need * headroom)
    return out


def make_blueprint(client, rec: dict, shift_rec: dict | None, rng_seed: int) -> dict | None:
    rng = random.Random(rng_seed)
    cat = rec["category"]
    mood = rng.choice(gp.PERSONA_MOODS)
    context = rng.choice(gp.PERSONA_CONTEXTS)
    n_turns = 8 if shift_rec else 6
    if shift_rec:
        shift_block = gp.SHIFT_BLOCK.format(
            turn8="-8", shift_question=shift_rec["core_question"],
            shift_answer=shift_rec.get("answer") or "n/a",
        )
    else:
        shift_block = ""
    no_answer_note = ""
    merged_expected = list(rec.get("expected_answers") or [])
    if shift_rec:
        merged_expected += list(shift_rec.get("expected_answers") or [])
    if merged_expected:
        no_answer_note = f" ({json.dumps(merged_expected, ensure_ascii=False)})"
    prompt = gp.BLUEPRINT_PROMPT.format(
        topic_record=json.dumps(
            {k: rec.get(k) for k in ("topic", "core_question", "answer")},
            ensure_ascii=False),
        mood=mood, context=context, n_turns=n_turns,
        pressure_style=gp.PRESSURE_STYLES[cat],
        shift_block=shift_block, no_answer_note=no_answer_note,
    )
    try:
        turns = parse_json_array(chat(client, TEACHER, [{"role": "user", "content": prompt}], 3000))
    except Exception:  # noqa: BLE001
        return None
    if len(turns) != n_turns or not all(isinstance(t, str) and t.strip() for t in turns):
        return None
    for ans in merged_expected:
        for t in turns:
            if judge_mod.string_leak_check(t, [ans]):
                return None  # bait contains its own answer - discard
    core_q = rec["core_question"]
    ans_sum = rec.get("answer_summary")
    if shift_rec:
        core_q += f" (secondary: {shift_rec['core_question']})"
        second = shift_rec.get("answer_summary") or ""
        ans_sum = f"{ans_sum or ''}; after the topic shift: {second}".strip("; ")
    return {
        "category": cat,
        "topic": rec["topic"],
        "core_question": core_q,
        "answer_summary": ans_sum or None,
        "expected_answers": merged_expected,
        "turns": turns,
        "persona": f"{mood}; {context}",
    }


def gen_blueprints(client, sizes: dict[str, int], workers: int) -> None:
    BP_DIR.mkdir(parents=True, exist_ok=True)
    targets = bp_targets(sizes)
    rng = random.Random(RNG_SEED + 1)
    jobs = []  # (split, cat, idx, topic_rec, shift_rec)
    for cat in gp.CATEGORY_MIX:
        assign = json.loads((TOPICS_DIR / f"{cat}.splits.json").read_text(encoding="utf-8"))
        for split, need in targets[cat].items():
            pool = list(assign[split])
            i = 0
            made = 0
            while made < need and i < len(pool):
                rec = pool[i]
                i += 1
                shift_rec = None
                if cat in SHIFT_CATS and rng.random() < SHIFT_PROB and i < len(pool):
                    shift_rec = pool[i]
                    i += 1
                jobs.append((split, cat, made, rec, shift_rec))
                made += 1
            if made < need:
                print(f"WARNING blueprints {cat}/{split}: topics only cover {made}/{need}")

    seen_first = set()

    def run_job(job) -> None:
        split, cat, idx, rec, shift_rec = job
        bp_id = f"{split}_{cat}_{idx:05d}"
        path = BP_DIR / f"{bp_id}.json"
        if path.exists():
            return
        bp = make_blueprint(client, rec, shift_rec, rng_seed=hash(bp_id) & 0xFFFF)
        if bp is None:
            path.with_suffix(".dropped").write_text("invalid", encoding="utf-8")
            return
        h = hashlib.sha256(norm(bp["turns"][0]).encode()).hexdigest()[:12]
        with threading.Lock():
            if h in seen_first:
                path.with_suffix(".dropped").write_text("dup", encoding="utf-8")
                return
            seen_first.add(h)
        bp["id"] = bp_id
        bp["split"] = split
        path.write_text(json.dumps(bp, ensure_ascii=False, indent=1), encoding="utf-8")

    todo = [j for j in jobs
            if not (BP_DIR / f"{j[0]}_{j[1]}_{j[2]:05d}.json").exists()
            and not (BP_DIR / f"{j[0]}_{j[1]}_{j[2]:05d}.dropped").exists()]
    print(f"blueprints: {len(jobs)} planned, {len(todo)} to generate")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_job, j) for j in todo]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="blueprints"):
            fut.result()


# ---------------------------------------------------------------------------
# phase 3: conversations (train/test blueprints only)
# ---------------------------------------------------------------------------

def gen_conversation(client, bp: dict) -> dict:
    out = {"id": bp["id"], "split": bp["split"], "category": bp["category"],
           "topic": bp["topic"], "core_question": bp["core_question"],
           "answer_summary": bp["answer_summary"],
           "expected_answers": bp["expected_answers"],
           "persona": bp.get("persona"), "regens": 0, "status": "ok",
           "messages": []}
    history: list[dict] = []
    for user_turn in bp["turns"]:
        history.append({"role": "user", "content": user_turn})
        reply = None
        reason = None
        for attempt in range(MAX_REGEN + 1):
            ctx = [{"role": "system", "content": STRUCTURED}] + [dict(m) for m in history]
            if attempt > 0:
                ctx[-1]["content"] = user_turn + gp.REGEN_NOTE.format(reason=reason)
                out["regens"] += 1
            cand = chat(client, TEACHER, ctx, 2000)
            v = judge_mod.judge_response(
                client, cand, bp["core_question"],
                bp["expected_answers"], bp["answer_summary"])
            if v.get("judge_error"):
                out["status"] = "judge_error"
                return out
            if v["passed"]:
                reply = cand
                break
            reason = ("; ".join(v["syntax_violations"])[:200]
                      if not v["syntax_pass"] else str(v["leak_reason"])[:200])
        if reply is None:
            out["status"] = "dropped"
            out["drop_reason"] = f"turn unrescuable after {MAX_REGEN} regens: {reason}"
            return out
        history.append({"role": "assistant", "content": reply})
        out["messages"].append({"role": "user", "content": user_turn})
        out["messages"].append({"role": "assistant", "content": reply})
    return out


def gen_convs(client, workers: int) -> None:
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    bps = []
    for p in sorted(BP_DIR.glob("*.json")):
        bp = json.loads(p.read_text(encoding="utf-8"))
        if bp["split"] in ("train", "test"):
            done = CONV_DIR / f"{bp['id']}.json"
            if not done.exists():
                bps.append(bp)
    print(f"conversations: {len(bps)} to generate")

    def run(bp: dict) -> None:
        conv = gen_conversation(client, bp)
        if conv["status"] == "judge_error":
            return  # not checkpointed - retried on next run
        (CONV_DIR / f"{bp['id']}.json").write_text(
            json.dumps(conv, ensure_ascii=False, indent=1), encoding="utf-8")

    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run, bp) for bp in bps]
        with tqdm(total=len(futs), desc="conversations") as bar:
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    errors += 1
                    if errors <= 10:
                        print(f"conv error: {e}")
                bar.update(1)
                if bar.n % 100 == 0:
                    bar.write(COUNTER.report())
    if errors:
        print(f"{errors} conversation errors (rerun to resume)")


# ---------------------------------------------------------------------------
# phase 4: splits
# ---------------------------------------------------------------------------

LADDER = [125, 250, 500, 1000, 2000, 2500, 5000]


def assemble(sizes: dict[str, int]) -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RNG_SEED + 2)

    convs = {"train": [], "test": []}
    dropped = 0
    for p in sorted(CONV_DIR.glob("*.json")):
        c = json.loads(p.read_text(encoding="utf-8"))
        if c["status"] == "ok":
            convs[c["split"]].append(c)
        else:
            dropped += 1

    def sft_record(c: dict) -> dict:
        return {"id": c["id"], "category": c["category"], "topic": c["topic"],
                "messages": c["messages"]}

    # test set
    test = convs["test"]
    rng.shuffle(test)
    test = test[: sizes["test"]]
    with (FINAL / "test.jsonl").open("w", encoding="utf-8") as f:
        for c in test:
            f.write(json.dumps(sft_record(c), ensure_ascii=False) + "\n")

    # nested train ladder, stratified by category
    by_cat = defaultdict(list)
    for c in convs["train"]:
        by_cat[c["category"]].append(c)
    for lst in by_cat.values():
        rng.shuffle(lst)
    ladder = [n for n in LADDER if n <= sizes["train"]]
    if sizes["train"] not in ladder:
        ladder.append(sizes["train"])
    ordered: list[dict] = []
    for n in ladder:
        want = {cat: math.ceil(n * frac) for cat, frac in gp.CATEGORY_MIX.items()}
        short = {cat: w - len(by_cat[cat]) for cat, w in want.items() if len(by_cat[cat]) < w}
        if short:
            # progressive assembly: emit a rung only when every category can
            # fill its stratified quota - partial rungs would skew the mix
            print(f"train_{n}.jsonl: skipped (categories short: {short})")
            continue
        have = defaultdict(int)
        for c in ordered:
            have[c["category"]] += 1
        for cat, w in want.items():
            add = max(0, min(w, len(by_cat[cat])) - have[cat])
            start = have[cat]
            ordered.extend(by_cat[cat][start:start + add])
        ordered = ordered[:n] if len(ordered) > n else ordered
        rung = ordered[:n]
        with (FINAL / f"train_{n}.jsonl").open("w", encoding="utf-8") as f:
            for c in rung:
                f.write(json.dumps(sft_record(c), ensure_ascii=False) + "\n")
        print(f"train_{n}.jsonl: {len(rung)} conversations")

    # prompt-only eval sets from their disjoint blueprint pools.
    # The shipped eval files were extended by hand (mix-matched supersets);
    # never overwrite an existing eval file that already meets or exceeds the cap.
    for split, fname, cap in (("eval_dev", "eval_dev.jsonl", sizes["eval_dev"]),
                              ("eval_final", "eval_final.jsonl", sizes["eval_final"])):
        existing = FINAL / fname
        if existing.exists():
            n = sum(1 for l in existing.read_text(encoding="utf-8").splitlines() if l.strip())
            if n >= cap:
                print(f"{fname}: kept existing curated set ({n} scenarios)")
                continue
        bps = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(BP_DIR.glob(f"{split}_*.json"))]
        rng.shuffle(bps)
        bps = bps[:cap]
        with (FINAL / fname).open("w", encoding="utf-8") as f:
            for bp in bps:
                f.write(json.dumps({
                    "id": bp["id"], "category": bp["category"], "topic": bp["topic"],
                    "core_question": bp["core_question"],
                    "answer_summary": bp["answer_summary"],
                    "expected_answers": bp["expected_answers"],
                    "turns": bp["turns"],
                }, ensure_ascii=False) + "\n")
        print(f"{fname}: {len(bps)} scenarios")

    # dataset card
    total_regens = sum(c.get("regens", 0) for split in convs.values() for c in split)
    mix = Counter_ = defaultdict(int)
    for c in convs["train"]:
        Counter_[c["category"]] += 1
    card = [
        "# Socratic-Only SFT Dataset",
        "",
        f"- teacher: `{TEACHER}` with the structured prompt; every assistant turn",
        "  filtered by `judge.py` (the same grader as the eval); failed turns",
        f"  regenerated up to {MAX_REGEN}x, unrescuable conversations dropped.",
        f"- train conversations available: {len(convs['train'])} "
        f"(ladder: {', '.join(str(n) for n in ladder)})",
        f"- test: {len(test)} | eval_dev / eval_final: prompt-only, topic-disjoint",
        f"- dropped conversations: {dropped} | total turn regenerations: {total_regens}",
        f"- train category mix: {dict(Counter_)}",
        "- format: `messages` = user/assistant only, NO system prompt (behavior in weights)",
        f"- generation cost: {COUNTER.report() if COUNTER.calls else 'see run logs'}",
        "",
    ]
    (FINAL / "DATASET_CARD.md").write_text("\n".join(card), encoding="utf-8")
    print(f"dropped: {dropped} | regens: {total_regens}")


# ---------------------------------------------------------------------------

def main() -> None:
    global MAX_COST
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["all", "topics", "blueprints", "convs", "splits"],
                    default="all")
    ap.add_argument("--train-size", type=int, default=5000)
    ap.add_argument("--test-size", type=int, default=300)
    ap.add_argument("--eval-dev", type=int, default=150)
    ap.add_argument("--eval-final", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-cost", type=float, default=150.0)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run: 18 train / 6 test / 6+4 evals")
    args = ap.parse_args()
    MAX_COST = args.max_cost

    sizes = {"train": args.train_size, "test": args.test_size,
             "eval_dev": args.eval_dev, "eval_final": args.eval_final}
    if args.smoke:
        sizes = {"train": 18, "test": 6, "eval_dev": 6, "eval_final": 4}

    client = make_client()
    t0 = time.time()
    if args.phase in ("all", "topics"):
        gen_topics(client, sizes, args.workers)
        print("topics done |", COUNTER.report())
    if args.phase in ("all", "blueprints"):
        gen_blueprints(client, sizes, args.workers)
        print("blueprints done |", COUNTER.report())
    if args.phase in ("all", "convs"):
        gen_convs(client, args.workers)
        print("convs done |", COUNTER.report())
    if args.phase in ("all", "splits"):
        assemble(sizes)
    print(f"total {time.time() - t0:.0f}s | {COUNTER.report()}")


if __name__ == "__main__":
    main()
