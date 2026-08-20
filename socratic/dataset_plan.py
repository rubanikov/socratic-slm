"""Deterministic bookkeeping for the agent-based dataset build.

- writes topic split assignments (train/test/eval_dev/eval_final, topic-disjoint)
- emits the generation manifest: one job per scenario (topic + persona + shift
  + turn plan), consumed by generation agents in waves
- reports build progress

No API calls. Usage:
  python socratic/dataset_plan.py assign   --train-size N --test-size N ...
  python socratic/dataset_plan.py manifest --train-size N ...
  python socratic/dataset_plan.py status
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen_prompts as gp  # noqa: E402
from generate_dataset import (  # noqa: E402
    BP_DIR, CONV_DIR, DATASET, RNG_SEED, SHIFT_CATS, SHIFT_PROB, TOPICS_DIR,
    bp_targets, topic_targets,
)

MANIFEST = DATASET / "manifest.jsonl"
RAW_DIR = DATASET / "convs_raw"
REPAIR_DIR = DATASET / "repair"


def sizes_from_args(args) -> dict[str, int]:
    if args.smoke:
        return {"train": 18, "test": 6, "eval_dev": 6, "eval_final": 4}
    return {"train": args.train_size, "test": args.test_size,
            "eval_dev": args.eval_dev, "eval_final": args.eval_final}


def assign_splits(sizes: dict[str, int]) -> None:
    """Split each category's topic pool into disjoint per-split pools."""
    rng = random.Random(RNG_SEED)
    targets = topic_targets(sizes)
    for cat in gp.CATEGORY_MIX:
        path = TOPICS_DIR / f"{cat}.jsonl"
        if not path.exists():
            sys.exit(f"missing {path} - generate topics first")
        records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        rng.shuffle(records)
        assign_path = TOPICS_DIR / f"{cat}.splits.json"
        if assign_path.exists():
            print(f"{cat}: split assignment already exists, keeping it")
            continue
        assign: dict[str, list[dict]] = {}
        i = 0
        for split in ("eval_final", "eval_dev", "test", "train"):
            n = targets[cat][split]
            assign[split] = records[i:i + n]
            i += n
        short = {s: targets[cat][s] - len(assign[s]) for s in assign if len(assign[s]) < targets[cat][s]}
        if short:
            print(f"WARNING {cat}: topic pool short by {short}")
        assign_path.write_text(json.dumps(assign, ensure_ascii=False), encoding="utf-8")
        print(f"{cat}: assigned {sum(len(v) for v in assign.values())} topics across splits")


def build_manifest(sizes: dict[str, int]) -> None:
    rng = random.Random(RNG_SEED + 1)
    targets = bp_targets(sizes)
    jobs = []
    for cat in gp.CATEGORY_MIX:
        assign = json.loads((TOPICS_DIR / f"{cat}.splits.json").read_text(encoding="utf-8"))
        for split, need in targets[cat].items():
            pool = list(assign[split])
            i = 0
            made = 0
            used: list[dict] = []
            while made < need and i < len(pool):
                rec = pool[i]
                i += 1
                shift_rec = None
                if cat in SHIFT_CATS and rng.random() < SHIFT_PROB:
                    # recycle an already-used topic as the shift secondary -
                    # costs no extra topics, harmless for diversity
                    if used:
                        shift_rec = rng.choice(used)
                    elif i < len(pool):
                        shift_rec = pool[i]
                        i += 1
                used.append(rec)
                jobs.append({
                    "id": f"{split}_{cat}_{made:05d}",
                    "split": split,
                    "category": cat,
                    "topic_record": rec,
                    "shift_record": shift_rec,
                    "mood": rng.choice(gp.PERSONA_MOODS),
                    "context": rng.choice(gp.PERSONA_CONTEXTS),
                    "n_turns": 8 if shift_rec else 6,
                    "pressure_style": gp.PRESSURE_STYLES[cat],
                })
                made += 1
            if made < need:
                print(f"WARNING manifest {cat}/{split}: {made}/{need} (topic pool exhausted)")
    with MANIFEST.open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(f"manifest: {len(jobs)} jobs -> {MANIFEST}")


def pending_jobs() -> list[dict]:
    """Jobs with no accepted output yet (and no exhausted repair)."""
    if not MANIFEST.exists():
        return []
    jobs = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for j in jobs:
        jid = j["id"]
        raw = (RAW_DIR / f"{jid}.json").exists()
        dropped = (REPAIR_DIR / f"{jid}.dropped").exists()
        if j["split"] in ("eval_dev", "eval_final"):
            if not (BP_DIR / f"{jid}.json").exists() and not raw and not dropped:
                out.append(j)
        else:
            if not (CONV_DIR / f"{jid}.json").exists() and not raw and not dropped:
                out.append(j)
    return out


def status() -> None:
    n_manifest = sum(1 for _ in MANIFEST.open(encoding="utf-8")) if MANIFEST.exists() else 0
    counts = {
        "manifest jobs": n_manifest,
        "convs accepted": len(list(CONV_DIR.glob("*.json"))) if CONV_DIR.exists() else 0,
        "raw awaiting filter": len(list(RAW_DIR.glob("*.json"))) if RAW_DIR.exists() else 0,
        "repair tickets": len(list(REPAIR_DIR.glob("*.json"))) if REPAIR_DIR.exists() else 0,
        "dropped": len(list(REPAIR_DIR.glob("*.dropped"))) if REPAIR_DIR.exists() else 0,
        "eval blueprints": len(list(BP_DIR.glob("eval_*.json"))) if BP_DIR.exists() else 0,
        "pending jobs": len(pending_jobs()),
    }
    by_split = defaultdict(int)
    if CONV_DIR.exists():
        for p in CONV_DIR.glob("*.json"):
            by_split[p.name.split("_")[0]] += 1
    for k, v in counts.items():
        print(f"{k}: {v}")
    if by_split:
        print(f"accepted by split: {dict(by_split)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["assign", "manifest", "status", "pending", "jobs"])
    ap.add_argument("ids", nargs="*", help="jobs: manifest ids to print")
    ap.add_argument("--train-size", type=int, default=5000)
    ap.add_argument("--test-size", type=int, default=300)
    ap.add_argument("--eval-dev", type=int, default=150)
    ap.add_argument("--eval-final", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="pending: cap listed jobs")
    args = ap.parse_args()
    sizes = sizes_from_args(args)
    for d in (RAW_DIR, REPAIR_DIR, BP_DIR, CONV_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if args.cmd == "assign":
        assign_splits(sizes)
    elif args.cmd == "manifest":
        build_manifest(sizes)
    elif args.cmd == "status":
        status()
    elif args.cmd == "pending":
        jobs = pending_jobs()
        if args.limit:
            jobs = jobs[: args.limit]
        print(json.dumps([j["id"] for j in jobs], indent=0))
    elif args.cmd == "jobs":
        want = set(args.ids)
        found = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines()
                 if l.strip() and json.loads(l)["id"] in want]
        print(json.dumps(found, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
