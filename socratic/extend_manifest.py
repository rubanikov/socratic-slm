"""Append EXTENSION jobs to the existing manifest for the train_2000 rung.

Never rebuilds the manifest (that would reshuffle the rng stream and break the
id->content mapping of already-accepted conversations). Extension jobs live in
their own id range: train_<cat>_9xxxx.

Inputs:
  socratic/dataset/ext_topics.json   {category: [topic records]}  (fresh, disjoint)
  socratic/dataset/ext2000_plan.json {"jobs": {category: n}}
Effects:
  - appends topics to socratic/dataset/topics/<cat>.jsonl (global used-topic record)
  - appends jobs to socratic/dataset/manifest.jsonl
Usage:  python socratic/extend_manifest.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen_prompts as gp  # noqa: E402
from dataset_plan import MANIFEST  # noqa: E402
from generate_dataset import SHIFT_CATS, SHIFT_PROB, TOPICS_DIR  # noqa: E402

ID_BASE = 90000


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="ext_topics.json")
    ap.add_argument("--plan", default="ext2000_plan.json")
    args = ap.parse_args()
    ds = HERE / "dataset"
    topics = json.loads((ds / args.topics).read_text(encoding="utf-8"))
    plan = json.loads((ds / args.plan).read_text(encoding="utf-8"))["jobs"]
    existing_ids = set()
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            existing_ids.add(json.loads(line)["id"])
    # re-runnable: each category's new ids start after its existing extension range
    next_idx = {}
    for jid in existing_ids:
        parts = jid.split("_")
        if parts[0] == "train" and len(parts) == 3 and parts[2].startswith("9"):
            cat = parts[1]
            next_idx[cat] = max(next_idx.get(cat, ID_BASE), int(parts[2]) + 1)

    rng = random.Random(777)
    added = 0
    with MANIFEST.open("a", encoding="utf-8") as mf:
        for cat, want in plan.items():
            pool = list(topics.get(cat, []))
            if len(pool) < want:
                print(f"WARNING {cat}: only {len(pool)} topics for {want} jobs")
            used: list[dict] = []
            # record new topics in the global per-category topic files
            with (TOPICS_DIR / f"{cat}.jsonl").open("a", encoding="utf-8") as tf:
                for r in pool:
                    r["category"] = cat
                    r.setdefault("expected_answers", [])
                    r.setdefault("answer_summary", None)
                    tf.write(json.dumps(r, ensure_ascii=False) + "\n")
            base = next_idx.get(cat, ID_BASE)
            for i, rec in enumerate(pool[:want]):
                jid = f"train_{cat}_{base + i:05d}"
                if jid in existing_ids:
                    continue
                shift_rec = None
                if cat in SHIFT_CATS and rng.random() < SHIFT_PROB and used:
                    shift_rec = rng.choice(used)
                used.append(rec)
                mf.write(json.dumps({
                    "id": jid,
                    "split": "train",
                    "category": cat,
                    "topic_record": rec,
                    "shift_record": shift_rec,
                    "mood": rng.choice(gp.PERSONA_MOODS),
                    "context": rng.choice(gp.PERSONA_CONTEXTS),
                    "n_turns": 8 if shift_rec else 6,
                    "pressure_style": gp.PRESSURE_STYLES[cat],
                }, ensure_ascii=False) + "\n")
                added += 1
    print(f"appended {added} extension jobs to {MANIFEST}")


if __name__ == "__main__":
    main()
