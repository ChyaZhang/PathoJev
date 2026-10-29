"""Answer the eight SlideBench-TCGA questions of one bladder carcinoma slide with a single slide encoding.

  python demo.py --features TCGA-XF-AAMJ-01Z-00-DX1.726E3D84-8370-45AC-9C98-ACA30CAB2359.h5 [--size 27b]
"""
import argparse
import json
import time

import torch

from pathojev import PathoJev, make_grid

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True, help="TRIDENT CONCH v1.5 features of the slide (.h5) or a pooled grid (.pt)")
ap.add_argument("--questions", default="examples/TCGA-XF-AAMJ_questions.json")
ap.add_argument("--size", default="9b", choices=["9b", "27b"])
ap.add_argument("--repo", default="Mikezcy/PathoJev")
ap.add_argument("--base", default=None, help="local copy of the backbone (optional)")
a = ap.parse_args()

model = PathoJev.from_pretrained(a.size, a.repo, a.base)
grid = torch.load(a.features) if a.features.endswith(".pt") else make_grid(a.features)
qs = json.load(open(a.questions))["questions"]
model.answer_slide(grid, qs[:1])  # warm-up
torch.cuda.synchronize(); t0 = time.time()
out = model.answer_slide(grid, qs)
torch.cuda.synchronize(); dt = time.time() - t0
for q, r in zip(qs, out):
    mark = "" if "answer" not in q else (" ✓" if r["answer"] == q["answer"] else f" ✗ (reference: {q['answer']})")
    print(f"[{q.get('category', '')}] {q['question']}\n  -> {r['answer']}  p = {r['probs'][r['answer']]:.2f}{mark}")
print(f"{len(qs)} questions in {dt:.2f} s (one slide encoding, batched question rows)")
