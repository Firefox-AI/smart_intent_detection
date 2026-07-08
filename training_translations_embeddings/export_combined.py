#!/usr/bin/env python3
"""Export FR + EN query-intent datasets into one ordered jsonl for embedding, plus per-group
labels/metadata. Groups (in order): fr_train, en_train, fr_val, en_eval, fr_test, en_test.

Source datasets (HuggingFace):
  Mozilla/query-intent-detection-dataset          (English; columns: Query, Label)
  Mozilla/query-intent-detection-dataset-french   (French)

Usage: export_combined.py [output_dir]   (default: firefox/artifacts/intent2)
"""
import json
import os
import sys
from datasets import load_dataset

OUT = sys.argv[1] if len(sys.argv) > 1 else "/firefox/artifacts/intent2"
os.makedirs(OUT, exist_ok=True)
L = {"Search": 1, "Chat": 0}

fr = load_dataset("Mozilla/query-intent-detection-dataset-french")
en = load_dataset("Mozilla/query-intent-detection-dataset")

groups = [
    ("fr_train", fr["train"]),
    ("en_train", en["train"]),
    ("fr_val", fr["validation"]),
    ("en_eval", en["eval"]),
    ("fr_test", fr["test"]),
    ("en_test", en["test"]),
]

meta = {"order": [], "counts": {}, "labels": {}}
with open(f"{OUT}/all.jsonl", "w") as f:
    n = 0
    for name, ds in groups:
        meta["order"].append(name)
        meta["counts"][name] = len(ds)
        meta["labels"][name] = [L[x] for x in ds["Label"]]
        for row in ds:
            f.write(json.dumps({"id": str(n), "text": row["Query"]}) + "\n")
            n += 1

open(f"{OUT}/placeholder.jsonl", "w").write(json.dumps({"id": "0", "text": "x"}) + "\n")
json.dump(meta, open(f"{OUT}/meta.json", "w"))
print("total texts:", n)
print("counts:", meta["counts"])
