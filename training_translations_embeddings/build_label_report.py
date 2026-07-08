#!/usr/bin/env python3
"""Full per-label evaluation report for the frozen-embedding + French-balanced tiny transformer.

Re-scores the saved model on each eval split and renders an HTML report with, per group:
AUC, accuracy, per-class precision/recall/F1/support (Chat & Search), macro/weighted averages,
and the confusion matrix.
"""
import json
import os

import numpy as np
import torch
from safetensors.numpy import load_file as load_safetensors
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from tokenizers import Tokenizer

import train_tiny_transformer as T

ART = os.path.dirname(os.path.abspath(__file__))
RUN = "intent_tiny_fr_oversample"
LABELS = ["Chat", "Search"]  # index 0, 1

ck = torch.load(f"{ART}/{RUN}/model.pt", map_location="cpu", weights_only=False)
args = ck["args"]
device = "cpu"

emb = np.ascontiguousarray(
    load_safetensors(f"{T.STATIC_MODEL}/model.safetensors")["embeddings"], dtype=np.float32
)
model = T.TinyTransformerClassifier(
    emb, args["nhead"], args["dim_ff"], args["dropout"], args["max_len"], freeze_emb=True
).to(device)
model.load_state_dict(ck["state_dict"])
model.eval()

tokenizer = Tokenizer.from_file(f"{T.STATIC_MODEL}/tokenizer.json")
groups = T.load_groups()

EVAL = [
    ("Dev combined (FR val + EN eval)", ["fr_val", "en_eval"]),
    ("Dev — French (validation)", ["fr_val"]),
    ("Dev — English (eval)", ["en_eval"]),
    ("Test — French", ["fr_test"]),
    ("Test — English", ["en_test"]),
]


def score_group(keys):
    texts, labels = [], []
    for k in keys:
        t, l = groups[k]
        texts += t
        labels += l
    ids, mask = T.encode_texts(texts, tokenizer, args["max_len"])
    prob = T.predict_proba(model, ids, mask, device)
    y = np.asarray(labels, dtype=int)
    pred = (prob >= 0.5).astype(int)
    rep = classification_report(
        y, pred, labels=[0, 1], target_names=LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, prob)),
        "accuracy": float(rep["accuracy"]),
        "report": rep,
        "cm": cm.tolist(),
    }


results = {name: score_group(keys) for name, keys in EVAL}
json.dump(results, open(f"{ART}/{RUN}/label_report.json", "w"), indent=1)


def heat(v, lo=0.80, hi=1.0):
    t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    r = int(255 - t * (255 - 46)); g = int(255 - t * (255 - 125)); b = int(255 - t * (255 - 50))
    return f"background:rgb({r},{g},{b});color:{'#fff' if t > 0.55 else '#111'}"


def metric_table(rep):
    rows = ""
    for key, disp in [("Chat", "Chat (0)"), ("Search", "Search (1)"),
                      ("macro avg", "macro avg"), ("weighted avg", "weighted avg")]:
        d = rep[key]
        cls = ' class="avg"' if "avg" in key else ""
        rows += (
            f'<tr{cls}><td class="lbl">{disp}</td>'
            f'<td style="{heat(d["precision"])}">{d["precision"]:.4f}</td>'
            f'<td style="{heat(d["recall"])}">{d["recall"]:.4f}</td>'
            f'<td style="{heat(d["f1-score"])}">{d["f1-score"]:.4f}</td>'
            f'<td>{int(d["support"]):,}</td></tr>'
        )
    return rows


sections = ""
for name, keys in EVAL:
    r = results[name]
    cm = r["cm"]  # [[TN, FP], [FN, TP]]
    is_fr = "French" in name
    flag = ' <span class="pri">★ primary</span>' if name == "Dev — French (validation)" else ""
    sections += f"""
<h2>{name}{flag}</h2>
<p class="hdr">n = {r['n']:,} &nbsp;·&nbsp; <b>AUC {r['auc']:.4f}</b> &nbsp;·&nbsp; <b>Accuracy {r['accuracy']:.4f}</b></p>
<div class="cols">
<table class="metrics">
<thead><tr><th>label</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
<tbody>{metric_table(r['report'])}</tbody>
</table>
<table class="cm">
<thead><tr><th></th><th>pred Chat</th><th>pred Search</th></tr></thead>
<tbody>
<tr><td class="lbl">true Chat</td><td class="tn">{cm[0][0]:,}</td><td class="fp">{cm[0][1]:,}</td></tr>
<tr><td class="lbl">true Search</td><td class="fn">{cm[1][0]:,}</td><td class="tp">{cm[1][1]:,}</td></tr>
</tbody>
</table>
</div>
"""

html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Intent classifier (frozen embeddings + FR-balanced) — full label report</title>
<style>
 body {{ font: 15px/1.55 -apple-system, system-ui, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color:#1a1a1a; }}
 h1 {{ font-size: 23px; margin-bottom: 2px; }} .sub {{ color:#666; margin-top:0; }}
 h2 {{ margin-top: 30px; border-bottom: 2px solid #eee; padding-bottom: 6px; font-size:18px; }}
 .hdr {{ margin: 6px 0 10px; }}
 .cols {{ display: flex; gap: 22px; flex-wrap: wrap; align-items: flex-start; }}
 table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; margin: 4px 0; }}
 table.metrics {{ flex: 1 1 460px; }} table.cm {{ flex: 0 0 auto; }}
 th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: center; }}
 th {{ background:#f6f6f6; font-weight:600; }}
 td.lbl {{ text-align:left; font-weight:600; white-space:nowrap; }}
 tr.avg td {{ font-style: italic; color:#444; }}
 .cm td.tp, .cm td.tn {{ background:#e9f6ec; }} .cm td.fp, .cm td.fn {{ background:#fdecec; }}
 .config {{ background:#fafafa; border-left:3px solid #ccc; padding:10px 16px; font-size:14px; color:#444; }}
 .pri {{ background:#0060df; color:#fff; border-radius:9px; padding:1px 8px; font-size:12px; vertical-align:middle; }}
 code {{ background:#f0f0f3; padding:1px 5px; border-radius:4px; }}
</style></head><body>

<h1>Query intent classifier — full per-label report</h1>
<p class="sub">Tiny 1-layer transformer over frozen static (SIF-scaled) en-fr embeddings, French-balanced training. Positive class = Search.</p>

<div class="config">
<b>Model:</b> frozen <code>Embedding(32000×384)</code> + fixed sinusoidal PE + 1 <code>TransformerEncoderLayer</code>
(d=384, heads={args['nhead']}, ff={args['dim_ff']}, GELU, norm-first) + mean-pool + <code>Linear(384→1)</code>.<br>
<b>Training:</b> fr_train ×{args['fr_oversample']} + en_train (≈ language-balanced), BCE (no class weighting),
embeddings frozen, max_len={args['max_len']}. Threshold 0.5.<br>
<b>Size:</b> ~889K trainable params (embeddings reused, not retrained).
</div>
{sections}
<p class="sub">Confusion matrix cells: green = correct (TN/TP), red = errors (FP/FN). Generated by artifacts/build_label_report.py.</p>
</body></html>"""

out = f"{ART}/{RUN}/label_report.html"
open(out, "w").write(html)
print(f"wrote {out}")
for name, _ in EVAL:
    r = results[name]
    print(f"{name:<34} n={r['n']:>6}  AUC={r['auc']:.4f}  acc={r['accuracy']:.4f}")
