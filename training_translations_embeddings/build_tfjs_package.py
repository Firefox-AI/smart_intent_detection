#!/usr/bin/env python3
"""Repackage the FR-bal ONNX variants as a transformers.js text-classification model dir,
and compute an onnxruntime reference (P(Search)) for sample queries to check Node against.

Layout produced (intent_tiny_fr_oversample/tfjs/):
  config.json, tokenizer.json, tokenizer_config.json
  onnx/model.onnx (fp32), onnx/model_fp16.onnx, onnx/model_quantized.onnx (q8)
"""
import json
import os
import shutil

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

import train_tiny_transformer as T

ART = os.path.dirname(os.path.abspath(__file__))
SRC = f"{ART}/intent_tiny_fr_oversample/onnx"
DST = f"{ART}/intent_tiny_fr_oversample/tfjs"
os.makedirs(f"{DST}/onnx", exist_ok=True)

# Only the three variants we're testing.
VARIANTS = {"fp32": "model.onnx", "fp16": "model_fp16.onnx", "q8": "model_quantized.onnx"}
for fn in VARIANTS.values():
    shutil.copy(f"{SRC}/{fn}", f"{DST}/onnx/{fn}")

# Root config: model_type=bert so transformers.js routes to its generic seq-classification
# ONNX runner (the JS class just tokenizes -> runs the session -> logits; our graph has the rest).
json.dump({
    "model_type": "bert",
    "architectures": ["BertForSequenceClassification"],
    "id2label": {"0": "Chat", "1": "Search"},
    "label2id": {"Chat": 0, "Search": 1},
    "num_labels": 2,
    "problem_type": "single_label_classification",
    "max_position_embeddings": 48,
}, open(f"{DST}/config.json", "w"), indent=2)

json.dump({
    "tokenizer_class": "PreTrainedTokenizerFast",
    "model_max_length": 48,
    "unk_token": "<unk>",
    "eos_token": "</s>",
    # The vocab has no dedicated pad token (SentencePiece pad_id = -1). Pad with
    # </s> (id 0); it's masked out by attention_mask, matching training. Without a
    # pad_token, transformers.js batch padding throws "can't convert undefined to BigInt".
    "pad_token": "</s>",
    "clean_up_tokenization_spaces": False,
}, open(f"{DST}/tokenizer_config.json", "w"), indent=2)

# Copy the tokenizer and strip the `Nmt` normalizer, which transformers.js can't load
# ("Unknown Normalizer type: Nmt"); it's a no-op for normal query text.
tok_json = json.load(open(f"{T.STATIC_MODEL}/tokenizer.json"))
norms = tok_json.get("normalizer", {}).get("normalizers")
if norms:
    tok_json["normalizer"]["normalizers"] = [n for n in norms if n.get("type") != "Nmt"]
json.dump(tok_json, open(f"{DST}/tokenizer.json", "w"), ensure_ascii=False)

# ---- onnxruntime reference on sample queries (batch of 1 each, no padding) ----
QUERIES = [
    ("meilleur restaurant italien à lyon", "fr"),
    ("raconte-moi une blague sur les chats", "fr"),
    ("horaires train paris marseille", "fr"),
    ("weather in paris tomorrow", "en"),
    ("write me a poem about the ocean", "en"),
    ("explain quantum entanglement like I'm five", "en"),
]
tok = Tokenizer.from_file(f"{DST}/tokenizer.json")

ref = {}
for dt, fn in VARIANTS.items():
    sess = ort.InferenceSession(f"{DST}/onnx/{fn}", providers=["CPUExecutionProvider"])
    scores = []
    for text, _lang in QUERIES:
        ids = np.array([tok.encode(text).ids], dtype=np.int64)
        attn = np.ones_like(ids)
        logits = sess.run(None, {"input_ids": ids, "attention_mask": attn})[0]
        p = float(1.0 / (1.0 + np.exp(-(logits[0, 1] - logits[0, 0]))))
        scores.append(p)
    ref[dt] = scores

json.dump({"queries": [q for q, _ in QUERIES], "p_search": ref},
          open(f"{ART}/tfjs_reference.json", "w"), indent=2)

print(f"packaged -> {DST}")
print(f"{'query':<46}{'fp32':>8}{'fp16':>8}{'q8':>8}")
for i, (q, _l) in enumerate(QUERIES):
    print(f"{q:<46}{ref['fp32'][i]:>8.4f}{ref['fp16'][i]:>8.4f}{ref['q8'][i]:>8.4f}")
