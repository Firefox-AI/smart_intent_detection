#!/usr/bin/env python3
"""Export the FR-bal tiny-transformer intent classifier to ONNX, slimmed and quantization-ready.

The model is a custom PyTorch module (not a HF model), so optimum can't export it — we use
torch.onnx.export directly, then onnxslim (same reason ml-form-autofill/onnx/export_onnx.py
slims: ORT's quantizer needs a shape-inferable graph).

I/O (standard text-classification shape so transformers.js can run/quantize it):
  inputs : input_ids [B,L] int64, attention_mask [B,L] int64 (1 = token, 0 = pad)
  output : logits [B,2]  ->  softmax = [P(Chat), P(Search)]   (label 0 = Chat, 1 = Search)

The single-sigmoid head is mapped to 2 logits as [0, z], so softmax([0,z]) = [1-sigmoid(z), sigmoid(z)].

Output: intent_tiny_fr_oversample/onnx/{model.onnx, config.json, tokenizer.json}
"""
import json
import os
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import onnxslim
import torch
import torch.nn as nn
from tokenizers import Tokenizer

import train_tiny_transformer as T

ART = os.path.dirname(os.path.abspath(__file__))
RUN = "intent_tiny_fr_oversample"
OUT = f"{ART}/{RUN}/onnx"
os.makedirs(OUT, exist_ok=True)

ck = torch.load(f"{ART}/{RUN}/model.pt", map_location="cpu", weights_only=False)
a = ck["args"]
dim = a.get("pca_dim") or 384

# Build with a generous PE length (queries are short; trained at max_len=48) and load weights.
model = T.TinyTransformerClassifier(
    np.zeros((32000, dim), np.float32), a["nhead"], a["dim_ff"], a["dropout"],
    max_len=512, freeze_emb=True
)
sd = dict(ck["state_dict"])
sd.pop("pe", None)  # fixed sinusoidal buffer; ours is recomputed at len 512
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not unexpected and missing == ["pe"], (missing, unexpected)
model.eval()


class ClassifierOnnx(nn.Module):
    """input_ids + attention_mask (1=token,0=pad) -> [B,2] logits (Chat, Search)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        pad_mask = attention_mask == 0          # True where padding
        z = self.m(input_ids, pad_mask).unsqueeze(1)   # [B,1] raw logit (P(Search) via sigmoid)
        return torch.cat([torch.zeros_like(z), z], dim=1)  # [B,2]


wrap = ClassifierOnnx(model).eval()

# Force the exportable attention path (the fused fast path doesn't trace to ONNX).
torch.backends.mha.set_fastpath_enabled(False)

ids = torch.zeros((2, 12), dtype=torch.long)
am = torch.ones((2, 12), dtype=torch.long)
model_path = f"{OUT}/model.onnx"
with torch.no_grad():
    torch.onnx.export(
        wrap, (ids, am), model_path,
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
print(f"Exported -> {model_path}")

slimmed = onnxslim.slim(model_path)
onnx.save(slimmed, model_path, save_as_external_data=False)
# torch's exporter may have written external weights; the slimmed save inlines them,
# so drop the now-stale .data leftover to keep a single self-contained model.onnx.
if os.path.exists(model_path + ".data"):
    os.remove(model_path + ".data")
print(f"Slimmed -> {model_path}")

# Companion files so it's a complete transformers.js text-classification model.
json.dump({
    "model_type": "static-embeddings-intent",
    "architectures": ["TinyTransformerForSequenceClassification"],
    "id2label": {"0": "Chat", "1": "Search"},
    "label2id": {"Chat": 0, "Search": 1},
    "num_labels": 2,
    "max_position_embeddings": 48,
    "note": "1-layer transformer over frozen en-fr static embeddings; softmax(logits)[:,1] = P(Search).",
}, open(f"{OUT}/config.json", "w"), indent=2)
shutil.copy(f"{T.STATIC_MODEL}/tokenizer.json", f"{OUT}/tokenizer.json")

# ---- Verify ONNX matches PyTorch (and the dumped probs) on real eval queries ----
tok = Tokenizer.from_file(f"{T.STATIC_MODEL}/tokenizer.json")
groups = T.load_groups()
texts = groups["fr_val"][0][:64] + groups["en_eval"][0][:64]
np_ids, pad = T.encode_texts(texts, tok, 48)          # pad: True where padding
attn = (~pad).astype(np.int64)

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
onnx_logits = sess.run(None, {"input_ids": np_ids.astype(np.int64), "attention_mask": attn})[0]
onnx_psearch = (1.0 / (1.0 + np.exp(-(onnx_logits[:, 1] - onnx_logits[:, 0]))))

with torch.no_grad():
    torch_logit = model(torch.from_numpy(np_ids), torch.from_numpy(pad)).numpy()
torch_psearch = 1.0 / (1.0 + np.exp(-torch_logit))

max_diff = float(np.max(np.abs(onnx_psearch - torch_psearch)))
print(f"max |ONNX - PyTorch| P(Search) over {len(texts)} queries: {max_diff:.2e}")
assert max_diff < 1e-4, "ONNX output does not match PyTorch"

size = os.path.getsize(model_path)
print(f"OK. model.onnx = {size/1e6:.2f} MB  ({size:,} bytes)")
print(f"Folder ready for quantization: {OUT}")
print("  e.g.  python -m scripts.quantize --input_folder %s --output_folder %s "
      "--modes fp16 q8 int8 uint8 q4 q4f16 bnb4 --per_channel" % (OUT, OUT))
