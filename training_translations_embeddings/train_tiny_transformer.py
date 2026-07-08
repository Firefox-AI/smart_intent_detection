#!/usr/bin/env python3
"""Tiny transformer intent classifier over static token embeddings.

A different architecture from the mean-pool + MLP head: keep the per-token static
embeddings, run them through ONE TransformerEncoderLayer (with fixed sinusoidal
positional encoding), masked mean-pool, then a single linear unit -> sigmoid (binary).

By default the token embeddings are the RAW Marian Wemb and are FROZEN, so the model
stays tiny and reuses the existing export. Positive class = Search (1), Chat = 0.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from safetensors.numpy import load_file as load_safetensors
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,
)
from tokenizers import Tokenizer

ART = os.path.dirname(os.path.abspath(__file__))

def load_groups():
    meta = json.load(open(f"{ART}/intent2/meta.json"))
    texts = [json.loads(line)["text"] for line in open(f"{ART}/intent2/all.jsonl")]
    groups, off = {}, 0
    for name in meta["order"]:
        n = meta["counts"][name]
        groups[name] = (texts[off:off + n], list(meta["labels"][name]))
        off += n
    return groups


def sinusoidal_pe(max_len, dim):
    pe = torch.zeros(max_len, dim)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TinyTransformerClassifier(nn.Module):
    def __init__(self, emb_weight, nhead, dim_ff, dropout, max_len, freeze_emb):
        super().__init__()
        vocab, dim = emb_weight.shape
        self.embed = nn.Embedding.from_pretrained(
            torch.from_numpy(emb_weight), freeze=freeze_emb
        )
        self.register_buffer("pe", sinusoidal_pe(max_len, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(dim, 1)

    def forward(self, ids, pad_mask):
        # ids: [B, L]; pad_mask: [B, L] True where padding.
        x = self.embed(ids) + self.pe[: ids.shape[1]].unsqueeze(0)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        keep = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)  # logit


def encode_texts(texts, tokenizer, max_len):
    encs = tokenizer.encode_batch([t if t else " " for t in texts])
    ids = np.zeros((len(texts), max_len), dtype=np.int64)
    mask = np.ones((len(texts), max_len), dtype=bool)  # True = pad
    for i, e in enumerate(encs):
        seq = e.ids[:max_len] or [0]
        ids[i, : len(seq)] = seq
        mask[i, : len(seq)] = False
    return ids, mask


@torch.no_grad()
def predict_proba(model, ids, mask, device, batch_size=1024):
    model.eval()
    out = []
    for s in range(0, len(ids), batch_size):
        bi = torch.from_numpy(ids[s:s + batch_size]).to(device)
        bm = torch.from_numpy(mask[s:s + batch_size]).to(device)
        out.append(torch.sigmoid(model(bi, bm)).cpu().numpy())
    return np.concatenate(out)


def metrics(name, y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {
        "group": name, "n": int(len(y)), "pos_Search": int(np.sum(y)),
        "auc": float(roc_auc_score(y, prob)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
    }


def main():
    global STATIC_MODEL, RAW_NPZ
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", choices=["raw", "scaled"], default="raw")
    p.add_argument("--train-embeddings", action="store_true", help="unfreeze token embeddings")
    p.add_argument("--max-len", type=int, default=48)
    p.add_argument("--nhead", type=int, default=6)
    p.add_argument("--dim-ff", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--no-pos-weight", action="store_true",
                   help="disable BCE class balancing (train at the natural base rate)")
    p.add_argument("--fr-oversample", type=int, default=1,
                   help="replicate fr_train this many times in the training set")
    p.add_argument("--pca-dim", type=int, default=0,
                   help="PCA-reduce the embedding dim to this many components (0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=f"{ART}/intent_tiny_transformer")
    p.add_argument("--static-model", required=True)
    p.add_argument("--npz", required=True)
    args = p.parse_args()

    RAW_NPZ = args.npz
    STATIC_MODEL = args.static_model

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    if args.embeddings == "scaled":
        emb = load_safetensors(f"{STATIC_MODEL}/model.safetensors")["embeddings"]
    else:
        emb = np.load(RAW_NPZ)["Wemb"]
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    if args.pca_dim:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=args.pca_dim, random_state=args.seed).fit_transform(emb)
        emb = np.ascontiguousarray(emb, dtype=np.float32)
    print(f"embeddings={args.embeddings} shape={emb.shape} device={device} "
          f"(embedding params: {emb.shape[0] * emb.shape[1]:,})")

    tokenizer = Tokenizer.from_file(f"{STATIC_MODEL}/tokenizer.json")
    groups = load_groups()

    enc = {}
    for name, (texts, labels) in groups.items():
        ids, mask = encode_texts(texts, tokenizer, args.max_len)
        enc[name] = (ids, mask, np.asarray(labels, dtype=np.float32))

    # Train = fr_train (optionally oversampled) + en_train; carve internal 5% val.
    fr, en = enc["fr_train"], enc["en_train"]
    k = args.fr_oversample
    Xi = np.concatenate([fr[0]] * k + [en[0]])
    Xm = np.concatenate([fr[1]] * k + [en[1]])
    y = np.concatenate([fr[2]] * k + [en[2]])
    print(f"fr_oversample={k}: fr={len(fr[2]) * k} en={len(en[2])}")
    perm = np.random.permutation(len(y))
    Xi, Xm, y = Xi[perm], Xm[perm], y[perm]
    nval = max(1, int(0.05 * len(y)))
    tr = slice(nval, None); va = slice(0, nval)
    pos_rate = float(y[tr].mean())
    print(f"train={len(y) - nval} internal-val={nval} pos={pos_rate:.3f}")

    model = TinyTransformerClassifier(
        emb, args.nhead, args.dim_ff, args.dropout, args.max_len,
        freeze_emb=not args.train_embeddings,
    ).to(device)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train_params:,}")

    if args.no_pos_weight:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )

    best_auc, best_state, bad = -1.0, None, 0
    idx = np.arange(nval, len(y))
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(idx)
        t0 = time.time()
        total = 0.0
        for s in range(0, len(idx), args.batch_size):
            b = idx[s:s + args.batch_size]
            bi = torch.from_numpy(Xi[b]).to(device)
            bm = torch.from_numpy(Xm[b]).to(device)
            by = torch.from_numpy(y[b]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(bi, bm), by)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        val_prob = predict_proba(model, Xi[va], Xm[va], device)
        val_auc = roc_auc_score(y[va], val_prob)
        print(f"epoch {epoch + 1}: loss={total / len(idx):.4f} val_auc={val_auc:.4f} "
              f"({time.time() - t0:.0f}s)")
        if val_auc > best_auc + 1e-4:
            best_auc, best_state, bad = val_auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad > args.patience:
                print("early stop")
                break
    if best_state:
        model.load_state_dict(best_state)

    results = []
    for name in ["fr_val", "en_eval", "fr_test", "en_test"]:
        ids, mask, yy = enc[name]
        prob = predict_proba(model, ids, mask, device)
        results.append(metrics(name, yy.astype(int), prob))
    # Dev combined
    ids = np.concatenate([enc["fr_val"][0], enc["en_eval"][0]])
    mask = np.concatenate([enc["fr_val"][1], enc["en_eval"][1]])
    yy = np.concatenate([enc["fr_val"][2], enc["en_eval"][2]]).astype(int)
    results.insert(0, metrics("Dev combined (FR val + EN eval)", yy, predict_proba(model, ids, mask, device)))

    label = {"fr_val": "Dev — French (validation)", "en_eval": "Dev — English (eval)",
             "fr_test": "Test — French", "en_test": "Test — English"}
    for r in results:
        r["group"] = label.get(r["group"], r["group"])

    json.dump(results, open(f"{args.out}/results.json", "w"), indent=1)
    print(f"\n{'group':<34}{'n':>8}{'AUC':>9}{'Prec':>8}{'Recall':>8}{'F1':>8}{'Acc':>8}")
    for r in results:
        print(f"{r['group']:<34}{r['n']:>8}{r['auc']:>9.4f}{r['precision']:>8.4f}"
              f"{r['recall']:>8.4f}{r['f1']:>8.4f}{r['accuracy']:>8.4f}")

    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, f"{args.out}/model.pt")
    print(f"\nsaved model + results to {args.out}")


if __name__ == "__main__":
    main()
