#!/usr/bin/env bash
# Fetch / regenerate the data files NOT shipped in this backup (too large to bundle):
#
#   source/            raw Marian student model (.npz) + vocab        [download from GCS]
#   static_model/      Model2Vec StaticModel (model.safetensors)      [built from the .npz]
#   dataset/intent2/   FR+EN query-intent dataset (all.jsonl+meta)     [download from HuggingFace]
#
# Run from anywhere; writes into this backup. Requires the venv described in README.md.
set -euo pipefail

PY="${PY:-/tmp/spm_venv/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # transformer_backup root

GCS="https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/models/en-fr/retrain_hr_NLIxDbE1TBGyOTI-zwZagw/student"

echo "==> [1/3] Raw Marian model + vocab -> source/"
mkdir -p "$HERE/source"
curl -sSL -o "$HERE/source/final.model.npz.best-chrf.npz"             "$GCS/final.model.npz.best-chrf.npz"
curl -sSL -o "$HERE/source/vocab.en.spm"                              "$GCS/vocab.en.spm"
curl -sSL -o "$HERE/source/final.model.npz.best-chrf.npz.decoder.yml" "$GCS/final.model.npz.best-chrf.npz.decoder.yml"

echo "==> [2/3] Static embedding model -> static_model/ (Nmt-free tokenizer)"
"$PY" "$HERE/scripts/export-embeddings-npz.py" /tmp/_emb.txt \
    --npz   "$HERE/source/final.model.npz.best-chrf.npz" \
    --vocab "$HERE/source/vocab.en.spm" \
    --write-model2vec --output-dir "$HERE" --dirname static_model

echo "==> [3/3] Dataset -> dataset/intent2/ (downloads Mozilla/query-intent-detection-dataset[-french])"
"$PY" "$HERE/scripts/export_combined.py" "$HERE/dataset/intent2"

echo "Done. Fetched: source/, static_model/, dataset/intent2/"
