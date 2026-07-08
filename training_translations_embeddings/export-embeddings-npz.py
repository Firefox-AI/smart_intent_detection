#!/usr/bin/env python3
"""Export static sentence embeddings directly from a Marian .npz model.

Faithfully replicates TranslationModel::embed from the inference engine:
gather the Wemb rows for a text's SentencePiece tokens, apply SIF weighting
(a / (a + exp(score))) using the vocab's unigram scores, mean-pool, then
L2-normalize. This works on the raw .npz that the buffer-based WASM engine
cannot load. Can also export a Model2Vec StaticModel (--write-model2vec).

Dependencies:
    pip install numpy sentencepiece
    pip install model2vec tokenizers   # only needed for --write-model2vec

Getting the data (a Marian student model .npz and its source SentencePiece vocab):
    These live in the Mozilla translations training data bucket on GCS. For the
    en-fr student model used by default, download the two files with:

      BASE="https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/models/en-fr/retrain_hr_NLIxDbE1TBGyOTI-zwZagw/student"
      mkdir -p models/enfr-npz
      curl -sSL -o models/enfr-npz/final.model.npz.best-chrf.npz "$BASE/final.model.npz.best-chrf.npz"
      curl -sSL -o models/enfr-npz/vocab.en.spm                  "$BASE/vocab.en.spm"

    Other language pairs and runs live under .../models/<pair>/<run>/student/ in
    the same bucket. Browse https://console.cloud.google.com/storage/browser/
    moz-fx-translations-data--303e-prod-translations-data/models to find them.
    Pass the downloaded paths via --npz and --vocab.

Examples:
    # Text export only (uses the default paths/URLs below):
    python export-embeddings-npz.py out.txt --npz <model.npz> --vocab <vocab.spm>

    # Also write a Model2Vec StaticModel into ./en-fr-static-model2vec:
    python export-embeddings-npz.py out.txt --npz <model.npz> --vocab <vocab.spm> \\
        --write-model2vec --output-dir .
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import sentencepiece as spm

MODEL_DIR = "models/enfr-npz"
DEFAULT_NPZ_PATH = f"{MODEL_DIR}/final.model.npz.best-chrf.npz"
DEFAULT_VOCAB_PATH = f"{MODEL_DIR}/vocab.en.spm"

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/"
    "models/en-fr/retrain_hr_NLIxDbE1TBGyOTI-zwZagw/student/final.model.npz.best-chrf.npz"
)
DEFAULT_VOCAB_URL = (
    "https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/"
    "models/en-fr/retrain_hr_NLIxDbE1TBGyOTI-zwZagw/student/vocab.en.spm"
)

# Matches the `a` constant in TranslationModel::embed.
SIF_A = 1e-3

# IDF scaling parameters baked into the Model2Vec vectors (see the config below).
IDF_EPSILON = 1e-12
IDF_CLIP_MIN = 1.0
IDF_CLIP_MAX = 15.0

TOKENS = ["queen", "king", "fork", "spoon", "roi", "reine"]


def die(message):
    print(message, file=sys.stderr)
    sys.exit(1)


def get_score(sp, token_id):
    # sentencepiece exposes the unigram log-prob; method name varies by version.
    if hasattr(sp, "GetScore"):
        return sp.GetScore(token_id)
    return sp.get_score(token_id)


def embed(wemb, sp, text):
    ids = sp.encode(text, out_type=int, add_bos=False, add_eos=False)
    if not ids:
        return None

    dim = wemb.shape[1]
    result = np.zeros(dim, dtype=np.float64)
    weight_sum = 0.0
    for token_id in ids:
        p = math.exp(get_score(sp, token_id))
        weight = SIF_A / (SIF_A + p)
        weight_sum += weight
        result += weight * wemb[token_id]

    if weight_sum > 0:
        result /= weight_sum

    norm = np.linalg.norm(result)
    if norm > 0:
        result /= norm

    return result, ids


def cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_idf_weights(sp):
    """Per-piece IDF weights from the SentencePiece unigram scores.

    Mirrors the formula recorded in the Model2Vec config:
    probabilities are the normalized exp(score); the weight is
    clip(-log(max(p, eps)), clip_min, clip_max), rescaled so the mean is 1.
    """
    scores = np.array(
        [sp.get_score(i) for i in range(sp.get_piece_size())], dtype=np.float64
    )
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum()

    idf = -np.log(np.maximum(probabilities, IDF_EPSILON))
    idf = np.clip(idf, IDF_CLIP_MIN, IDF_CLIP_MAX)
    return idf / idf.mean()


def write_model2vec_static_model(
    output_dir: Path,
    dirname: str,
    embeddings: np.ndarray,
    sp: spm.SentencePieceProcessor,
    model_url: str,
    vocab_url: str,
    idf_weights: np.ndarray,
    model_name: str,
    base_model_name: str,
    language: List[str],
) -> Path:
    """Write a Model2Vec StaticModel using Model2Vec's own save_pretrained API.

    This requires optional dependencies:
      pip install model2vec tokenizers
    """
    try:
        from model2vec import StaticModel  # type: ignore
        from tokenizers import (  # type: ignore
            Regex,
            Tokenizer,
            decoders,
            normalizers,
            pre_tokenizers,
        )
        from tokenizers.models import Unigram  # type: ignore
    except Exception as exc:
        die(
            "--write-model2vec requires optional dependencies. Install them with "
            "`pip install model2vec tokenizers` and rerun. "
            f"Original import error: {exc}"
        )

    vocab = [
        (sp.id_to_piece(i), float(sp.get_score(i))) for i in range(sp.get_piece_size())
    ]
    # Build the Unigram tokenizer with the vocab's unk token wired in, otherwise
    # encoding any out-of-vocab input raises "unknown token but unk_id is missing".
    # This mirrors tokenizers' SentencePieceUnigramTokenizer, plus unk_id.
    tokenizer = Tokenizer(Unigram(vocab, sp.unk_id()))
    # NFKC + collapse-runs-of-spaces. We intentionally omit SentencePiece's `Nmt`
    # normalizer: transformers.js does not implement it ("Unknown Normalizer type: Nmt")
    # and would fail to load the tokenizer, while Nmt is a no-op for normal query text.
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFKC(), normalizers.Replace(Regex(" {2,}"), " ")]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁", prepend_scheme="always"
    )
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")

    config = {
        "normalize": True,
        "embedding_source": "bergamot_marian_wemb",
        "model_url": model_url,
        "vocab_url": vocab_url,
        "embedding_dtype": str(embeddings.dtype),
        "idf_scaling": {
            "enabled": True,
            "probability_source": "normalized exp(SentencePiece score)",
            "formula": "clip(-log(max(probability, epsilon)), clip_min, clip_max) / mean(clipped_idf)",
            "epsilon": IDF_EPSILON,
            "clip_min": IDF_CLIP_MIN,
            "clip_max": IDF_CLIP_MAX,
            "normalized_mean": 1.0,
            "weight_min": float(idf_weights.min()),
            "weight_max": float(idf_weights.max()),
            "weight_mean": float(idf_weights.mean()),
        },
    }

    static_model = StaticModel(
        vectors=embeddings.astype(np.float32, copy=False),
        tokenizer=tokenizer,
        config=config,
        normalize=True,
        base_model_name=base_model_name,
        language=language,
    )

    model2vec_dir = output_dir / dirname
    static_model.save_pretrained(model2vec_dir, model_name=model_name)
    return model2vec_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_file",
        nargs="?",
        default="embeddings_export_npz.txt",
        help="Path for the text embeddings export.",
    )
    parser.add_argument(
        "--npz",
        default=DEFAULT_NPZ_PATH,
        help="Path to the Marian student model .npz (contains the Wemb matrix).",
    )
    parser.add_argument(
        "--vocab",
        default=DEFAULT_VOCAB_PATH,
        help="Path to the source SentencePiece vocab (.spm) for the model.",
    )
    parser.add_argument(
        "--write-model2vec",
        action="store_true",
        help="Also export a Model2Vec StaticModel (needs `pip install model2vec tokenizers`).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write the Model2Vec StaticModel into.",
    )
    parser.add_argument(
        "--dirname",
        default="en-fr-static-model2vec",
        help="Subdirectory name for the Model2Vec StaticModel.",
    )
    parser.add_argument("--model-name", default="bergamot-en-fr-static")
    parser.add_argument(
        "--base-model-name",
        default="mozilla/translations-en-fr-student",
    )
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL)
    parser.add_argument("--vocab-url", default=DEFAULT_VOCAB_URL)
    parser.add_argument(
        "--language",
        nargs="+",
        default=["en", "fr"],
        help="Language codes recorded in the StaticModel metadata.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_file = args.output_file

    wemb = np.load(args.npz)["Wemb"]
    sp = spm.SentencePieceProcessor(model_file=args.vocab)

    results = []
    for token in TOKENS:
        embedding, ids = embed(wemb, sp, token)
        pieces = sp.encode(token, out_type=str, add_bos=False, add_eos=False)
        results.append((token, embedding, ids, pieces))

    lines = []
    lines.append("# Static embeddings exported directly from a Marian .npz model")
    lines.append(f"# Model: {args.npz}")
    lines.append(f"# Source vocab: {args.vocab}")
    lines.append(f"# Wemb shape: {wemb.shape} (vocab x dim), dtype: {wemb.dtype}")
    lines.append(f"# Tokens: {', '.join(repr(t) for t in TOKENS)}")
    lines.append(
        "# Method: SIF-weighted (a=1e-3) mean-pool of Wemb rows, then L2-normalized "
        "(replicates TranslationModel::embed)."
    )
    lines.append("")

    lines.append("## Tokenization (token -> SentencePiece pieces / ids)")
    for token, _embedding, ids, pieces in results:
        lines.append(f"{token!r}: {pieces} {ids}")
    lines.append("")

    lines.append("## Embeddings (token then comma-separated values)")
    for token, embedding, _ids, _pieces in results:
        lines.append(repr(token))
        lines.append(",".join(repr(float(v)) for v in embedding))
    lines.append("")

    labels = [repr(token) for token, *_ in results]
    embeddings = [embedding for _t, embedding, *_ in results]

    lines.append("## Cosine similarity matrix")
    lines.append(",".join([""] + labels))
    for i in range(len(results)):
        row = [labels[i]]
        for j in range(len(results)):
            row.append(f"{cosine(embeddings[i], embeddings[j]):.6f}")
        lines.append(",".join(row))
    lines.append("")

    lines.append("## Cosine similarity (unique pairs)")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            sim = cosine(embeddings[i], embeddings[j])
            lines.append(f"{labels[i]} <-> {labels[j]}: {sim:.6f}")
    lines.append("")

    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(f"Wrote embeddings for {len(TOKENS)} tokens to {output_file}")

    if args.write_model2vec:
        # Model2Vec mean-pools per-token vectors with no runtime weighting, so the
        # IDF weighting is baked into the stored vectors (one weight per piece).
        idf_weights = compute_idf_weights(sp)
        scaled_vectors = wemb * idf_weights[:, None]

        model2vec_dir = write_model2vec_static_model(
            output_dir=args.output_dir,
            dirname=args.dirname,
            embeddings=scaled_vectors,
            sp=sp,
            model_url=args.model_url,
            vocab_url=args.vocab_url,
            idf_weights=idf_weights,
            model_name=args.model_name,
            base_model_name=args.base_model_name,
            language=args.language,
        )
        print(f"Wrote Model2Vec StaticModel to {model2vec_dir}")


if __name__ == "__main__":
    main()
