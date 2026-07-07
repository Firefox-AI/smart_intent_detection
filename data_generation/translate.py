#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import csv
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel


MODEL_NAME = "gpt-5.2"

SYSTEM_TRANSLATE_PROMPT = """
You are tasked to translate english browser queries into {}.
Example when language is french:
  Query: Hello world
  translation: Bonjour le monde
""".strip()


class TranslationFormat(BaseModel):
    translation: str


client: Optional[OpenAI] = None


def init_worker(model_name: str) -> None:
    global client, MODEL_NAME
    MODEL_NAME = model_name
    client = OpenAI()


def translate_query(query: str, lang: str) -> str:
    if client is None:
        raise RuntimeError("OpenAI client was not initialized")

    completion = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_TRANSLATE_PROMPT.format(lang)},
            {"role": "user", "content": f"Query: {query}"},
        ],
        response_format=TranslationFormat,
    )

    response = completion.choices[0].message.parsed
    ret = response.translation

    if ret.endswith("?"):
        ret = ret[:-1]

    return ret.strip()


def translate_query_with_retry(
    item: Dict[str, Any],
    lang: str,
    max_retries: int,
    sleep_seconds: float,
) -> Dict[str, Any]:
    row_id = item["row_id"]
    query = item["query"]

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            translation = translate_query(query, lang)
            return {
                "row_id": row_id,
                "translation": translation,
            }
        except Exception as e:
            last_error = str(e)

            if attempt < max_retries:
                time.sleep(sleep_seconds * attempt)

    return {
        "row_id": row_id,
        "translation": None,
        "error": last_error,
    }


def worker_translate(args):
    item, lang, max_retries, sleep_seconds = args
    return translate_query_with_retry(
        item=item,
        lang=lang,
        max_retries=max_retries,
        sleep_seconds=sleep_seconds,
    )


def die(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def chunked(items: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield i // batch_size, items[i : i + batch_size]


def write_batch_result(path: Path, results: List[Dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(".jsonl.tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tmp_path.replace(path)


def read_batch_result(path: Path) -> List[Dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate CSV data in restartable batches using OpenAI and multiprocessing."
    )

    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--postfix", required=True)

    parser.add_argument("--text-column", default="Query")
    parser.add_argument("--output-column", default="Translation")

    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=512)

    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        die(f"Input file not found: {input_path}")

    if args.batch_size <= 0:
        die("--batch-size must be positive")

    if args.workers <= 0:
        die("--workers must be positive")

    if not os.environ.get("OPENAI_API_KEY"):
        die("OPENAI_API_KEY is not set")

    output_column = args.output_column or f"{args.text_column}_{args.lang}"

    tmp_dir = Path(f"./tmp-translation-{args.lang}-{args.postfix}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading input CSV: {input_path}")
    df = pd.read_csv(input_path)

    if args.text_column not in df.columns:
        die(f"Column {args.text_column!r} not found. Available: {list(df.columns)}")

    df = df.reset_index(drop=True)
    df["row_id"] = df.index

    items = [
        {
            "row_id": int(row_id),
            "query": str(query),
        }
        for row_id, query in zip(df["row_id"], df[args.text_column])
    ]

    batch_files = []

    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.model,),
    ) as pool:
        for batch_id, batch_items in chunked(items, args.batch_size):
            batch_file = tmp_dir / f"batch-{batch_id:06d}.jsonl"
            batch_files.append(batch_file)

            if batch_file.exists():
                print(f"Skipping completed batch: {batch_file}")
                continue

            print(f"Processing batch {batch_id} with {len(batch_items)} rows")

            tasks = [
                (item, args.lang, args.max_retries, args.sleep_seconds)
                for item in batch_items
            ]

            results = list(pool.imap(worker_translate, tasks))

            failed = [r for r in results if r.get("translation") is None]
            if failed:
                print(f"WARNING: batch {batch_id} has {len(failed)} failed rows")

            write_batch_result(batch_file, results)
            print(f"Wrote batch result: {batch_file}")

    print("Consolidating batch files...")

    translations: Dict[int, str] = {}
    errors: Dict[int, str] = {}

    for batch_file in sorted(batch_files):
        if not batch_file.exists():
            die(f"Missing batch file: {batch_file}")

        for row in read_batch_result(batch_file):
            row_id = int(row["row_id"])

            if row.get("translation") is not None:
                translations[row_id] = row["translation"]
            else:
                errors[row_id] = row.get("error", "unknown error")

    if errors:
        error_file = tmp_dir / "errors.jsonl"
        with error_file.open("w", encoding="utf-8") as f:
            for row_id, error in sorted(errors.items()):
                f.write(
                    json.dumps(
                        {
                            "row_id": row_id,
                            "error": error,
                            "query": str(df.loc[row_id, args.text_column]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        die(f"{len(errors)} rows failed. See {error_file}")

    missing = [int(i) for i in df["row_id"] if int(i) not in translations]
    if missing:
        die(f"Missing translations for {len(missing)} rows. First missing row_id={missing[0]}")

    df[output_column] = df["row_id"].map(translations)
    df = df[[output_column, 'Label', 'source']]
    df.columns = ['Query', 'Label', 'source']
    #df = df.drop(columns=["row_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )

    print(f"Done. Wrote final output: {output_path}")


if __name__ == "__main__":
    main()

