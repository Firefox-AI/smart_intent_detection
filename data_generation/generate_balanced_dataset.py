import os
import json
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

from llm_module import get_category_of_batched_queries, generate_queries, get_category_of_query
from prompts import (
    SYSTEM_PROMPT_GENERAL,
    SYSTEM_PROMPT_SHORT,
    SYSTEM_PROMPT_PHRASE,
    SYSTEM_PROMPT_DYNAMIC,
    SYSTEM_PROMPT_MISSED_URL,
    SYSTEM_PROMPT_TIME_SENSITIVE,
    SYSTEM_PROMPT_CHAT,
    SYSTEM_PROMPT_CHAT_IMPERATIVE,
    SYSTEM_PROMPT_CHAT_AI_COMMON,
    SYSTEM_PROMPT_TAB_COMPARISON,
    DEF_CHAT_AI_COMMON,
)

RAW_DATA_DIR = "./data/raw_balanced"
LABELED_DATA_DIR = "./data/labeled_balanced"
ALL_DATA_FILE_NAME = "./data/labeled_queries_balanced.csv"

DATA_FILE_NAMES = {
    "general": "general.csv",
    "chat": "chat.csv",
    "phrases": "phrases.csv",
    "time_sensitive": "timebound.csv",
    "dynamic": "dynamic.csv",
    "short": "short.csv",
    "missed_navigation": "hybrid_urls.csv",
    "imperative_chat": "imperative_chat.csv",
    "ai_tool_common_query_explanation": "ai_tool_common_query_explanation.csv",
    "ai_tool_common_query_summarization": "ai_tool_common_query_summarization.csv",
    "ai_tool_common_query_translation": "ai_tool_common_query_translation.csv",
    "ai_tool_common_query_coding": "ai_tool_common_query_coding.csv",
    "ai_tool_common_query_assistance": "ai_tool_common_query_assistance.csv",
    "ai_tool_common_query_writing": "ai_tool_common_query_writing.csv",
    "ai_tool_common_query_math": "ai_tool_common_query_math.csv",
    "ai_tool_common_query_comparison": "ai_tool_common_query_comparison.csv",
    "ai_tool_common_query_rephrase": "ai_tool_common_query_rephrase.csv",
    "ai_tool_common_query_roleplay": "ai_tool_common_query_roleplay.csv",
    "ai_tool_common_query_analysis": "ai_tool_common_query_analysis.csv",
    "ai_tool_common_query_learning": "ai_tool_common_query_learning.csv",
    "ai_tool_common_query_task_management": "ai_tool_common_query_task_management.csv",
    "ai_tool_common_query_discussion": "ai_tool_common_query_discussion.csv",
    "ai_tool_common_query_casual_chat": "ai_tool_common_query_casual_chat.csv",
    "ai_tool_common_query_system_setting": "ai_tool_common_query_system_setting.csv",
    "comparison": "comparison.csv",
}


def get_labels_multiprocess_no_batch(
    df,
    max_workers=8
):
    labels = list()
    queries_to_label = df['Query'].tolist()
    while queries_to_label:
        batches = _get_batches_helper(queries_to_label, max_workers, batch_size=1)
        num_workers = min(max_workers, len(batches))
        with Pool(num_workers) as p:
            results = p.map(get_category_of_query_retry_till_success, [arr[0] for arr in batches])

        labels.extend(results)
        print(f"There are {len(queries_to_label)} more data to label...")

    return labels


def get_labels_multiprocess(
    df,
    batch_size=400,
    max_workers=8,
):
    labels = list()
    queries_to_label = df['Query'].tolist()
    while queries_to_label:
        batches = _get_batches_helper(queries_to_label, max_workers, batch_size)
        num_workers = min(max_workers, len(batches))
        with Pool(num_workers) as p:
            results = p.map(get_category_of_batched_queries, batches)

        for items in results:
            labels.extend(items)

    return labels


def get_category_of_query_retry_till_success(query):
    while True:
        try:
            ret = get_category_of_query(query)
            break
        except Exception as err:
            print(f"Got error: {err} while calling API. waiting for 5 seconds.", flush=True)
            time.sleep(5)
    return ret


def _get_batches_helper(queries_to_label, max_workers, batch_size):
    if not queries_to_label:
        return []

    ret = [[]]
    while queries_to_label and len(ret) <= max_workers and len(ret[-1]) <= batch_size:
        query = queries_to_label.pop(0)
        if len(ret[-1]) == batch_size:
            tmp = [query]
            ret.append(tmp)
            continue
        ret[-1].append(query)
    return ret


def generate_quries_to_file(system_prompt, number_of_samples, output_file):
    query_list = generate_queries((system_prompt, number_of_samples))

    df = pd.DataFrame({"Query": query_list})
    df.to_csv(output_file, index=False)
    print(f"Finished sample generation and wrote to {output_file}", flush=True)


def generate_queries_multiprocess(
    system_prompt: str,
    number_of_samples: int,
    output_file: str,
    batch_size: int = 400,
    max_workers: int = 8,
    max_tries: int = 50,
    queries_to_avoid: set = set(),
):
    collected = set()
    num_tries = 0

    while num_tries < max_tries and len(collected) < number_of_samples:
        with Pool(max_workers) as p:
            results = p.map(generate_queries, [(system_prompt, batch_size) for _ in range(max_workers)])

        for items in results:
            items = set(items) - queries_to_avoid
            collected.update(items)

        num_tries += 1

        print(f"Round {num_tries} done. Collected {len(collected)} samples.", flush=True)

    df = pd.DataFrame({"Query": list(collected)})
    df.to_csv(output_file, index=False)
    print(f"Wrote {df.shape[0]} queries to {output_file}")


def label_raw_data():
    for key, file_path in DATA_FILE_NAMES.items():
        print(f"Labeling {key} subset.")
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, file_path))
        if not key.startswith("comparison"):
            continue
        if key in ["general", 'chat', 'imperative_chat']:
            df['Label'] = get_labels_multiprocess_no_batch(df, max_workers=32)
        if key.startswith("ai_tool_common_query"):
            df['Label'] = get_labels_multiprocess_no_batch(df, max_workers=32)
        if key in ["comparison"]:
            df['Label'] = get_labels_multiprocess_no_batch(df, max_workers=32)
        else:
            df['Label'] = ['Search'] * df.shape[0]

        df.to_csv(os.path.join(LABELED_DATA_DIR, file_path), index=False)
        print(f"Finished {key} subset.")


def collect_labeled_data():
    collection = list()
    for file_name in os.listdir(LABELED_DATA_DIR):
        if not file_name.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(LABELED_DATA_DIR, file_name))
        df['source'] = file_name[:-4]
        collection.append(df)
    df = pd.concat(collection, ignore_index=True)
    df.drop_duplicates(subset=['Query', 'Label'], inplace=True)

    return df


def helper_postprocess(query):
    if not isinstance(query, str):
        return ""
    return query.replace("?", "")


def split_dataset(file_name, seed=666, test_ratio=0.2):
    df = pd.read_csv(file_name)
    index = list(df.index)
    np.random.seed(seed)
    np.random.shuffle(index)

    train_index, other_index = split_helper(index, split_ratio=0.8)
    eval_index, test_index = split_helper(other_index, split_ratio=0.5)

    train_set = df.loc[train_index, :]
    train_set.to_csv("./data/balanced_training_data.csv", index=False)
    eval_set = df.loc[eval_index, :]
    eval_set.to_csv("./data/balanced_evaluation_data.csv", index=False)
    test_set = df.loc[test_index, :]
    test_set.to_csv("./data/balanced_test_data.csv", index=False)


def split_helper(index_array, split_ratio=0.5):
    split = int(split_ratio * len(index_array)) + 1
    first = index_array[: split]
    second = index_array[split:]
    return first, second


def main():
    if not os.path.isdir(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
    if not os.path.isdir(LABELED_DATA_DIR):
        os.makedirs(LABELED_DATA_DIR)

    with open("./data/golden_chat_data_to_avoid.json") as f:
        queries_to_avoid = set(json.load(f))

    #generate_general_queries
    #generate_queries_multiprocess(SYSTEM_PROMPT_GENERAL, 20000, os.path.join(RAW_DATA_DIR, DATA_FILE_NAMES['general']))

    # only generate chat intent queries
    #generate_queries_multiprocess(SYSTEM_PROMPT_CHAT, 100000, os.path.join(RAW_DATA_DIR, DATA_FILE_NAMES['chat']))

    #generate_phrase_queries
    #generate_queries_multiprocess(SYSTEM_PROMPT_PHRASE, 20000, os.path.join(RAW_DATA_DIR, DATA_FILE_NAMES['phrases']))

    #generate_time_sensitive_queries
    #generate_queries_multiprocess(SYSTEM_PROMPT_TIME_SENSITIVE, 20000, os.path.join(RAW_DATA_DIR, DATA_FILE_NAMES['time_sensitive']))

    #generate_dynamic_information_queries
    #generate_queries_multiprocess(SYSTEM_PROMPT_DYNAMIC, 20000, os.path.join(RAW_DATA_DIR, DATA_FILE_NAMES['dynamic']))

    #generate_short_queries
    #generate_queries_multiprocess(SYSTEM_PROMPT_SHORT, 20000, os.path.join(RAW_DATA_DIR, DATA_FILE_NAMES['short']))

    #generate_missed_navigations
    #generate_queries_multiprocess(SYSTEM_PROMPT_MISSED_URL, 20000, os.path.join(RAW_DATA_DIR, DATA_FILE_NAMES['missed_navigation']))

    #generate imperative form chat queries
    #generate_queries_multiprocess(SYSTEM_PROMPT_CHAT_IMPERATIVE, 20000, os.path.join(RAW_DATA_DIR, DATA_FILE_NAMES['imperative_chat']), queries_to_avoid=queries_to_avoid)

    #generate tab comparison queries
    #generate_queries_multiprocess(SYSTEM_PROMPT_TAB_COMPARISON, 20000, os.path.join(RAW_DATA_DIR, DATA_FILE_NAMES['comparison']))

    #generate common types of AI tool queries specifically
    # for key, csv_path in DATA_FILE_NAMES.items():
    #     if not key.startswith("ai_tool_common_query"):
    #         continue
    #     system_prompt = SYSTEM_PROMPT_CHAT_AI_COMMON.format(DEF_CHAT_AI_COMMON[key])
    #     generate_queries_multiprocess(system_prompt, 2000, os.path.join(RAW_DATA_DIR, csv_path), queries_to_avoid=queries_to_avoid)

    label_raw_data()

    all_data = collect_labeled_data()
    all_data['Query'] = all_data['Query'].apply(helper_postprocess)
    all_data.to_csv(ALL_DATA_FILE_NAME, index=False)

    split_dataset(ALL_DATA_FILE_NAME)
    print("All done.", flush=True)


if __name__ == "__main__":
    main()

