from openai import OpenAI
from pydantic import BaseModel

from prompts import SYSTEM_LABEL_PROMPT, SYSTEM_BATCH_LABEL_PROMPT, SYSTEM_TRANSLATE_PROMPT


class LabelFormat(BaseModel):
    intent_category: str


class BatchLabelFormat(BaseModel):
    intent_category_list: list[str]


class GenerationFormat(BaseModel):
    query_list: list[str]

class TranslationFormat(BaseModel):
    translation: str


client = OpenAI()
MODEL_NAME = "gpt-5"


def get_category_of_batched_queries(batched_query):
    num_retry = 0
    while True:
        completion = client.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_BATCH_LABEL_PROMPT},
                {"role": "user", "content": f"Queries: {batched_query}"},
            ],
            response_format=BatchLabelFormat,
            #temperature=0
        )

        response = completion.choices[0].message.parsed
        ret = response.intent_category_list
        if len(ret) != len(batched_query):
            num_retry += 1
            print(f"Number of result mismatched, retrying... number of retry = {num_retry}", flush=True)
            continue

    return ret


def get_category_of_query(query):
    completion = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_LABEL_PROMPT},
            {"role": "user", "content": f"Query: {query}"},
        ],
        response_format=LabelFormat,
        #temperature=0
    )

    response = completion.choices[0].message.parsed
    ret = response.intent_category

    return ret



def generate_queries(args_list):
    system_prompt, number_of_samples = args_list
    completion = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate {number_of_samples} distinct quries."},
        ],
        response_format=GenerationFormat,
        #temperature=0
    )

    response = completion.choices[0].message.parsed
    return response.query_list


def translate_query(query, lang):
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
    ret = ret.strip()

    return ret

