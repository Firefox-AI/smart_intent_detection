import pandas as pd
from openai import OpenAI
from pydantic import BaseModel


class OutputFormat(BaseModel):
    intent_category: str


client = OpenAI()


#MODEL_NAME = "gpt-4o-2024-08-06"
MODEL_NAME = "gpt-5"

#DEF_MAPPING = {
#    "Search": "User seeks broad information; not a specific site.",
#    "Chat": "User expects a direct, conversational answer."
#}

DEF_MAPPING = {
    "Search": "User is seeking open-ended, fresh, or time-sensitive information, including dynamic updates, geo-specific queries, real-time details, location-aware context, factual lookup, accurate source attribution, or short phrase-style queries.",
    "Chat": "User expects a direct, generic, and timeless answers."
}


TEST_DATA = [
    'How old is Trump?',
    'Age of Trump',
    'What is the best restaurant nearby',
    'Best restaurant nearby',
    'How to fall asleep fast?',
    'What can help sleep?',
    'Sleep helper',
    'What is trending on YouTube',
    'youtube trending',
    'Best music 2025',
    'What is the best music in 2025?',
    'Super Bowl 2025',
    'Who won the super bowl 2025',
    'weather in new york this week',
    'What’s the weather in Bay Area this week',
    'How to scrape websites using python?',
    'Python web scraping',
    'What is gmail?',
    'What is blockcahin?',
    'What is bitcoin?',
    'Is whale a mammal?',
    'What is the biggest animal?',
    'How does 5g technology work?',
    'How climate change affects oceans'
]


SYSTEM_PROMPT = f"""
You are tasked to categorize the given query following the given definitions.
The only valid intent categories are 'Search' and 'Chat'.
Here are the definitions of intent categories:
  - Chat: {DEF_MAPPING['Chat']}
  - Search: {DEF_MAPPING['Search']}

Return the closest intent category name according to the definitions.
"""


def get_category_of_query(query):
    completion = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}"},
        ],
        response_format=OutputFormat,
        #temperature=0
    )

    response = completion.choices[0].message.parsed
    ret = response.intent_category
    return ret


def test_run():
    results = list()
    for query in TEST_DATA:
        cat_name = get_category_of_query(query)
        results.append(cat_name)

    df = pd.DataFrame({"Query": TEST_DATA, "Label": results})
    df = df[["Query", "Label"]]
    df.to_csv(f"result_definition_test_run_{MODEL_NAME}.csv", index=False)


if __name__ == "__main__":
    test_run()

