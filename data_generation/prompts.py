
SYSTEM_PROMPT_GENERAL = f"""
You are tasked to generate realistic web search queries users would type in web browser address bars.
Follow these rules strictly:
- Natural browser style; mix short and longer queries, not formal prose.
- No placeholders; use real cities, landmarks, companies, tools, venues, etc. if needed
- Replace generic words (e.g., “brands”, “companies”, “providers”, “resources”) with real names.
- All words must be complete; no fragments.
- Avoid generic filler like “what are the pros of…”.
- Output only queries; no commentary.
"""


SYSTEM_PROMPT_SHORT = f"""
You are tasked to generate realistic web search queries users would type in web browser address bars.
Follow these rules strictly:
- Natural browser style; only short queries in less than 4 words, not formal prose.
- No placeholders; use real cities, landmarks, companies, tools, venues, etc. if needed
- Output only queries; no commentary.

Example of output:
query_list: ["restaurants near me", "what is bitcoin"]
"""


SYSTEM_PROMPT_PHRASE = f"""
You are tasked to generate realistic web search queries users would type in web browser address bars.
Follow these rules strictly:
- Natural browser style; only phrases or single words, no complete sentences, not formal prose.
- Output only queries; no commentary.

Example of output:
query_list: ["blue skirts", "ramen"]
"""


SYSTEM_PROMPT_TIME_SENSITIVE = f"""
You are tasked to generate realistic web search queries users would type in web browser address bars.
Follow these rules strictly:
- Time-sensitive queries like asking stock price, sports scores or breaking news, etc.
- Natural browser style; mix short and longer queries, not formal prose.
- No placeholders; use real cities, landmarks, companies, tools, venues, etc. if needed
- Replace generic words (e.g., “brands”, “companies”, “providers”, “resources”) with real names.
- All words must be complete; no fragments.
- Avoid generic filler like “what are the pros of…”.
- Output only queries; no commentary.

Example of output:
query_list: ["Price of Tesla", "Tsla price", "top news", "Who won Super Bowl 2025"]
"""


SYSTEM_PROMPT_DYNAMIC = f"""
You are tasked to generate realistic web search queries users would type in web browser address bars.
Follow these rules strictly:
- The query is about dynamic information like the open time of stores, or the weather tomorrow, etc.
- Natural browser style; mix short and longer queries, not formal prose.
- No placeholders; use real cities, landmarks, companies, tools, venues, etc. if needed
- Replace generic words (e.g., “brands”, “companies”, “providers”, “resources”) with real names.
- All words must be complete; no fragments.
- Avoid generic filler like “what are the pros of…”.
- Output only queries; no commentary.

Example of output:
query_list: ["Is Safeway open right now?", "Weather tomorrow"]
"""


SYSTEM_PROMPT_MISSED_URL = f"""
You are tasked to generate realistic web search queries users would type in web browser address bars.
Follow these rules strictly:
- The query includes a URL.
- There can be additional words except for the URL in the query.
- The URL can include its own parameters and it can be either short and long.
- The URL might not be complete to navigate.
- Output only queries; no commentary.

Example of output:
query_list: ["go to https://en.wikipedia.org/wiki/Final_Destination_Bloodlines", "site:Mcdonald's", "localhost:8001"]
"""


DEF_MAPPING = {
    "Search": "User is seeking URL navigation, open-ended, fresh, or time-sensitive information, including dynamic updates, geo-specific queries, real-time details, location-aware context, accurate source attribution, or short phrase-style queries.",
    "Chat": "User expects a direct, generic, and timeless answers."
}


DEF_CHAT_AI_COMMON = {
    "ai_tool_common_query_explanation": "The user asks the AI to define, describe, or explain a specific concept or term.",
    "ai_tool_common_query_summarization": "The user asks the AI to summarize text, content, or ideas concisely.",
    "ai_tool_common_query_translation": "The user asks the AI to translate words, phrases, or sentences between languages.",
    "ai_tool_common_query_coding": "The user asks the AI to write, explain, or fix code in a programming language.",
    "ai_tool_common_query_assistance": "The user asks for help drafting, rewriting, or editing written content such as emails or reports.",
    "ai_tool_common_query_writing": "The user asks the AI to create imaginative or artistic text such as poems, stories, or scripts.",
    "ai_tool_common_query_math": "The user asks the AI to solve math problems, perform calculations, or explain mathematical reasoning.",
    "ai_tool_common_query_comparison": "The user asks the AI to compare two or more items, concepts, or ideas.",
    "ai_tool_common_query_rephrase": "The user asks the AI to rephrase, simplify, or restyle text while keeping the meaning.",
    "ai_tool_common_query_roleplay": "The user asks the AI to act as a particular persona, role, or expert in a simulated conversation.",
    "ai_tool_common_query_analysis": "The user asks the AI to analyze or extract insights from text, lists, or short data snippets.",
    "ai_tool_common_query_learning": "The user asks the AI to teach or explain a topic step-by-step for learning purposes.",
    "ai_tool_common_query_task_management": "The user asks the AI to plan, organize, or assist with personal tasks or productivity.",
    "ai_tool_common_query_discussion": "The user asks for the AI’s perspective, thoughts, or balanced discussion on a topic.",
    "ai_tool_common_query_casual_chat": "The user engages the AI in small talk, jokes, or lighthearted conversation.",
    "ai_tool_common_query_system_setting": "The user gives direct commands or requests related to system actions or assistant behavior."
}


SYSTEM_PROMPT_CHAT_AI_COMMON = """
You are tasked to generate common queries users would type in AI tools like chatgpt.
Follow these rules strictly:
- User's need: {}
- The query is expecting a direct, generic, and timeless answers.
- The query should be common conversation starter sentences for AI tools.
- The query can be just a command verb like "help" or "summarize" as long as it matches the above user's need
- The query can be imperative format like "write me a poem" as long as it matches the above user's need
- The query can be a complete sentence like "Hello there, how are you?" as long as it matches the above user's need
- DO NOT generate queries that are seeking URL navigation, fresh, or time-sensitive information.
- DO NOT include dynamic updates, geo-specific info, real-time details, location-aware context.
- The query should not expect accurate source attribution.
- Output only queries; no commentary.
- DO NOT use the given examples below directly.

Example of output:
query_list: ["Hello there, how are you?", "rewrite email professional tone", "Do you know if there are any day-offs next month?"]
"""


SYSTEM_PROMPT_CHAT = f"""
You are tasked to generate realistic web search queries users would type in web browser address bars.
Follow these rules strictly:
- The query is expecting a direct, generic, and timeless answers.
- DO NOT generate queries that are seeking URL navigation, fresh, or time-sensitive information.
- DO NOT include dynamic updates, geo-specific info, real-time details, location-aware context.
- The query should not expect accurate source attribution.
- The query cannot be short phrases.
- Output only queries; no commentary.

Example of output:
query_list: ["How to export chats from WhatsApp iOS to Android?", "Show me some algorithms for finding shortest path.", "Are Beyond Meat patties healthy?"]
"""


SYSTEM_PROMPT_CHAT_IMPERATIVE = f"""
You are tasked to generate realistic web search queries users would type in web browser address bars.
Follow these rules strictly:
- The query is expecting a direct, generic, and timeless answers.
- The user wants a constructed response or assistance (writing, explaining, summarizing, coding help, brainstorming, recommendations with reasoning).
- Imperatives are task-style: Write, Summarize, Explain, Help, Create, Rewrite, Brainstorm, Compare and recommend, Debug, Optimize, Outline, Generate, Give feedback, Make, Walk through.
- The query can be just a command verb like "help" or "summarize"
- DO NOT generate queries that are seeking URL navigation, fresh, or time-sensitive information.
- DO NOT include dynamic updates, geo-specific info, real-time details, location-aware context.
- The query should not expect accurate source attribution.
- Output only queries; no commentary.
- DO NOT use the given examples below directly.

Example of output:
query_list: ["simplify technical documentation", "rewrite email professional tone", "compare iPhone Samsung"]
"""


SYSTEM_BATCH_LABEL_PROMPT = f"""
You are tasked to categorize the given queries in a list following the given definitions.
The only valid intent categories are 'Search' and 'Chat'.
Here are the definitions of intent categories:
  - Chat: {DEF_MAPPING['Chat']}
  - Search: {DEF_MAPPING['Search']}

Return the closest intent category name for each query according to the definitions in the list respectively.
Note that a complete sentence can be either 'Chat' or 'Search'

Output format:
    intent_category_list: a list of intent type 'Chat' or 'Search'
"""


SYSTEM_LABEL_PROMPT = f"""
You are tasked to categorize a query following the given definitions.
The only valid intent categories are 'Search' and 'Chat'.
Here are the definitions of intent categories:
  - Chat: {DEF_MAPPING['Chat']}.
  - Search: {DEF_MAPPING['Search']}

Critical rule:
- If the query refers to existing browser context (e.g., "my tabs", "these tabs", "open tabs", "@mentions"),
  classify as 'Chat'.

Examples:
- "compare @nike @adidas" → Chat
- "compare my tabs about shoes" → Chat
- "which of these tabs is best" → Chat

- "compare nike vs adidas" → Search
- "best running shoes nike vs adidas" → Search

Return the closest intent category name according to the definitions.
Note that a complete sentence can be either 'Chat' or 'Search'

Output format:
    intent_category: string, intent type 'Chat' or 'Search'
"""


# Categorize tab comparison queries as chat
SYSTEM_PROMPT_TAB_COMPARISON = f"""
You are tasked to generate realistic user queries for an intent classifier.

Focus on comparison-style queries involving browser tabs.

Follow these rules strictly:

- Generate queries that involve comparing information.
- Mix two types of queries:
  1. Queries that refer to existing browser context (these should be Chat intent)
  2. Queries that do NOT refer to browser context (these should be Search intent)

- For Chat-style queries:
  - Include phrases like:
    - "my tabs"
    - "these tabs"
    - "open tabs"
    - "selected tabs"
    - "@mentions" (e.g. "@amazon", "@nike")
  - These queries imply the assistant should use existing context.

- For Search-style queries:
  - DO NOT include tab references or @mentions
  - These should look like normal web searches.

- Natural browser style; mix short and longer queries, not formal prose.
- No placeholders; use real brands, websites, or topics when appropriate.
- Avoid generic filler like “what are the pros of…”.
- Output only queries; no commentary.

Examples:

query_list: [
  "compare @amazon @bestbuy",
  "compare my tabs about running shoes",
  "which of these tabs is cheaper for nike shoes",
  "compare amazon vs bestbuy prices",
  "nike vs adidas running shoes comparison",
  "compare these tabs about laptops",
  "best running shoes nike vs adidas"
]
"""

SYSTEM_TRANSLATE_PROMPT = """
You are tasked to translate english browser queries into {}
Example:
  Query: Hello world
  translation: Bonjour le monde
"""
