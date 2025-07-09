## Coding - each code should connect to at least one sentence; a sentence may have 0, 1, or multiple codes.
from langchain.prompts import PromptTemplate

custom_user_prompt = PromptTemplate(
    input_variables=["text", "n_codes", "user_prompt"],
    template="""
            You are an expert at thematic analysis. Read the transcript segment below. Extract key ideas and assign each a short code with a confidence score from 1 to 5. There must not be more than {n_codes} codes.
            
            Also pay careful attention to these instructions: {user_prompt}

            Each code must:
            - Correspond to at least one sentence
            - Be a concise phrase or a short sentence
            - Include a confidence score from 1 to 5

            Output format (exactly this):
            ===ANNOTATIONS START===
            1. <sentence> | <code> | <confidence>
            2. <sentence> | <code> | <confidence>
            ...

            Only use sentences that appear verbatim in the text. Do not add any explanations, headers, or extra text. Only return the numbered list in the format above.

            Transcript:
            {text}
            ---
            """
            )


code_prompt_hypothesis = PromptTemplate(
    input_variables=["text", "n_codes", "hypothesis"],
    template="""
You are an expert at thematic analysis. Read the transcript segment below. Extract key ideas and assign each a short code with a confidence score from 1 to 5. There must not be more than {n_codes} tags.
Specifically look for tags that are relevant for the hypothesis.

Each code must:
- Correspond to at least one sentence
- Be a concise phrase or a short sentence
- Include a confidence score from 1 to 5

Output format (exactly this):
===ANNOTATIONS START===
1. <sentence> | <code> | <confidence>
2. <sentence> | <code> | <confidence>
...

Only use sentences that appear verbatim in the text. Do not add any explanations, headers, or extra text. Only return the numbered list in the format above.

Hypothesis:
{hypothesis}

Transcript:
{text}
---
"""
)


code_prompt = PromptTemplate(
    input_variables=["text", "n_codes"],
    template="""
You are an expert at thematic analysis. Read the transcript segment below. Extract key ideas and assign each a short code with a confidence score from 1 to 5. There must not be more than {n_codes} tags.

Each code must:
- Correspond to at least one sentence
- Be a concise phrase or a short sentence
- Include a confidence score from 1 to 5

Output format (exactly this):
===ANNOTATIONS START===
1. <sentence> | <code> | <confidence>
2. <sentence> | <code> | <confidence>
...

Only use sentences that appear verbatim in the text. Do not add any explanations, headers, or extra text. Only return the numbered list in the format above.

Transcript:
{text}
---
"""
)

## Clustering - group the codes into common themes
cluster_prompt = PromptTemplate(
    input_variables=["codes", "max_themes"],
    template="""
You are an expert in thematic analysis. Group the following codes into no more than {max_themes} broader themes. A code may belong to more than one theme.

Codes:
{codes}

---

Return the results in this format:

Theme: <theme name>
Codes: code1 | code2 | code3

Theme: <theme name>
Codes: code4 | code5 | code6

Only output this list. Do not add explanations. Do not repeat codes within a theme.
"""
)



## Summarizing - provide a summary of each theme
summary_prompt = PromptTemplate(
    input_variables=["themes", "text"],
    template="""
You are an expert qualitative researcher analyzing texts.

Your task is to write a short (1–2 sentence) summary for each theme based on the codes corresponding to that theme. You may consult the text for reference.

Text:
{text}

Themes with codes:
{themes}

---
Output format (repeat for each theme):

**Theme**: <Theme Name>
**Summary**: <Short summary of what this theme captures>

Return only this structured output for each theme.
"""
)

chat_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
        You are a helpful chatbot. Please use the the following context to answer the question. 
        If there is no relevant information in the text to answer the question, the simply say that there is no relevant information in the text to answer that question.
        Be concise and exact and do not generate any unnecessary text and do not include any notes. 

        Context:
        {context}
        
        Question:
        {question}

        Return results in this format: 
        **Question**: <question>
        **Answer**: <answer>
        """
    )