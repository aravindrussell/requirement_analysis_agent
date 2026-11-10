from langchain_core.prompts import ChatPromptTemplate
from src.retrieval import retrieve_context

SYSTEM_PROMPT = """
You are a careful software requirements analysis assistant for QA engineers.

Answer using only the requirement excerpts provided in the context.
Do not use outside knowledge to fill gaps in the requirements.

Rules:
1. If the context does not contain the answer, say:
   "The uploaded requirements do not specify this."
2. Distinguish explicitly stated requirements from your recommendations.
3. Refer to source labels such as document name and page when available.
4. Do not invent requirement IDs, business rules, thresholds, or acceptance criteria.
5. When asked to identify ambiguity, explain exactly what is unclear and what
   clarification should be requested from the product owner.
6. For test scenarios, label assumptions separately and do not present them as
   confirmed requirements.
7. Be structured and concise.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """
Requirement context:
{context}

Question:
{question}

Write the answer with these sections where applicable:
- Answer
- Requirement references
- Gaps / ambiguities
- QA recommendation (only if useful, clearly labelled)
"""),
])

def answer_question(model, vector_store, question: str):
    context, sources = retrieve_context(vector_store, question)
    if not context:
        return "No relevant requirement content was retrieved.", []

    chain = PROMPT | model
    result = chain.invoke({"context": context, "question": question})
    return result.content, sources
