
from langchain_core.prompts import ChatPromptTemplate

from src.retrieval import retrieve_context


TEST_DESIGN_SYSTEM_PROMPT = """
You are a Senior Software Test Architect and QA Test Design specialist.

Your task is to generate software test scenarios, detailed test cases,
acceptance criteria, and requirement quality observations using ONLY
the retrieved requirement context.

IMPORTANT RULES:

1. Never invent requirement IDs, business rules, validations, thresholds,
   workflows, or expected behavior.
2. If the source has no requirement ID, use a generated reference such as
   "SOURCE-1" based on the retrieved source label. Do not pretend it is an
   official requirement ID.
3. Clearly distinguish:
   - Explicitly stated requirements
   - Reasonable QA-derived test ideas
   - Assumptions requiring product-owner confirmation
4. Do not present an assumption as an approved expected result.
5. If an expected result is not specified, write:
   "Not specified in the uploaded requirements - clarification required."
6. Every test scenario and test case must reference its source label.
7. Include positive, negative, boundary, and validation scenarios where
   supported by the available requirements.
8. Do not create artificial boundary values when the requirement does not
   define a threshold. Identify the missing threshold instead.
9. Acceptance criteria must be measurable and testable.
10. Use Given / When / Then format for acceptance criteria.
11. Identify missing, conflicting, ambiguous, and incomplete requirements.
12. Be precise, structured, and useful to a professional QA team.

Do not claim that generated test cases have been executed.
"""


TEST_DESIGN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TEST_DESIGN_SYSTEM_PROMPT),
    ("human", """
Generate a QA test design from the following retrieved requirement context.

RETRIEVED REQUIREMENT CONTEXT:
{context}

Prepare the output using these sections:

# 1. Requirement Understanding
Summarize the explicit requirements found in the retrieved context.
Include source labels.

# 2. Functional Test Scenarios
Create a table with:
- Scenario ID (GEN-TS-001 format)
- Scenario description
- Test type (Positive / Negative / Boundary / Validation)
- Requirement source
- Classification (Explicitly supported / QA-derived)

# 3. Detailed Test Cases
Create a table with:
- Test Case ID (GEN-TC-001 format)
- Scenario ID
- Preconditions
- Test steps
- Test data
- Expected result
- Requirement source
- Classification

Do not invent expected results or test data.
If something is unspecified, explicitly say so.

# 4. Acceptance Criteria
Create measurable Given / When / Then criteria.
For each criterion, identify the supporting source.
If the requirement is insufficiently defined, explain what is missing.

# 5. Requirement Gaps and Ambiguities
For each gap, provide:
- Missing or ambiguous information
- Why it matters for testing
- A specific clarification question for the product owner

# 6. QA Recommendations
List additional recommendations separately from confirmed requirements.

Keep the output practical and suitable for review by a QA lead or Test Architect.
"""),
])


def generate_test_design(model, vector_store, k: int = 8):
    """
    Retrieve relevant requirements and generate a structured QA test design.

    Returns:
        answer: Generated test design in Markdown.
        sources: Retrieved source labels and excerpts.
    """

    retrieval_query = """
    Retrieve all available functional requirements, business rules,
    validations, input constraints, error handling, thresholds,
    workflows, user actions, session rules, acceptance criteria,
    and requirement dependencies from the uploaded documents.
    """

    context, sources = retrieve_context(
        vector_store,
        retrieval_query,
        k=k,
    )

    if not context:
        return (
            "No relevant requirement content was retrieved. "
            "Please upload and process requirement documents first.",
            [],
        )

    chain = TEST_DESIGN_PROMPT | model

    result = chain.invoke({
        "context": context,
    })

    answer = result.content

    if not isinstance(answer, str):
        answer = str(answer)

    return answer, sources
