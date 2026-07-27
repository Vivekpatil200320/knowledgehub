import logging

from app.models.orm import Message
from app.services.llm_service import complete

logger = logging.getLogger("knowledgehub.condensation")

CONDENSE_PROMPT_TEMPLATE = """Rewrite the follow-up message as a standalone question that can be understood without the conversation history.

Rules:
1. If the follow-up omits its subject, fill it in from the history. ("What about pricing?" after discussing Product X becomes "What is the pricing of Product X?")
2. If the follow-up names its OWN subject, keep that subject and do NOT attach the previous topic to it. Never claim a relationship between two subjects that the history did not state. ("What about Product Y?" after discussing Product X becomes "What about Product Y?" — never "Product Y, part of Product X".)
3. If the follow-up is already standalone, return it unchanged.
4. Add no facts that are not present in the history or the follow-up.

Do NOT answer the question. Return ONLY the rewritten question, with no preamble or explanation.

CONVERSATION HISTORY:
{history}

FOLLOW-UP MESSAGE:
{question}

STANDALONE QUESTION:"""


def format_history(messages: list[Message]) -> str:
    return "\n".join(f"{m.role.capitalize()}: {m.content}" for m in messages)


async def condense(history: list[Message], question: str) -> str:
    """Rewrite a follow-up into a self-contained retrieval query.

    Hand-rolled rather than using LangChain's ConversationalRetrievalChain so the condensed
    query stays an inspectable artifact we can persist and assert on in evals.
    """
    if not history:
        return question

    prompt = CONDENSE_PROMPT_TEMPLATE.format(
        history=format_history(history), question=question
    )
    try:
        condensed = (await complete(prompt, temperature=0.0)).strip().strip('"')
    except Exception:
        logger.exception("Condensation failed, falling back to the raw question")
        return question

    return condensed or question
