import logging

from app.models.orm import Message
from app.services.llm_service import complete

logger = logging.getLogger("knowledgehub.condensation")

CONDENSE_PROMPT_TEMPLATE = """Given the conversation history and a follow-up message, rewrite the follow-up as a standalone question that contains all the context needed to answer it on its own. Resolve any pronouns or implied subjects using the history. If the follow-up is already standalone, return it unchanged.

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
