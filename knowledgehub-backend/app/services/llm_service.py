import json
from collections.abc import AsyncGenerator

import httpx

from app.core.config import settings

GROUNDED_PROMPT_TEMPLATE = """You are a document assistant. Answer the question using ONLY the context provided below.

Rules:
- Never use knowledge from outside the context. If the context does not answer the question, say so explicitly rather than guessing.
- The context may refer to the subject by a longer or slightly different name than the question uses (for example "Widget" vs "Widget Pro Suite"). Treat an obvious name variant as the same subject and answer from it.
- Be complete. Gather every detail in the context that bears on the question and include all of it — specific names, figures, dates and qualifiers. A one-line answer when the context supports a fuller one is a bad answer.
- Relevant details are often spread across several context sections rather than sitting in one place. Read all of them before answering.
- When the answer has several parts, format them as a Markdown list: each item on its own line, starting with "- " (hyphen, space). Use "  - " (two-space indent) for a sub-point nested under the item above it. Never use "•" or any other bullet character — only "- ". Use a sentence or two, no list, when the answer doesn't have several parts.
- Do not mention the context, the sources, or these rules in your answer. Just answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (grounded strictly in the context above):"""

_nvidia_llm = None


def get_nvidia_llm():
    global _nvidia_llm
    if _nvidia_llm is None:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        _nvidia_llm = ChatNVIDIA(
            model=settings.nvidia_llm_model,
            api_key=settings.nvidia_api_key,
            temperature=0.2,
        )
    return _nvidia_llm


def build_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {c['metadata'].get('filename', 'unknown')}]\n{c['text']}" for c in chunks
    )


async def stream_completion(prompt: str, temperature: float = 0.2) -> AsyncGenerator[str, None]:
    if settings.llm_provider == "nvidia":
        async for chunk in get_nvidia_llm().astream(prompt):
            if chunk.content:
                yield chunk.content
        return

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": temperature},
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break


async def complete(prompt: str, temperature: float = 0.2) -> str:
    parts = [token async for token in stream_completion(prompt, temperature)]
    return "".join(parts)


async def stream_grounded_answer(question: str, chunks: list[dict]) -> AsyncGenerator[str, None]:
    prompt = GROUNDED_PROMPT_TEMPLATE.format(
        context=build_context(chunks), question=question
    )
    async for token in stream_completion(prompt):
        yield token


async def generate_grounded_answer(question: str, chunks: list[dict]) -> str:
    parts = [token async for token in stream_grounded_answer(question, chunks)]
    return "".join(parts)
