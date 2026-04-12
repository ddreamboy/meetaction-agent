import asyncio
from functools import lru_cache
from typing import TypeVar

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.config import settings

T = TypeVar("T", bound=BaseModel)


@lru_cache()
def get_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        timeout=settings.LLM_TIMEOUT,
    )


async def llm_call_with_retry(
    messages: list[dict],
    temperature: float,
    node: str,
    session_id: str,
) -> tuple[str, int, int]:
    """
    Выполняет LLM-вызов с exponential backoff retry

    Returns:
        (content, input_tokens, output_tokens)
    """
    client = get_llm_client()
    last_exc: Exception | None = None

    for attempt in range(settings.LLM_RETRY_COUNT + 1):
        try:
            response = await client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            return content, input_tokens, output_tokens
        except Exception as exc:
            last_exc = exc
            if attempt < settings.LLM_RETRY_COUNT:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "LLM retry | session_id={} node={} attempt={} wait={}s error={}",
                    session_id,
                    node,
                    attempt + 1,
                    wait,
                    type(exc).__name__,
                )
                await asyncio.sleep(wait)

    logger.error(
        "LLM failed after retries | session_id={} node={} error={}",
        session_id,
        node,
        str(last_exc),
    )
    raise last_exc


async def llm_structured_call(
    messages: list[dict],
    response_model: type[T],
    temperature: float,
    node: str,
    session_id: str,
) -> tuple[T, int, int]:
    """
    Выполняет LLM-вызов с exponential backoff retry и структурированным выводом response_model

    Returns:
        (parsed_model_instance, input_tokens, output_tokens)
    """
    client = get_llm_client()
    last_exc: Exception | None = None

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": response_model.model_json_schema(),
        },
    }

    for attempt in range(settings.LLM_RETRY_COUNT + 1):
        try:
            response = await client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
            content = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            parsed = response_model.model_validate_json(content)
            return parsed, input_tokens, output_tokens
        except Exception as exc:
            last_exc = exc
            if attempt < settings.LLM_RETRY_COUNT:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "LLM structured retry | session_id={} node={} attempt={} wait={}s error={}",
                    session_id,
                    node,
                    attempt + 1,
                    wait,
                    type(exc).__name__,
                )
                await asyncio.sleep(wait)

    logger.error(
        "LLM structured failed after retries | session_id={} node={} error={}",
        session_id,
        node,
        str(last_exc),
    )
    raise last_exc


async def get_embedding(text: str) -> list[float]:
    client = get_llm_client()
    response = await client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding
