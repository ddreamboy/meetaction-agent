import time

from loguru import logger

from app.agent.state import AgentState
from app.llm.client import llm_structured_call
from app.observability.token_counter import TokenCounter
from app.schemas.extraction import SConsultationGraph

SYSTEM_PROMPT = """\
You are a consultation analysis assistant. Extract key insights from the consultation transcript

IMPORTANT: Write all text fields in the SAME LANGUAGE as the transcript, do not translate

For each insight identify:
- topic: the subject being discussed
- insight: the key knowledge or finding
- recommendation: actionable recommendation (if any, otherwise omit)
- context_quote: verbatim quote (1–2 sentences) from the transcript that is the source of this insight
"""


async def consultation_extractor_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "")
    logger.info("Consultation extractor node | session_id={}", session_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"<transcript>\n{state['transcript_clean']}\n</transcript>",
        },
    ]

    t0 = time.monotonic()
    try:
        result, input_tok, output_tok = await llm_structured_call(
            messages=messages,
            response_model=SConsultationGraph,
            temperature=0.0,
            node="consultation_extractor",
            session_id=session_id,
        )
        TokenCounter(session_id).add(input_tok, output_tok, "consultation_extractor")

        items = [item.model_dump() for item in result.items]
        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Consultation extractor done | session_id={} items={} latency_ms={}",
            session_id,
            len(items),
            latency_ms,
        )
        return {**state, "knowledge_graph": items, "error_message": None}

    except Exception as exc:
        logger.error(
            "Consultation extractor failed | session_id={} error={}",
            session_id,
            str(exc),
        )
        return {**state, "error_message": "Не удалось извлечь инсайты из транскрипта"}
