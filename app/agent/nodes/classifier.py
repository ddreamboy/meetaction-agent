import time

from loguru import logger

from app.agent.state import AgentState
from app.config import settings
from app.llm.client import llm_structured_call
from app.observability.token_counter import TokenCounter
from app.schemas.common import SClassifierResult

SYSTEM_PROMPT = """\
You are a meeting classifier. Analyze the transcript and classify the meeting type.

Types:
- "work_meeting": operational meeting with tasks, decisions, assignments, deadlines
- "consultation": advisory/educational session with insights, recommendations, knowledge sharing
"""


async def classifier_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "")
    logger.info("Classifier node | session_id={}", session_id)

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
            response_model=SClassifierResult,
            temperature=0.0,
            node="classifier",
            session_id=session_id,
        )
        TokenCounter(session_id).add(input_tok, output_tok, "classifier")

        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Classifier done | session_id={} type={} confidence={} latency_ms={}",
            session_id,
            result.meeting_type.value,
            result.confidence,
            latency_ms,
        )
        return {**state, "meeting_type": result.meeting_type.value}

    except Exception as exc:
        logger.warning(
            "Classifier failed, fallback to default | session_id={} error={}",
            session_id,
            str(exc),
        )
        return {**state, "meeting_type": settings.CLASSIFIER_DEFAULT}
